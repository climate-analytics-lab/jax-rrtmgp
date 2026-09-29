# Copyright 2024 The swirl_jatmos Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Longwave heating of a model-top layer colder than RRTMGP's tables (#39).

RRTMGP's gas-optics tables start at 160 K. A model-top layer near 1 Pa in a
cold mesosphere can radiate itself down to and through that edge. When values
below a table were mirrored back into it, the layer's emission grew as it
cooled below 160 K, so its longwave cooling was V-shaped in its own
temperature with the minimum exactly at the table edge, and a column left to
itself ran away (jax-gcm #920: 35 K, then NaN). This test holds a realistic
column fixed, sweeps only the top layer's temperature from 200 K down to 100 K,
and checks that the layer's longwave heating has no such V.
"""

import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
from rrtmgp import constants
from rrtmgp import kernel_ops
from rrtmgp.config import radiative_transfer
from rrtmgp.optics import atmospheric_state
from rrtmgp.optics import optics
from rrtmgp.rte import two_stream

root = Path()
_ATMOSPHERIC_STATE_FILEPATH = root / 'rrtmgp/optics/test_data/clearsky_as.nc'
_VMR_GLOBAL_MEAN_FILEPATH = root / 'rrtmgp/optics/test_data/vmr_global_means.json'

_SECONDS_PER_DAY = 86400.0
_CP_DRY_AIR = 1004.64
_T_TABLE_MIN = 160.0


def _top_layer_lw_heating(temps: np.ndarray) -> np.ndarray:
  """LW heating [K/day] of the top layer, one column per swept temperature.

  The column is RFMIP site 0 (experiment 0) with one layer added on top of it,
  spanning 2 Pa to the RFMIP top level and centred at 1 Pa, like the top layer
  of the jax-gcm column in #920. That centre is just below RRTMGP's lowest
  reference pressure (1.005 Pa), so the pressure axis is (slightly) outside
  its table as well.
  """
  cfg = radiative_transfer.RadiativeTransfer(
      optics=radiative_transfer.OpticsParameters(
          optics=radiative_transfer.RRTMOptics(
              longwave_nc_filepath='rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g128.nc',
              shortwave_nc_filepath='rrtmgp/optics/rrtmgp_data/rrtmgp-gas-sw-g112.nc',
              cloud_longwave_nc_filepath='rrtmgp/optics/rrtmgp_data/cloudysky_lw.nc',
              cloud_shortwave_nc_filepath='rrtmgp/optics/rrtmgp_data/cloudysky_sw.nc',
          )
      ),
      atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
          sfc_emis=1.0,
          sfc_alb=0.2,
          zenith=0.0,
          irrad=0.0,
          toa_flux_lw=0.0,
          vmr_global_mean_filepath=_VMR_GLOBAL_MEAN_FILEPATH,
      ),
  )
  atm = atmospheric_state.from_config(cfg.atmospheric_state_cfg)
  optics_lib = optics.optics_factory(cfg.optics, atm.vmr)

  site = 0
  with nc.Dataset(_ATMOSPHERIC_STATE_FILEPATH, 'r') as ds:
    # The file is top first; the solver wants surface first.
    p_lay = np.flip(ds['pres_layer'][:].data[site], -1)
    p_lev = np.flip(ds['pres_level'][:].data[site], -1)
    t_lay = np.flip(ds['temp_layer'][:].data[0, site], -1)
    h2o = np.flip(ds['water_vapor'][:].data[0, site], -1)
    o3 = np.flip(ds['ozone'][:].data[0, site], -1)
    sfc_t = float(ds['surface_temperature'][:].data[0, site])

  p_lev = np.concatenate([p_lev[:-1], [2.0, p_lev[-1]]])
  p_lay = np.concatenate([p_lay, [1.0]])
  t_lay = np.concatenate([t_lay, [t_lay[-1]]])
  h2o = np.concatenate([h2o, [h2o[-1]]])
  o3 = np.concatenate([o3, [o3[-1]]])

  n = temps.size

  def column(f):
    # One halo cell at each end, one column per swept temperature.
    return np.broadcast_to(
        np.pad(f, (1, 1), mode='edge'), (n, 1, f.size + 2)
    ).copy()

  p = column(p_lay)
  p_face = np.broadcast_to(
      np.pad(p_lev, (1, 0), mode='edge'), (n, 1, p_lev.size + 1)
  ).copy()
  t = column(t_lay)
  t[:, 0, 0] = sfc_t
  t[:, 0, -2] = temps  # The top interior layer.
  t[:, 0, -1] = temps  # Its halo; the solver imposes a Neumann BC there.
  h = column(h2o)
  oz = column(o3)

  dtype = jnp.float_
  p, p_face, t, h, oz = (
      jnp.asarray(x, dtype=dtype) for x in (p, p_face, t, h, oz)
  )
  dp = kernel_ops.forward_difference(p_face, dim=2)
  molecules = -(dp / constants.G) * constants.AVOGADRO / (
      constants.DRY_AIR_MOL_MASS + constants.WATER_MOL_MASS * h
  )
  lw = jax.jit(
      lambda t: two_stream.solve_lw(
          p, t, molecules, optics_lib, atm, {'h2o': h, 'o3': oz},
          sfc_t * jnp.ones((n, 1), dtype=dtype),
      )
  )(t)
  # flux[..., k] is at the bottom face of cell k: the top layer is bounded by
  # faces -2 (bottom) and -1 (top of atmosphere).
  f_net = np.asarray(lw['flux_net'], np.float64)[:, 0]
  p_face = np.asarray(p_face, np.float64)[:, 0]
  return (
      constants.G / _CP_DRY_AIR
      * (f_net[:, -1] - f_net[:, -2]) / (p_face[:, -1] - p_face[:, -2])
      * _SECONDS_PER_DAY
  )


class ColdTopLayerTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.temps = np.arange(200.0, 99.0, -2.5)
    cls.heating = _top_layer_lw_heating(cls.temps)

  def _at(self, temperature):
    return self.heating[int(np.argmin(np.abs(self.temps - temperature)))]

  def test_heating_is_finite(self):
    self.assertTrue(np.all(np.isfinite(self.heating)))

  def test_heating_monotone_through_the_table_edge(self):
    """Heating rises steadily as the layer cools from 200 K through 160 K.

    Checked down to 145 K, one reference-temperature step below the edge.
    With the mirrored table the heating peaked just above 160 K and then fell
    again (for this column: -3.2 K/day at 165 K, -4.1 at 160, -4.9 at 155).
    """
    window = self.temps >= _T_TABLE_MIN - 15.0
    steps = np.diff(self.heating[window])  # temps are decreasing
    self.assertTrue(
        np.all(steps > 0),
        f'non-monotone heating: {dict(zip(self.temps, self.heating))}',
    )

  def test_no_v_below_the_table(self):
    """Below 160 K the layer never cools faster than it does at 160 K.

    The mirrored table made the cooling grow again below the edge, reaching
    -15.0 K/day at 100 K against -4.1 K/day at 160 K for this column.
    """
    below = self.heating[self.temps < _T_TABLE_MIN]
    self.assertTrue(np.all(below >= self._at(_T_TABLE_MIN)))

  def test_colder_layer_is_restored(self):
    """Well below the table the net LW effect on the layer is warming.

    This is the restoring force that stops a radiatively cooling top layer:
    the extrapolated Planck source falls with temperature, so absorption of
    the upwelling radiation from below wins.
    """
    self.assertGreater(self._at(120.0), 0.0)
    self.assertGreater(self._at(100.0), 0.0)
    self.assertLess(self._at(200.0), 0.0)


if __name__ == '__main__':
  unittest.main()
