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

"""End-to-end tests for `RRTMGP.compute_heating_rate`."""

import dataclasses
import functools
from pathlib import Path
from typing import TypeAlias

import unittest
import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
from rrtmgp import constants
from rrtmgp import rrtmgp
from rrtmgp import rrtmgp_common
from rrtmgp import test_util
from rrtmgp.config import radiative_transfer
from rrtmgp.optics import lookup_volume_mixing_ratio

Array: TypeAlias = jax.Array

_VMR_GLOBAL_MEAN_FILENAME = 'rrtmgp/optics/test_data/rcemip_global_mean_vmr.json'
_VMR_SOUNDING_FILENAME = 'rrtmgp/optics/test_data/rcemip_vmr_sounding.csv'
_ATMOSPHERIC_STATE_FILENAME = 'rrtmgp/optics/test_data/cloudysky_as.nc'
_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g128.nc'
_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-sw-g112.nc'
_CLD_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/cloudysky_lw.nc'
_CLD_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/cloudysky_sw.nc'

root = Path()


def _build_radiative_transfer_cfg() -> radiative_transfer.RadiativeTransfer:
  """Build a `RadiativeTransfer` config wired to the RCEMIP test data."""
  return radiative_transfer.RadiativeTransfer(
      optics=radiative_transfer.OpticsParameters(
          optics=radiative_transfer.RRTMOptics(
              longwave_nc_filepath=root / _LW_LOOKUP_TABLE_FILENAME,
              shortwave_nc_filepath=root / _SW_LOOKUP_TABLE_FILENAME,
              cloud_longwave_nc_filepath=root / _CLD_LW_LOOKUP_TABLE_FILENAME,
              cloud_shortwave_nc_filepath=root / _CLD_SW_LOOKUP_TABLE_FILENAME,
          )
      ),
      atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
          sfc_emis=0.98,
          sfc_alb=0.06,
          zenith=0.535526654,
          irrad=1360.8585174,
          toa_flux_lw=0.0,
          vmr_global_mean_filepath=root / _VMR_GLOBAL_MEAN_FILENAME,
          vmr_sounding_filepath=root / _VMR_SOUNDING_FILENAME,
      ),
  )


def _build_inputs(
    site: int = 0, n_horiz: int = 2
) -> dict[str, Array]:
  """Build a 3D input bundle for `RRTMGP.compute_heating_rate` from
  `cloudysky_as.nc`. Cloud condensate is zeroed so the test isolates the
  gas-VMR pathway.
  """
  ds = nc.Dataset(root / _ATMOSPHERIC_STATE_FILENAME, 'r')

  halo_width = 1
  paddings_2d = ((0, 0), (halo_width, halo_width))
  pressure = np.pad(
      np.transpose(ds['p_lay'][:].data), paddings_2d, mode='edge'
  )

  temp_internal = np.transpose(ds['t_lay'][:].data)
  temp_level = np.transpose(ds['t_lev'][:].data)
  nx, nz = temp_internal.shape
  nz_with_halos = nz + 2 * halo_width
  temperature = np.zeros((nx, nz_with_halos), dtype=jnp.float_)
  temperature[:, halo_width:-halo_width] = temp_internal
  temperature[:, 0] = 2 * temp_level[:, 0] - temp_internal[:, 0]
  temperature[:, -1] = 2 * temp_level[:, -1] - temp_internal[:, -1]

  vmr_h2o_profile = np.pad(
      np.transpose(ds['vmr_h2o'][:].data), paddings_2d, mode='edge'
  )
  sfc_temp_1d = ds['t_sfc'][:].data

  convert_to_3d = functools.partial(
      test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=n_horiz
  )
  p_ref_xxc = convert_to_3d(pressure[site, :])
  temperature_3d = convert_to_3d(temperature[site, :])
  vmr_h2o = convert_to_3d(vmr_h2o_profile[site, :])

  # Invert _humidity_to_volume_mixing_ratio to back out a consistent q_t (with
  # q_c = 0 since this is a clear-sky setup). vmr_h2o = mol_ratio * q_v / (1 -
  # q_t) with q_v = q_t, so q_t = vmr_h2o / (vmr_h2o + mol_ratio).
  mol_ratio = constants.R_V / constants.R_D
  q_t = vmr_h2o / (vmr_h2o + mol_ratio)
  zeros = jnp.zeros_like(q_t)

  # Density via ideal gas law, used for the cloud-path scaling (zeros here, so
  # the actual value is irrelevant — keep it physically sensible).
  rho = p_ref_xxc / (constants.R_D * temperature_3d)

  sfc_temperature = sfc_temp_1d[site] * jnp.ones(
      (n_horiz, n_horiz), dtype=jnp.float_
  )
  return {
      'rho_xxc': rho,
      'q_t': q_t,
      'q_liq': zeros,
      'q_ice': zeros,
      'q_c': zeros,
      'cloud_r_eff_liq': zeros,
      'cloud_r_eff_ice': zeros,
      'temperature': temperature_3d,
      'sfc_temperature': sfc_temperature,
      'p_ref_xxc': p_ref_xxc,
      'sg_map': {},
  }


class RRTMGPVMROverrideTest(unittest.TestCase):
  """Regression test for the `vmr_fields` override kwarg.

  Mirrors the failure mode reported in climate-analytics-lab/jax-gcm#483:
  before this kwarg existed, an upstream-supplied per-cell ozone profile was
  silently ignored and the library's pressure-reconstructed default was used
  instead. This test perturbs ozone by 10x via the new kwarg and asserts that
  the shortwave heating rate actually responds.
  """

  def test_ozone_override_changes_shortwave_heating(self):
    cfg = _build_radiative_transfer_cfg()
    rt = rrtmgp.RRTMGP(cfg, dz=500.0)
    inputs = _build_inputs()

    baseline = rt.compute_heating_rate(**inputs)

    o3_default = (
        lookup_volume_mixing_ratio.reconstruct_vmr_fields_from_pressure(
            rt.atmospheric_state.vmr, inputs['p_ref_xxc']
        )['o3']
    )
    perturbed = rt.compute_heating_rate(
        **inputs, vmr_fields={'o3': 10.0 * o3_default}
    )

    baseline_hr = baseline[rrtmgp_common.KEY_STORED_RADIATION]
    perturbed_hr = perturbed[rrtmgp_common.KEY_STORED_RADIATION]

    # Strip halos (top + bottom) before comparing, matching how downstream
    # consumers use the field.
    diff = np.asarray(perturbed_hr[:, :, 1:-1] - baseline_hr[:, :, 1:-1])
    max_abs_diff = np.max(np.abs(diff))
    # Pre-fix, the override was a no-op and this difference was bit-exact zero.
    # 1e-6 K/s ~ 0.09 K/day, well above noise but well below stratospheric SW
    # ozone-heating scales.
    self.assertGreater(
        max_abs_diff,
        1e-6,
        msg=(
            'Scaling ozone 10x via vmr_fields did not change the heating '
            f'rate (max |diff| = {max_abs_diff:.3e} K/s). The vmr_fields '
            'override is not being threaded through compute_heating_rate.'
        ),
    )

  def test_matching_ozone_override_matches_baseline(self):
    """Passing the reconstructed o3 as an override is a no-op."""
    cfg = _build_radiative_transfer_cfg()
    rt = rrtmgp.RRTMGP(cfg, dz=500.0)
    inputs = _build_inputs()

    baseline = rt.compute_heating_rate(**inputs)

    o3_default = (
        lookup_volume_mixing_ratio.reconstruct_vmr_fields_from_pressure(
            rt.atmospheric_state.vmr, inputs['p_ref_xxc']
        )['o3']
    )
    echoed = rt.compute_heating_rate(**inputs, vmr_fields={'o3': o3_default})

    np.testing.assert_array_equal(
        np.asarray(echoed[rrtmgp_common.KEY_STORED_RADIATION]),
        np.asarray(baseline[rrtmgp_common.KEY_STORED_RADIATION]),
    )


class RRTMGPAerosolTest(unittest.TestCase):
  """Regression tests for the per-band aerosol_optics_{lw,sw} kwargs."""

  def _make_zero_aerosol(self, n_bnd: int, shape: tuple[int, int, int]):
    z = jnp.zeros((n_bnd,) + shape, dtype=jnp.float_)
    return {'optical_depth': z, 'ssa': z, 'asymmetry_factor': z}

  def test_zero_aerosol_matches_baseline(self):
    """Passing an all-zero aerosol bundle is a no-op vs. not passing one."""
    cfg = _build_radiative_transfer_cfg()
    rt = rrtmgp.RRTMGP(cfg, dz=500.0)
    inputs = _build_inputs()

    baseline = rt.compute_heating_rate(**inputs)

    shape = inputs['p_ref_xxc'].shape
    aer_lw = self._make_zero_aerosol(rt.optics_lib.gas_optics_lw.n_bnd, shape)
    aer_sw = self._make_zero_aerosol(rt.optics_lib.gas_optics_sw.n_bnd, shape)
    zeroed = rt.compute_heating_rate(
        **inputs, aerosol_optics_lw=aer_lw, aerosol_optics_sw=aer_sw
    )

    # Zero tau contributes nothing in combine_optical_properties — but the
    # extra combination step touches the ssa via a `max(tau, EPSILON)` guard,
    # so the two paths agree only to floating-point precision, not bit-exact.
    np.testing.assert_allclose(
        np.asarray(zeroed[rrtmgp_common.KEY_STORED_RADIATION]),
        np.asarray(baseline[rrtmgp_common.KEY_STORED_RADIATION]),
        rtol=1e-5,
        atol=1e-9,
    )

  def test_sw_aerosol_reduces_surface_flux(self):
    """Increasing SW aerosol optical depth must reduce the surface
    down-flux (scattering + absorption attenuates the direct + diffuse beam).
    """
    cfg = dataclasses.replace(
        _build_radiative_transfer_cfg(),
        save_lw_sw_heating_rates=True,
    )
    rt = rrtmgp.RRTMGP(cfg, dz=500.0)
    inputs = _build_inputs()

    baseline = rt.compute_heating_rate(**inputs)

    shape = inputs['p_ref_xxc'].shape
    n_bnd_sw = rt.optics_lib.gas_optics_sw.n_bnd
    # A modestly absorbing/scattering aerosol uniformly distributed in the
    # column: tau=0.05 per layer per band, ssa=0.9, g=0.7. This is sulfate-ish
    # and well above floating-point noise.
    aer_sw = {
        'optical_depth': 0.05 * jnp.ones((n_bnd_sw,) + shape, dtype=jnp.float_),
        'ssa': 0.9 * jnp.ones((n_bnd_sw,) + shape, dtype=jnp.float_),
        'asymmetry_factor': 0.7 * jnp.ones(
            (n_bnd_sw,) + shape, dtype=jnp.float_
        ),
    }
    perturbed = rt.compute_heating_rate(**inputs, aerosol_optics_sw=aer_sw)

    # Surface SW down flux at the first physical level (above the surface halo)
    # must drop when aerosols are added.
    hw = 1
    baseline_surf = np.asarray(baseline['sw_flux_down_full'][:, :, 0])
    perturbed_surf = np.asarray(perturbed['sw_flux_down_full'][:, :, 0])
    self.assertTrue(
        np.all(perturbed_surf < baseline_surf),
        msg=(
            'SW aerosol did not reduce surface down-flux everywhere. '
            f'min(baseline - perturbed) = {np.min(baseline_surf - perturbed_surf):.3e} W/m²'
        ),
    )
    del hw

  def test_lw_aerosol_changes_toa_outgoing(self):
    """Adding an absorbing LW aerosol must change TOA outgoing LW flux."""
    cfg = _build_radiative_transfer_cfg()
    rt = rrtmgp.RRTMGP(cfg, dz=500.0)
    inputs = _build_inputs()

    baseline = rt.compute_heating_rate(**inputs)

    shape = inputs['p_ref_xxc'].shape
    n_bnd_lw = rt.optics_lib.gas_optics_lw.n_bnd
    # Pure absorber: tau=0.1, ssa=0 (no scattering — typical for LW dust).
    aer_lw = {
        'optical_depth': 0.1 * jnp.ones((n_bnd_lw,) + shape, dtype=jnp.float_),
        'ssa': jnp.zeros((n_bnd_lw,) + shape, dtype=jnp.float_),
        'asymmetry_factor': jnp.zeros(
            (n_bnd_lw,) + shape, dtype=jnp.float_
        ),
    }
    perturbed = rt.compute_heating_rate(**inputs, aerosol_optics_lw=aer_lw)

    baseline_toa = np.asarray(baseline['lw_flux_up_full'][:, :, -1])
    perturbed_toa = np.asarray(perturbed['lw_flux_up_full'][:, :, -1])
    max_diff = np.max(np.abs(baseline_toa - perturbed_toa))
    self.assertGreater(
        max_diff,
        1e-3,
        msg=(
            'LW aerosol left TOA outgoing flux unchanged '
            f'(max |Δ| = {max_diff:.3e} W/m²).'
        ),
    )


if __name__ == '__main__':
  unittest.main()
