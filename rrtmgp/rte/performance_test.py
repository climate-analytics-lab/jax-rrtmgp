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

"""Performance regression guards for the radiative transfer solve (issue #22).

A change to the two-stream inner loop can multiply the cost of the whole model
without changing any answer, so nothing else in the suite notices. That is what
happened in #22: the solve got several times more expensive and it was only
caught much later, end to end, in a downstream GCM.

The two ways that happens have different signatures, so they are guarded
separately, and both guards are deterministic -- they assert on the *compiled
program*, not on wall-clock time, so they behave identically on a laptop and on
a loaded CI runner:

  1. **More arithmetic per element.** Extra transcendentals or flops in the
     per-g-point body. Caught by `cost_analysis()` budgets below.

  2. **More loop iterations.** A `scan` given a longer trip count than the
     physics needs. `cost_analysis()` is blind to this -- it reports the loop
     *body* cost, so a scan of length 10 and one of length 100 look identical.
     Caught structurally, by pinning the scan length itself.

The budgets are ceilings with headroom, not exact values; they are meant to
catch a multiplicative regression, not to freeze the implementation. If a change
genuinely needs more arithmetic, raise the number here deliberately and say why
in the commit -- that is the point of the guard.
"""

import functools
from pathlib import Path
from typing import TypeAlias

import unittest
import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np

from rrtmgp import constants
from rrtmgp import kernel_ops
from rrtmgp import test_util
from rrtmgp.config import radiative_transfer
from rrtmgp.optics import atmospheric_state
from rrtmgp.optics import gas_optics
from rrtmgp.optics import lookup_gas_optics_longwave
from rrtmgp.optics import lookup_gas_optics_shortwave
from rrtmgp.optics import optics
from rrtmgp.rte import two_stream

Array: TypeAlias = jax.Array

_ROOT = Path()
_LW_LOOKUP = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g256.nc'
_SW_LOOKUP = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-sw-g224.nc'
_CLOUD_LW = 'rrtmgp/optics/rrtmgp_data/cloudysky_lw.nc'
_CLOUD_SW = 'rrtmgp/optics/rrtmgp_data/cloudysky_sw.nc'
_ATMOS_STATE = 'rrtmgp/optics/test_data/clearsky_as.nc'
_VMR_GLOBAL_MEANS = 'rrtmgp/optics/test_data/vmr_global_means.json'

_HALO = 1
_N_HORIZ = 2

# Per-g-point-body budgets. Reference values on the implementation these were
# written against (g256 longwave / g224 shortwave, 2x2 columns):
#
#            flops        transcendentals
#   LW    2,358,866            19,840
#   SW    2,221,804            18,848
#
# The ceilings carry ~35% headroom. For scale, the arithmetic here is already
# ~6x the transcendental count of the 0.2.1 release, so these are not tight.
_MAX_FLOPS = {'lw': 3_200_000, 'sw': 3_000_000}
_MAX_TRANSCENDENTALS = {'lw': 27_000, 'sw': 26_000}


def _radiation_setup():
    """Build the optics library, atmospheric state, and a small column batch."""
    ds = nc.Dataset(_ROOT / _ATMOS_STATE, 'r')
    params = radiative_transfer.RadiativeTransfer(
        optics=radiative_transfer.OpticsParameters(
            optics=radiative_transfer.RRTMOptics(
                longwave_nc_filepath=str(_ROOT / _LW_LOOKUP),
                shortwave_nc_filepath=str(_ROOT / _SW_LOOKUP),
                cloud_longwave_nc_filepath=str(_ROOT / _CLOUD_LW),
                cloud_shortwave_nc_filepath=str(_ROOT / _CLOUD_SW),
            )
        ),
        atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
            sfc_emis=0.98, sfc_alb=0.06, zenith=0.5, irrad=1360.0,
            toa_flux_lw=0.0,
            vmr_global_mean_filepath=_ROOT / _VMR_GLOBAL_MEANS,
        ),
    )
    atmos_state = atmospheric_state.from_config(params.atmospheric_state_cfg)
    optics_lib = optics.optics_factory(params.optics, atmos_state.vmr)

    site, expt = 0, 0
    p_layer = np.flip(ds['pres_layer'][:].data, axis=-1)
    p_level = np.flip(ds['pres_level'][:].data, axis=-1)
    pressure = np.pad(p_layer, ((0, 0), (_HALO, _HALO)), mode='edge')
    pressure_level = np.pad(p_level, ((0, 0), (_HALO, _HALO - 1)), mode='edge')
    t_layer = np.flip(ds['temp_layer'][:].data, axis=-1)
    t_level = np.flip(ds['temp_level'][:].data, axis=-1)
    nx, ny, nz = t_layer.shape
    temperature = np.zeros((nx, ny, nz + 2 * _HALO), dtype=jnp.float_)
    temperature[:, :, _HALO:-_HALO] = t_layer
    temperature[:, :, 0] = 2 * t_level[:, :, 0] - t_layer[:, :, 0]
    temperature[:, :, -1] = 2 * t_level[:, :, -1] - t_layer[:, :, -1]
    h2o = np.pad(np.flip(ds['water_vapor'][:].data, axis=-1),
                 ((0, 0), (0, 0), (_HALO, _HALO)), mode='edge')
    o3 = np.pad(np.flip(ds['ozone'][:].data, axis=-1),
                ((0, 0), (0, 0), (_HALO, _HALO)), mode='edge')

    convert = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=_N_HORIZ
    )
    sfc_temperature = ds['surface_temperature'][:].data[expt, site] * jnp.ones(
        (_N_HORIZ, _N_HORIZ), dtype=jnp.float_
    )
    p = convert(pressure[site, :])
    p_lev = convert(pressure_level[site, :])
    t = convert(temperature[expt, site, :])
    vmr_h2o = convert(h2o[expt, site, :])
    vmr_o3 = convert(o3[expt, site, :])
    dp = kernel_ops.forward_difference(p_lev, dim=2)
    molecules = (
        -(dp / constants.G) * constants.AVOGADRO
        / (constants.DRY_AIR_MOL_MASS + constants.WATER_MOL_MASS * vmr_h2o)
    )
    return (optics_lib, atmos_state, p, t, molecules,
            {'h2o': vmr_h2o, 'o3': vmr_o3}, sfc_temperature)


def _compiled_cost(band: str) -> dict[str, float]:
    """Compile the solve for `band` and return XLA's cost analysis."""
    (optics_lib, atmos_state, p, t, molecules, vmr_fields,
     sfc_temperature) = _radiation_setup()
    if band == 'lw':
        fn = lambda temp: two_stream.solve_lw(
            p, temp, molecules, optics_lib, atmos_state, vmr_fields,
            sfc_temperature, use_scan=True,
        )['flux_net']
    else:
        fn = lambda temp: two_stream.solve_sw(
            p, temp, molecules, optics_lib, atmos_state, vmr_fields,
            use_scan=True,
        )['flux_net']
    return jax.jit(fn).lower(t).compile().cost_analysis()


class PerformanceTest(unittest.TestCase):

    def test_minor_gas_scan_is_bounded_by_widest_band(self):
        """The minor-absorber scan must not walk the whole table.

        Each band uses a contiguous run of minor intervals, so the scan only
        needs to cover the widest band. Scanning the full table instead costs
        an interpolation per surplus interval per g-point -- invisible to every
        correctness test, because the surplus contributions are masked to zero.
        """
        lw = lookup_gas_optics_longwave.from_nc_file(str(_ROOT / _LW_LOOKUP))
        sw = lookup_gas_optics_shortwave.from_nc_file(str(_ROOT / _SW_LOOKUP))

        cases = [
            ('lw lower', lw.minor_lower_bnd_start, lw.minor_lower_bnd_end,
             lw.n_minor_absrb_lower),
            ('lw upper', lw.minor_upper_bnd_start, lw.minor_upper_bnd_end,
             lw.n_minor_absrb_upper),
            ('sw lower', sw.minor_lower_bnd_start, sw.minor_lower_bnd_end,
             sw.n_minor_absrb_lower),
            ('sw upper', sw.minor_upper_bnd_start, sw.minor_upper_bnd_end,
             sw.n_minor_absrb_upper),
        ]
        for label, start, end, n_intervals in cases:
            with self.subTest(label):
                length = gas_optics.minor_scan_length(start, end, n_intervals)
                # It must cover the widest band...
                widths = np.where(
                    np.asarray(start) >= 0,
                    np.asarray(end) - np.asarray(start) + 1,
                    0,
                )
                self.assertGreaterEqual(length, int(widths.max()))
                # ...and must be a real saving against the full table, which is
                # the regression this guards. The shipped tables are 4-7x wider
                # than their widest band; require at least 2x.
                self.assertLessEqual(
                    length, n_intervals // 2,
                    msg=(f'{label}: minor scan length {length} is not '
                         f'meaningfully shorter than the table dimension '
                         f'{n_intervals}; the scan is walking intervals no '
                         f'band uses.'),
                )

    def test_longwave_solve_arithmetic_within_budget(self):
        self._assert_within_budget('lw')

    def test_shortwave_solve_arithmetic_within_budget(self):
        self._assert_within_budget('sw')

    def _assert_within_budget(self, band: str):
        """Per-g-point-body flops and transcendentals stay under budget.

        Note this counts the *body* of the g-point loop, so it catches extra
        arithmetic per element but says nothing about trip counts -- which is
        why the scan length is pinned separately above.
        """
        cost = _compiled_cost(band)
        flops = cost.get('flops', 0.0)
        transcendentals = cost.get('transcendentals', 0.0)

        self.assertGreater(flops, 0.0, 'cost analysis reported no flops')
        self.assertLessEqual(
            flops, _MAX_FLOPS[band],
            msg=(f'{band.upper()} solve costs {flops:,.0f} flops per g-point '
                 f'body, over the {_MAX_FLOPS[band]:,} budget. If this is a '
                 f'deliberate trade, raise the budget and justify it.'),
        )
        self.assertLessEqual(
            transcendentals, _MAX_TRANSCENDENTALS[band],
            msg=(f'{band.upper()} solve costs {transcendentals:,.0f} '
                 f'transcendentals per g-point body, over the '
                 f'{_MAX_TRANSCENDENTALS[band]:,} budget. Note that a '
                 f'`jnp.where` evaluates both branches, so a "safe" branch '
                 f'guarding a sqrt/exp costs the same as taking it.'),
        )


if __name__ == '__main__':
    unittest.main()
