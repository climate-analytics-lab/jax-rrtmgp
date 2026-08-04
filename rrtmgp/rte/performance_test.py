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
# Column count matters for what these budgets actually measure. The production
# use case (a GCM radiation call) solves a large batch of independent columns,
# where cost is dominated by per-element work in the g-point loop body. At a
# handful of columns, fixed per-call overhead dilutes exactly the per-element
# costs that hurt in production, so a budget calibrated there would be
# insensitive to the regressions worth catching. 136x136 = 18,496 columns is
# the T63L47 grid (18,432) to within rounding.
#
# This costs nothing to run: the guards below only *compile* the solve and read
# XLA's cost model. Nothing is executed, so the large shape adds no runtime.
_N_HORIZ = 136

# Per-g-point-body budgets at the column count above (g256 longwave / g224
# shortwave). Reference values on the implementation these were written
# against:
#
#            flops          transcendentals
#   LW    2,463,834,112         9,174,016
#   SW    2,354,763,008        16,054,528
#
# The ceilings carry ~30% headroom. They are set from the *current* numbers
# rather than left slack at a historical high-water mark: the transcendental
# counts were ~2.5x (longwave) higher before the hyperbolic quantities were
# computed together, and a budget loose enough to admit that guards nothing.
#
# These scale with the column count, so changing `_N_HORIZ` means recomputing
# them. Per element the figures above are ~8 transcendentals; at 2x2 columns
# the same code measures ~32, because fixed per-call work is then spread over
# far fewer elements -- which is the reason the guard is calibrated at the
# production shape rather than a token one.
_MAX_FLOPS = {'lw': 3_200_000_000, 'sw': 3_100_000_000}
_MAX_TRANSCENDENTALS = {'lw': 12_000_000, 'sw': 21_000_000}


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


def _scan_lengths(jaxpr) -> list[int]:
    """Every `scan` trip count in `jaxpr`, including nested sub-jaxprs.

    Reads the trip count straight off the traced program, so it reflects what
    the solver actually does rather than what a helper reports.
    """
    lengths = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == 'scan':
            lengths.append(int(eqn.params['length']))
        for value in eqn.params.values():
            for sub in (value if isinstance(value, (tuple, list)) else (value,)):
                inner = getattr(sub, 'jaxpr', sub)
                if hasattr(inner, 'eqns'):
                    lengths.extend(_scan_lengths(inner))
    return lengths


def _minor_optical_depth_scan_lengths(lookup, atmos_state, molecules, p, t,
                                      vmr_fields) -> list[int]:
    """Trace the real minor-gas optical depth and collect its scan lengths."""
    vmr_by_index = {
        lookup.idx_gases[name]: field for name, field in vmr_fields.items()
    }
    jaxpr = jax.make_jaxpr(
        lambda temp: gas_optics.compute_minor_optical_depth(
            lookup, atmos_state.vmr, molecules, temp, p, 0, vmr_by_index
        )
    )(t)
    return _scan_lengths(jaxpr.jaxpr)


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

    def test_minor_gas_scan_trip_count_in_traced_solver(self):
        """The trip count the solver actually uses, not what a helper returns.

        The check above pins `minor_scan_length`, which is only useful while
        `_compute_minor_optical_depth` keeps calling it. Reverting that call
        site to scan the whole table would restore the regression with the
        helper left untouched, and the check above would still pass. This one
        reads the trip count off the traced program, so it follows the code
        that actually runs.
        """
        (optics_lib, atmos_state, p, t, molecules, vmr_fields,
         _) = _radiation_setup()
        lookup = optics_lib.gas_optics_lw

        lengths = _minor_optical_depth_scan_lengths(
            lookup, atmos_state, molecules, p, t, vmr_fields
        )
        self.assertTrue(
            lengths, 'no scan found in the minor-gas optical depth; the '
                     'traced structure changed and this guard needs updating'
        )

        # The lower and upper atmosphere are accumulated separately, so the
        # widest band of either bounds every scan here.
        widest = 0
        for start, end in ((lookup.minor_lower_bnd_start,
                            lookup.minor_lower_bnd_end),
                           (lookup.minor_upper_bnd_start,
                            lookup.minor_upper_bnd_end)):
            widths = np.where(
                np.asarray(start) >= 0,
                np.asarray(end) - np.asarray(start) + 1,
                0,
            )
            widest = max(widest, int(widths.max()))

        table_dim = max(lookup.n_minor_absrb_lower, lookup.n_minor_absrb_upper)
        self.assertLessEqual(
            max(lengths), widest,
            msg=(f'minor-gas scan runs {max(lengths)} iterations, more than '
                 f'the widest band ({widest}) needs. Scanning the full table '
                 f'({table_dim}) evaluates interpolations that are then masked '
                 f'away -- the issue #22 regression.'),
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

        # Both metrics must actually be reported. Defaulting a missing key to
        # zero would leave the corresponding budget vacuously satisfied, so a
        # backend or JAX version that stops reporting one would silently
        # disable the guard rather than fail visibly.
        for metric in ('flops', 'transcendentals'):
            self.assertIn(
                metric, cost,
                msg=(f'cost analysis did not report {metric!r}, so its budget '
                     f'cannot be enforced. Keys present: {sorted(cost)}'),
            )
        flops = cost['flops']
        transcendentals = cost['transcendentals']

        self.assertGreater(flops, 0.0, 'cost analysis reported no flops')
        self.assertGreater(
            transcendentals, 0.0,
            'cost analysis reported no transcendentals; the solve uses exp and '
            'sqrt, so a zero here means the metric is not being measured',
        )
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
