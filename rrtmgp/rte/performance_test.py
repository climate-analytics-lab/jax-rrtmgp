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

The ways that happens have different signatures, so they are guarded
separately, and every guard is deterministic -- they assert on the *compiled
program*, not on wall-clock time, so they behave identically on a laptop and on
a loaded CI runner:

  1. **More arithmetic per element.** Extra transcendentals or flops in the
     per-g-point body. Caught by `cost_analysis()` budgets below.

  2. **More loop iterations.** A `scan` given a longer trip count than the
     physics needs. `cost_analysis()` is blind to this -- it reports the loop
     *body* cost, so a scan of length 10 and one of length 100 look identical.
     Caught structurally, by pinning the scan length itself.

  3. **Arithmetic in a cell kernel, diluted below the noise floor of a
     whole-solve budget.** See below.

The budgets are ceilings with headroom, not exact values; they are meant to
catch a multiplicative regression, not to freeze the implementation. If a change
genuinely needs more arithmetic, raise the number here deliberately and say why
in the commit -- that is the point of the guard.

Two blind spots in the original version of this file let the issue #27
regression through, both worth stating because they are easy to reintroduce:

  * **Only the whole solve was measured.** A solve is dominated by gas optics,
    so a cell kernel can several-fold and still move the total by a few
    percent. The 0.3.0 two-stream rewrite made `lw_cell_source_and_properties`
    4.8x the flops and 2.7x the transcendentals of 0.2.1 and never came close
    to tripping a whole-solve ceiling. The kernels are therefore now measured
    *in isolation* as well, where nothing dilutes them.

  * **Only `use_scan=True` was measured.** That is not the default and not
    what downstream callers get: `solve_lw` / `solve_sw` default to
    `use_scan=False`, which unrolls the vertical recurrence instead of
    emitting a `scan`. The two configurations do not merely differ by a
    constant -- at 0.3.0 the same change that cost 1.35x (LW) / 2.0x (SW) of
    the solve under `use_scan=True` cost 3.1x / 7.7x under `use_scan=False`,
    because the unrolled form multiplies per-cell arithmetic in a way the
    scan form does not. A budget is only meaningful for the configuration it
    was measured in, so both are now pinned.
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
from rrtmgp import jaxpr_cost
from rrtmgp import kernel_ops
from rrtmgp import test_util
from rrtmgp.config import radiative_transfer
from rrtmgp.optics import atmospheric_state
from rrtmgp.optics import gas_optics
from rrtmgp.optics import lookup_gas_optics_longwave
from rrtmgp.optics import lookup_gas_optics_shortwave
from rrtmgp.optics import optics
from rrtmgp.rte import monochromatic_two_stream
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
#                        flops          transcendentals
#   LW  use_scan=True  2,415,670,272        3,440,256
#   SW  use_scan=True  2,245,821,696       11,467,520
#   LW  use_scan=False 6,818,315,776        3,440,256
#   SW  use_scan=False 10,076,223,488      11,467,520
#
# The ceilings carry ~30% headroom. They are set from the *current* numbers
# rather than left slack at a historical high-water mark: the transcendental
# counts were ~2.5x (longwave) higher before the hyperbolic quantities were
# computed together, and a budget loose enough to admit that guards nothing.
#
# The transcendental budgets in particular are now tight: the longwave solve
# sits at 0.2.1's count exactly, so there is no room left in which a repeat of
# the issue #27 regression could hide.
#
# These scale with the column count, so changing `_N_HORIZ` means recomputing
# them. Per element the figures above are ~4 (longwave) and ~13 (shortwave)
# transcendentals; at 2x2 columns the same code measures several times that,
# because fixed per-call work is then spread over far fewer elements -- which
# is the reason the guard is calibrated at the production shape rather than a
# token one.
#
# Keyed by (band, use_scan). The `use_scan=False` figures are much larger
# because that setting unrolls the vertical recurrence into the g-point body;
# they are budgets for a different program, not a looser bound on the same one.
_MAX_FLOPS = {
    ('lw', True): 3_200_000_000,
    ('sw', True): 3_100_000_000,
    ('lw', False): 8_200_000_000,
    ('sw', False): 12_100_000_000,
}
_MAX_TRANSCENDENTALS = {
    ('lw', True): 4_500_000,
    ('sw', True): 15_000_000,
    ('lw', False): 4_500_000,
    ('sw', False): 15_000_000,
}

# Per-cell two-stream kernel budgets, measured with nothing else in the
# program. This is the granularity the issue #27 regression actually lived at,
# and the ceilings are correspondingly tight -- ~20% headroom rather than the
# ~30% carried above, because there is no gas optics here to move underneath
# them. Reference values on the implementation these were written against, at
# the `_N_HORIZ` shape above (136 x 136 x 49 = 906,304 elements):
#
#                                       flops     transcendentals   per element
#   sw_cell_properties                315,393,792      9,063,040       348 / 10
#   lw_cell_source_and_properties     308,143,360      2,718,912       340 /  3
#
# For scale, the same two kernels at 0.2.1 measured 160 and 73 flops per
# element and 6 and 3 transcendentals. The longwave kernel is back to 0.2.1's
# transcendental count exactly; the shortwave one keeps 4 more per element than
# 0.2.1 -- two `sqrt` for the smooth energy-conservation clamps, and the extra
# `exp`/`expm1` pair that the float32-stable direct-beam form needs. Those are
# the numerics 0.3.0 was for, so they are budgeted for rather than removed.
_MAX_KERNEL_FLOPS = {'sw': 380_000_000, 'lw': 370_000_000}
_MAX_KERNEL_TRANSCENDENTALS = {'sw': 10_900_000, 'lw': 3_300_000}


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


def _minor_optical_depth_scan_lengths(lookup, atmos_state, molecules, p, t,
                                      vmr_fields) -> list[int]:
    """Trace the real minor-gas optical depth and collect its scan lengths.

    Reads the trip counts off the traced program via `rrtmgp.jaxpr_cost`, so
    this reflects what the solver actually does rather than what a helper
    reports.
    """
    vmr_by_index = {
        lookup.idx_gases[name]: field for name, field in vmr_fields.items()
    }
    return jaxpr_cost.scan_lengths(
        lambda temp: gas_optics.compute_minor_optical_depth(
            lookup, atmos_state.vmr, molecules, temp, p, 0, vmr_by_index
        ),
        t,
    )


def _cost_of(fn, *args) -> dict[str, float]:
    """Compile `fn` at `args` and return XLA's cost analysis."""
    cost = jax.jit(fn).lower(*args).compile().cost_analysis()
    # Some backends report a list of per-computation dicts.
    return cost[0] if isinstance(cost, list) else cost


def _compiled_cost(band: str, use_scan: bool) -> dict[str, float]:
    """Compile the solve for `band` and return XLA's cost analysis."""
    (optics_lib, atmos_state, p, t, molecules, vmr_fields,
     sfc_temperature) = _radiation_setup()
    if band == 'lw':
        fn = lambda temp: two_stream.solve_lw(
            p, temp, molecules, optics_lib, atmos_state, vmr_fields,
            sfc_temperature, use_scan=use_scan,
        )['flux_net']
    else:
        fn = lambda temp: two_stream.solve_sw(
            p, temp, molecules, optics_lib, atmos_state, vmr_fields,
            use_scan=use_scan,
        )['flux_net']
    return _cost_of(fn, t)


def _kernel_cost(band: str) -> dict[str, float]:
    """Cost of one two-stream cell kernel, compiled on its own.

    Deliberately not routed through the solve: the point of this measurement is
    that nothing else is in the program to dilute it. Inputs span the physical
    ranges (optical depth over several decades, single-scattering albedo up to
    and including 1) so no branch is optimised away as unreachable.
    """
    shape = (_N_HORIZ, _N_HORIZ, 49)
    rng = np.random.default_rng(0)
    f32 = jnp.float32
    tau = jnp.asarray(10.0 ** rng.uniform(-6, 0.5, shape), f32)
    ssa = jnp.asarray(rng.uniform(0.0, 1.0, shape), f32)
    asymmetry = jnp.asarray(rng.uniform(0.0, 0.9, shape), f32)
    if band == 'sw':
        fn = lambda tau, ssa, g: monochromatic_two_stream.sw_cell_properties(
            0.5, tau, ssa, g
        )
        return _cost_of(fn, tau, ssa, asymmetry)
    src = jnp.asarray(rng.uniform(0.0, 10.0, shape), f32)
    fn = lambda tau, ssa, s, g: (
        monochromatic_two_stream.lw_cell_source_and_properties(tau, ssa, s, s, g)
    )
    return _cost_of(fn, tau, ssa, src, asymmetry)


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
        """Both `use_scan` settings, because they are different programs.

        `use_scan=False` is the default that downstream callers get, and it
        unrolls the vertical recurrence, so a change to the per-cell kernels
        lands on it far harder than on the `use_scan=True` form. Measuring only
        the latter is how the issue #27 regression cleared this file.
        """
        for use_scan in (True, False):
            with self.subTest(use_scan=use_scan):
                self._assert_within_budget('lw', use_scan)

    def test_shortwave_solve_arithmetic_within_budget(self):
        for use_scan in (True, False):
            with self.subTest(use_scan=use_scan):
                self._assert_within_budget('sw', use_scan)

    def test_longwave_cell_kernel_within_budget(self):
        self._assert_kernel_within_budget('lw')

    def test_shortwave_cell_kernel_within_budget(self):
        self._assert_kernel_within_budget('sw')

    def _assert_metrics_present(self, cost, what: str):
        """Both metrics must actually be reported.

        Defaulting a missing key to zero would leave the corresponding budget
        vacuously satisfied, so a backend or JAX version that stops reporting
        one would silently disable the guard rather than fail visibly.
        """
        for metric in ('flops', 'transcendentals'):
            self.assertIn(
                metric, cost,
                msg=(f'cost analysis did not report {metric!r} for {what}, so '
                     f'its budget cannot be enforced. Keys: {sorted(cost)}'),
            )
        self.assertGreater(cost['flops'], 0.0, f'{what}: no flops reported')
        self.assertGreater(
            cost['transcendentals'], 0.0,
            f'{what}: cost analysis reported no transcendentals, but this code '
            'uses exp and sqrt -- a zero here means the metric is not being '
            'measured rather than that the work is not being done',
        )

    def _assert_within_budget(self, band: str, use_scan: bool):
        """Per-g-point-body flops and transcendentals stay under budget.

        Note this counts the *body* of the g-point loop, so it catches extra
        arithmetic per element but says nothing about trip counts -- which is
        why the scan length is pinned separately above.
        """
        label = f'{band.upper()} solve (use_scan={use_scan})'
        cost = _compiled_cost(band, use_scan)
        self._assert_metrics_present(cost, label)
        flops = cost['flops']
        transcendentals = cost['transcendentals']

        self.assertLessEqual(
            flops, _MAX_FLOPS[band, use_scan],
            msg=(f'{label} costs {flops:,.0f} flops per g-point body, over the '
                 f'{_MAX_FLOPS[band, use_scan]:,} budget. If this is a '
                 f'deliberate trade, raise the budget and justify it.'),
        )
        self.assertLessEqual(
            transcendentals, _MAX_TRANSCENDENTALS[band, use_scan],
            msg=(f'{label} costs {transcendentals:,.0f} transcendentals per '
                 f'g-point body, over the '
                 f'{_MAX_TRANSCENDENTALS[band, use_scan]:,} budget. Note that '
                 f'a `jnp.where` evaluates both branches, so a "safe" branch '
                 f'guarding a sqrt/exp costs the same as taking it.'),
        )

    def _assert_kernel_within_budget(self, band: str):
        """The two-stream cell kernel on its own, with nothing to dilute it.

        The whole-solve budgets above are dominated by gas optics, which is
        most of their cost and none of their risk: at 0.3.0 the longwave cell
        kernel took 4.8x the flops and 2.7x the transcendentals of 0.2.1 while
        moving the longwave solve total by ~35%. This guard sees the kernel at
        full amplitude, which is the only way a change of that shape shows up
        as a number somebody has to justify.
        """
        label = f'{band.upper()} cell kernel'
        cost = _kernel_cost(band)
        self._assert_metrics_present(cost, label)
        flops = cost['flops']
        transcendentals = cost['transcendentals']

        self.assertLessEqual(
            flops, _MAX_KERNEL_FLOPS[band],
            msg=(f'{label} costs {flops:,.0f} flops '
                 f'({flops / (_N_HORIZ * _N_HORIZ * 49):.0f} per element), over '
                 f'the {_MAX_KERNEL_FLOPS[band]:,} budget.'),
        )
        self.assertLessEqual(
            transcendentals, _MAX_KERNEL_TRANSCENDENTALS[band],
            msg=(f'{label} costs {transcendentals:,.0f} transcendentals '
                 f'({transcendentals / (_N_HORIZ * _N_HORIZ * 49):.0f} per '
                 f'element), over the {_MAX_KERNEL_TRANSCENDENTALS[band]:,} '
                 f'budget. Both branches of a `jnp.where` are evaluated, so '
                 f'guarding a transcendental costs as much as taking it.'),
        )


if __name__ == '__main__':
    unittest.main()
