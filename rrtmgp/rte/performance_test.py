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

  4. **More kernel launches for the same arithmetic.** On an accelerator this
     workload is launch-bound, not flop-bound, so this is a distinct failure
     mode that (1)-(3) cannot see at all. Caught by the launch-count budgets
     below.

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

**Why arithmetic budgets alone are not enough.** Flops and transcendentals
turned out not to predict throughput on this workload at all, and the evidence
is unambiguous:

  * Cutting the longwave solve's transcendentals back to 0.2.1's exact count
    (whole-solve 9.17M -> 3.44M) changed end-to-end GCM throughput by nothing:
    20.68 -> 20.45 s/simulated-day, the same product, inside run-to-run noise.
  * A follow-up that deduplicated exponentials and cut `select_n` counts (SW
    13 -> 9, LW 7 -> 4 per element) also changed throughput by nothing.
  * XLA's cost model said `use_scan=True` should shrink the 0.3.0 solve
    regression from 3.15x/7.74x to 1.34x/1.99x. Measured end to end it was
    44-49% *slower*, because a scan serialises kernel launches that the
    unrolled form issues back to back.

What did track the regression is the number of kernel launches. Between 0.2.1
and 0.3.0-era main the GCM's radiation call went from 89,733 to 132,832
launches (+48%) against a measured radiation time of 850.5 -> 1193.5 ms
(+40%), while the *average* kernel got slightly faster (9.48 -> 8.99 us). The
GPU is busy only 64-74% of the wall time at 86% occupancy: the solve is
launch-bound, and a change that adds launches costs time even if it removes
arithmetic. Hence guard 4.
"""

import collections
import functools
import re
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

# Kernel-launch budgets for a whole solve, keyed by (platform, band,
# use_scan). This is the number of kernels the backend issues per call -- the
# launches in the g-point loop body multiplied by `n_gpt`, plus the handful
# outside it -- which is what the GCM actually pays for on an accelerator (see
# the module docstring for why flops do not predict it).
#
# Reference values on the implementation these were written against, CPU
# backend, jax 0.10.2, at the `_N_HORIZ` shape:
#
#                      launches/solve   g-point body   n_gpt
#   LW use_scan=False      117,036          457         256
#   LW use_scan=True       275,244        1,075         256
#   SW use_scan=False       82,674          369         224
#   SW use_scan=True       273,970        1,223         224
#
# Both `use_scan` settings are pinned for the same reason the arithmetic
# budgets are, and the numbers show why it matters: `use_scan=True` issues
# 2.4x / 3.3x the launches of the unrolled form while XLA's cost model reports
# it as *cheaper* in flops. That inversion is exactly the one that made the
# scan 44-49% slower in the GCM, and only this metric has the sign right.
#
# **Backend dependence, and how far this is trusted.** Kernel count is a
# property of the compiled executable, so it is backend-specific: CPU and GPU
# do not fuse identically, and the absolute CPU numbers here are ~1.5x the GPU
# ones. The budgets are therefore keyed by platform and a platform with no
# entry skips rather than asserting something it has not been calibrated for.
#
# What was checked before trusting the CPU numbers as a guard is that they
# reproduce the *ratio* that the GPU traces measured. Counting the solve this
# way on CPU at 0.2.1 and at 0.3.0-era main gives 134,640 -> 201,824 launches,
# a factor of 1.499, against the 1.484 measured on GPU with a profiler. The
# split also matches: the CPU count attributes 78,144 of main's launches to the
# minor-gas scan where the GPU trace attributes 78,594. So the CPU metric is
# not the GPU's number, but it tracks the GPU's *change* to about 1%, which is
# what a regression guard needs.
#
# The ceilings carry ~20% headroom -- looser than the arithmetic budgets,
# because fusion decisions do shift a few percent across XLA releases, and
# tighter than the +48% regression that motivated the guard, so a repeat of it
# cannot fit underneath.
#
# They are set from the *current* count, which is still ~1.5x 0.2.1's. The
# guard freezes today's cost so the next increment has to be argued for; it
# does not certify that today's cost is right. Most of what is being frozen is
# one structural change: 0.3.0 replaced the minor-gas absorber loop -- a
# `while_loop` that ran each band's own handful of intervals -- with a
# fixed-length `scan`, because a data-dependent trip count is not
# reverse-mode differentiable. Even bounded to the widest band it now runs the
# worst case for every g-point, which is 73% of the launches 0.3.0 added.
_MAX_LAUNCHES = {
    ('cpu', 'lw', False): 140_000,
    ('cpu', 'lw', True): 330_000,
    ('cpu', 'sw', False): 99_000,
    ('cpu', 'sw', True): 329_000,
}

# HLO text parsing for the launch count. Instruction lines look like
#   %name = f32[16,16,62]{2,1,0} fusion(%a, %b), kind=kLoop, calls=%fused.1
# or, for a tuple-shaped result,
#   %name = (s32[], f32[8]{0}) while(%t), condition=%c, body=%b, ...
# The opcode is the first `word(` in the line: the result shape that precedes
# it uses brackets and braces, or is a bare parenthesised tuple, so neither
# form can match.
_HLO_DECL = re.compile(r'^\s*(ENTRY\s+)?%?([\w.\-]+)\s*\(')
_HLO_INSTR = re.compile(r'^\s+%?([\w.\-]+)\s*=\s*(.*)$')
_HLO_OPCODE = re.compile(r'(?:^|[\s)])([a-z][\w\-]*)\(')
_HLO_TRIP = re.compile(r'"known_trip_count":\{"n":"(\d+)"\}')

# Pure metadata and addressing: never a kernel launch on any backend.
_NOT_A_LAUNCH = frozenset({
    'parameter', 'constant', 'tuple', 'get-tuple-element', 'bitcast',
    'after-all', 'token', 'partition-id', 'replica-id', 'get-dimension-size',
    'set-dimension-size', 'domain', 'opt-barrier', 'optimization-barrier',
})
# Control flow: the launches are those of the sub-computations, not of the
# instruction itself. Note `to_apply` (the reducer of a `reduce`, the
# comparator of a `sort`) is deliberately *not* treated as control flow -- it
# is inlined into the parent kernel, not launched separately.
_HLO_CONTROL = frozenset({'while', 'conditional', 'call', 'async-start'})


def _parse_hlo(text: str) -> tuple[dict[str, list], str | None]:
    """Split an HLO module into computations.

    Returns `({name: [(opcode, line), ...]}, entry_name)`. Nested `{...}`
    inside an instruction (fusion bodies are separate computations, but
    sharding and backend-config braces are not) is handled by only closing a
    computation on a line that is exactly `}`.
    """
    comps: dict[str, list] = {}
    entry = None
    cur, instrs = None, []
    for line in text.splitlines():
        stripped = line.strip()
        if cur is None:
            m = _HLO_DECL.match(line)
            if m and stripped.endswith('{'):
                cur, instrs = m.group(2), []
                if m.group(1):
                    entry = cur
            continue
        if stripped == '}':
            comps[cur] = instrs
            cur = None
            continue
        m = _HLO_INSTR.match(line)
        if m:
            rest = m.group(2)
            op = _HLO_OPCODE.search(rest)
            instrs.append((op.group(1) if op else '?', rest))
    return comps, entry


def _hlo_callees(opcode: str, line: str) -> list[tuple[str, int]]:
    """`(computation, repeat count)` pairs a control instruction invokes."""
    out = []
    if opcode == 'while':
        body = re.search(r'body=%?([\w.\-]+)', line)
        trip = _HLO_TRIP.search(line)
        if body:
            # A `while` whose trip count XLA could not determine is charged a
            # single iteration. That under-counts, so it makes the budget a
            # floor rather than a ceiling for such a program -- worth knowing,
            # but the current solves contain no such loop (the minor-gas loop
            # became a fixed-length scan in 0.3.0).
            out.append((body.group(1), int(trip.group(1)) if trip else 1))
    elif opcode == 'conditional':
        m = re.search(r'branch_computations=\{([^}]*)\}', line)
        names = list(m.group(1).split(',')) if m else []
        for key in ('true_computation', 'false_computation'):
            b = re.search(key + r'=%?([\w.\-]+)', line)
            if b:
                names.append(b.group(1))
        out += [(n.strip().lstrip('%'), 1) for n in names if n.strip()]
    else:
        m = re.search(r'calls=\{?%?([\w.\-]+)', line)
        if m:
            out.append((m.group(1), 1))
    return out


def _count_launches(text: str, histogram=None) -> int:
    """Kernel launches issued per call of a compiled module.

    Walks the *optimized* HLO from its entry computation. A `fusion` is one
    launch and its body is not descended into; a `while` costs its static trip
    count times its body, which is what makes this count the quantity the GCM
    pays -- for a solve that is (g-point body launches) x `n_gpt` rather than
    the body size alone. Only one branch of a `conditional` runs, but which one
    is data-dependent, so the most expensive is charged.
    """
    comps, entry = _parse_hlo(text)
    memo: dict[str, int] = {}

    def walk(name: str, weight: int) -> int:
        if name not in comps:
            return 0
        if name in memo and histogram is None:
            return memo[name]
        total = 0
        for opcode, line in comps[name]:
            if opcode in _NOT_A_LAUNCH:
                continue
            if opcode == 'fusion':
                total += 1
                if histogram is not None:
                    histogram['fusion'] += weight
                continue
            if opcode in _HLO_CONTROL:
                callees = _hlo_callees(opcode, line)
                if opcode == 'conditional' and callees:
                    total += max(walk(c, weight) for c, _ in callees)
                else:
                    for callee, trip in callees:
                        total += trip * walk(callee, weight * trip)
                continue
            total += 1
            if histogram is not None:
                histogram[opcode] += weight
        memo[name] = total
        return total

    return walk(entry, 1) if entry else 0


@functools.lru_cache(maxsize=None)
def _radiation_setup():
    """Build the optics library, atmospheric state, and a small column batch.

    Cached: every guard in this file wants the same inputs, and reading the
    g256/g224 lookup tables off disk each time dominates the non-compile cost.
    """
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


@functools.lru_cache(maxsize=None)
def _compiled_solve(band: str, use_scan: bool):
    """The compiled solve for `band`, cached.

    Compiling the solve at the production shape is the expensive part of this
    file, and the flops, transcendental and launch-count guards all want the
    same four executables. Caching keeps the suite to one compile each.
    """
    (optics_lib, atmos_state, p, t, molecules, vmr_fields,
     sfc_temperature) = _radiation_setup()
    def solve(temp):
        if band == 'lw':
            return two_stream.solve_lw(
                p, temp, molecules, optics_lib, atmos_state, vmr_fields,
                sfc_temperature, use_scan=use_scan,
            )['flux_net']
        return two_stream.solve_sw(
            p, temp, molecules, optics_lib, atmos_state, vmr_fields,
            use_scan=use_scan,
        )['flux_net']

    return jax.jit(solve).lower(t).compile()


def _compiled_cost(band: str, use_scan: bool) -> dict[str, float]:
    """Compile the solve for `band` and return XLA's cost analysis."""
    cost = _compiled_solve(band, use_scan).cost_analysis()
    return cost[0] if isinstance(cost, list) else cost


def _solve_launch_count(band: str, use_scan: bool):
    """Kernel launches per solve, and a histogram of what issues them."""
    histogram = collections.Counter()
    text = _compiled_solve(band, use_scan).as_text()
    return _count_launches(text, histogram), histogram


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
        def sw_kernel(tau, ssa, g):
            return monochromatic_two_stream.sw_cell_properties(0.5, tau, ssa, g)

        return _cost_of(sw_kernel, tau, ssa, asymmetry)

    def lw_kernel(tau, ssa, s, g):
        return monochromatic_two_stream.lw_cell_source_and_properties(
            tau, ssa, s, s, g
        )

    src = jnp.asarray(rng.uniform(0.0, 10.0, shape), f32)
    return _cost_of(lw_kernel, tau, ssa, src, asymmetry)


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

    def test_longwave_solve_launch_count_within_budget(self):
        """Launches, not arithmetic -- the metric that tracked issue #27.

        Both `use_scan` settings again, and here the two are not merely
        different budgets for the same program: the scan form issues several
        times the launches while reporting *fewer* flops. Pinning only one of
        them would leave the cheaper-looking, slower configuration unguarded.
        """
        for use_scan in (True, False):
            with self.subTest(use_scan=use_scan):
                self._assert_launches_within_budget('lw', use_scan)

    def test_shortwave_solve_launch_count_within_budget(self):
        for use_scan in (True, False):
            with self.subTest(use_scan=use_scan):
                self._assert_launches_within_budget('sw', use_scan)

    def test_launch_count_sees_loop_trip_counts(self):
        """The counter must multiply a loop body by its trip count.

        This is the property that makes the metric worth having over a plain
        instruction count, and the one a refactor of `_count_launches` could
        silently drop -- leaving a guard that reports a few hundred launches
        for a solve that issues a hundred thousand, and passes forever.
        """
        module = '\n'.join([
            'HloModule m',
            '%body (p: f32[4]) -> f32[4] {',
            '  %p = f32[4] parameter(0)',
            '  %f = f32[4] fusion(%p), kind=kLoop, calls=%fused',
            '}',
            'ENTRY %main (a: f32[4]) -> f32[4] {',
            '  %a = f32[4] parameter(0)',
            '  %w = f32[4] while(%a), condition=%cond, body=%body, '
            'backend_config={"known_trip_count":{"n":"256"}}',
            '}',
        ])
        self.assertEqual(_count_launches(module), 256)

        # ...and a loop XLA could not bound is charged one iteration, so the
        # number stays a lower bound rather than silently becoming zero.
        self.assertEqual(
            _count_launches(module.replace(
                ', backend_config={"known_trip_count":{"n":"256"}}', '')),
            1,
        )

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

    def _assert_launches_within_budget(self, band: str, use_scan: bool):
        """Kernel launches per solve stay under budget on this backend.

        The budgets are calibrated per platform because fusion is, so a
        platform that has not been calibrated skips with instructions rather
        than asserting a number measured somewhere else.
        """
        platform = jax.devices()[0].platform
        key = (platform, band, use_scan)
        if key not in _MAX_LAUNCHES:
            self.skipTest(
                f'no launch budget calibrated for platform {platform!r}. '
                f'Kernel count is a property of the compiled executable and '
                f'differs between backends, so asserting the CPU number here '
                f'would be meaningless. To calibrate, measure the current '
                f'count on this backend and add ({platform!r}, {band!r}, '
                f'{use_scan}) to _MAX_LAUNCHES with ~20% headroom.'
            )

        label = f'{band.upper()} solve (use_scan={use_scan}) on {platform}'
        launches, histogram = _solve_launch_count(band, use_scan)
        self.assertGreater(
            launches, 0,
            msg=(f'{label}: no launches counted. The optimized-HLO text no '
                 f'longer parses -- the guard is disabled, not satisfied.'),
        )
        self.assertLessEqual(
            launches, _MAX_LAUNCHES[key],
            msg=(f'{label} issues {launches:,} kernel launches, over the '
                 f'{_MAX_LAUNCHES[key]:,} budget. This is the metric that '
                 f'moved with the issue #27 regression while the flops and '
                 f'transcendental budgets above stayed green, so do not raise '
                 f'it on the strength of an arithmetic saving -- the GCM is '
                 f'launch-bound here and pays per launch. Biggest '
                 f'contributors: {histogram.most_common(5)}.'),
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
