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

"""A library for solving the two-stream radiative transfer equation."""

from collections.abc import Iterator, Mapping, Sequence
import contextlib
import contextvars
import dataclasses
from typing import Callable, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from rrtmgp import constants
from rrtmgp import backend
from rrtmgp import kernel_ops
from rrtmgp.optics import atmospheric_state
from rrtmgp.optics import cloud_optics
from rrtmgp.optics import lookup_gas_optics_base
from rrtmgp.optics import lookup_gas_optics_longwave
from rrtmgp.optics import optics
from rrtmgp.optics import optics_base
from rrtmgp.optics import optics_utils
from rrtmgp.rte import longwave_noscat
from rrtmgp.rte import monochromatic_two_stream

Array: TypeAlias = jax.Array
AbstractLookupGasOptics: TypeAlias = (
    lookup_gas_optics_base.AbstractLookupGasOptics
)
AtmosphericState: TypeAlias = atmospheric_state.AtmosphericState

# Number of g-points solved per iteration of the spectral loop (see
# `_solve_over_gpoints`). One is the historical behaviour and is bit-for-bit
# identical to it; a larger value trades memory for kernel launches and is the
# lever for the launch-bound GPU regression documented there. Left at 1 until a
# GPU A/B confirms the win -- the recommendation, on the structural evidence, is
# 8.
DEFAULT_GPT_CHUNK = 1

# The longwave solvers `solve_lw` can use: the two-stream solver (the default;
# gas, cloud and aerosol scattering) and the absorption-only solver of RTE's
# `lw_solver_noscat_oneangle` (see `rrtmgp.rte.longwave_noscat`).
LW_SOLVER_TWO_STREAM = 'two_stream'
LW_SOLVER_NOSCAT = 'noscat'
LW_SOLVERS = (LW_SOLVER_TWO_STREAM, LW_SOLVER_NOSCAT)
# How the spectral loop obtains the gas optics (see `solve_lw`):
#   'per_gpt'  -- evaluated inside the loop, one g-point (or chunk) at a time.
#   'all_gpts' -- evaluated for every g-point before the loop, with the
#                 g-point innermost and every table read a row gather, and read
#                 back inside the loop.
GAS_OPTICS_MODE_PER_GPT = 'per_gpt'
GAS_OPTICS_MODE_ALL_GPTS = 'all_gpts'
GAS_OPTICS_MODES = (GAS_OPTICS_MODE_PER_GPT, GAS_OPTICS_MODE_ALL_GPTS)

# Prefix of the clear-sky entries `solve_{lw,sw}(with_clear_sky=True)` add to
# their result.
_CLEAR_PREFIX = 'clear_'
# Prefix of the aerosol-free entries `solve_{lw,sw}(with_aerosol_free=True)`
# add to their result. With both flags the aerosol-free clear sky is under
# `aerosol_free_clear_`, so `split_aerosol_free_fluxes` first and
# `split_clear_sky_fluxes` on each half.
_AEROSOL_FREE_PREFIX = 'aerosol_free_'


def _problem_prefixes(
    with_clear_sky: bool, with_aerosol_free: bool
) -> list[str]:
  """Key prefixes of the problems a solve returns, the all-sky one first."""
  prefixes = ['']
  if with_clear_sky:
    prefixes.append(_CLEAR_PREFIX)
  if with_aerosol_free:
    prefixes += [_AEROSOL_FREE_PREFIX + prefix for prefix in prefixes]
  return prefixes


def _split_prefixed(
    fluxes: dict[str, Array], prefix: str
) -> tuple[dict[str, Array], dict[str, Array]]:
  rest = {k: v for k, v in fluxes.items() if not k.startswith(prefix)}
  split = {
      k[len(prefix):]: v for k, v in fluxes.items() if k.startswith(prefix)
  }
  return rest, split


def split_clear_sky_fluxes(
    fluxes: dict[str, Array],
) -> tuple[dict[str, Array], dict[str, Array]]:
  """Splits the result of a `with_clear_sky` solve into all-sky and clear-sky.

  Args:
    fluxes: The dictionary returned by `solve_lw` / `solve_sw` called with
      `with_clear_sky=True`.

  Returns:
    The all-sky dictionary and the clear-sky one, each with the keys a plain
    `solve_{lw,sw}` call returns (no `clear_` prefix).
  """
  return _split_prefixed(fluxes, _CLEAR_PREFIX)


def split_aerosol_free_fluxes(
    fluxes: dict[str, Array],
) -> tuple[dict[str, Array], dict[str, Array]]:
  """Splits the result of a `with_aerosol_free` solve into with and without.

  Args:
    fluxes: The dictionary returned by `solve_lw` / `solve_sw` called with
      `with_aerosol_free=True`.

  Returns:
    The dictionary of the problems with the aerosol and the dictionary of
    their aerosol-free twins, each with the keys the same call without
    `with_aerosol_free` returns (so still with `clear_` entries when the call
    had `with_clear_sky=True`; `split_clear_sky_fluxes` takes those apart).
  """
  return _split_prefixed(fluxes, _AEROSOL_FREE_PREFIX)


def _solve_problems(
    solve: Callable[
        [Mapping[str, Array], Mapping[str, Array] | None], dict[str, Array]
    ],
    all_sky_props: Mapping[str, Array],
    clear_props: Mapping[str, Array] | None,
    aer_slice: Mapping[str, Array] | None,
    with_clear_sky: bool,
    with_aerosol_free: bool,
) -> dict[str, Array]:
  """Runs the transport of one g-point for every requested problem.

  The problems are the all-sky one (`all_sky_props` with `aer_slice` mixed
  in), the clear-sky one (`clear_props`, the same without the cloud), and the
  aerosol-free twin of each (the same properties with no aerosol mixed in).
  They share everything upstream of the transport -- the gas optics, the
  Planck sources, the cloud mix-in -- so this is where the sharing pays:
  `solve` is the cell properties and the transport only. Two problems that
  are the same one (no cloud to remove: `clear_props is all_sky_props`; no
  aerosol to remove: `aer_slice is None`) are solved once and returned under
  both prefixes, so asking for a twin that cannot differ costs nothing.

  Args:
    solve: `(optical_props, aer_slice) -> fluxes`, the transport of one
      problem; `aer_slice` is the g-point's band of the aerosol optics, or
      `None` for none.
    all_sky_props: The gas + cloud optical properties.
    clear_props: The gas optical properties, when `with_clear_sky`.
    aer_slice: The aerosol optical properties of this g-point's band, if any.
    with_clear_sky: Also solve the clear-sky problem(s).
    with_aerosol_free: Also solve the aerosol-free twin of every problem.

  Returns:
    The fluxes of every problem, keyed by `_problem_prefixes` order: the
    all-sky keys bare, the others under their prefix.
  """
  solved: dict[tuple[int, bool], dict[str, Array]] = {}
  fluxes: dict[str, Array] = {}
  for prefix in _problem_prefixes(with_clear_sky, with_aerosol_free):
    props = clear_props if prefix.endswith(_CLEAR_PREFIX) else all_sky_props
    aer = None if prefix.startswith(_AEROSOL_FREE_PREFIX) else aer_slice
    key = (id(props), aer is None)
    if key not in solved:
      solved[key] = solve(props, aer)
    fluxes.update({prefix + k: v for k, v in solved[key].items()})
  return fluxes


def _check_gas_optics_mode(
    gas_optics_mode: str, optics_lib: optics_base.OpticsScheme
) -> bool:
  """Validates `gas_optics_mode`; returns whether the gas optics is hoisted."""
  if gas_optics_mode not in GAS_OPTICS_MODES:
    raise ValueError(
        f'gas_optics_mode must be one of {GAS_OPTICS_MODES}, got'
        f' {gas_optics_mode!r}.'
    )
  if gas_optics_mode == GAS_OPTICS_MODE_PER_GPT:
    return False
  if not isinstance(optics_lib, optics.RRTMOptics):
    raise ValueError(
        f"gas_optics_mode={GAS_OPTICS_MODE_ALL_GPTS!r} requires the RRTMOptics"
        ' scheme (it reads the correlated-k gas-optics tables).'
    )
  return True


def _gpt_chunk_plan(n_gpt: int, gpt_chunk: int) -> tuple[int, int, bool]:
  """Resolve a requested g-point chunk size against a spectrum of `n_gpt`.

  Returns the chunk size actually used, the number of loop iterations, and
  whether the last iteration needs masking because `n_gpt` is not a multiple of
  the chunk.

  The remainder is handled by *padding* (the tail iteration re-solves already
  valid g-points and discards them) rather than by a separate remainder
  iteration: a second, differently-shaped loop body would be a second compiled
  program, doubling compile time and defeating the point of the exercise, which
  is to emit *fewer* distinct kernels. The shipped RRTMGP tables have
  `n_gpt` = 256 / 224, so every chunk size that is a power of two up to 16
  divides exactly and no masking is emitted at all.
  """
  if gpt_chunk < 1:
    raise ValueError(f'gpt_chunk must be >= 1, got {gpt_chunk}.')
  chunk = min(int(gpt_chunk), n_gpt)
  n_chunks = -(-n_gpt // chunk)  # ceil
  return chunk, n_chunks, n_chunks * chunk != n_gpt


def _solve_over_gpoints(
    one_gpt: Callable[[Array], dict[str, Array]],
    n_gpt: int,
    gpt_chunk: int,
    init_val: dict[str, Array],
) -> dict[str, Array]:
  """Accumulate `one_gpt` over the spectrum, `gpt_chunk` g-points at a time.

  Every g-point is an independent radiative transfer problem, so the spectral
  loop can process any number of them per iteration; only the order in which
  their fluxes are summed changes.

  **Why this exists.** The solve is *launch bound* on GPU, not flop bound. The
  0.3.0 two-stream rewrite cost ~30-34% of end-to-end GCM throughput while the
  average kernel *duration* fell (9.48 -> 8.99 us): the radiation call went from
  ~9.0k to ~13.3k kernel launches per model step and the device sat idle 26-36%
  of the wall time. With `use_scan=False` (the default) the vertical recurrence
  is unrolled over ~47 levels, so one iteration of this loop issues hundreds of
  kernels over small `(nx, ny)` slices, and there are `n_gpt` ~ 240 iterations.
  Processing `G` g-points per iteration issues the *same* kernels on arrays that
  are `G` times larger, `n_gpt / G` times -- roughly a `G`-fold cut in launches
  for identical arithmetic, which is exactly what an 8.5-of-80-GiB,
  64-74%-utilised workload has headroom for.

  **Other levers on the same bottleneck, and what each measured.**

  * *Rearranging the arithmetic.* Collapsing the rewritten two-stream's
    two-branch selects and deduplicating its transcendentals (the parent commit
    of this one) measured 20.45 vs 20.68 s/sim-day -- no recovery. Cutting the
    number of `select`s did not cut the number of fusion roots.
  * *`use_scan=True` at small `gpt_chunk`.* `use_scan` replaces the unrolled
    level recurrences by a `lax.scan` over levels (the loop over g-point chunks
    stays a `fori_loop`). XLA's cost model predicts a large win; the
    measurement at `gpt_chunk=1` is 44-49% *slower* end to end, because a
    `scan` serialises the very kernel launches that are the bottleneck and each
    step is a tiny kernel. A scan step is a kernel over `gpt_chunk` x columns
    elements, so the trade reverses once the chunk is large: with
    `gas_optics_mode='all_gpts'` and `gpt_chunk=64` (T63L47 ECHAM column, A100,
    f32) the transport of one radiation call measured 37 ms against 126 ms for
    the unrolled loop at `gpt_chunk=4`, and the whole call ~139 ms against
    ~202 ms. The unrolled loop does not profit from a large chunk (its fused
    level chains are the cost), the scan does.

  The corollary is that XLA's flop/byte cost model is not a throughput proxy on
  this workload. The only quantity that tracked the regression is the launch
  count, which is what this function reduces.

  Args:
    one_gpt: Solves a single g-point, given its (traced, scalar) index, and
      returns a dict of flux fields.
    n_gpt: Number of g-points in the spectrum.
    gpt_chunk: Number of g-points to solve per loop iteration.
    init_val: Zeroed accumulator with the same structure as `one_gpt`'s result.

  Returns:
    The spectral sum of `one_gpt` over all `n_gpt` g-points.
  """
  chunk, n_chunks, needs_mask = _gpt_chunk_plan(n_gpt, gpt_chunk)

  if chunk == 1:
    # Identical program to the unchunked solver, not merely equivalent: no
    # leading axis is introduced and the spectral sum is accumulated in the same
    # order, so this path is bit-for-bit the historical result.
    def step_fn(igpt, cumulative):
      per_gpt = one_gpt(igpt)
      with jax.named_scope('spectral_sum'):
        return jax.tree.map(jnp.add, per_gpt, cumulative)

    return jax.lax.fori_loop(0, n_gpt, step_fn, init_val)

  # `vmap` over the g-point index is what batches the chunk. It is used here in
  # preference to threading a g-axis through the gas-optics lookups by hand for
  # two reasons: it is exact by construction (the batched program computes the
  # same values as the scalar one), and -- critically -- the *inside* of the
  # optics and cell kernels keeps its original rank, so the positional `[:, :,
  # k]` slicing and `dim=2` shift operators that are written throughout
  # `optics.py`, `gas_optics.py` and `monochromatic_two_stream.py` stay correct
  # with no edits. Only `igpt` is mapped, so every loop-invariant operand --
  # above all the multi-megabyte `kmajor` / `kminor` tables -- stays a single
  # shared, *unbatched* operand rather than being broadcast across the chunk.
  # That is the issue #8 failure mode, and avoiding it is the reason the mapped
  # axis is the g-point index and nothing else.
  def step_fn(ichunk, cumulative):
    igpt = ichunk * chunk + jnp.arange(chunk)
    if needs_mask:
      valid = igpt < n_gpt
      # Clamp rather than let the gather run out of bounds: the surplus
      # iterations re-solve the last g-point and are zeroed below.
      igpt = jnp.minimum(igpt, n_gpt - 1)
    per_gpt = jax.vmap(one_gpt)(igpt)

    def reduce_and_add(chunked: Array, cumulative: Array) -> Array:
      if needs_mask:
        mask = valid.reshape((chunk,) + (1,) * (chunked.ndim - 1))
        chunked = jnp.where(mask, chunked, jnp.zeros_like(chunked))
      return jnp.sum(chunked, axis=0) + cumulative

    with jax.named_scope('spectral_sum'):
      return jax.tree.map(reduce_and_add, per_gpt, cumulative)

  return jax.lax.fori_loop(0, n_chunks, step_fn, init_val)


def _cloud_tau_scale_kwargs(
    cloud_tau_scale_liq: Array | float | None,
    cloud_tau_scale_ice: Array | float | None,
) -> dict[str, Array | float]:
  """Optics keyword arguments for the per-phase cloud optical-depth scaling.

  Only the factors that were supplied are forwarded, so an `OpticsScheme`
  written against the interface without them keeps working when they are not
  used.
  """
  kwargs = {}
  if cloud_tau_scale_liq is not None:
    kwargs['cloud_tau_scale_liq'] = cloud_tau_scale_liq
  if cloud_tau_scale_ice is not None:
    kwargs['cloud_tau_scale_ice'] = cloud_tau_scale_ice
  return kwargs


def _lw_planck_sources(
    pressure: Array,
    temperature: Array,
    igpt: Array,
    optics_lib: optics_base.OpticsScheme,
    vmr_fields: dict[int, Array] | None,
    sfc_temperature: Array | float | None,
    lw_solver: str = LW_SOLVER_TWO_STREAM,
    planck_fraction: Array | None = None,
) -> tuple[dict[str, Array], Array]:
  """Planck sources and the surface source of one g-point.

  These depend on the temperature, pressure and gas amounts but not on any
  condensate or aerosol, so a clear-sky and an all-sky solve of the same
  g-point share them (see `solve_lw`, `with_clear_sky`).

  Returns:
    The layer-centre source (`planck_src`) and the combined face sources
    (`planck_src_bottom`, `planck_src_top`: the geometric mean of the two
    adjacent layers' values at each face, so both layers share one source per
    level), and the surface source. The two-stream solver reads only the face
    sources; the no-scattering solver reads the layer-centre one as well.
  """
  if isinstance(sfc_temperature, float):
    # Create a plane for the surface temperature representation.
    nx, ny, _ = temperature.shape
    sfc_temperature = sfc_temperature * jnp.ones(
        (nx, ny), dtype=temperature.dtype
    )

  # Compute Planck sources: `planck_src`, `planck_src_bottom`, `planck_src_top`,
  # and `planck_src_sfc`.
  extra = {} if planck_fraction is None else {'planck_fraction': planck_fraction}
  planck_srcs = optics_lib.compute_planck_sources(
      pressure,
      temperature,
      igpt,
      vmr_fields,
      sfc_temperature=sfc_temperature,
      **extra,
  )

  halo_width = 1
  sfc_src = planck_srcs.get(
      'planck_src_sfc', planck_srcs['planck_src_bottom'][:, :, halo_width]
  )

  # Compute combined Planck sources.  Output keys are `planck_src_bottom` and
  # `planck_src_top`.
  with jax.named_scope('cell'):
    combined_srcs = monochromatic_two_stream.lw_combine_sources(planck_srcs)
  if lw_solver == LW_SOLVER_NOSCAT:
    # One source per level, shared by the two layers that meet there: current
    # upstream RTE feeds `lw_solver_noscat_oneangle` the same geometric-mean
    # level source as the two-stream solver
    # (`mo_gas_optics_rrtmgp_kernels.F90::compute_Planck_source`), so the two
    # solvers differ only in their transport. The layer-centre value is kept
    # for the linear-in-tau emission of `lw_source_noscat`. (RRTMG, RTE <= 1.6
    # and ECHAM6's `rtrnmc` instead give each layer its own face values,
    # `pfrac(layer) * B(T_face)`, which are discontinuous across a face; the
    # two conventions differ by at most 0.04 W/m^2 on the RFMIP clear-sky
    # profiles, so this follows current RTE rather than changing the physics.)
    return {
        'planck_src': planck_srcs['planck_src'],
        'planck_src_top': combined_srcs['planck_src_top'],
        'planck_src_bottom': combined_srcs['planck_src_bottom'],
    }, sfc_src
  return combined_srcs, sfc_src


def _lw_cell_properties(
    lw_optical_props: dict[str, Array],
    combined_srcs: dict[str, Array],
    sfc_src: Array,
    optics_lib: optics_base.OpticsScheme,
    aerosol_optics_slice: dict[str, Array] | None = None,
) -> dict[str, Array]:
  """Two-stream cell properties and sources from a g-point's optical depth."""
  # Mix in aerosol contributions for this band, if supplied. Aerosol tau/ssa/g
  # are passed through as-is (no delta-scaling applied), matching the convention
  # of simple aerosol parameterisations.
  if aerosol_optics_slice is not None:
    with jax.named_scope('aerosol_optics'):
      lw_optical_props = optics_lib.combine_optical_properties(
          lw_optical_props, aerosol_optics_slice
      )

  # Compute `t_diff`, `r_diff`, `src_up`, and `src_down`.
  with jax.named_scope('cell'):
    src_and_properties = monochromatic_two_stream.lw_cell_source_and_properties(
        lw_optical_props['optical_depth'],
        lw_optical_props['ssa'],
        combined_srcs['planck_src_bottom'],
        combined_srcs['planck_src_top'],
        lw_optical_props['asymmetry_factor'],
    )
  src_and_properties['sfc_src'] = sfc_src

  return src_and_properties


def _reindex_vmr_fields(
    vmr_fields: Mapping[str | int, Array], gas_optics_lib: AbstractLookupGasOptics
) -> dict[int, Array]:
  """Converts the chemical formulas of the gas species to RRTM indices."""
  return {
      k if isinstance(k, int) else gas_optics_lib.idx_gases[k]: v
      for k, v in vmr_fields.items()
  }


def solve_lw(
    pressure: Array,
    temperature: Array,
    molecules: Array,
    optics_lib: optics_base.OpticsScheme,
    atmos_state: AtmosphericState,
    vmr_fields: dict[str, Array] | None = None,
    sfc_temperature: Array | float | None = None,
    cloud_r_eff_liq: Array | None = None,
    cloud_path_liq: Array | None = None,
    cloud_r_eff_ice: Array | None = None,
    cloud_path_ice: Array | None = None,
    use_scan: bool = False,
    cloud_path_liq_per_gpt: Array | None = None,
    cloud_path_ice_per_gpt: Array | None = None,
    aerosol_optics: dict[str, Array] | None = None,
    gpt_chunk: int = DEFAULT_GPT_CHUNK,
    cloud_tau_scale_liq: Array | float | None = None,
    cloud_tau_scale_ice: Array | float | None = None,
    with_clear_sky: bool = False,
    lw_solver: str = LW_SOLVER_TWO_STREAM,
    lw_secant: float = longwave_noscat.LW_NOSCAT_DEFAULT_SECANT,
    gas_optics_mode: str = GAS_OPTICS_MODE_PER_GPT,
    with_aerosol_free: bool = False,
) -> dict[str, Array]:
  """Solves the radiative transfer equation over the longwave spectrum.

  By default this is the two-stream solver; `lw_solver='noscat'` selects the
  absorption-only solver instead (see below).

  Local optical properties like optical depth, single-scattering albedo, and
  asymmetry factor are computed using an optics library and transformed to
  two-stream approximations of reflectance and transmittance. The sources of
  longwave radiation are the Planck sources, which are a function only of
  temperature. To obtain the cell-centered directional Planck sources, the
  sources are first computed at the cell boundaries and the net source
  emanating from the grid cell is determined. Each spectral interval,
  represented by a g-point, is a separate radiative transfer problem, and can
  be computed in parallel. Finally, the independently solved fluxes are summed
  over the full spectrum to yield the final upwelling and downwelling fluxes.

  Args:
    pressure: The pressure field [Pa].
    temperature: The temperature field [K].
    molecules: The number of molecules in an atmospheric grid cell per area
      [molecules/m²].
    optics_lib: An instance of an optics library.
    atmos_state: An instance containing the atmospheric state.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by the chemical formula.
    sfc_temperature: The optional surface temperature represented as either a 2D
      field or as a scalar [K].
    cloud_r_eff_liq: The effective radius of cloud droplets [m].
    cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
      [kg/m²].
    cloud_r_eff_ice: The effective radius of cloud ice particles [m].
    cloud_path_ice: The cloud ice water path in each atmospheric grid cell
      [kg/m²].
    use_scan: Whether to use scan or for loops for the recurrent operation.
    cloud_path_liq_per_gpt: Optional per-g-point cloud liquid water path with
      leading axis of size `n_gpt_lw`, e.g. shape `[n_gpt_lw, nx, ny, nz]`.
      When provided, the per-g-point slice replaces `cloud_path_liq` inside the
      g-point loop. Intended for McICA, where the upstream sub-column generator
      produces a separate binary cloud profile per g-point.
    cloud_path_ice_per_gpt: Same as above, for ice water path.
    aerosol_optics: Optional dictionary of per-band aerosol optical properties.
      Must contain keys `'optical_depth'`, `'ssa'`, and `'asymmetry_factor'`,
      each shaped `[n_bnd_lw, nx, ny, nz]`. The per-g-point band slice (using
      `gas_optics_lw.g_point_to_bnd`) is combined with the gas+cloud optical
      properties via the mass-weighted mix in
      `OpticsScheme.combine_optical_properties`. Aerosol values are passed
      through as-is (no delta-scaling). Requires the `RRTMOptics` scheme.
    gpt_chunk: Number of g-points solved per iteration of the spectral loop.
      Purely a performance knob -- g-points are independent problems, so the
      only thing it changes is the order in which their fluxes are summed. See
      `_solve_over_gpoints` for why it exists and what it trades.
    cloud_tau_scale_liq: Optional multiplier of the liquid cloud optical depth,
      e.g. a sub-grid inhomogeneity factor: a scalar or an array broadcastable
      against the cloud path. It scales the optical depth after the table
      lookup while the single-scattering albedo and asymmetry factor keep the
      physical per-phase weighting (see
      `cloud_optics.compute_optical_properties`). It applies to the per-g-point
      cloud paths too. Has no effect in the gray atmosphere, which has no cloud
      optics.
    cloud_tau_scale_ice: Same as above, for the ice cloud optical depth.
    gas_optics_mode: `'per_gpt'` (default) evaluates the gas optics inside
      the spectral loop, a g-point (or `gpt_chunk` of them) at a time, as every
      earlier version did. `'all_gpts'` evaluates it for every g-point before
      the loop, with the g-point innermost and each table read a row gather over
      the g-points, and the loop takes its g-points' slices from the result
      (`RRTMOptics.compute_gas_optics_all_gpts`). The fluxes are the same; the
      cost moves from per-g-point gather throughput to the memory of the
      precomputed arrays (`n_gpt` fields each of the optical depth and of the
      Planck fraction -- the Rayleigh optical depth in the shortwave).
      Requires the `RRTMOptics` scheme. The precomputed stage reads the tables
      by row gathers, or with the TPU kernels (`rrtmgp.backend`) by
      contractions on the matrix unit.

      **With `'all_gpts'`, `use_scan=True` and `gpt_chunk > 1` a 3D
      atmosphere is solved by `solve_lw_zc`**, on every backend: the same
      equations and outputs, with the atmosphere laid out levels-leading and
      columns-minor for the solve (`_uses_column_minor`). It is the faster
      program on each backend measured -- e.g. 108 against 243 ms per T63L47
      radiation call (LW + SW) on an A100 -- and agrees with the level-minor
      solve to float32 rounding. `two_stream._column_minor(False)` forces the
      level-minor solve, for comparisons.
    with_clear_sky: Also solve the cloud-free problem (gas + aerosol, the same
      aerosol mix-in as the all-sky problem) and return it under `clear_`
      prefixed keys. The gas optics and the Planck sources, which dominate the
      cost of a g-point and are identical in the two problems, are evaluated
      once per g-point; only the cloud mix-in, the cell properties and the
      transport are done twice. The clear-sky fluxes equal those of a separate
      call with no cloud arguments, and the all-sky fluxes are those of a call
      without this flag. Requires an optics scheme that implements
      `add_cloud_optical_properties` (both shipped ones do). When there is no
      cloud to remove (no cloud paths, or the gray atmosphere) the two problems
      are one and the all-sky fluxes are returned for both.
    lw_solver: `'two_stream'` (the default) solves the two-stream equations
      with the gas, cloud and aerosol single-scattering albedo and asymmetry
      factor. `'noscat'` solves RTE's no-scattering equation instead
      (`rrtmgp.rte.longwave_noscat`, a transcription of
      `lw_solver_noscat_oneangle`): one propagation angle, and an absorption
      optical depth `tau (1 - ssa)` formed from the combined gas, cloud and
      aerosol properties, so cloud and aerosol scattering is dropped, as in
      ECHAM6's `mo_lrtm_solver.f90`. Everything upstream of the transport (gas
      optics, Planck sources, cloud and aerosol mix-in, McICA paths, the
      clear-sky sharing, `gpt_chunk`) is common to both.
    lw_secant: Secant of the propagation angle of the `'noscat'` solver; the
      default is upstream RTE's one-point quadrature, `1 / 0.6096748751`.
      Ignored by the two-stream solver, which has the diffusivity factor 1.66
      built in.
    with_aerosol_free: Also solve the aerosol-free twin of every problem (the
      all-sky one, and the clear-sky one under `with_clear_sky`): the same
      optical properties with no `aerosol_optics` mixed in, and return it
      under `aerosol_free_` prefixed keys. As for the clear sky, the gas
      optics, Planck sources and cloud mix-in are shared and only the cell
      properties and the transport are repeated. The twin equals a separate
      call without `aerosol_optics`; without `aerosol_optics` it is the
      original problem, returned under both names at no extra cost.

  Returns:
    A dictionary with the following entries (in units of W/m²):
      `flux_up`: The upwelling longwave radiative flux at cell face i - 1/2.
      `flux_down`: The downwelling longwave radiative flux at face i - 1/2.
      `flux_net`: The net longwave radiative flux at face i - 1/2.
    With `with_clear_sky`, also `clear_flux_up`, `clear_flux_down` and
    `clear_flux_net`: the same three fields for the cloud-free atmosphere.
    With `with_aerosol_free`, also every returned field once more under
    `aerosol_free_` (so `aerosol_free_flux_up`, ..., and with both flags
    `aerosol_free_clear_flux_up`, ...) for the problem without aerosol.
  """
  if lw_solver not in LW_SOLVERS:
    raise ValueError(
        f'lw_solver must be one of {LW_SOLVERS}, got {lw_solver!r}.'
    )
  optics_lib = cast(optics.RRTMOptics | optics.GrayAtmosphereOptics, optics_lib)
  hoist_gas_optics = _check_gas_optics_mode(gas_optics_mode, optics_lib)
  if vmr_fields is not None:
    # Convert the chemical formulas of the gas species to RRTM-consistent
    # numerical identifiers.
    vmr_fields = _reindex_vmr_fields(vmr_fields, optics_lib.gas_optics_lw)

  if _uses_column_minor(
      optics_lib, temperature, hoist_gas_optics, use_scan, gpt_chunk
  ):
    return _solve_3d_via_zc(
        solve_lw_zc,
        pressure,
        temperature,
        molecules,
        optics_lib,
        atmos_state,
        vmr_fields=vmr_fields,
        sfc_temperature=sfc_temperature,
        cloud_r_eff_liq=cloud_r_eff_liq,
        cloud_path_liq=cloud_path_liq,
        cloud_r_eff_ice=cloud_r_eff_ice,
        cloud_path_ice=cloud_path_ice,
        cloud_path_liq_per_gpt=cloud_path_liq_per_gpt,
        cloud_path_ice_per_gpt=cloud_path_ice_per_gpt,
        aerosol_optics=aerosol_optics,
        gpt_chunk=gpt_chunk,
        cloud_tau_scale_liq=cloud_tau_scale_liq,
        cloud_tau_scale_ice=cloud_tau_scale_ice,
        with_clear_sky=with_clear_sky,
        lw_solver=lw_solver,
        lw_secant=lw_secant,
        with_aerosol_free=with_aerosol_free,
    )

  # The gas optics of every g-point, ahead of the spectral loop. Everything in
  # it is independent of the cloud, the aerosol and the g-point loop's order.
  gas_all_gpts = None
  if hoist_gas_optics:
    gas_all_gpts = optics_lib.compute_gas_optics_all_gpts(
        pressure, temperature, molecules, vmr_fields, is_lw=True
    )

  if aerosol_optics is not None:
    assert isinstance(optics_lib, optics.RRTMOptics), (
        'aerosol_optics requires the RRTMOptics scheme (needs per-band'
        ' g_point_to_bnd mapping).'
    )
    g_point_to_bnd_lw = optics_lib.gas_optics_lw.g_point_to_bnd

  def transport(lw_optical_props, srcs, sfc_src, aer_slice):
    # Boundary conditions. `toa_flux_lw` prescribes the *broadband* downwelling
    # flux at the top of the atmosphere, but each g-point is solved as a
    # separate radiative transfer problem and the results are summed over the
    # spectrum below. Feeding the full broadband value to every g-point would
    # therefore return `n_gpt_lw` times the prescribed flux. Split it across the
    # g-points so the spectral sum reproduces it exactly. The shortwave path
    # does the same thing with the physically-derived
    # `solar_fraction_by_gpt` weights; a prescribed broadband longwave flux
    # carries no such spectral information, so it is divided uniformly.
    toa_flux_down_lw = (
        atmos_state.toa_flux_lw / optics_lib.n_gpt_lw
    ) * jnp.ones_like(sfc_src)
    sfc_emissivity_lw = atmos_state.sfc_emis * jnp.ones_like(sfc_src)

    if lw_solver == LW_SOLVER_NOSCAT:
      # Mix in aerosol contributions for this band, if supplied, exactly as the
      # two-stream path does (`_lw_cell_properties`); the solver then keeps only
      # the absorption part of the combined optical depth.
      if aer_slice is not None:
        with jax.named_scope('aerosol_optics'):
          lw_optical_props = optics_lib.combine_optical_properties(
              lw_optical_props, aer_slice
          )
      with jax.named_scope('transport'):
        return longwave_noscat.lw_noscat_transport(
            lw_optical_props['optical_depth'],
            lw_optical_props['ssa'],
            srcs['planck_src'],
            srcs['planck_src_top'],
            srcs['planck_src_bottom'],
            toa_flux_down_lw,
            sfc_src,
            sfc_emissivity_lw,
            use_scan,
            secant=lw_secant,
            # The fence on the unrolled update chain pays only while the gas
            # optics sits inside the g-point loop; see `lw_noscat_transport`.
            fence_chain=gas_optics_mode == GAS_OPTICS_MODE_PER_GPT,
        )

    optical_props_2stream = _lw_cell_properties(
        lw_optical_props, srcs, sfc_src, optics_lib, aer_slice
    )
    with jax.named_scope('transport'):
      fluxes = monochromatic_two_stream.lw_transport(
          optical_props_2stream['t_diff'],
          optical_props_2stream['r_diff'],
          optical_props_2stream['src_up'],
          optical_props_2stream['src_down'],
          toa_flux_down_lw,
          sfc_src,
          sfc_emissivity_lw,
          use_scan,
      )
    # keys: 'flux_up', 'flux_down', 'flux_net'
    return fluxes

  def one_gpt(igpt):
    cpl = (
        cloud_path_liq_per_gpt[igpt]
        if cloud_path_liq_per_gpt is not None
        else cloud_path_liq
    )
    cpi = (
        cloud_path_ice_per_gpt[igpt]
        if cloud_path_ice_per_gpt is not None
        else cloud_path_ice
    )
    aer_slice = None
    if aerosol_optics is not None:
      ibnd = g_point_to_bnd_lw[igpt]
      aer_slice = {k: v[ibnd] for k, v in aerosol_optics.items()}

    srcs, sfc_src = _lw_planck_sources(
        pressure,
        temperature,
        igpt,
        optics_lib,
        vmr_fields,
        sfc_temperature,
        lw_solver,
        planck_fraction=(
            None if gas_all_gpts is None else gas_all_gpts['planck_fraction'][igpt]
        ),
    )

    def gas_properties():
      if gas_all_gpts is not None:
        return optics_lib.gas_optical_properties_from_all_gpts(
            gas_all_gpts, igpt, True
        )
      return optics_lib.compute_gas_optical_properties(
          pressure, temperature, molecules, igpt, True, vmr_fields
      )

    def solve(lw_optical_props, aer):
      return transport(lw_optical_props, srcs, sfc_src, aer)

    if not with_clear_sky:
      if gas_all_gpts is None:
        lw_optical_props = optics_lib.compute_lw_optical_properties(
            pressure,
            temperature,
            molecules,
            igpt,
            vmr_fields,
            cloud_r_eff_liq,
            cpl,
            cloud_r_eff_ice,
            cpi,
            **_cloud_tau_scale_kwargs(cloud_tau_scale_liq, cloud_tau_scale_ice),
        )
      else:
        # The same composition `compute_lw_optical_properties` makes: the gas
        # properties, with the cloud mixed in when there is cloud.
        lw_optical_props = gas_properties()
        if cpl is not None or cpi is not None:
          lw_optical_props = optics_lib.add_cloud_optical_properties(
              igpt,
              lw_optical_props,
              True,
              cloud_r_eff_liq,
              cpl,
              cloud_r_eff_ice,
              cpi,
              **_cloud_tau_scale_kwargs(
                  cloud_tau_scale_liq, cloud_tau_scale_ice
              ),
          )
      return _solve_problems(
          solve, lw_optical_props, None, aer_slice, False, with_aerosol_free
      )

    # The gas optics is the expensive, cloud-independent part: do it once and
    # form the clear-sky and all-sky optical properties from it.
    gas_props = gas_properties()
    has_cloud = cpl is not None or cpi is not None
    all_sky_props = gas_props
    if has_cloud:
      all_sky_props = optics_lib.add_cloud_optical_properties(
          igpt,
          gas_props,
          True,
          cloud_r_eff_liq,
          cpl,
          cloud_r_eff_ice,
          cpi,
          **_cloud_tau_scale_kwargs(cloud_tau_scale_liq, cloud_tau_scale_ice),
      )
    # `all_sky_props is gas_props` when there is no cloud to remove, and
    # `_solve_problems` then solves the one problem once.
    return _solve_problems(
        solve, all_sky_props, gas_props, aer_slice, True, with_aerosol_free
    )

  flux_keys = [
      f'{prefix}{key}'
      for prefix in _problem_prefixes(with_clear_sky, with_aerosol_free)
      for key in ('flux_up', 'flux_down', 'flux_net')
  ]
  init_val = {key: jnp.zeros_like(temperature) for key in flux_keys}

  # The top halo face is the top of the atmosphere, and the solver already
  # produces the correct value there: the downward recurrence seeds
  # `flux_down[..., -1]` with the prescribed incoming flux, and `flux_up` at
  # that face follows from the shifted albedo and aggregate emission of the
  # whole column below. It therefore must not be overwritten -- an earlier
  # quadratic extrapolation from the interior did, which replaced the exact
  # downwelling boundary value with a spurious nonzero flux and degraded the
  # net flux (and hence the top layer's heating rate) at the top of the
  # atmosphere. See `toa_flux_test.py` and issue #19.
  return _solve_over_gpoints(
      one_gpt, optics_lib.n_gpt_lw, gpt_chunk, init_val
  )


def _sw_sfc_albedo_by_band(
    albedo: Array | float,
    plane: tuple[int, ...],
    optics_lib: optics_base.OpticsScheme,
    name: str,
) -> Array | None:
  """Resolve a shortwave surface albedo into a per-band table, if it is one.

  A per-column albedo -- anything that broadcasts against the `[nx, ny]`
  surface plane, as `sfc_alb` always has -- returns `None` and is broadcast
  inside the g-point loop exactly as `sfc_alb` is. Otherwise the albedo must
  carry a trailing band axis, `(..., n_bnd_sw)` with the leading axes
  broadcastable against the plane, and is returned as a `[n_bnd_sw, nx, ny]`
  table so the g-point loop takes its band with a single gather along the
  leading axis. A rank-3 array never broadcasts against the plane, so
  `(nx, ny, n_bnd_sw)` is always read as per band. A `(n_bnd_sw,)` vector is
  one spectrum for every column (for instance a single column's albedos under
  a column `vmap`), except on a plane whose last extent happens to be
  `n_bnd_sw`, where it broadcasts and is therefore read as per column.
  """
  shape = tuple(np.shape(albedo))
  try:
    if np.broadcast_shapes(shape, plane) == plane:
      return None
  except ValueError:
    pass
  if not isinstance(optics_lib, optics.RRTMOptics):
    raise ValueError(
        f'A per-band {name} requires the RRTMOptics scheme (needs the'
        ' g_point_to_bnd mapping).'
    )
  n_bnd = optics_lib.gas_optics_sw.n_bnd
  try:
    band_leading_ok = (
        len(shape) >= 1
        and shape[-1] == n_bnd
        and np.broadcast_shapes(shape[:-1], plane) == plane
    )
  except ValueError:
    band_leading_ok = False
  if not band_leading_ok:
    raise ValueError(
        f'{name} has shape {shape}, which is neither broadcastable against the'
        f' surface plane {plane} nor per band with a trailing axis of'
        f' n_bnd_sw = {n_bnd} entries, (..., {n_bnd}).'
    )
  by_band = jnp.broadcast_to(jnp.asarray(albedo), plane + (n_bnd,))
  return jnp.moveaxis(by_band, -1, 0)


def solve_sw(
    pressure: Array,
    temperature: Array,
    molecules: Array,
    optics_lib: optics_base.OpticsScheme,
    atmos_state: AtmosphericState,
    vmr_fields: dict[str, Array] | None = None,
    cloud_r_eff_liq: Array | None = None,
    cloud_path_liq: Array | None = None,
    cloud_r_eff_ice: Array | None = None,
    cloud_path_ice: Array | None = None,
    use_scan: bool = False,
    cloud_path_liq_per_gpt: Array | None = None,
    cloud_path_ice_per_gpt: Array | None = None,
    aerosol_optics: dict[str, Array] | None = None,
    gpt_chunk: int = DEFAULT_GPT_CHUNK,
    cloud_tau_scale_liq: Array | float | None = None,
    cloud_tau_scale_ice: Array | float | None = None,
    sfc_alb_dir: Array | float | None = None,
    sfc_alb_dif: Array | float | None = None,
    with_clear_sky: bool = False,
    gas_optics_mode: str = GAS_OPTICS_MODE_PER_GPT,
    with_aerosol_free: bool = False,
) -> dict[str, Array]:
  """Solves the two-stream radiative transfer equation for shortwave.

  Local optical properties like optical depth, single-scattering albedo, and
  asymmetry factor are computed using an optics library and transformed to
  two-stream approximations of reflectance and transmittance. The sources of
  shortwave radiation are determined by the diffuse propagation of direct
  solar radiation through the layered atmosphere. Each spectral interval,
  represented by a g-point, is a separate radiative transfer problem, and can
  be computed in parallel. Finally, the independently solved fluxes are summed
  over the full spectrum to yield the final upwelling and downwelling fluxes.

  Args:
    pressure: The pressure field [Pa].
    temperature: The temperature field [K].
    molecules: The number of molecules in an atmospheric grid cell per area
      [molecules/m²].
    optics_lib: An instance of an optics library.
    atmos_state: An instance containing the atmospheric state.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index.
    cloud_r_eff_liq: The effective radius of cloud droplets [m].
    cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
      [kg/m²].
    cloud_r_eff_ice: The effective radius of cloud ice particles [m].
    cloud_path_ice: The cloud ice water path in each atmospheric grid cell
      [kg/m²].
    use_scan: Whether to use scan or for loops for the recurrent operation.
    cloud_path_liq_per_gpt: Optional per-g-point cloud liquid water path with
      leading axis of size `n_gpt_sw`, e.g. shape `[n_gpt_sw, nx, ny, nz]`.
      When provided, the per-g-point slice replaces `cloud_path_liq` inside the
      g-point loop. Intended for McICA, where the upstream sub-column generator
      produces a separate binary cloud profile per g-point.
    cloud_path_ice_per_gpt: Same as above, for ice water path.
    aerosol_optics: Optional dictionary of per-band aerosol optical properties.
      Must contain keys `'optical_depth'`, `'ssa'`, and `'asymmetry_factor'`,
      each shaped `[n_bnd_sw, nx, ny, nz]`. The per-g-point band slice (using
      `gas_optics_sw.g_point_to_bnd`) is combined with the gas+cloud optical
      properties via the mass-weighted mix in
      `OpticsScheme.combine_optical_properties`. Aerosol values are passed
      through as-is (no delta-scaling). Requires the `RRTMOptics` scheme.
    gpt_chunk: Number of g-points solved per iteration of the spectral loop.
      Purely a performance knob -- g-points are independent problems, so the
      only thing it changes is the order in which their fluxes are summed. See
      `_solve_over_gpoints` for why it exists and what it trades.
    cloud_tau_scale_liq: Optional multiplier of the liquid cloud optical depth,
      e.g. a sub-grid inhomogeneity factor: a scalar or an array broadcastable
      against the cloud path. It scales the optical depth after the table
      lookup while the single-scattering albedo and asymmetry factor keep the
      physical per-phase weighting (see
      `cloud_optics.compute_optical_properties`). It applies to the per-g-point
      cloud paths too. Has no effect in the gray atmosphere, which has no cloud
      optics.
    cloud_tau_scale_ice: Same as above, for the ice cloud optical depth.
    sfc_alb_dir: Optional surface albedo for the direct solar beam. Either a
      per-column field, broadcastable against the `[nx, ny]` surface plane
      like `atmos_state.sfc_alb`, or a per-band one with a trailing axis of
      `n_bnd_sw` entries, `(..., n_bnd_sw)`, e.g. `[nx, ny, n_bnd_sw]` or
      `[n_bnd_sw]` (one spectrum for every column, e.g. a single column under
      a column `vmap`); each g-point uses its band's value. An array that broadcasts against the plane is read as
      per column, so a rank-3 array is the unambiguous per-band form. When
      omitted, `atmos_state.sfc_alb` is used. A per-band albedo requires the
      `RRTMOptics` scheme.
    sfc_alb_dif: Same as above, for diffuse light (the surface reflection of
      the downwelling diffuse flux).
    gas_optics_mode: `'per_gpt'` (default) evaluates the gas optics inside
      the spectral loop, a g-point (or `gpt_chunk` of them) at a time, as every
      earlier version did. `'all_gpts'` evaluates it for every g-point before
      the loop, with the g-point innermost and each table read a row gather over
      the g-points, and the loop takes its g-points' slices from the result
      (`RRTMOptics.compute_gas_optics_all_gpts`). The fluxes are the same; the
      cost moves from per-g-point gather throughput to the memory of the
      precomputed arrays (`n_gpt` fields each of the optical depth and of the
      Planck fraction -- the Rayleigh optical depth in the shortwave).
      Requires the `RRTMOptics` scheme. The precomputed stage reads the tables
      by row gathers, or with the TPU kernels (`rrtmgp.backend`) by
      contractions on the matrix unit.

      **With `'all_gpts'`, `use_scan=True` and `gpt_chunk > 1` a 3D
      atmosphere is solved by `solve_sw_zc`**, on every backend: the same
      equations and outputs, with the atmosphere laid out levels-leading and
      columns-minor for the solve (`_uses_column_minor`). It is the faster
      program on each backend measured -- e.g. 108 against 243 ms per T63L47
      radiation call (LW + SW) on an A100 -- and agrees with the level-minor
      solve to float32 rounding. `two_stream._column_minor(False)` forces the
      level-minor solve, for comparisons.
    with_clear_sky: Also solve the cloud-free problem (gas + aerosol, the same
      aerosol mix-in, surface albedos and night masking as the all-sky problem)
      and return it under `clear_` prefixed keys. The gas optics, which
      dominate the cost of a g-point and are identical in the two problems, are
      evaluated once per g-point; only the cloud mix-in, the cell properties
      and the transport are done twice. The clear-sky fluxes equal those of a
      separate call with no cloud arguments, and the all-sky fluxes are those of
      a call without this flag. Requires an optics scheme that implements
      `add_cloud_optical_properties` (both shipped ones do). When there is no
      cloud to remove (no cloud paths, or the gray atmosphere) the two problems
      are one and the all-sky fluxes are returned for both.

  Returns:
    A dictionary with the following entries (in units of W/m²):
      `flux_up`: The upwelling shortwave radiative flux at cell face i - 1/2.
      `flux_down`: The downwelling shortwave radiative flux at face i - 1/2.
        This is the total, direct beam plus diffuse.
      `flux_net`: The net shortwave radiative flux at face i - 1/2.
      `flux_down_dir_sfc`: The direct-beam downwelling flux incident on the
        surface, a `[nx, ny]` field. It is the part of the surface value of
        `flux_down` (`flux_down[:, :, 1]`) that arrives unscattered, so the
        diffuse part is their difference.
    With `with_clear_sky`, also `clear_flux_up`, `clear_flux_down`,
    `clear_flux_net` and `clear_flux_down_dir_sfc`: the same four fields for
    the cloud-free atmosphere.
    With `with_aerosol_free`, also every returned field once more under
    `aerosol_free_` (with both flags `aerosol_free_clear_...` as well) for
    the same problem with no `aerosol_optics` mixed in; see `solve_lw`.
  """
  zenith = atmos_state.zenith
  optics_lib = cast(optics.RRTMOptics | optics.GrayAtmosphereOptics, optics_lib)
  hoist_gas_optics = _check_gas_optics_mode(gas_optics_mode, optics_lib)
  if vmr_fields is not None:
    # Convert the chemical formulas of the gas species to RRTM-consistent
    # numerical identifiers.
    vmr_fields = _reindex_vmr_fields(vmr_fields, optics_lib.gas_optics_sw)

  if _uses_column_minor(
      optics_lib, temperature, hoist_gas_optics, use_scan, gpt_chunk
  ):
    plane = temperature.shape[:-1]

    def albedo(alb, name):
      by_band = (
          None
          if alb is None
          else _sw_sfc_albedo_by_band(alb, plane, optics_lib, name)
      )
      return alb if by_band is None else by_band

    return _solve_3d_via_zc(
        solve_sw_zc,
        pressure,
        temperature,
        molecules,
        optics_lib,
        atmos_state,
        vmr_fields=vmr_fields,
        cloud_r_eff_liq=cloud_r_eff_liq,
        cloud_path_liq=cloud_path_liq,
        cloud_r_eff_ice=cloud_r_eff_ice,
        cloud_path_ice=cloud_path_ice,
        cloud_path_liq_per_gpt=cloud_path_liq_per_gpt,
        cloud_path_ice_per_gpt=cloud_path_ice_per_gpt,
        aerosol_optics=aerosol_optics,
        gpt_chunk=gpt_chunk,
        cloud_tau_scale_liq=cloud_tau_scale_liq,
        cloud_tau_scale_ice=cloud_tau_scale_ice,
        sfc_alb_dir=albedo(sfc_alb_dir, 'sfc_alb_dir'),
        sfc_alb_dif=albedo(sfc_alb_dif, 'sfc_alb_dif'),
        with_clear_sky=with_clear_sky,
        with_aerosol_free=with_aerosol_free,
    )

  # The gas optics of every g-point, ahead of the spectral loop (see `solve_lw`).
  gas_all_gpts = None
  if hoist_gas_optics:
    gas_all_gpts = optics_lib.compute_gas_optics_all_gpts(
        pressure, temperature, molecules, vmr_fields, is_lw=False
    )

  if aerosol_optics is not None:
    assert isinstance(optics_lib, optics.RRTMOptics), (
        'aerosol_optics requires the RRTMOptics scheme (needs per-band'
        ' g_point_to_bnd mapping).'
    )
    g_point_to_bnd_sw = optics_lib.gas_optics_sw.g_point_to_bnd

  # Surface albedos. Each of the direct and diffuse albedos is either per
  # column, broadcast inside the g-point loop as `sfc_alb` always has been, or
  # a per-band table resolved here, outside the loop, into a `[n_bnd_sw, nx,
  # ny]` layout, so the only per-g-point work it adds is one gather of the
  # g-point's band.
  plane = tuple(temperature.shape[:2])
  alb_dir_by_band = alb_dif_by_band = None
  if sfc_alb_dir is not None:
    alb_dir_by_band = _sw_sfc_albedo_by_band(
        sfc_alb_dir, plane, optics_lib, 'sfc_alb_dir'
    )
  if sfc_alb_dif is not None:
    alb_dif_by_band = _sw_sfc_albedo_by_band(
        sfc_alb_dif, plane, optics_lib, 'sfc_alb_dif'
    )
  if alb_dir_by_band is not None or alb_dif_by_band is not None:
    g_point_to_bnd_sw = optics_lib.gas_optics_sw.g_point_to_bnd

  # Sun-at-or-below-horizon handling. This used to be a `jax.lax.cond` that
  # skipped the entire g-point solve at night. That short-circuit is silently
  # destroyed the moment a caller `vmap`s this solver over columns with a
  # per-column `zenith` (the GCM use case): `vmap` lowers a `cond` with a
  # batched predicate into a `select` that runs *both* branches and broadcasts
  # every operand the day branch captures across the whole column batch --
  # including the loop-invariant gas-optics `kmajor` table (~3 MB), which
  # balloons to ~31 GB at ~9000 columns (issue #8). Computing unconditionally
  # keeps those tables a single shared operand under `vmap`. To stay finite for
  # the night columns (the raw `exp(-tau / cos(zenith))` terms in the shortwave
  # solve diverge to +inf once `cos(zenith) <= 0`), the solve is fed a
  # horizon-clamped zenith and the night columns are zeroed out afterwards.
  if jnp.ndim(zenith) > 0:
    zenith = jnp.broadcast_to(jnp.asarray(zenith), plane)
  night = zenith >= 0.5 * jnp.pi
  # Overhead sun (cos = 1) is the most benign valid angle; the resulting night
  # fluxes are masked to zero below, so this placeholder never reaches output.
  safe_zenith = jnp.where(night, jnp.zeros_like(jnp.asarray(zenith)), zenith)

  def one_gpt(igpt):
    cpl = (
        cloud_path_liq_per_gpt[igpt]
        if cloud_path_liq_per_gpt is not None
        else cloud_path_liq
    )
    cpi = (
        cloud_path_ice_per_gpt[igpt]
        if cloud_path_ice_per_gpt is not None
        else cloud_path_ice
    )
    aer_slice = None
    if aerosol_optics is not None:
      ibnd = g_point_to_bnd_sw[igpt]
      aer_slice = {k: v[ibnd] for k, v in aerosol_optics.items()}

    # Create an xy plane for the surface albedo and top-of-atmospehre flux, but
    # keep the same horizontal sharding as the temperature. Without a separate
    # direct or diffuse albedo both are this one plane, as they always were.
    sfc_albedo = atmos_state.sfc_alb * jnp.ones_like(temperature)[:, :, 0]

    def resolve_albedo(albedo, by_band):
      if albedo is None:
        return sfc_albedo
      if by_band is not None:
        return by_band[g_point_to_bnd_sw[igpt]]
      return albedo * jnp.ones_like(temperature)[:, :, 0]

    sfc_albedo_dir = resolve_albedo(sfc_alb_dir, alb_dir_by_band)
    sfc_albedo_dif = resolve_albedo(sfc_alb_dif, alb_dif_by_band)

    # Monochromatic top of atmosphere flux.
    solar_flux = atmos_state.irrad * optics_lib.solar_fraction_by_gpt[igpt]
    toa_flux = solar_flux * jnp.ones_like(temperature)[:, :, 0]

    def solve_optical_props(sw_optical_props, aer):
      # Mix in aerosol contributions for this band, if supplied. Done after the
      # gas+cloud combination (where cloud has already been delta-scaled in SW),
      # so the aerosol tau/ssa/g are passed through as-is.
      if aer is not None:
        with jax.named_scope('aerosol_optics'):
          sw_optical_props = optics_lib.combine_optical_properties(
              sw_optical_props, aer
          )
      with jax.named_scope('cell'):
        optical_props_2stream = monochromatic_two_stream.sw_cell_properties(
            safe_zenith,
            sw_optical_props['optical_depth'],
            sw_optical_props['ssa'],
            sw_optical_props['asymmetry_factor'],
        )

      with jax.named_scope('sources'):
        sources_2stream = monochromatic_two_stream.sw_cell_source(
            t_dir=optical_props_2stream['t_dir'],
            r_dir=optical_props_2stream['r_dir'],
            optical_depth=sw_optical_props['optical_depth'],
            toa_flux=toa_flux,
            sfc_albedo_direct=sfc_albedo_dir,
            zenith=safe_zenith,
            use_scan=use_scan,
        )

      with jax.named_scope('transport'):
        sw_fluxes = monochromatic_two_stream.sw_transport(
            t_diff=optical_props_2stream['t_diff'],
            r_diff=optical_props_2stream['r_diff'],
            src_up=sources_2stream['src_up'],
            src_down=sources_2stream['src_down'],
            sfc_src=sources_2stream['sfc_src'],
            sfc_albedo=sfc_albedo_dif,
            flux_down_dir=sources_2stream['flux_down_dir'],
            use_scan=use_scan,
        )
      # The direct beam reaching the surface, for the host model's direct /
      # diffuse partition of the surface flux. Only the surface plane is kept,
      # so the spectral sum carries one extra 2D field rather than a 3D one.
      halo_width = 1
      sw_fluxes['flux_down_dir_sfc'] = sources_2stream['flux_down_dir'][
          :, :, halo_width
      ]
      return sw_fluxes

    def gas_properties():
      if gas_all_gpts is not None:
        return optics_lib.gas_optical_properties_from_all_gpts(
            gas_all_gpts, igpt, False
        )
      return optics_lib.compute_gas_optical_properties(
          pressure, temperature, molecules, igpt, False, vmr_fields
      )

    if not with_clear_sky:
      if gas_all_gpts is None:
        sw_optical_props = optics_lib.compute_sw_optical_properties(
            pressure,
            temperature,
            molecules,
            igpt,
            vmr_fields,
            cloud_r_eff_liq,
            cpl,
            cloud_r_eff_ice,
            cpi,
            **_cloud_tau_scale_kwargs(cloud_tau_scale_liq, cloud_tau_scale_ice),
        )
      else:
        sw_optical_props = gas_properties()
        if cpl is not None or cpi is not None:
          sw_optical_props = optics_lib.add_cloud_optical_properties(
              igpt,
              sw_optical_props,
              False,
              cloud_r_eff_liq,
              cpl,
              cloud_r_eff_ice,
              cpi,
              **_cloud_tau_scale_kwargs(
                  cloud_tau_scale_liq, cloud_tau_scale_ice
              ),
          )
      return _solve_problems(
          solve_optical_props, sw_optical_props, None, aer_slice, False,
          with_aerosol_free,
      )

    # The gas optics (absorption + Rayleigh) is the expensive, cloud-independent
    # part: do it once and form the clear-sky and all-sky optical properties
    # from it.
    gas_props = gas_properties()
    has_cloud = cpl is not None or cpi is not None
    all_sky_props = gas_props
    if has_cloud:
      all_sky_props = optics_lib.add_cloud_optical_properties(
          igpt,
          gas_props,
          False,
          cloud_r_eff_liq,
          cpl,
          cloud_r_eff_ice,
          cpi,
          **_cloud_tau_scale_kwargs(cloud_tau_scale_liq, cloud_tau_scale_ice),
      )
    # `all_sky_props is gas_props` when there is no cloud to remove, and
    # `_solve_problems` then solves the one problem once.
    return _solve_problems(
        solve_optical_props, all_sky_props, gas_props, aer_slice, True,
        with_aerosol_free,
    )

  one_problem = {
      key: jnp.zeros_like(temperature)
      for key in ('flux_up', 'flux_down', 'flux_net')
  }
  one_problem['flux_down_dir_sfc'] = jnp.zeros_like(temperature)[:, :, 0]
  fluxes_0 = {
      f'{prefix}{key}': val
      for prefix in _problem_prefixes(with_clear_sky, with_aerosol_free)
      for key, val in one_problem.items()
  }

  # As in `solve_lw`, the top halo face is the top of the atmosphere and the
  # solver already produces the correct value there, so it is left untouched.
  fluxes = _solve_over_gpoints(
      one_gpt, optics_lib.n_gpt_sw, gpt_chunk, fluxes_0
  )
  # Zero out columns where the sun is at or below the horizon. `night` is a
  # scalar (single column / column `vmap`) or a 2D `[nx, ny]` mask over the
  # horizontal plane.
  fluxes = {
      key: jnp.where(
          night[..., jnp.newaxis]
          if (jnp.ndim(night) == 2 and val.ndim == 3)
          else night,
          jnp.zeros_like(val),
          val,
      )
      for key, val in fluxes.items()
  }
  return fluxes


def compute_heating_rate(
    flux_net: Array,
    pressure: Array,
    q_v: Array | None = None,
    *,
    axis: int = 2,
) -> Array:
  """Computes cell-center heating rate from pressure and net radiative flux.

  The net radiative flux corresponds to the bottom cell face. The difference
  of the net flux at the top face and that at the bottom face gives the total
  net flux out of the grid cell. Using the pressure difference across the grid
  cell, the net flux can be converted to a heating rate, in K/s.

  Args:
    flux_net: The net flux at the bottom face [W/m²].
    pressure: The pressure field [Pa].
    q_v: Water vapor specific humidity [kg/kg], used for the moist heat
      capacity. When omitted the dry-air value is used, which overestimates
      the heating rate by 0.84·q_v — 1.7% at 20 g/kg.
    axis: The vertical axis: 2 for `(nx, ny, nz)` fields, 0 for the
      column-minor `(nz, n_columns)` ones of `solve_lw_zc`.

  Returns:
    The heating rate of the grid cell [K/s].
  """
  # Compute the centered pressure difference in z:
  #   dp_{i,j,k} = (p_{i,j,k+1} - p_{i,j,k-1}) / 2.
  # This is an approximation to the pressure difference across the cell, which
  # would use the pressure at the upper and lower faces, but it should be ok.
  dp = 0.5 * kernel_ops.centered_difference(pressure, dim=axis)

  # Compute the forward pressure difference of fluxes on faces (like a
  # derivative of face_to_node).
  dflux = kernel_ops.forward_difference(flux_net, dim=axis)

  # Moist heat capacity: cp = cp_d·(1 - q_v) + cp_v·q_v. Using the dry value
  # makes the heating rate too large by (cp_v/cp_d - 1)·q_v = 0.84·q_v, which
  # is ~1.7% in a tropical boundary layer and biases the low-level longwave
  # cooling everywhere it is moist.
  cp = constants.CP_D
  if q_v is not None:
    cp = cp + (constants.CP_V - constants.CP_D) * q_v

  # Compute the heating rate at the grid cell center in K/s.
  return constants.G * dflux / dp / cp


# ---------------------------------------------------------------------------
# The column-minor solver: `solve_lw_zc` and `solve_sw_zc`.
#
# The same physics as `solve_lw` / `solve_sw`, for an atmosphere laid out
# `(n_levels, n_columns)`: levels leading, columns minor. Keeping the columns
# on the minor axis through the gas optics, the spectral loop and the
# vertical recurrences is what lets every backend vectorise across columns
# (SIMD lanes on a CPU, the 128-lane minor tile dimension of a TPU) and what
# removes the transposes a level-minor layout needs around every vertical
# scan. A host model that keeps its columns minor calls these directly
# (`RRTMGP.compute_heating_rate_zc`); `solve_lw` / `solve_sw` route a 3D
# solve here when the gas optics is hoisted, the recurrences are scanned and
# more than one g-point is solved per iteration (`_uses_column_minor`).
#
# Ahead of the spectral loop: the gas optics of every g-point (levels
# leading, `(nz, n_gpt, nc)`), the face temperatures, the Planck function of
# every band and the cloud Mie properties of every band. Each iteration of the
# loop solves `chunk` g-points as `(nz, chunk, nc)` arrays: it slices its
# g-points out of the gas optics, projects the per-band fields onto its
# g-points (`_band_selector`), forms the cell properties and runs the
# transport (`monochromatic_two_stream.solve_rte_2stream_zc` and siblings).
#
# Measured (one LW + SW call with clear-sky fluxes and McICA clouds, T63L47,
# 18432 columns, jax 0.11): on an A100 at `gpt_chunk=64` 97 ms through
# `compute_heating_rate_zc` and 108 ms through `compute_heating_rate`, against
# 243 ms for `main`'s level-minor scan; on a CPU (64 cores, 2048 columns)
# 1.19 s against 4.08 s; on a TPU v5e 342 ms for two 9216-column calls at
# `gpt_chunk=16`.
# ---------------------------------------------------------------------------

# Override of `_uses_column_minor` for tests and comparisons: `None` decides
# by the placement, `True` / `False` force the column-minor / level-minor
# solve of a 3D atmosphere.
_column_minor_override: contextvars.ContextVar[bool | None] = (
    contextvars.ContextVar('rrtmgp_column_minor', default=None)
)


@contextlib.contextmanager
def _column_minor(enabled: bool | None) -> Iterator[None]:
  """Force (`True`), forbid (`False`) or restore (`None`) the column-minor solve.

  Applies to `solve_lw` / `solve_sw` traced inside the block, and only where
  the column-minor solve can run at all (the RRTMOptics scheme and a 3D
  atmosphere). Meant for tests and A/B comparisons of the two layouts, which
  solve the same equations.

  Args:
    enabled: The routing to use inside the block.

  Yields:
    Nothing; the routing is active until the block exits.
  """
  token = _column_minor_override.set(enabled)
  try:
    yield
  finally:
    _column_minor_override.reset(token)


def _uses_column_minor(
    optics_lib: optics_base.OpticsScheme,
    temperature: Array,
    hoist_gas_optics: bool,
    use_scan: bool,
    gpt_chunk: int,
) -> bool:
  """Whether `solve_lw` / `solve_sw` hand a 3D atmosphere to the `_zc` solve.

  They do for the placement the column-minor solve is built for -- the gas
  optics hoisted out of the spectral loop, the recurrences scanned, and more
  than one g-point per iteration -- on every backend: it is the faster
  program on each one measured (see the comment above). A single g-point per
  iteration, or the unrolled recurrences, keep the level-minor solve, whose
  programs those settings were tuned for.
  """
  if not isinstance(optics_lib, optics.RRTMOptics) or temperature.ndim != 3:
    return False
  if temperature.size == 0:
    return False
  override = _column_minor_override.get()
  if override is not None:
    return override
  return hoist_gas_optics and use_scan and gpt_chunk > 1


# The TPU vector memory one `(nz, chunk, nc)` float32 array of the spectral
# loop may take. The TPU kernels cap `gpt_chunk` to fit, so the vertical
# recurrences of a chunk stay in vector memory instead of round-tripping HBM.
# 32 MiB suits a v5e; set it to `None` to use `gpt_chunk` as given.
TPU_CHUNK_VMEM_BYTES: int | None = 32 * 1024 * 1024


def _column_minor_chunk_plan(
    n_gpt: int, gpt_chunk: int, nz: int, nc: int
) -> tuple[int, int, bool]:
  """`_gpt_chunk_plan` for the column-minor solve.

  With the TPU kernels the chunk is first capped so that a `(nz, chunk, nc)`
  float32 array fits in `TPU_CHUNK_VMEM_BYTES`, in multiples of 8 (the
  second-minor tile dimension). Then, if the spectrum takes more than one
  iteration, the chunk is evened out across them in multiples of 8 when that
  still covers `n_gpt` in the same number of iterations: 112 g-points at
  `gpt_chunk=64` are two chunks of 56, not 64 and a padded 48.

  Returns:
    The chunk size, the number of iterations and whether the last one is
    padded, as `_gpt_chunk_plan`.
  """
  chunk = gpt_chunk
  if backend.use_tpu_kernels() and TPU_CHUNK_VMEM_BYTES and chunk > 8:
    per_gpt = max(nz * nc * 4, 1)
    chunk = min(chunk, max(8, TPU_CHUNK_VMEM_BYTES // per_gpt // 8 * 8))
  cap = chunk
  chunk, n_chunks, needs_mask = _gpt_chunk_plan(n_gpt, cap)
  if n_chunks > 1:
    even = -(-(-(-n_gpt // n_chunks)) // 8) * 8
    if even <= cap and even * n_chunks >= n_gpt:
      chunk = even
      needs_mask = n_chunks * chunk != n_gpt
  return chunk, n_chunks, needs_mask


def _interior(arr: Array, nz: int, axis: int = 0) -> Array:
  """`arr` without its halo levels, if it has them (`nz + 2` on `axis`)."""
  if arr.shape[axis] == nz + 2:
    return jax.lax.slice_in_dim(arr, 1, nz + 1, axis=axis)
  if arr.shape[axis] != nz:
    raise ValueError(
        f'Expected {nz} or {nz + 2} levels on axis {axis}, got shape'
        f' {arr.shape}.'
    )
  return arr


def _pack_chunks(
    arr: Array, gpt_axis: int, n_chunks: int, chunk: int
) -> Array:
  """`(nz, n_gpt, nc)` (or `(n_gpt, nz, nc)`) as `(n_chunks, nz, chunk, nc)`.

  The spectrum is padded to `n_chunks * chunk` by repeating its last g-point,
  which the spectral sum masks out.
  """
  if gpt_axis == 0:
    arr = jnp.swapaxes(arr, 0, 1)
  nz, n_gpt, nc = arr.shape
  pad = n_chunks * chunk - n_gpt
  if pad:
    arr = jnp.concatenate(
        [arr, jnp.repeat(arr[:, -1:, :], pad, axis=1)], axis=1
    )
  return jnp.swapaxes(arr.reshape(nz, n_chunks, chunk, nc), 0, 1)


def _per_column(val: Array | float, nc: int, dtype: jnp.dtype) -> Array:
  """A scalar or `(nc,)` field as `(1, nc)` (a scalar stays `(1, 1)`)."""
  v = jnp.asarray(val, dtype=dtype)
  if v.ndim == 0:
    return v.reshape(1, 1)
  return jnp.broadcast_to(v, (nc,)).reshape(1, nc)


def _tau_scale_zc(
    scale: Array | float | None, nz: int, nc: int
) -> Array | float | None:
  """A `cloud_tau_scale_*` in a shape that broadcasts against `(nz, 1, nc)`."""
  if scale is None or jnp.ndim(scale) == 0:
    return scale
  s = jnp.asarray(scale)
  if s.ndim == 1:
    s = jnp.broadcast_to(s, (nc,))[jnp.newaxis, :]
  elif s.shape[0] not in (1, nz):
    s = _interior(s, nz)
  return s[:, jnp.newaxis, :]


def _band_selector(
    g_point_to_bnd: Array, n_bnd: int, n_chunks: int, chunk: int,
    dtype: jnp.dtype,
) -> Array:
  """Which band each g-point of each chunk is in, for `_by_gpt`.

  Static: `(n_chunks, chunk)` band indices on CPU/GPU, and on the TPU kernels
  `(n_chunks, chunk, n_bnd)` one-hot rows, so the projection is a small
  matmul on the matrix unit rather than a gather.
  """
  g2b = np.asarray(g_point_to_bnd)
  n_gpt = g2b.shape[0]
  bands = g2b[np.minimum(np.arange(n_chunks * chunk), n_gpt - 1)]
  if not backend.use_tpu_kernels():
    return jnp.asarray(bands, dtype=jnp.int32).reshape(n_chunks, chunk)
  one_hot = np.eye(n_bnd, dtype=np.float32)[bands]
  return jnp.asarray(one_hot, dtype=dtype).reshape(n_chunks, chunk, n_bnd)


def _by_gpt(selector: Array, field: Array) -> Array:
  """A per-band field on a chunk's g-points: `(..., n_bnd, nc) -> (..., chunk, nc)`.

  Args:
    selector: One chunk of `_band_selector`.
    field: `(n_bnd, nc)` or `(nz, n_bnd, nc)`.

  Returns:
    `(chunk, nc)` or `(nz, chunk, nc)`.
  """
  if jnp.issubdtype(selector.dtype, jnp.integer):
    return field[..., selector, :]
  highest = jax.lax.Precision.HIGHEST
  if field.ndim == 2:
    return jnp.matmul(selector, field, precision=highest)
  lhs = jnp.broadcast_to(selector, (field.shape[0],) + selector.shape)
  return jax.lax.dot_general(
      lhs,
      field,
      dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      precision=highest,
  )


class _CloudZc:
  """The cloud optics of a column-minor solve, by band and then per chunk.

  The Mie properties of each phase are looked up for every band before the
  spectral loop. With bulk cloud paths the cloud optical properties are formed
  per band there too, and a chunk projects them onto its g-points (and applies
  its McICA masks, if any). With per-g-point (McICA) cloud paths a chunk
  projects the Mie properties and forms the cloud optics of its own g-points.
  Either way the physics is `cloud_optics.combine_phases`.
  """

  def __init__(
      self,
      optics_lib: optics.RRTMOptics,
      is_lw: bool,
      nz: int,
      nc: int,
      n_chunks: int,
      chunk: int,
      dtype: jnp.dtype,
      *,
      r_eff_liq: Array | None,
      path_liq: Array | None,
      r_eff_ice: Array | None,
      path_ice: Array | None,
      path_liq_per_gpt: Array | None,
      path_ice_per_gpt: Array | None,
      mask_per_gpt: Array | None,
      per_gpt_axis: int,
      tau_scale_liq: Array | float | None,
      tau_scale_ice: Array | float | None,
  ):
    self._optics_lib = optics_lib
    self._is_lw = is_lw
    lookup = optics_lib.cloud_optics_lw if is_lw else optics_lib.cloud_optics_sw
    assert lookup is not None
    has_liq = path_liq_per_gpt is not None or path_liq is not None
    has_ice = path_ice_per_gpt is not None or path_ice is not None
    self.per_gpt = path_liq_per_gpt is not None or path_ice_per_gpt is not None
    if self.per_gpt and mask_per_gpt is not None:
      raise ValueError(
          'cloud_mask_per_gpt masks the bulk cloud paths; it cannot be combined'
          ' with per-g-point cloud paths (cloud_path_*_per_gpt), which already'
          ' are per g-point.'
      )
    zeros = jnp.zeros((nz, nc), dtype=dtype)

    def radius(r):
      return zeros if r is None else _interior(r, nz)

    # `(n_bnd, nz, nc)` per field.
    liq, ice = cloud_optics.phase_properties_all_bands(
        lookup,
        radius(r_eff_liq) if has_liq else None,
        radius(r_eff_ice) if has_ice else None,
    )
    self._tau_scale = (
        _tau_scale_zc(tau_scale_liq, nz, nc),
        _tau_scale_zc(tau_scale_ice, nz, nc),
    )

    def per_gpt_or_bulk(per_gpt, bulk):
      if per_gpt is not None:
        return _pack_chunks(
            _interior(per_gpt, nz, axis=1 - per_gpt_axis),
            per_gpt_axis,
            n_chunks,
            chunk,
        )
      return None if bulk is None else _interior(bulk, nz)[:, jnp.newaxis]

    if self.per_gpt:
      # `(nz, n_bnd, nc)`, ready for `_by_gpt`.
      self._phases = tuple(
          None
          if p is None
          else cloud_optics.PhaseProperties(
              *(jnp.swapaxes(x, 0, 1) for x in p)
          )
          for p in (liq, ice)
      )
      # `(n_chunks, nz, chunk, nc)`, or a bulk path `(nz, 1, nc)`.
      self._paths = (
          per_gpt_or_bulk(path_liq_per_gpt, path_liq),
          per_gpt_or_bulk(path_ice_per_gpt, path_ice),
      )
      self._by_band = None
      self._mask = None
    else:
      # The scales broadcast against `(n_bnd, nz, nc)` here.
      scale = tuple(
          s if s is None or jnp.ndim(s) == 0 else jnp.swapaxes(s, 0, 1)
          for s in self._tau_scale
      )
      by_band = self._combine(
          liq,
          ice,
          None if path_liq is None else _interior(path_liq, nz)[jnp.newaxis],
          None if path_ice is None else _interior(path_ice, nz)[jnp.newaxis],
          *scale,
      )
      # `(nz, n_bnd, nc)`.
      self._by_band = {k: jnp.swapaxes(v, 0, 1) for k, v in by_band.items()}
      self._mask = (
          None
          if mask_per_gpt is None
          else _pack_chunks(
              _interior(
                  jnp.asarray(mask_per_gpt, dtype=jnp.bool_),
                  nz,
                  axis=1 - per_gpt_axis,
              ),
              per_gpt_axis,
              n_chunks,
              chunk,
          )
      )

  def _combine(self, liq, ice, path_liq, path_ice, scale_liq, scale_ice):
    props = cloud_optics.combine_phases(
        liq, ice, path_liq, path_ice, scale_liq, scale_ice
    )
    if not self._is_lw:
      props = self._optics_lib._apply_delta_scaling_for_cloud(props)  # pylint: disable=protected-access
    return props

  def chunk_props(self, ichunk: Array, selector: Array) -> dict[str, Array]:
    """The cloud optical properties of chunk `ichunk`, `(nz, chunk, nc)`."""
    if not self.per_gpt:
      props = {k: _by_gpt(selector, v) for k, v in self._by_band.items()}
      if self._mask is not None:
        mask = jax.lax.dynamic_index_in_dim(
            self._mask, ichunk, axis=0, keepdims=False
        )
        props = {
            k: jnp.where(mask, v, jnp.zeros_like(v)) for k, v in props.items()
        }
      return props
    phases = tuple(
        None
        if p is None
        else cloud_optics.PhaseProperties(*(_by_gpt(selector, x) for x in p))
        for p in self._phases
    )
    paths = tuple(
        p
        if p is None or p.ndim == 3
        else jax.lax.dynamic_index_in_dim(p, ichunk, axis=0, keepdims=False)
        for p in self._paths
    )
    return self._combine(*phases, *paths, *self._tau_scale)


def _aerosol_zc(
    aerosol_optics: Mapping[str, Array] | None, nz: int
) -> dict[str, Array] | None:
  """`(n_bnd, nz[+2], nc)` per-band aerosol optics as `(nz, n_bnd, nc)`."""
  if aerosol_optics is None:
    return None
  return {
      k: jnp.swapaxes(_interior(v, nz, axis=1), 0, 1)
      for k, v in aerosol_optics.items()
  }


def _spectral_sum_zc(
    solve_chunk: Callable[[Array, Array], dict[str, Array]],
    selector: Array,
    n_gpt: int,
    n_chunks: int,
    chunk: int,
    needs_mask: bool,
    init: dict[str, Array],
) -> dict[str, Array]:
  """`_solve_over_gpoints` for the column-minor solve.

  Args:
    solve_chunk: `(ichunk, selector) -> fluxes`, each `(..., chunk, nc)`: the
      g-point axis is second-minor.
    selector: `_band_selector` of the whole spectrum.
    n_gpt: The number of g-points.
    n_chunks: The number of chunks.
    chunk: The g-points per chunk.
    needs_mask: Whether the last chunk is padded past `n_gpt`.
    init: Zeroed accumulators, each `solve_chunk` output with the g-point axis
      summed out.

  Returns:
    The sum over the spectrum.
  """
  valid = None
  if needs_mask:
    valid = jnp.asarray(
        (np.arange(n_chunks * chunk) < n_gpt).reshape(n_chunks, chunk, 1),
        dtype=jax.tree.leaves(init)[0].dtype,
    )

  def step(ichunk, total):
    fluxes = solve_chunk(
        ichunk,
        jax.lax.dynamic_index_in_dim(selector, ichunk, axis=0, keepdims=False),
    )
    mask = (
        None
        if valid is None
        else jax.lax.dynamic_index_in_dim(valid, ichunk, axis=0, keepdims=False)
    )

    def add(flux: Array, acc: Array) -> Array:
      if mask is not None:
        flux = flux * mask
      return jnp.sum(flux, axis=-2) + acc

    return jax.tree.map(add, fluxes, total)

  return jax.lax.fori_loop(0, n_chunks, step, init)


def _problem_keys(
    with_clear_sky: bool, with_aerosol_free: bool, keys: Sequence[str]
) -> list[str]:
  return [
      f'{prefix}{key}'
      for prefix in _problem_prefixes(with_clear_sky, with_aerosol_free)
      for key in keys
  ]


def solve_lw_zc(
    pressure: Array,
    temperature: Array,
    molecules: Array,
    optics_lib: optics.RRTMOptics,
    atmos_state: AtmosphericState,
    vmr_fields: Mapping[str | int, Array] | None = None,
    sfc_temperature: Array | float | None = None,
    cloud_r_eff_liq: Array | None = None,
    cloud_path_liq: Array | None = None,
    cloud_r_eff_ice: Array | None = None,
    cloud_path_ice: Array | None = None,
    cloud_path_liq_per_gpt: Array | None = None,
    cloud_path_ice_per_gpt: Array | None = None,
    aerosol_optics: Mapping[str, Array] | None = None,
    gpt_chunk: int = 64,
    cloud_tau_scale_liq: Array | float | None = None,
    cloud_tau_scale_ice: Array | float | None = None,
    with_clear_sky: bool = False,
    lw_solver: str = LW_SOLVER_TWO_STREAM,
    lw_secant: float = longwave_noscat.LW_NOSCAT_DEFAULT_SECANT,
    with_aerosol_free: bool = False,
    *,
    per_gpt_axis: int = 1,
    cloud_mask_per_gpt: Array | None = None,
) -> dict[str, Array]:
  """`solve_lw` for a column-minor `(nz + 2, nc)` atmosphere.

  The same physics, problems and outputs as `solve_lw` with the gas optics
  hoisted and the recurrences scanned, with the levels leading and the
  columns minor throughout (see the comment above `_column_minor`). A clear,
  aerosol-free problem (`ssa = 0`, no reflection) is solved with the
  no-scattering recurrences, which are what the two-stream ones reduce to.

  Args:
    pressure: `(nz + 2, nc)` pressure [Pa], with a halo level at each end.
    temperature: `(nz + 2, nc)` temperature [K], with halos.
    molecules: `(nz + 2, nc)` molecules per area [molecules/m^2], with halos.
    optics_lib: The `RRTMOptics` scheme.
    atmos_state: The atmospheric state; its `zenith`, `irrad`, `sfc_alb`,
      `sfc_emis` and `toa_flux_lw` may be scalars or `(nc,)`.
    vmr_fields: Optional volume mixing ratio fields, keyed by gas name or
      index, each `(nz + 2, nc)`.
    cloud_r_eff_liq: Effective droplet radius [m], `(nz[+2], nc)`.
    cloud_path_liq: Liquid water path [kg/m^2], `(nz[+2], nc)`.
    cloud_r_eff_ice: Effective ice radius [m], `(nz[+2], nc)`.
    cloud_path_ice: Ice water path [kg/m^2], `(nz[+2], nc)`.
    cloud_path_liq_per_gpt: Per-g-point (McICA) liquid water path,
      `(nz[+2], n_gpt, nc)`, or `(n_gpt, nz[+2], nc)` with `per_gpt_axis=0`.
      Replaces `cloud_path_liq` for the liquid phase.
    cloud_path_ice_per_gpt: Same as above, for the ice water path.
    aerosol_optics: Per-band aerosol `'optical_depth'`, `'ssa'` and
      `'asymmetry_factor'`, each `(n_bnd, nz[+2], nc)`.
    gpt_chunk: g-points per iteration of the spectral loop. With the TPU
      kernels it is capped to fit `TPU_CHUNK_VMEM_BYTES` (see
      `_column_minor_chunk_plan`, which also evens chunks out to multiples
      of 8).
    cloud_tau_scale_liq: Multiplier of the liquid cloud optical depth: a
      scalar, `(nc,)` or `(nz[+2], nc)`.
    cloud_tau_scale_ice: Same as above, for ice.
    sfc_temperature: Surface temperature [K], a scalar or `(nc,)`.
    with_clear_sky: Also solve the cloud-free problem (`solve_lw`).
    lw_solver: `'two_stream'` or `'noscat'` (`solve_lw`).
    lw_secant: Secant of the `'noscat'` solver's propagation angle.
    with_aerosol_free: Also solve the aerosol-free twin of every problem.
    per_gpt_axis: The g-point axis of the per-g-point inputs, 1 (default,
      `(nz, n_gpt, nc)`) or 0 (`(n_gpt, nz, nc)`).
    cloud_mask_per_gpt: Optional boolean McICA masks, `(nz[+2], n_gpt, nc)`
      (or `(n_gpt, nz[+2], nc)` with `per_gpt_axis=0`): the bulk cloud paths
      are seen by a g-point only where its mask is set. Cannot be combined
      with per-g-point cloud paths. One byte per (level, g-point, column)
      instead of the four floats of per-g-point paths.

  Returns:
    `(nz + 2, nc)` face fluxes [W/m^2], `flux_up`, `flux_down` and
    `flux_net`: entry `k` is the bottom face of level `k`, so entry 1 is the
    surface and entry `nz + 1` the top of the atmosphere; entry 0 (the face
    below the bottom halo) is zero. The clear-sky and aerosol-free problems
    under their prefixes, as `solve_lw` / `solve_sw`.
  """
  if lw_solver not in LW_SOLVERS:
    raise ValueError(
        f'lw_solver must be one of {LW_SOLVERS}, got {lw_solver!r}.'
    )
  if sfc_temperature is None:
    raise ValueError('solve_lw_zc requires sfc_temperature.')
  nzh, nc = temperature.shape
  nz = nzh - 2
  dtype = temperature.dtype
  lookup = cast(
      lookup_gas_optics_longwave.LookupGasOpticsLongwave,
      optics_lib.gas_optics_lw,
  )
  n_gpt = optics_lib.n_gpt_lw
  chunk, n_chunks, needs_mask = _column_minor_chunk_plan(
      n_gpt, gpt_chunk, nz, nc
  )

  gas = optics_lib.compute_gas_optics_all_gpts(
      pressure,
      temperature,
      molecules,
      None if vmr_fields is None else _reindex_vmr_fields(vmr_fields, lookup),
      is_lw=True,
      z_leading_output=True,
  )
  tau_packed = _pack_chunks(gas['optical_depth'][1:-1], 1, n_chunks, chunk)
  pfrac_packed = _pack_chunks(gas['planck_fraction'], 1, n_chunks, chunk)

  # The Planck function of every band at the faces, the surface and (for the
  # no-scattering solver) the layer centres, in one interpolation.
  sfc_temp = jnp.broadcast_to(jnp.asarray(sfc_temperature, dtype=dtype), (nc,))
  t_face = optics_base.reconstruct_face_values(temperature, sfc_temp, axis=0)[
      0
  ][1:]
  need_layer_src = lw_solver == LW_SOLVER_NOSCAT
  t_stack = [t_face, sfc_temp[jnp.newaxis]]
  if need_layer_src:
    t_stack.append(temperature[1:-1])
  planck = optics_utils.floor_at_zero(
      optics_utils.interpolate_rows_1d(
          lookup.totplnk,
          optics_utils.create_linear_interpolant(
              jnp.concatenate(t_stack, axis=0), lookup.t_planck
          ),
      )
  )
  # `(nz + 1, n_bnd, nc)`, `(n_bnd, nc)` and `(nz, n_bnd, nc)`.
  b_face = jnp.swapaxes(planck[:, : nz + 1], 0, 1)
  b_sfc = planck[:, nz + 1]
  b_layer = jnp.swapaxes(planck[:, nz + 2 :], 0, 1) if need_layer_src else None
  selector = _band_selector(
      lookup.g_point_to_bnd, lookup.n_bnd, n_chunks, chunk, dtype
  )

  has_cloud = any(
      p is not None
      for p in (
          cloud_path_liq,
          cloud_path_ice,
          cloud_path_liq_per_gpt,
          cloud_path_ice_per_gpt,
      )
  )
  cloud = None
  if has_cloud:
    cloud = _CloudZc(
        optics_lib, True, nz, nc, n_chunks, chunk, dtype,
        r_eff_liq=cloud_r_eff_liq, path_liq=cloud_path_liq,
        r_eff_ice=cloud_r_eff_ice, path_ice=cloud_path_ice,
        path_liq_per_gpt=cloud_path_liq_per_gpt,
        path_ice_per_gpt=cloud_path_ice_per_gpt,
        mask_per_gpt=cloud_mask_per_gpt, per_gpt_axis=per_gpt_axis,
        tau_scale_liq=cloud_tau_scale_liq, tau_scale_ice=cloud_tau_scale_ice,
    )
  aerosol = _aerosol_zc(aerosol_optics, nz)

  # The boundary conditions; see `solve_lw` for the split of `toa_flux_lw`.
  toa_flux_down = jnp.broadcast_to(
      _per_column(atmos_state.toa_flux_lw / n_gpt, nc, dtype), (chunk, nc)
  )
  sfc_emissivity = jnp.broadcast_to(
      _per_column(atmos_state.sfc_emis, nc, dtype), (chunk, nc)
  )
  sfc_reflectance = 1.0 - sfc_emissivity

  def solve_chunk(ichunk, sel):
    tau = jax.lax.dynamic_index_in_dim(tau_packed, ichunk, 0, keepdims=False)
    pfrac = jax.lax.dynamic_index_in_dim(
        pfrac_packed, ichunk, 0, keepdims=False
    )
    # The Planck sources of every face from each adjacent layer's spectral
    # mapping, combined into one per level (`lw_combine_sources`).
    b_face_g = _by_gpt(sel, b_face)
    level_src = monochromatic_two_stream.level_source(
        pfrac[:-1] * b_face_g, pfrac[1:] * b_face_g
    )
    level_src_bottom = level_src[:-1]
    level_src_top = level_src[1:]
    sfc_src = pfrac[1] * _by_gpt(sel, b_sfc)
    layer_src = (
        pfrac[1:-1] * _by_gpt(sel, b_layer) if need_layer_src else None
    )

    zeros = jnp.zeros_like(tau)
    gas_props = {'optical_depth': tau, 'ssa': zeros, 'asymmetry_factor': zeros}
    all_sky_props = gas_props
    if cloud is not None:
      all_sky_props = optics_lib.combine_optical_properties(
          gas_props, cloud.chunk_props(ichunk, sel)
      )
    aer = (
        None
        if aerosol is None
        else {k: _by_gpt(sel, v) for k, v in aerosol.items()}
    )
    sfc_emission = np.pi * sfc_emissivity * sfc_src

    def solve(props, aer_props):
      gas_only = props is gas_props and aer_props is None
      if aer_props is not None:
        props = optics_lib.combine_optical_properties(props, aer_props)
      if lw_solver == LW_SOLVER_NOSCAT:
        return longwave_noscat.lw_noscat_transport_zc(
            props['optical_depth'],
            props['ssa'],
            layer_src,
            level_src_top,
            level_src_bottom,
            toa_flux_down,
            sfc_src,
            sfc_emissivity,
            secant=lw_secant,
        )
      if gas_only:
        # No scattering anywhere: the two-stream equations without
        # reflection (`lw_pure_absorption_two_stream_cell`).
        cell = monochromatic_two_stream.lw_pure_absorption_two_stream_cell(
            props['optical_depth'], level_src_bottom, level_src_top
        )
        return monochromatic_two_stream.solve_rte_absorbing_zc(
            cell['t_diff'],
            cell['src_up'],
            cell['src_down'],
            toa_flux_down,
            sfc_emission,
            sfc_reflectance,
        )
      cell = monochromatic_two_stream.lw_cell_source_and_properties(
          props['optical_depth'],
          props['ssa'],
          level_src_bottom,
          level_src_top,
          props['asymmetry_factor'],
      )
      return monochromatic_two_stream.solve_rte_2stream_zc(
          cell['t_diff'],
          cell['r_diff'],
          cell['src_up'],
          cell['src_down'],
          toa_flux_down,
          sfc_emission,
          sfc_reflectance,
      )

    return _solve_problems(
        solve,
        all_sky_props,
        gas_props if with_clear_sky else None,
        aer,
        with_clear_sky,
        with_aerosol_free,
    )

  init = {
      key: jnp.zeros((nz + 1, nc), dtype=dtype)
      for key in _problem_keys(
          with_clear_sky, with_aerosol_free,
          ('flux_up', 'flux_down', 'flux_net'),
      )
  }
  fluxes = _spectral_sum_zc(
      solve_chunk, selector, n_gpt, n_chunks, chunk, needs_mask, init
  )
  # The face below the bottom halo carries no flux.
  zero_halo = jnp.zeros((1, nc), dtype=dtype)
  return {
      k: jnp.concatenate([zero_halo, v], axis=0) for k, v in fluxes.items()
  }



def _sw_sfc_albedo_by_band_zc(
    albedo: Array | float,
    nc: int,
    n_bnd: int,
    name: str,
    dtype: jnp.dtype,
) -> Array | None:
  """`_sw_sfc_albedo_by_band` for `(nc,)` columns: `(n_bnd, nc)` or `None`.

  `None` for a per-column albedo (a scalar, `(1,)` or `(nc,)`); a per-band
  one is `(n_bnd, nc)` or `(n_bnd, 1)`, or `(..., n_bnd)` with the leading
  axes broadcastable against `(nc,)`.
  """
  shape = tuple(np.shape(albedo))
  if len(shape) == 0 or shape in ((nc,), (1,)):
    return None
  if len(shape) == 2 and shape[0] == n_bnd and shape[1] in (1, nc):
    return jnp.broadcast_to(jnp.asarray(albedo, dtype=dtype), (n_bnd, nc))
  if (
      len(shape) >= 1
      and shape[-1] == n_bnd
      and np.broadcast_shapes(shape[:-1], (nc,)) == (nc,)
  ):
    return jnp.broadcast_to(jnp.asarray(albedo, dtype=dtype), (nc, n_bnd)).T
  raise ValueError(
      f'{name} has shape {shape}, incompatible with nc={nc}, n_bnd_sw={n_bnd}.'
  )


def solve_sw_zc(
    pressure: Array,
    temperature: Array,
    molecules: Array,
    optics_lib: optics.RRTMOptics,
    atmos_state: AtmosphericState,
    vmr_fields: Mapping[str | int, Array] | None = None,
    cloud_r_eff_liq: Array | None = None,
    cloud_path_liq: Array | None = None,
    cloud_r_eff_ice: Array | None = None,
    cloud_path_ice: Array | None = None,
    cloud_path_liq_per_gpt: Array | None = None,
    cloud_path_ice_per_gpt: Array | None = None,
    aerosol_optics: Mapping[str, Array] | None = None,
    gpt_chunk: int = 64,
    cloud_tau_scale_liq: Array | float | None = None,
    cloud_tau_scale_ice: Array | float | None = None,
    sfc_alb_dir: Array | float | None = None,
    sfc_alb_dif: Array | float | None = None,
    with_clear_sky: bool = False,
    with_aerosol_free: bool = False,
    *,
    per_gpt_axis: int = 1,
    cloud_mask_per_gpt: Array | None = None,
) -> dict[str, Array]:
  """`solve_sw` for a column-minor `(nz + 2, nc)` atmosphere.

  The same physics, problems and outputs as `solve_sw` with the gas optics
  hoisted and the recurrences scanned, with the levels leading and the
  columns minor throughout (see the comment above `_column_minor`). The gas
  optics are evaluated on the interior levels only (the shortwave never reads
  the halos). Columns with the sun at or below the horizon return zero.

  Args:
    pressure: `(nz + 2, nc)` pressure [Pa], with a halo level at each end.
    temperature: `(nz + 2, nc)` temperature [K], with halos.
    molecules: `(nz + 2, nc)` molecules per area [molecules/m^2], with halos.
    optics_lib: The `RRTMOptics` scheme.
    atmos_state: The atmospheric state; its `zenith`, `irrad`, `sfc_alb`,
      `sfc_emis` and `toa_flux_lw` may be scalars or `(nc,)`.
    vmr_fields: Optional volume mixing ratio fields, keyed by gas name or
      index, each `(nz + 2, nc)`.
    cloud_r_eff_liq: Effective droplet radius [m], `(nz[+2], nc)`.
    cloud_path_liq: Liquid water path [kg/m^2], `(nz[+2], nc)`.
    cloud_r_eff_ice: Effective ice radius [m], `(nz[+2], nc)`.
    cloud_path_ice: Ice water path [kg/m^2], `(nz[+2], nc)`.
    cloud_path_liq_per_gpt: Per-g-point (McICA) liquid water path,
      `(nz[+2], n_gpt, nc)`, or `(n_gpt, nz[+2], nc)` with `per_gpt_axis=0`.
      Replaces `cloud_path_liq` for the liquid phase.
    cloud_path_ice_per_gpt: Same as above, for the ice water path.
    aerosol_optics: Per-band aerosol `'optical_depth'`, `'ssa'` and
      `'asymmetry_factor'`, each `(n_bnd, nz[+2], nc)`.
    gpt_chunk: g-points per iteration of the spectral loop. With the TPU
      kernels it is capped to fit `TPU_CHUNK_VMEM_BYTES` (see
      `_column_minor_chunk_plan`, which also evens chunks out to multiples
      of 8).
    cloud_tau_scale_liq: Multiplier of the liquid cloud optical depth: a
      scalar, `(nc,)` or `(nz[+2], nc)`.
    cloud_tau_scale_ice: Same as above, for ice.
    sfc_alb_dir: Direct-beam surface albedo: per column (a scalar or `(nc,)`)
      or per band (`(n_bnd_sw, nc)`, or `(nc, n_bnd_sw)`). Defaults to
      `atmos_state.sfc_alb`.
    sfc_alb_dif: Same as above, for the diffuse albedo.
    with_clear_sky: Also solve the cloud-free problem (`solve_sw`).
    with_aerosol_free: Also solve the aerosol-free twin of every problem.
    per_gpt_axis: The g-point axis of the per-g-point inputs, 1 (default,
      `(nz, n_gpt, nc)`) or 0 (`(n_gpt, nz, nc)`).
    cloud_mask_per_gpt: Optional boolean McICA masks, `(nz[+2], n_gpt, nc)`
      (or `(n_gpt, nz[+2], nc)` with `per_gpt_axis=0`): the bulk cloud paths
      are seen by a g-point only where its mask is set. Cannot be combined
      with per-g-point cloud paths. One byte per (level, g-point, column)
      instead of the four floats of per-g-point paths.

  Returns:
    `(nz + 2, nc)` face fluxes [W/m^2], `flux_up`, `flux_down` and
    `flux_net`: entry `k` is the bottom face of level `k`, so entry 1 is the
    surface and entry `nz + 1` the top of the atmosphere; entry 0 (the face
    below the bottom halo) is zero. The clear-sky and aerosol-free problems
    under their prefixes, as `solve_lw` / `solve_sw`.
    Also `flux_down_dir_sfc`, the direct beam at the surface, `(nc,)` (and its
    prefixed twins).
  """
  nzh, nc = temperature.shape
  nz = nzh - 2
  dtype = temperature.dtype
  lookup = optics_lib.gas_optics_sw
  n_gpt = optics_lib.n_gpt_sw
  n_bnd = lookup.n_bnd
  chunk, n_chunks, needs_mask = _column_minor_chunk_plan(
      n_gpt, gpt_chunk, nz, nc
  )

  vmr = None
  if vmr_fields is not None:
    vmr = {
        k: _interior(v, nz) if jnp.ndim(v) else v
        for k, v in _reindex_vmr_fields(vmr_fields, lookup).items()
    }
  # Nothing in the shortwave reads the halos: the gas optics of the interior.
  gas = optics_lib.compute_gas_optics_all_gpts(
      pressure[1:-1],
      temperature[1:-1],
      molecules[1:-1],
      vmr,
      is_lw=False,
      z_leading_output=True,
  )
  tau_packed = _pack_chunks(gas['optical_depth'], 1, n_chunks, chunk)
  rayleigh_packed = _pack_chunks(gas['rayleigh'], 1, n_chunks, chunk)
  selector = _band_selector(
      lookup.g_point_to_bnd, n_bnd, n_chunks, chunk, dtype
  )
  solar_fraction = jnp.asarray(optics_lib.solar_fraction_by_gpt, dtype=dtype)
  pad = n_chunks * chunk - n_gpt
  if pad:
    solar_fraction = jnp.concatenate(
        [solar_fraction, jnp.repeat(solar_fraction[-1:], pad)]
    )
  solar_fraction = solar_fraction.reshape(n_chunks, chunk, 1)
  irrad = _per_column(atmos_state.irrad, nc, dtype)

  # Night columns: see `solve_sw`. The solve is fed an overhead sun there and
  # their fluxes are zeroed at the end.
  zenith = atmos_state.zenith
  if jnp.ndim(zenith) > 0:
    zenith = jnp.broadcast_to(jnp.asarray(zenith, dtype=dtype), (nc,))
  night = zenith >= 0.5 * jnp.pi
  safe_zenith = jnp.where(
      night, jnp.zeros_like(jnp.asarray(zenith, dtype=dtype)), zenith
  )
  mu = jnp.cos(safe_zenith)
  if jnp.ndim(safe_zenith) > 0:
    safe_zenith = safe_zenith.reshape(1, 1, nc)
    mu = mu.reshape(1, nc)
  mu_layer = mu[jnp.newaxis] if jnp.ndim(mu) else mu

  def albedo_by_gpt(albedo, name):
    """`albedo` (or `sfc_alb`) as a function of the chunk's band selector."""
    if albedo is None:
      albedo, name = atmos_state.sfc_alb, 'sfc_alb'
    by_band = _sw_sfc_albedo_by_band_zc(albedo, nc, n_bnd, name, dtype)
    if by_band is not None:
      return lambda sel: _by_gpt(sel, by_band)
    per_col = jnp.broadcast_to(_per_column(albedo, nc, dtype), (chunk, nc))
    return lambda sel: per_col

  sfc_albedo_dir = albedo_by_gpt(sfc_alb_dir, 'sfc_alb_dir')
  sfc_albedo_dif = albedo_by_gpt(sfc_alb_dif, 'sfc_alb_dif')

  has_cloud = any(
      p is not None
      for p in (
          cloud_path_liq,
          cloud_path_ice,
          cloud_path_liq_per_gpt,
          cloud_path_ice_per_gpt,
      )
  )
  cloud = None
  if has_cloud:
    cloud = _CloudZc(
        optics_lib, False, nz, nc, n_chunks, chunk, dtype,
        r_eff_liq=cloud_r_eff_liq, path_liq=cloud_path_liq,
        r_eff_ice=cloud_r_eff_ice, path_ice=cloud_path_ice,
        path_liq_per_gpt=cloud_path_liq_per_gpt,
        path_ice_per_gpt=cloud_path_ice_per_gpt,
        mask_per_gpt=cloud_mask_per_gpt, per_gpt_axis=per_gpt_axis,
        tau_scale_liq=cloud_tau_scale_liq, tau_scale_ice=cloud_tau_scale_ice,
    )
  aerosol = _aerosol_zc(aerosol_optics, nz)
  zeros_chunk = jnp.zeros((chunk, nc), dtype=dtype)

  def solve_chunk(ichunk, sel):
    tau = jax.lax.dynamic_index_in_dim(tau_packed, ichunk, 0, keepdims=False)
    rayleigh = jax.lax.dynamic_index_in_dim(
        rayleigh_packed, ichunk, 0, keepdims=False
    )
    frac = jax.lax.dynamic_index_in_dim(
        solar_fraction, ichunk, 0, keepdims=False
    )
    toa_flux = jnp.broadcast_to(irrad * frac, (chunk, nc))
    flux_down_direct_toa = toa_flux * mu
    albedo_dir = sfc_albedo_dir(sel)
    albedo_dif = sfc_albedo_dif(sel)

    gas_props = optics_lib._sw_gas_properties(tau, rayleigh)  # pylint: disable=protected-access
    all_sky_props = gas_props
    if cloud is not None:
      all_sky_props = optics_lib.combine_optical_properties(
          gas_props, cloud.chunk_props(ichunk, sel)
      )
    aer = (
        None
        if aerosol is None
        else {k: _by_gpt(sel, v) for k, v in aerosol.items()}
    )

    def solve(props, aer_props):
      if aer_props is not None:
        props = optics_lib.combine_optical_properties(props, aer_props)
      cell = monochromatic_two_stream.sw_cell_properties(
          safe_zenith,
          props['optical_depth'],
          props['ssa'],
          props['asymmetry_factor'],
          cos_zenith=mu_layer,
          return_t_noscat=True,
      )
      # The direct beam and the diffuse sources it feeds (`sw_cell_source`).
      flux_sfc, flux_bottom, flux_top = monochromatic_two_stream.sw_direct_beam_zc(
          cell['t_noscat'], flux_down_direct_toa
      )
      diffuse = monochromatic_two_stream.solve_rte_2stream_zc(
          cell['t_diff'],
          cell['r_diff'],
          cell['r_dir'] * flux_top,
          cell['t_dir'] * flux_top,
          zeros_chunk,
          albedo_dir * flux_sfc,
          albedo_dif,
      )
      direct = jnp.concatenate(
          [flux_bottom, flux_down_direct_toa[jnp.newaxis]], axis=0
      )
      flux_up = diffuse['flux_up']
      flux_down = diffuse['flux_down'] + direct
      return {
          'flux_up': flux_up,
          'flux_down': flux_down,
          'flux_net': flux_up - flux_down,
          'flux_down_dir_sfc': flux_sfc,
      }

    return _solve_problems(
        solve,
        all_sky_props,
        gas_props if with_clear_sky else None,
        aer,
        with_clear_sky,
        with_aerosol_free,
    )

  init = {}
  for prefix in _problem_prefixes(with_clear_sky, with_aerosol_free):
    for key in ('flux_up', 'flux_down', 'flux_net'):
      init[f'{prefix}{key}'] = jnp.zeros((nz + 1, nc), dtype=dtype)
    init[f'{prefix}flux_down_dir_sfc'] = jnp.zeros((nc,), dtype=dtype)
  fluxes = _spectral_sum_zc(
      solve_chunk, selector, n_gpt, n_chunks, chunk, needs_mask, init
  )
  zero_halo = jnp.zeros((1, nc), dtype=dtype)
  out = {}
  for key, val in fluxes.items():
    if val.ndim == 2:
      val = jnp.concatenate([zero_halo, val], axis=0)
    out[key] = jnp.where(night, jnp.zeros_like(val), val)
  return out



def _solve_3d_via_zc(
    solve_zc: Callable[..., dict[str, Array]],
    pressure: Array,
    temperature: Array,
    molecules: Array,
    optics_lib: optics.RRTMOptics,
    atmos_state: AtmosphericState,
    **kwargs,
) -> dict[str, Array]:
  """A 3D `(nx, ny, nz + 2)` solve by `solve_zc` on `(nz + 2, nx * ny)`.

  The inputs are laid out column-minor, solved, and the outputs laid back
  out: `solve_lw` / `solve_sw` route here (`_uses_column_minor`).

  Args:
    solve_zc: `solve_lw_zc` or `solve_sw_zc`.
    pressure: `(nx, ny, nz + 2)`.
    temperature: `(nx, ny, nz + 2)`.
    molecules: `(nx, ny, nz + 2)`.
    optics_lib: The `RRTMOptics` scheme.
    atmos_state: The atmospheric state; non-scalar fields broadcast against
      `(nx, ny)`.
    **kwargs: The remaining arguments of `solve_zc`, in the layout of
      `solve_lw` / `solve_sw`: 3D fields `(nx, ny, nz + 2)`, per-g-point
      paths and per-band aerosol `(n, nx, ny, nz + 2)`, surface fields
      broadcastable against `(nx, ny)` and per-band albedos `(n_bnd, nx, ny)`
      (as `_sw_sfc_albedo_by_band` returns them).

  Returns:
    The fluxes of `solve_zc`, `(nx, ny, nz + 2)` (and `(nx, ny)` for
    `flux_down_dir_sfc`).
  """
  nx, ny, nzh = temperature.shape
  nc = nx * ny

  def field(x):
    if x is None or jnp.ndim(x) == 0:
      return x
    x = jnp.broadcast_to(x, (nx, ny, nzh))
    return jnp.moveaxis(x, 2, 0).reshape(nzh, nc)

  def leading(x):
    """`(n, nx, ny, nzh)` -> `(n, nzh, nc)`."""
    if x is None:
      return None
    return jnp.moveaxis(x, 3, 1).reshape(x.shape[0], nzh, nc)

  def plane(x):
    if x is None or jnp.ndim(x) == 0:
      return x
    return jnp.broadcast_to(jnp.asarray(x), (nx, ny)).reshape(nc)

  state = {
      name: plane(getattr(atmos_state, name))
      for name in ('zenith', 'irrad', 'sfc_alb', 'sfc_emis', 'toa_flux_lw')
      if jnp.ndim(getattr(atmos_state, name)) > 0
  }
  args = dict(kwargs)
  for name in (
      'cloud_r_eff_liq', 'cloud_path_liq', 'cloud_r_eff_ice', 'cloud_path_ice',
      'cloud_tau_scale_liq', 'cloud_tau_scale_ice',
  ):
    args[name] = field(args.get(name))
  for name in ('cloud_path_liq_per_gpt', 'cloud_path_ice_per_gpt'):
    args[name] = leading(args.get(name))
  if args.get('aerosol_optics') is not None:
    args['aerosol_optics'] = {
        k: leading(v) for k, v in args['aerosol_optics'].items()
    }
  if args.get('vmr_fields') is not None:
    args['vmr_fields'] = {k: field(v) for k, v in args['vmr_fields'].items()}
  if 'sfc_temperature' in args:
    args['sfc_temperature'] = plane(args['sfc_temperature'])
  for name in ('sfc_alb_dir', 'sfc_alb_dif'):
    if name in args:
      a = args[name]
      args[name] = (
          a.reshape(a.shape[0], nc) if jnp.ndim(a) == 3 else plane(a)
      )
  out = solve_zc(
      field(pressure),
      field(temperature),
      field(molecules),
      optics_lib,
      dataclasses.replace(atmos_state, **state) if state else atmos_state,
      per_gpt_axis=0,
      **args,
  )
  return {
      k: (
          v.reshape(nx, ny)
          if v.ndim == 1
          else jnp.moveaxis(v.reshape(nzh, nx, ny), 0, 2)
      )
      for k, v in out.items()
  }
