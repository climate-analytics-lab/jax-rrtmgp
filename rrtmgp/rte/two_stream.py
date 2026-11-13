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

import dataclasses
from collections.abc import Mapping
from typing import Callable, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from rrtmgp import constants
from rrtmgp import kernel_ops
from rrtmgp import smooth_ops
from rrtmgp.optics import atmospheric_state
from rrtmgp.optics import lookup_cloud_optics
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
      by row gathers on CPU/GPU and factored MXU contractions on TPU.
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

  if (
      jax.default_backend() == 'tpu'
      and hoist_gas_optics
      and use_scan
      and gpt_chunk > 1
      and isinstance(optics_lib, optics.RRTMOptics)
      and temperature.ndim == 3
      and temperature.size > 0
  ):
    return _solve_lw_3d_via_zc(
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


# Capability flag advertising native 2-D `(nx, ny)` solar zenith support in
# `solve_sw` so downstream callers can skip scalar-zenith shims.
_SUPPORTS_2D_ZENITH = True

# Capability flag advertising native column-minor `(nzh, nc)` TPU solvers
# (`solve_lw_zc`, `solve_sw_zc`, `RRTMGP.compute_heating_rate_zc`).
_SUPPORTS_TPU_COL_MINOR_SOLVE = True


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
      by row gathers on CPU/GPU and factored MXU contractions on TPU.
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

  if (
      jax.default_backend() == 'tpu'
      and hoist_gas_optics
      and use_scan
      and gpt_chunk > 1
      and isinstance(optics_lib, optics.RRTMOptics)
      and temperature.ndim == 3
      and temperature.size > 0
  ):
    return _solve_sw_3d_via_zc(
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
        sfc_alb_dir=sfc_alb_dir,
        sfc_alb_dif=sfc_alb_dif,
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

  Returns:
    The heating rate of the grid cell [K/s].
  """
  # Compute the centered pressure difference in z:
  #   dp_{i,j,k} = (p_{i,j,k+1} - p_{i,j,k-1}) / 2.
  # This is an approximation to the pressure difference across the cell, which
  # would use the pressure at the upper and lower faces, but it should be ok.
  dp = 0.5 * kernel_ops.centered_difference(pressure, dim=2)

  # Compute the forward pressure difference of fluxes on faces (like a
  # derivative of face_to_node).
  dflux = kernel_ops.forward_difference(flux_net, dim=2)

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
# Column-minor `(nzh, nc)` TPU solvers (`solve_lw_zc`, `solve_sw_zc`,
# `compute_heating_rate_zc`).
#
# Keeping the column axis `nc` minor (`nc % 128 == 0` on standard Gaussian
# grids) and `chunk` second-minor (`chunk % 8 == 0`) fills `(8, 128)` TPU tiles
# at 100%, eliminates `moveaxis` transposes around vertical `lax.scan`s, fuses
# the upward albedo and emission recurrences into a single scan, precomputes
# WENO5 face temperatures, Planck `totplnk`, and cloud Mie tables once outside
# the spectral loop, and replaces all per-chunk gathers with `dynamic_slice`
# and tiny static band-projection matmuls.
# ---------------------------------------------------------------------------


def _tpu_gpt_chunk_plan(n_gpt: int, gpt_chunk: int) -> tuple[int, int, bool]:
  """Choose an 8-aligned `chunk <= gpt_chunk` when `n_chunks` stays unchanged."""
  chunk, n_chunks, needs_mask = _gpt_chunk_plan(n_gpt, gpt_chunk)
  if n_chunks > 1:
    even8 = ((-(-n_gpt // n_chunks) + 7) // 8) * 8
    if even8 <= gpt_chunk and even8 * n_chunks >= n_gpt:
      chunk = even8
      needs_mask = n_chunks * chunk != n_gpt
  return chunk, n_chunks, needs_mask


def _reconstruct_face_values_zc(f_zc: Array, f_lower_bc_c: Array) -> Array:
  """Reconstruct `f_bottom[1:]` (`(nz + 1, nc)`) on `(nzh, nc)` via WENO5-JS."""
  nzh, _ = f_zc.shape
  f = jnp.concatenate(
      [f_lower_bc_c[jnp.newaxis, :], f_zc[1:-1, :], f_zc[-2:-1, :]],
      axis=0,
  )
  f_m3 = jnp.roll(f, 3, axis=0)
  f_m2 = jnp.roll(f, 2, axis=0)
  f_m1 = jnp.roll(f, 1, axis=0)
  f_p1 = jnp.roll(f, -1, axis=0)
  f_p2 = jnp.roll(f, -2, axis=0)

  beta1_p = (
      13 / 12 * (f_m3 - 2 * f_m2 + f_m1) ** 2
      + 1 / 4 * (f_m3 - 4 * f_m2 + 3 * f_m1) ** 2
  )
  beta2_p = 13 / 12 * (f_m2 - 2 * f_m1 + f) ** 2 + 1 / 4 * (f_m2 - f) ** 2
  beta3_p = (
      13 / 12 * (f_m1 - 2 * f + f_p1) ** 2
      + 1 / 4 * (3 * f_m1 - 4 * f + f_p1) ** 2
  )

  beta1_m = (
      13 / 12 * (f - 2 * f_p1 + f_p2) ** 2
      + 1 / 4 * (3 * f - 4 * f_p1 + f_p2) ** 2
  )
  beta2_m = 13 / 12 * (f_m1 - 2 * f + f_p1) ** 2 + 1 / 4 * (f_m1 - f_p1) ** 2
  beta3_m = (
      13 / 12 * (f_m2 - 2 * f_m1 + f) ** 2
      + 1 / 4 * (f_m2 - 4 * f_m1 + 3 * f) ** 2
  )

  c1, c2, c3 = 0.1, 0.6, 0.3
  eps = 1e-5
  a1_p = c1 / (beta1_p + eps) ** 2
  a2_p = c2 / (beta2_p + eps) ** 2
  a3_p = c3 / (beta3_p + eps) ** 2
  a1_m = c1 / (beta1_m + eps) ** 2
  a2_m = c2 / (beta2_m + eps) ** 2
  a3_m = c3 / (beta3_m + eps) ** 2

  mask_p = jnp.asarray((np.arange(nzh) != 2)[:, np.newaxis])
  mask_m = jnp.asarray((np.arange(nzh) != (nzh - 2))[:, np.newaxis])
  a1_p = jnp.where(mask_p, a1_p, jnp.zeros_like(a1_p))
  a1_m = jnp.where(mask_m, a1_m, jnp.zeros_like(a1_m))

  sum_p = a1_p + a2_p + a3_p
  w1_p, w2_p, w3_p = a1_p / sum_p, a2_p / sum_p, a3_p / sum_p
  sum_m = a1_m + a2_m + a3_m
  w1_m, w2_m, w3_m = a1_m / sum_m, a2_m / sum_m, a3_m / sum_m

  f1_p = 1 / 3 * f_m3 - 7 / 6 * f_m2 + 11 / 6 * f_m1
  f2_p = -1 / 6 * f_m2 + 5 / 6 * f_m1 + 1 / 3 * f
  f3_p = 1 / 3 * f_m1 + 5 / 6 * f - 1 / 6 * f_p1
  f1_m = 11 / 6 * f - 7 / 6 * f_p1 + 1 / 3 * f_p2

  face_p = w1_p * f1_p + w2_p * f2_p + w3_p * f3_p
  face_m = w1_m * f1_m + w2_m * f3_p + w3_m * f2_p
  face_m_tail = jnp.concatenate([face_m[1:-1, :], f[-2:-1, :]], axis=0)
  return 0.5 * (face_p[1:, :] + face_m_tail)


def _interp_1d_table_mxu(table_2d: Array, x: Array, ref_grid: Array) -> Array:
  """1-D linear interpolation of all rows of `table_2d` `(n_rows, n_nodes)`."""
  interp = optics_utils.create_linear_interpolant(x, ref_grid)
  n_nodes = table_2d.shape[1]
  ncells = x.size
  oh = interp.interp_low.weight.reshape(1, ncells) * jax.nn.one_hot(
      interp.interp_low.idx.reshape(ncells),
      n_nodes,
      dtype=table_2d.dtype,
      axis=0,
  ) + interp.interp_high.weight.reshape(1, ncells) * jax.nn.one_hot(
      interp.interp_high.idx.reshape(ncells),
      n_nodes,
      dtype=table_2d.dtype,
      axis=0,
  )
  return jnp.matmul(
      table_2d, oh, precision=jax.lax.Precision.HIGHEST
  ).reshape(table_2d.shape[0], *x.shape)


def _planck_all_bands_mxu(
    lookup: lookup_gas_optics_longwave.LookupGasOpticsLongwave,
    t_face_zc: Array,
    sfc_temp_c: Array,
    t_lay_zc: Array | None = None,
) -> tuple[Array, Array, Array | None]:
  """Precompute `(b_face, b_sfc, b_lay)` across all LW bands in one matmul."""
  n_face = t_face_zc.shape[0]
  parts = [t_face_zc, sfc_temp_c[jnp.newaxis, :]]
  if t_lay_zc is not None:
    parts.append(t_lay_zc)
  t_stack = jnp.concatenate(parts, axis=0)
  b_all = optics_utils.floor_at_zero(
      _interp_1d_table_mxu(lookup.totplnk, t_stack, lookup.t_planck)
  )
  b_face = b_all[:, :n_face, :]
  b_sfc = b_all[:, n_face, :]
  b_lay = None if t_lay_zc is None else b_all[:, n_face + 1 :, :]
  return b_face, b_sfc, b_lay


def _cloud_mie_tables_mxu(
    cld_lkp: lookup_cloud_optics.LookupCloudOptics,
    r_eff_liq_zc: Array | None,
    r_eff_ice_zc: Array | None,
    need_liq: bool,
    need_ice: bool,
    nz: int,
    nc: int,
    dtype: jnp.dtype,
) -> tuple[tuple[Array, Array, Array] | None, tuple[Array, Array, Array] | None]:
  """Precompute `(ext, ssa, asy)` `(n_bnd, nz, nc)` Mie tables in <=2 matmuls."""
  n_bnd = cld_lkp.ext_liq.shape[0]
  liq_tables = None
  if need_liq:
    r_liq = (
        jnp.zeros((nz, nc), dtype=dtype)
        if r_eff_liq_zc is None
        else r_eff_liq_zc
    )
    size_liq = jnp.clip(
        1.0e6 * r_liq, cld_lkp.radius_liq_lower, cld_lkp.radius_liq_upper
    )
    tbl_liq = jnp.concatenate(
        [cld_lkp.ext_liq, cld_lkp.ssa_liq, cld_lkp.asy_liq], axis=0
    )
    grid_liq = jnp.linspace(
        cld_lkp.radius_liq_lower,
        cld_lkp.radius_liq_upper,
        cld_lkp.n_size_liq,
        dtype=tbl_liq.dtype,
    )
    mie_l = _interp_1d_table_mxu(tbl_liq, size_liq, grid_liq)
    liq_tables = (
        mie_l[:n_bnd],
        mie_l[n_bnd : 2 * n_bnd],
        mie_l[2 * n_bnd :],
    )

  ice_tables = None
  if need_ice:
    r_ice = (
        jnp.zeros((nz, nc), dtype=dtype)
        if r_eff_ice_zc is None
        else r_eff_ice_zc
    )
    size_ice = jnp.clip(
        2.0e6 * r_ice, cld_lkp.diameter_ice_lower, cld_lkp.diameter_ice_upper
    )
    rough = cld_lkp.ice_roughness.value
    tbl_ice = jnp.concatenate(
        [
            cld_lkp.ext_ice[rough],
            cld_lkp.ssa_ice[rough],
            cld_lkp.asy_ice[rough],
        ],
        axis=0,
    )
    grid_ice = jnp.linspace(
        cld_lkp.diameter_ice_lower,
        cld_lkp.diameter_ice_upper,
        cld_lkp.n_size_ice,
        dtype=tbl_ice.dtype,
    )
    mie_i = _interp_1d_table_mxu(tbl_ice, size_ice, grid_ice)
    ice_tables = (
        mie_i[:n_bnd],
        mie_i[n_bnd : 2 * n_bnd],
        mie_i[2 * n_bnd :],
    )
  return liq_tables, ice_tables


def _band_to_zgc(bnd_oh_chunk: Array, field_bzc: Array) -> Array:
  """Project `(n_bnd, nz, nc)` to `(nz, chunk, nc)` keeping `nc` minor."""
  n_bnd, nz, nc = field_bzc.shape
  chunk = bnd_oh_chunk.shape[0]
  out_gzc = jnp.matmul(
      bnd_oh_chunk,
      field_bzc.reshape(n_bnd, nz * nc),
      precision=jax.lax.Precision.HIGHEST,
  ).reshape(chunk, nz, nc)
  return jnp.swapaxes(out_gzc, 0, 1)


def _cloud_props_from_mie(
    optics_lib: optics.RRTMOptics,
    liq_tables: tuple[Array, Array, Array] | None,
    ice_tables: tuple[Array, Array, Array] | None,
    cpl: Array | None,
    cpi: Array | None,
    tau_scale_liq: Array | float | None,
    tau_scale_ice: Array | float | None,
    is_lw: bool,
    bnd_oh_chunk: Array | None = None,
) -> dict[str, Array]:
  """Combine liquid + ice cloud optical properties on `(n_bnd|chunk)` layout."""
  proj = (
      (lambda x: x)
      if bnd_oh_chunk is None
      else (lambda x: _band_to_zgc(bnd_oh_chunk, x))
  )
  tau_l = tau_ssa_l = tau_ssa_g_l = None
  if liq_tables is not None and cpl is not None:
    ext_l, ssa_l, asy_l = liq_tables
    cld_l = cpl * 1.0e3
    mask_l = smooth_ops.smooth_gate(cld_l, 0.0, 2.0e-6)
    tau_l = proj(ext_l) * cld_l * mask_l
    tau_ssa_l = proj(ssa_l) * tau_l
    tau_ssa_g_l = proj(asy_l) * tau_ssa_l

  tau_i = tau_ssa_i = tau_ssa_g_i = None
  if ice_tables is not None and cpi is not None:
    ext_i, ssa_i, asy_i = ice_tables
    cld_i = cpi * 1.0e3
    mask_i = smooth_ops.smooth_gate(cld_i, 0.0, 2.0e-6)
    tau_i = proj(ext_i) * cld_i * mask_i
    tau_ssa_i = proj(ssa_i) * tau_i
    tau_ssa_g_i = proj(asy_i) * tau_ssa_i

  if tau_l is not None and tau_i is not None:
    tau = tau_l + tau_i
    tau_ssa = tau_ssa_l + tau_ssa_i
    tau_ssa_g = tau_ssa_g_l + tau_ssa_g_i
  elif tau_l is not None:
    tau, tau_ssa, tau_ssa_g = tau_l, tau_ssa_l, tau_ssa_g_l
  else:
    assert tau_i is not None and tau_ssa_i is not None and tau_ssa_g_i is not None
    tau, tau_ssa, tau_ssa_g = tau_i, tau_ssa_i, tau_ssa_g_i

  tau_nz = tau != 0
  tau_ssa_nz = tau_ssa != 0
  safe_tau = jnp.where(tau_nz, tau, 1.0)
  safe_tau_ssa = jnp.where(tau_ssa_nz, tau_ssa, 1.0)

  optical_depth = tau
  if tau_scale_liq is not None or tau_scale_ice is not None:
    tl = jnp.zeros_like(tau) if tau_l is None else (
        tau_l if tau_scale_liq is None else tau_scale_liq * tau_l
    )
    ti = jnp.zeros_like(tau) if tau_i is None else (
        tau_i if tau_scale_ice is None else tau_scale_ice * tau_i
    )
    optical_depth = tl + ti

  cld_props = {
      'optical_depth': optical_depth,
      'ssa': jnp.where(tau_nz, tau_ssa / safe_tau, 0.0),
      'asymmetry_factor': jnp.where(tau_ssa_nz, tau_ssa_g / safe_tau_ssa, 0.0),
  }
  if not is_lw:
    cld_props = optics_lib._apply_delta_scaling_for_cloud(cld_props)  # pylint: disable=protected-access
  return cld_props


def _pack_gpt_zc_to_chunks(
    arr_gzc: Array, n_chunks: int, chunk: int
) -> Array:
  """Pack `(n_gpt, nz_or_nzh, nc)` into `(n_chunks, nz_or_nzh, chunk, nc)`."""
  n_gpt, nz_dim, nc = arr_gzc.shape
  pad_g = n_chunks * chunk - n_gpt
  if pad_g:
    arr_gzc = jnp.concatenate(
        [arr_gzc, jnp.repeat(arr_gzc[-1:, :, :], pad_g, axis=0)], axis=0
    )
  return jnp.swapaxes(arr_gzc.reshape(n_chunks, chunk, nz_dim, nc), 1, 2)


def _pack_per_gpt_cloud_zc(
    arr: Array,
    nz: int,
    nzh: int,
    n_chunks: int,
    chunk: int,
    per_gpt_axis: int,
) -> Array:
  """Pack per-g-point cloud path into `(n_chunks, nz, chunk, nc)`."""
  if per_gpt_axis == 0:
    if arr.shape[1] == nzh:
      arr = arr[:, 1:-1, :]
    return _pack_gpt_zc_to_chunks(arr, n_chunks, chunk)
  # `per_gpt_axis == 1`: `(nz_or_nzh, n_gpt, nc)` -> `(n_chunks, nz, chunk, nc)`.
  if arr.shape[0] == nzh:
    arr = arr[1:-1, :, :]
  _, n_gpt, nc = arr.shape
  pad_g = n_chunks * chunk - n_gpt
  if pad_g:
    arr = jnp.concatenate(
        [arr, jnp.repeat(arr[:, -1:, :], pad_g, axis=1)], axis=1
    )
  return jnp.swapaxes(arr.reshape(nz, n_chunks, chunk, nc), 0, 1)


def _interior_zc(arr: Array | None, nz: int, nzh: int) -> Array | None:
  """Slice optional `(nzh, nc)` or `(nz, nc)` field to interior `(nz, nc)`."""
  if arr is None:
    return None
  if arr.shape[0] == nzh:
    return arr[1:-1, :]
  assert arr.shape[0] == nz
  return arr


def _interior_bzc(arr: Array, nz: int, nzh: int) -> Array:
  """Slice `(n_bnd, nzh, nc)` or `(n_bnd, nz, nc)` field to `(n_bnd, nz, nc)`."""
  if arr.shape[1] == nzh:
    return arr[:, 1:-1, :]
  assert arr.shape[1] == nz
  return arr


def _scale_to_interior(
    scale: Array | float | None,
    nz: int,
    nzh: int,
    nc: int,
    with_middle_axis: bool,
) -> Array | float | None:
  """Normalize `cloud_tau_scale_{liq,ice}` to broadcast cleanly against cloud."""
  if scale is None or jnp.ndim(scale) == 0:
    return scale
  s = jnp.asarray(scale)
  if s.ndim == 1:
    s = jnp.broadcast_to(s, (nc,))[jnp.newaxis, :]
  elif s.shape[0] == nzh:
    s = s[1:-1, :]
  elif s.shape[0] not in (1, nz):
    s = jnp.broadcast_to(s, (nz, nc))
  if with_middle_axis:
    return s[:, jnp.newaxis, :]
  return s[jnp.newaxis, :, :]


def _to_1c(val: Array | float, nc: int, dtype: jnp.dtype) -> Array:
  """Convert scalar or `(nc,)` surface/TOA field to `(1, nc)`."""
  v = jnp.asarray(val, dtype=dtype)
  if v.ndim == 0:
    return v.reshape(1, 1)
  return jnp.broadcast_to(v, (nc,)).reshape(1, nc)


def _band_oh_packed(
    g_point_to_bnd: Array, n_bnd: int, n_chunks: int, chunk: int, dtype: jnp.dtype
) -> Array:
  """Static `(n_chunks, chunk, n_bnd)` band-projection matrix."""
  g2b = np.asarray(g_point_to_bnd)
  n_gpt = g2b.shape[0]
  padded = np.minimum(np.arange(n_chunks * chunk), n_gpt - 1)
  bnds = g2b[padded]
  oh = np.eye(n_bnd, dtype=np.float32)[bnds]
  return jnp.asarray(oh, dtype=dtype).reshape(n_chunks, chunk, n_bnd)


def _upward_op_zc(carry, inputs):
  alb_below, emiss_below = carry
  t, r, sup, sdn = inputs
  beta = 1.0 / (1.0 - r * alb_below)
  alb_next = r + t * t * beta * alb_below
  emiss_next = sup + t * beta * (emiss_below + sdn * alb_below)
  return (alb_next, emiss_next), (alb_below, emiss_below, beta)


def _downward_op_zc(fdn_above, inputs):
  t, r, sdn, emiss_b, b = inputs
  fdn = (t * fdn_above + r * emiss_b + sdn) * b
  return fdn, fdn


def _direct_op_zc(carry_above, w):
  below = w * carry_above
  return below, (below, carry_above)


def _solve_rte_2stream_zc(
    t_diff: Array,
    r_diff: Array,
    src_up: Array,
    src_down: Array,
    toa_flux_down: Array,
    sfc_emission: Array,
    sfc_reflectance: Array,
) -> dict[str, Array]:
  """Fused two-stream transport on interior `(nz, chunk, nc)` arrays.

  Returns `(nz + 1, chunk, nc)` face fluxes from the surface face (`0`) to the
  TOA face (`nz`). Bottom-halo face (`0` in `(nzh, nc)`) is identically zero and
  prepended once after the spectral sum.
  """
  (alb_toa, emiss_toa), (alb_below, emiss_below, beta) = jax.lax.scan(
      _upward_op_zc,
      (sfc_reflectance, sfc_emission),
      (t_diff, r_diff, src_up, src_down),
  )

  _, fdn_int = jax.lax.scan(
      _downward_op_zc,
      toa_flux_down,
      (t_diff, r_diff, src_down, emiss_below, beta),
      reverse=True,
  )

  flux_down_faces = jnp.concatenate(
      [fdn_int, toa_flux_down[jnp.newaxis, :, :]], axis=0
  )
  alb_faces = jnp.concatenate(
      [alb_below, alb_toa[jnp.newaxis, :, :]], axis=0
  )
  emiss_faces = jnp.concatenate(
      [emiss_below, emiss_toa[jnp.newaxis, :, :]], axis=0
  )
  flux_up_faces = flux_down_faces * alb_faces + emiss_faces
  return {
      'flux_up': flux_up_faces,
      'flux_down': flux_down_faces,
      'flux_net': flux_up_faces - flux_down_faces,
  }


def _lw_noscat_transport_zc(
    optical_depth: Array,
    ssa: Array,
    planck_src: Array,
    planck_src_top: Array,
    planck_src_bottom: Array,
    toa_flux_down: Array,
    sfc_src: Array,
    sfc_emissivity: Array,
    secant: float | Array = longwave_noscat.LW_NOSCAT_DEFAULT_SECANT,
    weight: float = longwave_noscat.LW_NOSCAT_DEFAULT_WEIGHT,
) -> dict[str, Array]:
  """No-scattering LW transport on interior `(nz, chunk, nc)` arrays."""
  tau_abs = jnp.maximum(optical_depth * (1.0 - ssa), 0.0)
  trans, source_up, source_down = longwave_noscat.lw_noscat_layer_sources(
      secant * tau_abs, planck_src, planck_src_top, planck_src_bottom
  )
  flux_scale = np.pi * weight
  source_up = flux_scale * source_up
  source_down = flux_scale * source_down

  def fdn_op(fdn_above, inputs):
    tr, sdn = inputs
    out = tr * fdn_above + sdn
    return out, out

  sfc_flux_down, fdn_int = jax.lax.scan(
      fdn_op, toa_flux_down, (trans, source_down), reverse=True
  )
  sfc_flux_up = (1.0 - sfc_emissivity) * sfc_flux_down + (
      sfc_emissivity * flux_scale * sfc_src
  )

  def fup_op(fup_below, inputs):
    tr, sup = inputs
    out = tr * fup_below + sup
    return out, out

  _, fup_int = jax.lax.scan(fup_op, sfc_flux_up, (trans, source_up))
  flux_down_faces = jnp.concatenate(
      [fdn_int, toa_flux_down[jnp.newaxis, :, :]], axis=0
  )
  flux_up_faces = jnp.concatenate(
      [sfc_flux_up[jnp.newaxis, :, :], fup_int], axis=0
  )
  return {
      'flux_up': flux_up_faces,
      'flux_down': flux_down_faces,
      'flux_net': flux_up_faces - flux_down_faces,
  }


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
) -> dict[str, Array]:
  """Column-minor `(nzh, nc)` longwave radiative transfer solver for TPU."""
  if lw_solver not in LW_SOLVERS:
    raise ValueError(
        f'lw_solver must be one of {LW_SOLVERS}, got {lw_solver!r}.'
    )
  assert sfc_temperature is not None, 'sfc_temperature is required.'
  nzh, nc = temperature.shape
  nz = nzh - 2
  dtype = temperature.dtype
  lookup_lw = cast(
      lookup_gas_optics_longwave.LookupGasOpticsLongwave,
      optics_lib.gas_optics_lw,
  )
  n_gpt = optics_lib.n_gpt_lw
  n_bnd = lookup_lw.n_bnd
  chunk, n_chunks, needs_mask = _tpu_gpt_chunk_plan(n_gpt, gpt_chunk)

  vmr_idx = (
      None
      if vmr_fields is None
      else _reindex_vmr_fields(vmr_fields, lookup_lw)
  )
  gas_all = optics_lib.compute_gas_optics_all_gpts_zc(
      pressure, temperature, molecules, vmr_idx, is_lw=True
  )
  tau_packed = _pack_gpt_zc_to_chunks(
      gas_all['optical_depth'][:, 1:-1, :], n_chunks, chunk
  )
  pfrac_packed = _pack_gpt_zc_to_chunks(
      gas_all['planck_fraction'], n_chunks, chunk
  )

  sfc_temp_c = jnp.broadcast_to(
      jnp.asarray(sfc_temperature, dtype=dtype), (nc,)
  )
  t_face_zc = _reconstruct_face_values_zc(temperature, sfc_temp_c)
  need_lay_src = lw_solver == LW_SOLVER_NOSCAT
  b_face, b_sfc, b_lay = _planck_all_bands_mxu(
      lookup_lw,
      t_face_zc,
      sfc_temp_c,
      t_lay_zc=temperature[1:-1, :] if need_lay_src else None,
  )
  bnd_oh_packed = _band_oh_packed(
      lookup_lw.g_point_to_bnd, n_bnd, n_chunks, chunk, dtype
  )

  has_liq = cloud_path_liq_per_gpt is not None or cloud_path_liq is not None
  has_ice = cloud_path_ice_per_gpt is not None or cloud_path_ice is not None
  has_cloud = has_liq or has_ice
  per_gpt_cloud = (
      cloud_path_liq_per_gpt is not None or cloud_path_ice_per_gpt is not None
  )

  liq_mie = ice_mie = None
  cld_props_by_band = None
  cpl_packed = cpi_packed = None
  cpl_int_z1c = cpi_int_z1c = None
  scale_liq_z1c = scale_ice_z1c = None

  if has_cloud:
    assert optics_lib.cloud_optics_lw is not None
    liq_mie, ice_mie = _cloud_mie_tables_mxu(
        optics_lib.cloud_optics_lw,
        _interior_zc(cloud_r_eff_liq, nz, nzh),
        _interior_zc(cloud_r_eff_ice, nz, nzh),
        has_liq,
        has_ice,
        nz,
        nc,
        dtype,
    )
    if per_gpt_cloud:
      scale_liq_z1c = _scale_to_interior(
          cloud_tau_scale_liq, nz, nzh, nc, with_middle_axis=True
      )
      scale_ice_z1c = _scale_to_interior(
          cloud_tau_scale_ice, nz, nzh, nc, with_middle_axis=True
      )
      if cloud_path_liq_per_gpt is not None:
        cpl_packed = _pack_per_gpt_cloud_zc(
            cloud_path_liq_per_gpt, nz, nzh, n_chunks, chunk, per_gpt_axis
        )
      elif cloud_path_liq is not None:
        cpl_int_z1c = _interior_zc(cloud_path_liq, nz, nzh)[:, jnp.newaxis, :]
      if cloud_path_ice_per_gpt is not None:
        cpi_packed = _pack_per_gpt_cloud_zc(
            cloud_path_ice_per_gpt, nz, nzh, n_chunks, chunk, per_gpt_axis
        )
      elif cloud_path_ice is not None:
        cpi_int_z1c = _interior_zc(cloud_path_ice, nz, nzh)[:, jnp.newaxis, :]
    else:
      cpl_b = (
          None
          if cloud_path_liq is None
          else _interior_zc(cloud_path_liq, nz, nzh)[jnp.newaxis, :, :]
      )
      cpi_b = (
          None
          if cloud_path_ice is None
          else _interior_zc(cloud_path_ice, nz, nzh)[jnp.newaxis, :, :]
      )
      cld_props_by_band = _cloud_props_from_mie(
          optics_lib,
          liq_mie,
          ice_mie,
          cpl_b,
          cpi_b,
          _scale_to_interior(
              cloud_tau_scale_liq, nz, nzh, nc, with_middle_axis=False
          ),
          _scale_to_interior(
              cloud_tau_scale_ice, nz, nzh, nc, with_middle_axis=False
          ),
          is_lw=True,
      )

  aer_int = (
      None
      if aerosol_optics is None
      else {k: _interior_bzc(v, nz, nzh) for k, v in aerosol_optics.items()}
  )

  toa_flux_down_chunk = jnp.broadcast_to(
      _to_1c(atmos_state.toa_flux_lw / n_gpt, nc, dtype), (chunk, nc)
  )
  sfc_emis_chunk = jnp.broadcast_to(
      _to_1c(atmos_state.sfc_emis, nc, dtype), (chunk, nc)
  )
  sfc_refl_chunk = 1.0 - sfc_emis_chunk

  valid_mask_4d = None
  if needs_mask:
    valid_np = (np.arange(n_chunks * chunk) < n_gpt).astype(np.float32)
    valid_mask_4d = jnp.asarray(valid_np, dtype=dtype).reshape(
        n_chunks, 1, chunk, 1
    )

  def step_fn(ichunk, cumulative):
    bnd_oh = jax.lax.dynamic_index_in_dim(
        bnd_oh_packed, ichunk, axis=0, keepdims=False
    )
    tau_g = jax.lax.dynamic_index_in_dim(
        tau_packed, ichunk, axis=0, keepdims=False
    )
    pfrac = jax.lax.dynamic_index_in_dim(
        pfrac_packed, ichunk, axis=0, keepdims=False
    )

    b_face_g = _band_to_zgc(bnd_oh, b_face)
    psrc_top = pfrac[:-1] * b_face_g
    psrc_bot_next = pfrac[1:] * b_face_g
    prod = psrc_top * psrc_bot_next
    pos = prod > 0.0
    level_src = jnp.where(pos, jnp.sqrt(jnp.where(pos, prod, 1.0)), 0.0)
    level_src_bottom = level_src[:-1]
    level_src_top = level_src[1:]

    b_sfc_g = jnp.matmul(
        bnd_oh, b_sfc, precision=jax.lax.Precision.HIGHEST
    )
    sfc_src = pfrac[1] * b_sfc_g
    lay_src = None
    if need_lay_src:
      assert b_lay is not None
      lay_src = pfrac[1:-1] * _band_to_zgc(bnd_oh, b_lay)

    zeros_g = jnp.zeros_like(tau_g)
    gas_props = {
        'optical_depth': tau_g,
        'ssa': zeros_g,
        'asymmetry_factor': zeros_g,
    }
    all_sky_props = gas_props
    if has_cloud:
      if per_gpt_cloud:
        cpl_c = (
            cpl_int_z1c
            if cpl_packed is None
            else jax.lax.dynamic_index_in_dim(
                cpl_packed, ichunk, axis=0, keepdims=False
            )
        )
        cpi_c = (
            cpi_int_z1c
            if cpi_packed is None
            else jax.lax.dynamic_index_in_dim(
                cpi_packed, ichunk, axis=0, keepdims=False
            )
        )
        cld_props = _cloud_props_from_mie(
            optics_lib,
            liq_mie,
            ice_mie,
            cpl_c,
            cpi_c,
            scale_liq_z1c,
            scale_ice_z1c,
            is_lw=True,
            bnd_oh_chunk=bnd_oh,
        )
      else:
        assert cld_props_by_band is not None
        cld_props = {
            k: _band_to_zgc(bnd_oh, v) for k, v in cld_props_by_band.items()
        }
      all_sky_props = optics_lib.combine_optical_properties(
          gas_props, cld_props
      )

    aer_chunk = (
        None
        if aer_int is None
        else {k: _band_to_zgc(bnd_oh, v) for k, v in aer_int.items()}
    )

    def solve_one(props, aer):
      if aer is not None:
        props = optics_lib.combine_optical_properties(props, aer)
      if lw_solver == LW_SOLVER_NOSCAT:
        assert lay_src is not None
        return _lw_noscat_transport_zc(
            props['optical_depth'],
            props['ssa'],
            lay_src,
            level_src_top,
            level_src_bottom,
            toa_flux_down_chunk,
            sfc_src,
            sfc_emis_chunk,
            secant=lw_secant,
        )
      cell = monochromatic_two_stream.lw_cell_source_and_properties(
          props['optical_depth'],
          props['ssa'],
          level_src_bottom,
          level_src_top,
          props['asymmetry_factor'],
      )
      sfc_emission = np.pi * sfc_emis_chunk * sfc_src
      return _solve_rte_2stream_zc(
          cell['t_diff'],
          cell['r_diff'],
          cell['src_up'],
          cell['src_down'],
          toa_flux_down_chunk,
          sfc_emission,
          sfc_refl_chunk,
      )

    per_chunk = _solve_problems(
        solve_one,
        all_sky_props,
        gas_props if with_clear_sky else None,
        aer_chunk,
        with_clear_sky,
        with_aerosol_free,
    )
    mask_g = (
        None
        if valid_mask_4d is None
        else jax.lax.dynamic_index_in_dim(
            valid_mask_4d, ichunk, axis=0, keepdims=False
        )
    )

    def reduce_add(chunked: Array, acc: Array) -> Array:
      if mask_g is not None:
        chunked = chunked * mask_g
      return jnp.sum(chunked, axis=1) + acc

    return jax.tree.map(reduce_add, per_chunk, cumulative)

  flux_keys = [
      f'{prefix}{key}'
      for prefix in _problem_prefixes(with_clear_sky, with_aerosol_free)
      for key in ('flux_up', 'flux_down', 'flux_net')
  ]
  init_faces = {
      key: jnp.zeros((nz + 1, nc), dtype=dtype) for key in flux_keys
  }
  summed_faces = jax.lax.fori_loop(0, n_chunks, step_fn, init_faces)
  zero_halo = jnp.zeros((1, nc), dtype=dtype)
  return {
      key: jnp.concatenate([zero_halo, val], axis=0)
      for key, val in summed_faces.items()
  }


def _sw_sfc_albedo_by_band_zc(
    albedo: Array | float,
    nc: int,
    n_bnd: int,
    name: str,
    dtype: jnp.dtype,
) -> Array | None:
  """Resolve SW surface albedo into `(n_bnd, nc)` when per-band, else `None`."""
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
) -> dict[str, Array]:
  """Column-minor `(nzh, nc)` shortwave radiative transfer solver for TPU."""
  nzh, nc = temperature.shape
  nz = nzh - 2
  dtype = temperature.dtype
  lookup_sw = optics_lib.gas_optics_sw
  assert lookup_sw is not None
  n_gpt = optics_lib.n_gpt_sw
  n_bnd = lookup_sw.n_bnd
  chunk, n_chunks, needs_mask = _tpu_gpt_chunk_plan(n_gpt, gpt_chunk)

  alb_dir_by_band = (
      None
      if sfc_alb_dir is None
      else _sw_sfc_albedo_by_band_zc(
          sfc_alb_dir, nc, n_bnd, 'sfc_alb_dir', dtype
      )
  )
  alb_dif_by_band = (
      None
      if sfc_alb_dif is None
      else _sw_sfc_albedo_by_band_zc(
          sfc_alb_dif, nc, n_bnd, 'sfc_alb_dif', dtype
      )
  )

  vmr_idx = (
      None
      if vmr_fields is None
      else {
          k: _interior_zc(v, nz, nzh)
          for k, v in _reindex_vmr_fields(vmr_fields, lookup_sw).items()
      }
  )
  # Nothing in `solve_sw` ever reads the z halos: evaluate SW gas optics on
  # interior `(nz, nc)` cells directly.
  gas_all = optics_lib.compute_gas_optics_all_gpts_zc(
      pressure[1:-1, :],
      temperature[1:-1, :],
      molecules[1:-1, :],
      vmr_idx,
      is_lw=False,
  )
  tau_packed = _pack_gpt_zc_to_chunks(gas_all['optical_depth'], n_chunks, chunk)
  rayl_packed = _pack_gpt_zc_to_chunks(gas_all['rayleigh'], n_chunks, chunk)

  bnd_oh_packed = _band_oh_packed(
      lookup_sw.g_point_to_bnd, n_bnd, n_chunks, chunk, dtype
  )
  pad_g = n_chunks * chunk - n_gpt
  sfrac = jnp.asarray(optics_lib.solar_fraction_by_gpt, dtype=dtype)
  if pad_g:
    sfrac = jnp.concatenate([sfrac, jnp.repeat(sfrac[-1:], pad_g)], axis=0)
  solar_frac_packed = sfrac.reshape(n_chunks, chunk, 1)
  irrad_1c = _to_1c(atmos_state.irrad, nc, dtype)

  zenith = atmos_state.zenith
  if jnp.ndim(zenith) > 0:
    zenith = jnp.broadcast_to(jnp.asarray(zenith, dtype=dtype), (nc,))
  night = zenith >= 0.5 * jnp.pi
  safe_zenith = jnp.where(
      night, jnp.zeros_like(jnp.asarray(zenith, dtype=dtype)), zenith
  )
  if jnp.ndim(safe_zenith) > 0:
    safe_zenith_3d = safe_zenith.reshape(1, 1, nc)
    mu_1c = jnp.cos(safe_zenith).reshape(1, nc)
    mu_z1c = mu_1c[jnp.newaxis, :, :]
  else:
    safe_zenith_3d = safe_zenith
    mu_1c = jnp.cos(safe_zenith)
    mu_z1c = mu_1c

  sfc_alb_default = jnp.broadcast_to(
      _to_1c(atmos_state.sfc_alb, nc, dtype), (chunk, nc)
  )
  alb_dir_const = (
      sfc_alb_default
      if sfc_alb_dir is None
      else (
          None
          if alb_dir_by_band is not None
          else jnp.broadcast_to(_to_1c(sfc_alb_dir, nc, dtype), (chunk, nc))
      )
  )
  alb_dif_const = (
      sfc_alb_default
      if sfc_alb_dif is None
      else (
          None
          if alb_dif_by_band is not None
          else jnp.broadcast_to(_to_1c(sfc_alb_dif, nc, dtype), (chunk, nc))
      )
  )

  has_liq = cloud_path_liq_per_gpt is not None or cloud_path_liq is not None
  has_ice = cloud_path_ice_per_gpt is not None or cloud_path_ice is not None
  has_cloud = has_liq or has_ice
  per_gpt_cloud = (
      cloud_path_liq_per_gpt is not None or cloud_path_ice_per_gpt is not None
  )

  liq_mie = ice_mie = None
  cld_props_by_band = None
  cpl_packed = cpi_packed = None
  cpl_int_z1c = cpi_int_z1c = None
  scale_liq_z1c = scale_ice_z1c = None

  if has_cloud:
    assert optics_lib.cloud_optics_sw is not None
    liq_mie, ice_mie = _cloud_mie_tables_mxu(
        optics_lib.cloud_optics_sw,
        _interior_zc(cloud_r_eff_liq, nz, nzh),
        _interior_zc(cloud_r_eff_ice, nz, nzh),
        has_liq,
        has_ice,
        nz,
        nc,
        dtype,
    )
    if per_gpt_cloud:
      scale_liq_z1c = _scale_to_interior(
          cloud_tau_scale_liq, nz, nzh, nc, with_middle_axis=True
      )
      scale_ice_z1c = _scale_to_interior(
          cloud_tau_scale_ice, nz, nzh, nc, with_middle_axis=True
      )
      if cloud_path_liq_per_gpt is not None:
        cpl_packed = _pack_per_gpt_cloud_zc(
            cloud_path_liq_per_gpt, nz, nzh, n_chunks, chunk, per_gpt_axis
        )
      elif cloud_path_liq is not None:
        cpl_int_z1c = _interior_zc(cloud_path_liq, nz, nzh)[:, jnp.newaxis, :]
      if cloud_path_ice_per_gpt is not None:
        cpi_packed = _pack_per_gpt_cloud_zc(
            cloud_path_ice_per_gpt, nz, nzh, n_chunks, chunk, per_gpt_axis
        )
      elif cloud_path_ice is not None:
        cpi_int_z1c = _interior_zc(cloud_path_ice, nz, nzh)[:, jnp.newaxis, :]
    else:
      cpl_b = (
          None
          if cloud_path_liq is None
          else _interior_zc(cloud_path_liq, nz, nzh)[jnp.newaxis, :, :]
      )
      cpi_b = (
          None
          if cloud_path_ice is None
          else _interior_zc(cloud_path_ice, nz, nzh)[jnp.newaxis, :, :]
      )
      cld_props_by_band = _cloud_props_from_mie(
          optics_lib,
          liq_mie,
          ice_mie,
          cpl_b,
          cpi_b,
          _scale_to_interior(
              cloud_tau_scale_liq, nz, nzh, nc, with_middle_axis=False
          ),
          _scale_to_interior(
              cloud_tau_scale_ice, nz, nzh, nc, with_middle_axis=False
          ),
          is_lw=False,
      )

  aer_int = (
      None
      if aerosol_optics is None
      else {k: _interior_bzc(v, nz, nzh) for k, v in aerosol_optics.items()}
  )

  zeros_chunk = jnp.zeros((chunk, nc), dtype=dtype)
  valid_mask_4d = valid_mask_3d = None
  if needs_mask:
    valid_np = (np.arange(n_chunks * chunk) < n_gpt).astype(np.float32)
    valid_arr = jnp.asarray(valid_np, dtype=dtype)
    valid_mask_4d = valid_arr.reshape(n_chunks, 1, chunk, 1)
    valid_mask_3d = valid_arr.reshape(n_chunks, chunk, 1)

  def step_fn(ichunk, cumulative):
    bnd_oh = jax.lax.dynamic_index_in_dim(
        bnd_oh_packed, ichunk, axis=0, keepdims=False
    )
    tau_abs = jax.lax.dynamic_index_in_dim(
        tau_packed, ichunk, axis=0, keepdims=False
    )
    rayl = jax.lax.dynamic_index_in_dim(
        rayl_packed, ichunk, axis=0, keepdims=False
    )
    solar_frac = jax.lax.dynamic_index_in_dim(
        solar_frac_packed, ichunk, axis=0, keepdims=False
    )
    toa_flux = jnp.broadcast_to(irrad_1c * solar_frac, (chunk, nc))
    flux_down_direct_bc = toa_flux * mu_1c

    sfc_albedo_dir = (
        alb_dir_const
        if alb_dir_const is not None
        else jnp.matmul(
            bnd_oh, alb_dir_by_band, precision=jax.lax.Precision.HIGHEST
        )
    )
    sfc_albedo_dif = (
        alb_dif_const
        if alb_dif_const is not None
        else jnp.matmul(
            bnd_oh, alb_dif_by_band, precision=jax.lax.Precision.HIGHEST
        )
    )

    gas_props = optics_lib._sw_gas_properties(tau_abs, rayl)  # pylint: disable=protected-access
    all_sky_props = gas_props
    if has_cloud:
      if per_gpt_cloud:
        cpl_c = (
            cpl_int_z1c
            if cpl_packed is None
            else jax.lax.dynamic_index_in_dim(
                cpl_packed, ichunk, axis=0, keepdims=False
            )
        )
        cpi_c = (
            cpi_int_z1c
            if cpi_packed is None
            else jax.lax.dynamic_index_in_dim(
                cpi_packed, ichunk, axis=0, keepdims=False
            )
        )
        cld_props = _cloud_props_from_mie(
            optics_lib,
            liq_mie,
            ice_mie,
            cpl_c,
            cpi_c,
            scale_liq_z1c,
            scale_ice_z1c,
            is_lw=False,
            bnd_oh_chunk=bnd_oh,
        )
      else:
        assert cld_props_by_band is not None
        cld_props = {
            k: _band_to_zgc(bnd_oh, v) for k, v in cld_props_by_band.items()
        }
      all_sky_props = optics_lib.combine_optical_properties(
          gas_props, cld_props
      )

    aer_chunk = (
        None
        if aer_int is None
        else {k: _band_to_zgc(bnd_oh, v) for k, v in aer_int.items()}
    )

    def solve_one(props, aer):
      if aer is not None:
        props = optics_lib.combine_optical_properties(props, aer)
      cell = monochromatic_two_stream.sw_cell_properties(
          safe_zenith_3d,
          props['optical_depth'],
          props['ssa'],
          props['asymmetry_factor'],
      )
      tau_pos = jnp.maximum(props['optical_depth'], 0.0)
      t_noscat = jnp.exp(-tau_pos / mu_z1c)

      flux_down_sfc, (fdir_int, fdir_above) = jax.lax.scan(
          _direct_op_zc, flux_down_direct_bc, t_noscat, reverse=True
      )
      src_up = cell['r_dir'] * fdir_above
      src_down = cell['t_dir'] * fdir_above
      sfc_src = sfc_albedo_dir * flux_down_sfc

      diff_fluxes = _solve_rte_2stream_zc(
          cell['t_diff'],
          cell['r_diff'],
          src_up,
          src_down,
          zeros_chunk,
          sfc_src,
          sfc_albedo_dif,
      )
      fdir_faces = jnp.concatenate(
          [fdir_int, flux_down_direct_bc[jnp.newaxis, :, :]], axis=0
      )
      flux_up = diff_fluxes['flux_up']
      flux_down = diff_fluxes['flux_down'] + fdir_faces
      return {
          'flux_up': flux_up,
          'flux_down': flux_down,
          'flux_net': flux_up - flux_down,
          'flux_down_dir_sfc': flux_down_sfc,
      }

    per_chunk = _solve_problems(
        solve_one,
        all_sky_props,
        gas_props if with_clear_sky else None,
        aer_chunk,
        with_clear_sky,
        with_aerosol_free,
    )
    mask_4d = (
        None
        if valid_mask_4d is None
        else jax.lax.dynamic_index_in_dim(
            valid_mask_4d, ichunk, axis=0, keepdims=False
        )
    )
    mask_3d = (
        None
        if valid_mask_3d is None
        else jax.lax.dynamic_index_in_dim(
            valid_mask_3d, ichunk, axis=0, keepdims=False
        )
    )

    def reduce_add(chunked: Array, acc: Array) -> Array:
      if chunked.ndim == 3:
        if mask_4d is not None:
          chunked = chunked * mask_4d
        return jnp.sum(chunked, axis=1) + acc
      if mask_3d is not None:
        chunked = chunked * mask_3d
      return jnp.sum(chunked, axis=0) + acc

    return jax.tree.map(reduce_add, per_chunk, cumulative)

  init_faces = {}
  for prefix in _problem_prefixes(with_clear_sky, with_aerosol_free):
    for key in ('flux_up', 'flux_down', 'flux_net'):
      init_faces[f'{prefix}{key}'] = jnp.zeros((nz + 1, nc), dtype=dtype)
    init_faces[f'{prefix}flux_down_dir_sfc'] = jnp.zeros((nc,), dtype=dtype)

  summed = jax.lax.fori_loop(0, n_chunks, step_fn, init_faces)
  zero_halo = jnp.zeros((1, nc), dtype=dtype)
  out = {}
  for key, val in summed.items():
    if val.ndim == 2:
      val = jnp.concatenate([zero_halo, val], axis=0)
    mask_n = night[jnp.newaxis, :] if (jnp.ndim(night) == 1 and val.ndim == 2) else night
    out[key] = jnp.where(mask_n, jnp.zeros_like(val), val)
  return out


def compute_heating_rate_zc(
    flux_net_zc: Array,
    pressure_zc: Array,
    q_v_zc: Array | None = None,
) -> Array:
  """Column-minor `(nzh, nc)` counterpart of `compute_heating_rate`."""
  dp = 0.5 * (
      jnp.roll(pressure_zc, -1, axis=0) - jnp.roll(pressure_zc, 1, axis=0)
  )
  dflux = jnp.roll(flux_net_zc, -1, axis=0) - flux_net_zc
  cp = constants.CP_D
  if q_v_zc is not None:
    cp = cp + (constants.CP_V - constants.CP_D) * q_v_zc
  return constants.G * dflux / dp / cp


def _atmos_state_3d_to_c(
    atmos_state: AtmosphericState, nx: int, ny: int
) -> AtmosphericState:
  """Flatten any non-scalar `(nx, ny)` fields on `atmos_state` to `(nx * ny,)`."""
  nc = nx * ny
  updates = {}
  for field_name in ('zenith', 'irrad', 'sfc_alb', 'sfc_emis', 'toa_flux_lw'):
    val = getattr(atmos_state, field_name)
    if jnp.ndim(val) > 0:
      updates[field_name] = jnp.broadcast_to(jnp.asarray(val), (nx, ny)).reshape(nc)
  return dataclasses.replace(atmos_state, **updates) if updates else atmos_state


def _to_zc_3d(x: Array | None, nx: int, ny: int, nzh: int) -> Array | None:
  """Transpose `(nx, ny, nzh)` (or broadcastable) to `(nzh, nx * ny)`."""
  if x is None:
    return None
  if x.shape != (nx, ny, nzh):
    x = jnp.broadcast_to(x, (nx, ny, nzh))
  return jnp.moveaxis(x, 2, 0).reshape(nzh, nx * ny)


def _scale_3d_to_zc(
    scale: Array | float | None, nx: int, ny: int, nzh: int
) -> Array | float | None:
  if scale is None or jnp.ndim(scale) == 0:
    return scale
  return _to_zc_3d(jnp.asarray(scale), nx, ny, nzh)


def _solve_lw_3d_via_zc(
    pressure: Array,
    temperature: Array,
    molecules: Array,
    optics_lib: optics.RRTMOptics,
    atmos_state: AtmosphericState,
    *,
    vmr_fields: dict[int, Array] | None,
    sfc_temperature: Array | float | None,
    cloud_r_eff_liq: Array | None,
    cloud_path_liq: Array | None,
    cloud_r_eff_ice: Array | None,
    cloud_path_ice: Array | None,
    cloud_path_liq_per_gpt: Array | None,
    cloud_path_ice_per_gpt: Array | None,
    aerosol_optics: dict[str, Array] | None,
    gpt_chunk: int,
    cloud_tau_scale_liq: Array | float | None,
    cloud_tau_scale_ice: Array | float | None,
    with_clear_sky: bool,
    lw_solver: str,
    lw_secant: float,
    with_aerosol_free: bool,
) -> dict[str, Array]:
  """Bridge 3-D `(nx, ny, nzh)` `solve_lw` onto `solve_lw_zc`."""
  nx, ny, nzh = temperature.shape
  nc = nx * ny
  sfc_t = (
      sfc_temperature
      if sfc_temperature is None or jnp.ndim(sfc_temperature) == 0
      else jnp.broadcast_to(jnp.asarray(sfc_temperature), (nx, ny)).reshape(nc)
  )
  vmr_zc = (
      None
      if vmr_fields is None
      else {k: _to_zc_3d(v, nx, ny, nzh) for k, v in vmr_fields.items()}
  )
  cpl_pg = (
      None
      if cloud_path_liq_per_gpt is None
      else jnp.moveaxis(cloud_path_liq_per_gpt, 3, 1).reshape(
          cloud_path_liq_per_gpt.shape[0], nzh, nc
      )
  )
  cpi_pg = (
      None
      if cloud_path_ice_per_gpt is None
      else jnp.moveaxis(cloud_path_ice_per_gpt, 3, 1).reshape(
          cloud_path_ice_per_gpt.shape[0], nzh, nc
      )
  )
  aer_zc = (
      None
      if aerosol_optics is None
      else {
          k: jnp.moveaxis(v, 3, 1).reshape(v.shape[0], nzh, nc)
          for k, v in aerosol_optics.items()
      }
  )
  out_zc = solve_lw_zc(
      _to_zc_3d(pressure, nx, ny, nzh),
      _to_zc_3d(temperature, nx, ny, nzh),
      _to_zc_3d(molecules, nx, ny, nzh),
      optics_lib,
      _atmos_state_3d_to_c(atmos_state, nx, ny),
      vmr_fields=vmr_zc,
      sfc_temperature=sfc_t,
      cloud_r_eff_liq=_to_zc_3d(cloud_r_eff_liq, nx, ny, nzh),
      cloud_path_liq=_to_zc_3d(cloud_path_liq, nx, ny, nzh),
      cloud_r_eff_ice=_to_zc_3d(cloud_r_eff_ice, nx, ny, nzh),
      cloud_path_ice=_to_zc_3d(cloud_path_ice, nx, ny, nzh),
      cloud_path_liq_per_gpt=cpl_pg,
      cloud_path_ice_per_gpt=cpi_pg,
      aerosol_optics=aer_zc,
      gpt_chunk=gpt_chunk,
      cloud_tau_scale_liq=_scale_3d_to_zc(cloud_tau_scale_liq, nx, ny, nzh),
      cloud_tau_scale_ice=_scale_3d_to_zc(cloud_tau_scale_ice, nx, ny, nzh),
      with_clear_sky=with_clear_sky,
      lw_solver=lw_solver,
      lw_secant=lw_secant,
      with_aerosol_free=with_aerosol_free,
      per_gpt_axis=0,
  )
  return {
      k: jnp.moveaxis(v.reshape(nzh, nx, ny), 0, 2) for k, v in out_zc.items()
  }


def _solve_sw_3d_via_zc(
    pressure: Array,
    temperature: Array,
    molecules: Array,
    optics_lib: optics.RRTMOptics,
    atmos_state: AtmosphericState,
    *,
    vmr_fields: dict[int, Array] | None,
    cloud_r_eff_liq: Array | None,
    cloud_path_liq: Array | None,
    cloud_r_eff_ice: Array | None,
    cloud_path_ice: Array | None,
    cloud_path_liq_per_gpt: Array | None,
    cloud_path_ice_per_gpt: Array | None,
    aerosol_optics: dict[str, Array] | None,
    gpt_chunk: int,
    cloud_tau_scale_liq: Array | float | None,
    cloud_tau_scale_ice: Array | float | None,
    sfc_alb_dir: Array | float | None,
    sfc_alb_dif: Array | float | None,
    with_clear_sky: bool,
    with_aerosol_free: bool,
) -> dict[str, Array]:
  """Bridge 3-D `(nx, ny, nzh)` `solve_sw` onto `solve_sw_zc`."""
  nx, ny, nzh = temperature.shape
  nc = nx * ny
  plane = (nx, ny)
  alb_dir_by_band = (
      None
      if sfc_alb_dir is None
      else _sw_sfc_albedo_by_band(sfc_alb_dir, plane, optics_lib, 'sfc_alb_dir')
  )
  alb_dif_by_band = (
      None
      if sfc_alb_dif is None
      else _sw_sfc_albedo_by_band(sfc_alb_dif, plane, optics_lib, 'sfc_alb_dif')
  )

  def _resolve_alb_arg(alb, by_band):
    if alb is None:
      return None
    if by_band is not None:
      return by_band.reshape(by_band.shape[0], nc)
    if jnp.ndim(alb) == 0:
      return alb
    return jnp.broadcast_to(jnp.asarray(alb), plane).reshape(nc)

  vmr_zc = (
      None
      if vmr_fields is None
      else {k: _to_zc_3d(v, nx, ny, nzh) for k, v in vmr_fields.items()}
  )
  cpl_pg = (
      None
      if cloud_path_liq_per_gpt is None
      else jnp.moveaxis(cloud_path_liq_per_gpt, 3, 1).reshape(
          cloud_path_liq_per_gpt.shape[0], nzh, nc
      )
  )
  cpi_pg = (
      None
      if cloud_path_ice_per_gpt is None
      else jnp.moveaxis(cloud_path_ice_per_gpt, 3, 1).reshape(
          cloud_path_ice_per_gpt.shape[0], nzh, nc
      )
  )
  aer_zc = (
      None
      if aerosol_optics is None
      else {
          k: jnp.moveaxis(v, 3, 1).reshape(v.shape[0], nzh, nc)
          for k, v in aerosol_optics.items()
      }
  )
  out_zc = solve_sw_zc(
      _to_zc_3d(pressure, nx, ny, nzh),
      _to_zc_3d(temperature, nx, ny, nzh),
      _to_zc_3d(molecules, nx, ny, nzh),
      optics_lib,
      _atmos_state_3d_to_c(atmos_state, nx, ny),
      vmr_fields=vmr_zc,
      cloud_r_eff_liq=_to_zc_3d(cloud_r_eff_liq, nx, ny, nzh),
      cloud_path_liq=_to_zc_3d(cloud_path_liq, nx, ny, nzh),
      cloud_r_eff_ice=_to_zc_3d(cloud_r_eff_ice, nx, ny, nzh),
      cloud_path_ice=_to_zc_3d(cloud_path_ice, nx, ny, nzh),
      cloud_path_liq_per_gpt=cpl_pg,
      cloud_path_ice_per_gpt=cpi_pg,
      aerosol_optics=aer_zc,
      gpt_chunk=gpt_chunk,
      cloud_tau_scale_liq=_scale_3d_to_zc(cloud_tau_scale_liq, nx, ny, nzh),
      cloud_tau_scale_ice=_scale_3d_to_zc(cloud_tau_scale_ice, nx, ny, nzh),
      sfc_alb_dir=_resolve_alb_arg(sfc_alb_dir, alb_dir_by_band),
      sfc_alb_dif=_resolve_alb_arg(sfc_alb_dif, alb_dif_by_band),
      with_clear_sky=with_clear_sky,
      with_aerosol_free=with_aerosol_free,
      per_gpt_axis=0,
  )
  return {
      k: (
          v.reshape(nx, ny)
          if v.ndim == 1
          else jnp.moveaxis(v.reshape(nzh, nx, ny), 0, 2)
      )
      for k, v in out_zc.items()
  }
