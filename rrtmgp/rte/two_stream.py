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

from typing import Callable, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from rrtmgp import constants
from rrtmgp import kernel_ops
from rrtmgp.optics import atmospheric_state
from rrtmgp.optics import lookup_gas_optics_base
from rrtmgp.optics import optics
from rrtmgp.optics import optics_base
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

  **Two other levers were tried and failed; do not retry them.**

  * *Rearranging the arithmetic.* Collapsing the rewritten two-stream's
    two-branch selects and deduplicating its transcendentals (the parent commit
    of this one) measured 20.45 vs 20.68 s/sim-day -- no recovery. Cutting the
    number of `select`s did not cut the number of fusion roots.
  * *`use_scan=True`.* XLA's cost model predicts a large win; the measurement is
    44-49% *slower* end to end, because a `scan` serialises the very kernel
    launches that are the bottleneck.

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
      return jax.tree.map(jnp.add, one_gpt(igpt), cumulative)

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

    return jax.tree.map(reduce_and_add, per_gpt, cumulative)

  return jax.lax.fori_loop(0, n_chunks, step_fn, init_val)


def _compute_local_properties_lw(
    pressure: Array,
    temperature: Array,
    molecules: Array,
    igpt: Array,
    optics_lib: optics_base.OpticsScheme,
    vmr_fields: dict[int, Array] | None = None,
    sfc_temperature: Array | float | None = None,
    cloud_r_eff_liq: Array | None = None,
    cloud_path_liq: Array | None = None,
    cloud_r_eff_ice: Array | None = None,
    cloud_path_ice: Array | None = None,
    aerosol_optics_slice: dict[str, Array] | None = None,
) -> dict[str, Array]:
  """Compute local optical properties for longwave radiative transfer."""
  if isinstance(sfc_temperature, float):
    # Create a plane for the surface temperature representation.
    nx, ny, _ = temperature.shape
    sfc_temperature = sfc_temperature * jnp.ones(
        (nx, ny), dtype=temperature.dtype
    )

  # Compute optical properties: `optical_depth`, `ssa`, & `asymmetry_factor`.
  lw_optical_props = optics_lib.compute_lw_optical_properties(
      pressure,
      temperature,
      molecules,
      igpt,
      vmr_fields,
      cloud_r_eff_liq,
      cloud_path_liq,
      cloud_r_eff_ice,
      cloud_path_ice,
  )

  # Mix in aerosol contributions for this band, if supplied. Aerosol tau/ssa/g
  # are passed through as-is (no delta-scaling applied), matching the convention
  # of simple aerosol parameterisations.
  if aerosol_optics_slice is not None:
    lw_optical_props = optics_lib.combine_optical_properties(
        lw_optical_props, aerosol_optics_slice
    )

  # Compute Planck sources: `planck_src`, `planck_src_bottom`, `planck_src_top`,
  # and `planck_src_sfc`.
  planck_srcs = optics_lib.compute_planck_sources(
      pressure, temperature, igpt, vmr_fields, sfc_temperature=sfc_temperature
  )

  halo_width = 1
  sfc_src = planck_srcs.get(
      'planck_src_sfc', planck_srcs['planck_src_bottom'][:, :, halo_width]
  )

  # Compute combined Planck sources.  Output keys are `planck_src_bottom` and
  # `planck_src_top`.
  combined_srcs = monochromatic_two_stream.lw_combine_sources(planck_srcs)

  # Compute `t_diff`, `r_diff`, `src_up`, and `src_down`.
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
    vmr_fields: dict[str, Array], gas_optics_lib: AbstractLookupGasOptics
) -> dict[int, Array]:
  """Converts the chemical formulas of the gas species to RRTM indices."""
  return {gas_optics_lib.idx_gases[k]: v for k, v in vmr_fields.items()}


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
) -> dict[str, Array]:
  """Solves two-stream radiative transfer equation over the longwave spectrum.

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

  Returns:
    A dictionary with the following entries (in units of W/m²):
      `flux_up`: The upwelling longwave radiative flux at cell face i - 1/2.
      `flux_down`: The downwelling longwave radiative flux at face i - 1/2.
      `flux_net`: The net longwave radiative flux at face i - 1/2.
  """
  optics_lib = cast(optics.RRTMOptics | optics.GrayAtmosphereOptics, optics_lib)
  if vmr_fields is not None:
    # Convert the chemical formulas of the gas species to RRTM-consistent
    # numerical identifiers.
    vmr_fields = _reindex_vmr_fields(vmr_fields, optics_lib.gas_optics_lw)

  if aerosol_optics is not None:
    assert isinstance(optics_lib, optics.RRTMOptics), (
        'aerosol_optics requires the RRTMOptics scheme (needs per-band'
        ' g_point_to_bnd mapping).'
    )
    g_point_to_bnd_lw = optics_lib.gas_optics_lw.g_point_to_bnd

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
    optical_props_2stream = _compute_local_properties_lw(
        pressure,
        temperature,
        molecules,
        igpt,
        optics_lib,
        vmr_fields,
        sfc_temperature,
        cloud_r_eff_liq,
        cpl,
        cloud_r_eff_ice,
        cpi,
        aerosol_optics_slice=aer_slice,
    )

    # Boundary conditions. `toa_flux_lw` prescribes the *broadband* downwelling
    # flux at the top of the atmosphere, but each g-point is solved as a
    # separate radiative transfer problem and the results are summed over the
    # spectrum below. Feeding the full broadband value to every g-point would
    # therefore return `n_gpt_lw` times the prescribed flux. Split it across the
    # g-points so the spectral sum reproduces it exactly. The shortwave path
    # does the same thing with the physically-derived
    # `solar_fraction_by_gpt` weights; a prescribed broadband longwave flux
    # carries no such spectral information, so it is divided uniformly.
    sfc_src = optical_props_2stream['sfc_src']
    toa_flux_down_lw = (
        atmos_state.toa_flux_lw / optics_lib.n_gpt_lw
    ) * jnp.ones_like(sfc_src)
    sfc_emissivity_lw = atmos_state.sfc_emis * jnp.ones_like(sfc_src)

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

  flux_keys = ['flux_up', 'flux_down', 'flux_net']
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

  Returns:
    A dictionary with the following entries (in units of W/m²):
      `flux_up`: The upwelling shortwave radiative flux at cell face i - 1/2.
      `flux_down`: The downwelling shortwave radiative flux at face i - 1/2.
      `flux_net`: The net shortwave radiative flux at face i - 1/2.
  """
  zenith = atmos_state.zenith
  optics_lib = cast(optics.RRTMOptics | optics.GrayAtmosphereOptics, optics_lib)
  if vmr_fields is not None:
    # Convert the chemical formulas of the gas species to RRTM-consistent
    # numerical identifiers.
    vmr_fields = _reindex_vmr_fields(vmr_fields, optics_lib.gas_optics_sw)

  if aerosol_optics is not None:
    assert isinstance(optics_lib, optics.RRTMOptics), (
        'aerosol_optics requires the RRTMOptics scheme (needs per-band'
        ' g_point_to_bnd mapping).'
    )
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
    )
    # Mix in aerosol contributions for this band, if supplied. Done after the
    # gas+cloud combination (where cloud has already been delta-scaled in SW),
    # so the aerosol tau/ssa/g are passed through as-is.
    if aerosol_optics is not None:
      ibnd = g_point_to_bnd_sw[igpt]
      aer_slice = {k: v[ibnd] for k, v in aerosol_optics.items()}
      sw_optical_props = optics_lib.combine_optical_properties(
          sw_optical_props, aer_slice
      )
    optical_props_2stream = monochromatic_two_stream.sw_cell_properties(
        safe_zenith,
        sw_optical_props['optical_depth'],
        sw_optical_props['ssa'],
        sw_optical_props['asymmetry_factor'],
    )

    # Create an xy plane for the surface albedo and top-of-atmospehre flux, but
    # keep the same horizontal sharding as the temperature.
    sfc_albedo = atmos_state.sfc_alb * jnp.ones_like(temperature)[:, :, 0]

    # Monochromatic top of atmosphere flux.
    solar_flux = atmos_state.irrad * optics_lib.solar_fraction_by_gpt[igpt]
    toa_flux = solar_flux * jnp.ones_like(temperature)[:, :, 0]

    sources_2stream = monochromatic_two_stream.sw_cell_source(
        t_dir=optical_props_2stream['t_dir'],
        r_dir=optical_props_2stream['r_dir'],
        optical_depth=sw_optical_props['optical_depth'],
        toa_flux=toa_flux,
        sfc_albedo_direct=sfc_albedo,
        zenith=safe_zenith,
        use_scan=use_scan,
    )

    sw_fluxes = monochromatic_two_stream.sw_transport(
        t_diff=optical_props_2stream['t_diff'],
        r_diff=optical_props_2stream['r_diff'],
        src_up=sources_2stream['src_up'],
        src_down=sources_2stream['src_down'],
        sfc_src=sources_2stream['sfc_src'],
        sfc_albedo=sfc_albedo,
        flux_down_dir=sources_2stream['flux_down_dir'],
        use_scan=use_scan,
    )
    return sw_fluxes

  flux_keys = ['flux_up', 'flux_down', 'flux_net']
  fluxes_0 = {key: jnp.zeros_like(temperature) for key in flux_keys}

  # As in `solve_lw`, the top halo face is the top of the atmosphere and the
  # solver already produces the correct value there, so it is left untouched.
  fluxes = _solve_over_gpoints(
      one_gpt, optics_lib.n_gpt_sw, gpt_chunk, fluxes_0
  )
  # Zero out columns where the sun is at or below the horizon. `night` is a
  # scalar (single column) or, under a column `vmap`, a per-column scalar; it
  # broadcasts over the trailing spatial/vertical axes of each flux field.
  fluxes = {
      key: jnp.where(night, jnp.zeros_like(val), val)
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
