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

"""Utility functions for computing optical properties of atmospheric gases."""

import collections
import dataclasses
from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from rrtmgp.optics import lookup_gas_optics_base
from rrtmgp.optics import lookup_gas_optics_longwave
from rrtmgp.optics import lookup_gas_optics_shortwave
from rrtmgp.optics import lookup_volume_mixing_ratio
from rrtmgp.optics import optics_utils


Array: TypeAlias = jax.Array
Interpolant: TypeAlias = optics_utils.Interpolant
IndexAndWeight: TypeAlias = optics_utils.IndexAndWeight
# pylint: disable=line-too-long
AbstractLookupGasOptics: TypeAlias = (
    lookup_gas_optics_base.AbstractLookupGasOptics
)
LookupGasOpticsLongwave: TypeAlias = (
    lookup_gas_optics_longwave.LookupGasOpticsLongwave
)
LookupGasOpticsShortwave: TypeAlias = (
    lookup_gas_optics_shortwave.LookupGasOpticsShortwave
)
LookupVolumeMixingRatio: TypeAlias = (
    lookup_volume_mixing_ratio.LookupVolumeMixingRatio
)
# pylint: enable=line-too-long

_PASCAL_TO_HPASCAL_FACTOR = 0.01
_M2_TO_CM2_FACTOR = 1e4


def _pressure_interpolant(
    p: Array,
    p_ref: Array,
    troposphere_offset: Array | None = None,
    out_of_range: str = optics_utils.EXTRAPOLATE,
) -> Interpolant:
  """Create a pressure interpolant based on reference pressure values.

  Outside the table (above ~1096 hPa or below ~1 Pa) the log-pressure axis is
  extrapolated along its end interval, as RRTMGP's `fpress` is
  (`mo_gas_optics_rrtmgp_kernels.F90::interpolation`: the index is limited to
  the interior, the fraction is not). The Planck fraction passes
  `out_of_range=CLAMP` instead; see `compute_planck_fraction`.
  """
  log_p = jnp.log(p)
  log_p_ref = jnp.log(p_ref)
  return optics_utils.create_linear_interpolant(
      log_p, log_p_ref, offset=troposphere_offset, out_of_range=out_of_range
  )


def _mixing_fraction_interpolant(
    f: Array, n_mixing_fraction: int
) -> Interpolant:
  """Create a mixing fraction interpolant based on desired number of points.

  The binary-species parameter eta lies in [0, 1] by construction for
  non-negative mixing ratios, so this axis only meets its end nodes, where the
  interpolant is continuous (eta = 1 selects the last entry with unit weight).
  A negative mixing ratio supplied by a caller would be extrapolated like the
  other gas-optics axes.
  """
  return optics_utils.create_linear_interpolant(
      f, jnp.linspace(0.0, 1.0, n_mixing_fraction, dtype=jnp.float_)
  )


def get_vmr(
    lookup_gas_optics: AbstractLookupGasOptics,
    vmr_lib: LookupVolumeMixingRatio,
    species_idx: Array,
    vmr_fields: dict[int, Array] | None = None,
) -> Array:
  """Get the volume mixing ratio, given major gas species index and pressure.

  Args:
    lookup_gas_optics: An `AbstractLookupGasOptics` object containing an index
      for the gas species.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    species_idx: An `Array` containing indices of gas species whose VMR will be
      computed.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    An `Array` of the same shape as `species_idx` or `pressure_idx` containing
    the pointwise volume mixing ratios of the corresponding gas species at that
    pressure level.
  """
  idx_gases = lookup_gas_optics.idx_gases
  # Indices of background gases for which a global mean VMR is available.
  vmr_gm = [0.0] * len(idx_gases)
  # Map the gas names in `vmr_lib.global_means` dict to indices consistent with
  # the RRTMGP `key_species` table.
  for k, v in vmr_lib.global_means.items():
    vmr_gm[idx_gases[k]] = v

  vmr = optics_utils.lookup_values(
      jnp.stack(vmr_gm, dtype=jnp.float_), (species_idx,)
  )

  # Overwrite with available precomputed vmr.
  if vmr_fields is not None:
    for gas_idx, vmr_field in vmr_fields.items():
      vmr = jnp.where(species_idx == gas_idx, vmr_field, vmr)

  # Note: Skipping these checks for now. Cannot have these boolean checks inside
  # of a traced function.
  # if jnp.any(vmr < 0.0):
  #   raise ValueError('At least one volume mixing ratio (VMR) is negative.')
  # if jnp.any(vmr > 1.0):
  #   raise ValueError('At least one volume mixing ratio (VMR) is above 1.')

  return vmr


def _compute_relative_abundance_interpolant(
    lookup_gas_optics: AbstractLookupGasOptics,
    vmr_lib: LookupVolumeMixingRatio,
    troposphere_idx: Array,
    temperature_idx: Array,
    ibnd: Array,
    scale_by_mixture: bool,
    vmr_fields: dict[int, Array] | None = None,
) -> Interpolant:
  """Create an `Interpolant` object for relative abundance of a major species.

  Args:
    lookup_gas_optics: An `AbstractLookupGasOptics` object containing a RRTMGP
      index for all relevant gas species.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    troposphere_idx: An `Array` that is 1 where the corresponding pressure level
      is below the troposphere limit and 0 otherwise. This informs whether an
      offset should be added to the reference pressure indices when indexing
      into the `kmajor` table.
    temperature_idx: An `Array` containing indices of reference temperature
      values.
    ibnd: The frequency band for which the relative abundance is computed.
    scale_by_mixture: Whether to scale the weights by the gas mixture.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    An `Interpolant` object for the relative abundance of the major gas species
      in a particular electromagnetic frequency band.
  """
  major_species_idx = []
  vmr_for_interp = []
  vmr_ref = []
  for i in range(2):
    major_species_idx.append(
        optics_utils.lookup_values(
            lookup_gas_optics.key_species[ibnd, :, i], (troposphere_idx,)
        )
    )
    vmr_for_interp.append(
        get_vmr(lookup_gas_optics, vmr_lib, major_species_idx[i], vmr_fields)
    )
    vmr_ref.append(
        optics_utils.lookup_values(
            lookup_gas_optics.vmr_ref,
            (temperature_idx, major_species_idx[i], troposphere_idx),
        )
    )
  vmr_ref_ratio = vmr_ref[0] / vmr_ref[1]
  combined_vmr = vmr_for_interp[0] + vmr_ref_ratio * vmr_for_interp[1]
  # Consistent with how the RRTM absorption coefficient tables are designed, the
  # relative abundance defaults to 0.5 when the volume mixing ratio of both
  # dominant species is exactly 0.
  #
  # The denominator is substituted before the division rather than selected
  # after it. `jnp.where` evaluates both arms, so dividing by the unguarded
  # `combined_vmr` produces a NaN in the discarded arm -- harmless to the value,
  # but reverse-mode differentiation propagates `0 * NaN` back through the
  # select and poisons the gradient with respect to every gas concentration.
  # That made `jax.grad` of the longwave solve non-finite wherever a cell had
  # both dominant species at exactly zero.
  is_present = combined_vmr > 0
  safe_combined_vmr = jnp.where(is_present, combined_vmr, 1.0)
  relative_abundance = jnp.where(
      is_present, vmr_for_interp[0] / safe_combined_vmr, 0.5
  )
  interpolant = _mixing_fraction_interpolant(
      relative_abundance, lookup_gas_optics.n_mixing_fraction
  )
  if scale_by_mixture:
    interpolant.interp_low.weight *= combined_vmr
    interpolant.interp_high.weight *= combined_vmr
  return interpolant


def compute_major_optical_depth(
    lookup_gas_optics: AbstractLookupGasOptics,
    vmr: LookupVolumeMixingRatio,
    molecules: Array,
    temperature: Array,
    p: Array,
    igpt: Array,
    vmr_fields: dict[int, Array] | None = None,
) -> Array:
  """Compute the optical depth contributions from major gases.

  Args:
    lookup_gas_optics: An `AbstractLookupGasOptics` object containing a RRTMGP
      index for all major gas species.
    vmr: A `LookupVolumeMixingRatio` object containing the volume mixing ratio
      of all relevant atmospheric gases.
    molecules: The number of molecules in an atmospheric grid cell per area
      [molecules/m^2]
    temperature: An `Array` containing temperature values (in K).
    p: An `Array` containing pressure values (in Pa).
    igpt: The absorption variable index (g-point) for which the optical depth is
      computed.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    An `Array` with the pointwise optical depth contributions from the major
    species.
  """
  # Take the troposphere limit into account when indexing into the major species
  # and absorption coefficients.
  # The troposphere index is 1 for levels above the troposphere limit and 0
  # otherwise.
  troposphere_idx = jnp.where(p <= lookup_gas_optics.p_ref_tropo, 1, 0)
  t_interp = optics_utils.create_linear_interpolant(
      temperature, lookup_gas_optics.t_ref
  )
  p_interp = _pressure_interpolant(
      p=p, p_ref=lookup_gas_optics.p_ref, troposphere_offset=troposphere_idx
  )
  # The frequency band for which the optical depth is computed.
  ibnd = lookup_gas_optics.g_point_to_bnd[igpt]

  def mix_interpolant_fn(t: IndexAndWeight) -> Interpolant:
    """Relative abundance interpolant function that depends on temperature.

    The arg name 't' is used throughout the interpolation logic to refer to the
    temperature index variable. The relative abundance variable is the only
    indexing variable that depends on another variable used to index the lookup
    tables.

    Args:
      t: An instance of `IndexAndWeight` encapsulating a point on the uniform
      temperature grid and its interpolation weight.

    Returns:
      An instance of `Interpolant` encapsulating the linear interpolation
      interval and weights on the relative abundance uniform grid.
    """
    return _compute_relative_abundance_interpolant(
        lookup_gas_optics,
        vmr,
        troposphere_idx,
        t.idx,
        ibnd,
        scale_by_mixture=True,
        vmr_fields=vmr_fields,
    )

  # Interpolant functions ordered according to the axes in `kmajor`.
  interpolant_fn_dict = collections.OrderedDict((
      ('t', lambda: t_interp),
      ('p', lambda: p_interp),
      ('m', mix_interpolant_fn),
  ))
  # Temperature and pressure outside the table are extrapolated along its end
  # interval (see `create_linear_interpolant`), which can go negative for a
  # large enough excursion; an absorption coefficient cannot.
  return (
      molecules
      / _M2_TO_CM2_FACTOR
      * optics_utils.floor_at_zero(
          optics_utils.interpolate(
              lookup_gas_optics.kmajor[..., igpt],
              interpolant_fns=interpolant_fn_dict,
          )
      )
  )


def minor_interval_count(
    minor_bnd_start: Array,
    minor_bnd_end: Array,
    minor_absorber_intervals: int,
) -> int:
  """Static number of minor-absorber intervals to evaluate: the widest band.

  Each band draws on a contiguous range of minor-absorber intervals. How many
  intervals are evaluated must be a compile-time constant (a data-dependent
  `while_loop` is not reverse-mode differentiable), but that count only has to
  cover the *widest* band, not the whole table: the evaluation walks the offset
  within a band, not the absolute interval index.

  This distinction is worth a lot. For the shipped tables the widest band spans
  5-11 intervals against table dimensions of 24-60, so covering the full table
  would evaluate roughly 12-16x more interpolations than any band actually
  uses, all of them masked away afterwards. That was the dominant cost in the
  runtime regression of issue #22.

  Args:
    minor_bnd_start: Per-band index of the first contributing minor interval,
      negative for a band with no minor absorbers.
    minor_bnd_end: Per-band index of the last contributing minor interval.
    minor_absorber_intervals: Size of the table's minor-interval dimension.

  Returns:
    The interval count, as a Python int so it stays a compile-time constant.
  """
  # These are loaded lookup constants, never traced values, so converting to
  # numpy here is safe under `jit` and keeps the result static.
  starts = np.asarray(minor_bnd_start)
  ends = np.asarray(minor_bnd_end)
  if starts.size == 0:
    return 0
  widths = np.where(starts >= 0, ends - starts + 1, 0)
  # Clamp into range: guards a degenerate/empty table and any stray limits.
  return int(np.clip(widths.max(), 0, minor_absorber_intervals))


class _MinorAbsorberTables(NamedTuple):
  """The lower- and upper-atmosphere minor-absorber tables as a single list.

  RRTMGP splits every minor-absorber quantity in two, one table for the
  atmosphere below the reference troposphere pressure and one for above, and a
  cell uses exactly one of them. The two are structurally identical -- same
  reference temperature and mixing-fraction axes, same per-interval metadata --
  so concatenating them along the interval axis (and shifting the upper
  contributor offsets past the end of the lower coefficient table) gives one
  list that a single pass can walk, with a per-cell mask deciding which half of
  it applies. See `compute_minor_optical_depth` for why that is worth doing.
  """

  # Minor absorption coefficients `(n_t_ref, n_eta, n_contrib_lower +
  # n_contrib_upper)`.
  kminor: Array
  # Per-interval metadata, each `(n_minor_absrb_lower + n_minor_absrb_upper)`.
  idx_gases: Array
  scales_with_density: Array
  idx_scaling_gas: Array
  scale_by_complement: Array
  # Offset into `kminor`, already shifted for the upper-atmosphere half.
  gpt_shift: Array
  # Where the upper-atmosphere half starts in the interval axis, and the total
  # interval count. Python ints: they bound indices at trace time.
  n_lower: int
  n_total: int


def _merge_minor_absorber_tables(
    lookup: AbstractLookupGasOptics,
) -> _MinorAbsorberTables:
  """Concatenate a lookup's lower- and upper-atmosphere minor tables."""

  def cat(lower: Array, upper: Array) -> Array:
    return jnp.concatenate([lower, upper], axis=-1)

  # The upper half's contributor offsets address `kminor_upper`, which now sits
  # after `kminor_lower` in the concatenated coefficient table.
  n_contrib_lower = lookup.kminor_lower.shape[-1]
  return _MinorAbsorberTables(
      kminor=cat(lookup.kminor_lower, lookup.kminor_upper),
      idx_gases=cat(
          lookup.idx_minor_gases_lower, lookup.idx_minor_gases_upper
      ),
      scales_with_density=cat(
          lookup.minor_lower_scales_with_density,
          lookup.minor_upper_scales_with_density,
      ),
      idx_scaling_gas=cat(
          lookup.idx_scaling_gases_lower, lookup.idx_scaling_gases_upper
      ),
      scale_by_complement=cat(
          lookup.lower_scale_by_complement, lookup.upper_scale_by_complement
      ),
      gpt_shift=cat(
          lookup.minor_lower_gpt_shift,
          lookup.minor_upper_gpt_shift + n_contrib_lower,
      ),
      n_lower=lookup.n_minor_absrb_lower,
      n_total=lookup.n_minor_absrb_lower + lookup.n_minor_absrb_upper,
  )


def compute_minor_optical_depth(
    lookup: AbstractLookupGasOptics,
    vmr_lib: LookupVolumeMixingRatio,
    molecules: Array,
    temperature: Array,
    p: Array,
    igpt: Array,
    vmr_fields: dict[int, Array] | None = None,
) -> Array:
  """Compute the optical depth contributions from minor gases.

  Every minor absorber that contributes to the g-point's band is evaluated in
  a single batched pass, structured around two observations about the cost of
  this function. It is the hottest piece of the radiative transfer solve on an
  accelerator, where the workload is launch-bound rather than flop-bound (the
  GPU is busy only 64-74% of the wall time and the average kernel is already
  small), and a bisect of the 0.3.0 throughput regression attributed 73% of the
  kernel launches it added to exactly this code. Two earlier attempts to
  recover that regression by cutting *arithmetic* out of the two-stream kernels
  recovered nothing at all, which is the evidence that launches, not flops, are
  what this costs.

  The two observations:

  * **The absorbers are a batch, not a sequence.** Each contributes an
    independent term to a sum, so the per-interval quantities carry the
    interval as a leading axis, the coefficient lookup gathers every interval
    in one go, and one reduction sums the axis away. The number of intervals
    must still be a compile-time constant -- a `while_loop` with a
    data-dependent trip count is not reverse-mode differentiable, and this code
    path exists to be differentiated -- but a static batch width is as
    differentiable as a static loop length while costing the launches of a
    single interval instead of one set per interval. Only the *widest* band's
    range needs to fit: covering the whole table would evaluate 12-16x more
    interpolations than any band uses, all masked away, which was the issue #22
    regression.

  * **Lower and upper atmosphere are one pass, not two.** Which of RRTMGP's two
    minor-absorber tables a cell uses depends only on its pressure, not on the
    g-point, so evaluating both and selecting afterwards doubled the work for
    every cell. The tables are concatenated (`_merge_minor_absorber_tables`)
    into one interval list whose slots carry which half they came from, and the
    per-cell mask keeps only the slots on that cell's side of the reference
    troposphere pressure. Same total number of slots as the two passes had
    between them, half the number of passes.

  Args:
    lookup: An instance of `AbstractLookupGasOptics` containing a RRTMGP index
      for all relevant gases and a lookup table for minor absorption
      coefficients.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    molecules: The number of molecules in an atmospheric grid cell per area
      [molecules/m^2]
    temperature: The temperature of the flow field [K].
    p: The pressure field (in Pa).
    igpt: The absorption rank (g-point) index for which the optical depth will
      be computed.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    An `Array` with the pointwise optical depth contributions from the minor
    species.
  """
  # How many intervals the widest band of each half spans. Both are Python
  # ints, so the batch width below is a compile-time constant.
  width_lower = minor_interval_count(
      lookup.minor_lower_bnd_start,
      lookup.minor_lower_bnd_end,
      lookup.n_minor_absrb_lower,
  )
  width_upper = minor_interval_count(
      lookup.minor_upper_bnd_start,
      lookup.minor_upper_bnd_end,
      lookup.n_minor_absrb_upper,
  )
  n_slots = width_lower + width_upper
  if n_slots == 0:
    # No band in either table has any minor absorber.
    return jnp.zeros_like(temperature)

  tables = _merge_minor_absorber_tables(lookup)

  # Which side of the reference troposphere pressure each cell is on. The
  # troposphere index is 1 for levels above the troposphere limit and 0
  # otherwise.
  is_lower_atmos = p > lookup.p_ref_tropo
  tropo_idx = jnp.where(is_lower_atmos, 0, 1)

  ibnd = lookup.g_point_to_bnd[igpt]
  loc_in_bnd = igpt - lookup.bnd_lims_gpt[ibnd, 0]
  temperature_interpolant = optics_utils.create_linear_interpolant(
      temperature, lookup.t_ref
  )

  if vmr_fields is not None and lookup.idx_h2o in vmr_fields:
    dry_factor = 1.0 / (1.0 + vmr_fields[lookup.idx_h2o])
  else:
    dry_factor = 1.0

  def mix_interpolant_fn(t: IndexAndWeight) -> Interpolant:
    """Relative abundance interpolant that depends on `t`."""
    return _compute_relative_abundance_interpolant(
        lookup, vmr_lib, tropo_idx, t.idx, ibnd, False, vmr_fields
    )

  def per_slot(values: Array) -> Array:
    """Shape a per-slot vector to broadcast against the field's axes.

    Keeping per-slot quantities at `(n_slots, 1, ..., 1)` rather than
    broadcasting them to the field's shape is deliberate: only the quantities
    that genuinely vary over both slot *and* cell -- the gathered absorption
    coefficients and the scaling that multiplies them -- are ever materialised
    at `(n_slots, *field)`.
    """
    return values.reshape(values.shape + (1,) * temperature.ndim)

  # The slot layout is static: slots `[0, width_lower)` walk the lower
  # atmosphere's band range and the rest walk the upper's, so each slot's
  # interval index depends only on the band, never on the cell. That is what
  # keeps the metadata lookups below per-slot vectors rather than per-cell
  # fields.
  slot = np.arange(n_slots)
  slot_is_lower = slot < width_lower
  index_dtype = lookup.minor_lower_bnd_start.dtype
  offset_in_side = jnp.asarray(
      np.where(slot_is_lower, slot, slot - width_lower), dtype=index_dtype
  )

  # Each half's contributing intervals form the contiguous range
  # ``[bnd_start[ibnd], bnd_end[ibnd]]``; a negative start flags a band with no
  # minor absorbers in that half, so no slot contributes.
  lower_start = lookup.minor_lower_bnd_start[ibnd]
  upper_start = lookup.minor_upper_bnd_start[ibnd]
  slot_start = jnp.where(
      slot_is_lower, lower_start, upper_start + tables.n_lower
  )
  slot_end = jnp.where(
      slot_is_lower,
      lookup.minor_lower_bnd_end[ibnd],
      lookup.minor_upper_bnd_end[ibnd] + tables.n_lower,
  )
  slot_has_minor = jnp.where(
      slot_is_lower, lower_start >= 0, upper_start >= 0
  )
  # A slot may only address intervals within its own half of the merged list.
  slot_limit = jnp.where(slot_is_lower, tables.n_lower, tables.n_total)

  i_unclamped = slot_start + offset_in_side
  in_range = jnp.logical_and(
      jnp.logical_and(i_unclamped <= slot_end, slot_has_minor),
      i_unclamped < slot_limit,
  )
  # The index is only read where `in_range`; clamp it so the surplus slots of a
  # narrower-than-widest band stay inside the table.
  i = jnp.clip(i_unclamped, 0, tables.n_total - 1)
  # A slot contributes to a cell only if it came from the half of the tables
  # that cell's pressure selects.
  contributes = jnp.logical_and(
      per_slot(in_range), per_slot(jnp.asarray(slot_is_lower)) == is_lower_atmos
  )

  # Map each minor contributor to the RRTMGP gas index.
  vmr_minor = get_vmr(
      lookup, vmr_lib, per_slot(tables.idx_gases[i]), vmr_fields
  )
  scaling = vmr_minor * molecules / _M2_TO_CM2_FACTOR

  # Density scaling, and the scaling gas it may in turn be scaled by. The
  # sequential form selected these with `lax.cond` on a scalar per-interval
  # flag; batched over slots the flags are a vector, so the selection is a
  # `where` over both arms. Neither arm can produce a non-finite value (no
  # division by a possibly-zero quantity, no fractional power), so evaluating
  # both is safe for reverse-mode differentiation as well as for the value.
  scaling_vmr = get_vmr(
      lookup,
      vmr_lib,
      per_slot(jnp.maximum(tables.idx_scaling_gas[i], 0)),
      vmr_fields,
  )
  gas_scaling = jnp.where(
      per_slot(tables.scale_by_complement[i] == 1),
      1.0 - scaling_vmr * dry_factor,
      scaling_vmr * dry_factor,
  )
  gas_scaling = jnp.where(
      per_slot(tables.idx_scaling_gas[i] > 0), gas_scaling, 1.0
  )
  density_scaling = _PASCAL_TO_HPASCAL_FACTOR * p / temperature * gas_scaling
  scaling = scaling * jnp.where(
      per_slot(tables.scales_with_density[i] == 1), density_scaling, 1.0
  )

  # Global contributor indices into the merged `kminor` table, one per slot.
  # The table is sliced down to just those contributors -- `(n_t, n_eta,
  # n_slots)` -- so the slot becomes an ordinary, exactly-indexed table axis
  # that the temperature/mixing-fraction interpolation gathers alongside the
  # axes it interpolates. Indexing the slot axis rather than broadcasting the
  # table across the field is what keeps this cheap in memory.
  k_loc = tables.gpt_shift[i] + loc_in_bnd
  coeffs = tables.kminor[..., k_loc]
  slot_idx = per_slot(jnp.arange(n_slots))
  # Floored for the same reason as the major coefficient: temperature outside
  # the table is extrapolated along its end interval.
  contribution = (
      optics_utils.floor_at_zero(
          optics_utils.interpolate(
              coeffs,
              collections.OrderedDict((
                  ('t', lambda: temperature_interpolant),
                  ('m', mix_interpolant_fn),
                  (
                      'i',
                      lambda: optics_utils.exact_index(slot_idx, coeffs.dtype),
                  ),
              )),
          )
      )
      * scaling
  )
  return jnp.sum(jnp.where(contributes, contribution, 0.0), axis=0)


def compute_rayleigh_optical_depth(
    lkp: LookupGasOpticsShortwave,
    vmr_lib: LookupVolumeMixingRatio,
    molecules: Array,
    temperature: Array,
    p: Array,
    igpt: Array,
    vmr_fields: dict[int, Array] | None = None,
) -> Array:
  """Compute the optical depth contribution from Rayleigh scattering.

  Args:
    lkp: An instance of `AbstractLookupGasOptics` containing a RRTMGP index
      for all relevant gases and a lookup table for Rayleigh absorption
      coefficients.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    molecules: The number of molecules in an atmospheric grid cell per area
      [molecules/m^2].
    temperature: Temperature variable (in K).
    p: The pressure field (in Pa).
    igpt: The absorption variable index (g-point) for which the optical depth
      will be computed.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    An `Array` with the pointwise optical depth contributions from Rayleigh
      scattering.
  """
  # The troposphere index is 1 for levels above the troposphere limit and 0
  # otherwise.
  tropo_idx = jnp.where(p <= lkp.p_ref_tropo, 1, 0)
  temperature_interpolant = optics_utils.create_linear_interpolant(
      temperature, lkp.t_ref
  )
  ibnd = lkp.g_point_to_bnd[igpt]

  def mix_interpolant_fn(t: IndexAndWeight) -> Interpolant:
    """Relative abundance interpolant function that depends on `t` and `p`."""
    return _compute_relative_abundance_interpolant(
        lkp, vmr_lib, tropo_idx, t.idx, ibnd, False, vmr_fields
    )

  interpolant_fns = collections.OrderedDict(
      (('t', lambda: temperature_interpolant), ('m', mix_interpolant_fn))
  )
  # Floored for the same reason as the absorption coefficients: temperature
  # outside the table is extrapolated along its end interval.
  rayl_tau_lower = optics_utils.floor_at_zero(
      optics_utils.interpolate(lkp.rayl_lower[..., igpt], interpolant_fns)
  )
  rayl_tau_upper = optics_utils.floor_at_zero(
      optics_utils.interpolate(lkp.rayl_upper[..., igpt], interpolant_fns)
  )
  if vmr_fields is not None and lkp.idx_h2o in vmr_fields:
    factor = 1.0 + vmr_fields[lkp.idx_h2o]
  else:
    factor = 1.0

  return (
      factor
      * molecules
      / _M2_TO_CM2_FACTOR
      * jnp.where(tropo_idx == 1, rayl_tau_upper, rayl_tau_lower)
  )


def compute_planck_fraction(
    lookup: LookupGasOpticsLongwave,
    vmr_lib: LookupVolumeMixingRatio,
    p: Array,
    temperature: Array,
    igpt: Array,
    vmr_fields: dict[int, Array] | None = None,
) -> Array:
  """Computes the Planck fraction that will be used to weight the Planck source.

  Args:
    lookup: An `LookupGasOpticsLongwave` object containing a RRTMGP index for
      all relevant gases and a lookup table for the Planck source.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    p: The pressure of the flow field [Pa].
    temperature: The temperature at the grid cell center [K].
    igpt: The absorption rank (g-point) index for which the optical depth will
      be computed.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    The pointwise Planck fraction associated with the temperature field.
  """
  # The troposphere index is 1 for levels above the troposphere limit and 0
  # otherwise.
  tropo_idx = jnp.where(p <= lookup.p_ref_tropo, 1, 0)
  # The Planck fractions of a band partition the band's Planck source among its
  # g-points: they sum to one over the band, and that sum is what conserves the
  # band's emitted energy once the solver adds the g-points up. Temperature and
  # pressure outside the table are therefore CLAMPED here, not extrapolated
  # like the absorption coefficients:
  #
  # * With every weight in [0, 1] the fraction is a convex combination of
  #   non-negative table entries, so it is non-negative and its band sum is a
  #   convex combination of the table's own band sums (1 to ~1e-4) -- with no
  #   floor. Extrapolating instead drives single fractions negative (band 10
  #   of the shipped tables: -0.08 at 40 K), and flooring those element-wise
  #   inflates the band sum, by up to 8% (g128) / 15% (g256) at 40 K, i.e. a
  #   cold layer would emit more band energy than its band source.
  # * The temperature dependence of the emitted energy is carried by the band
  #   source (`compute_planck_sources`), which still extrapolates. RRTMG's
  #   Planck fractions do not depend on temperature or pressure at all, only on
  #   the binary-species parameter (ECHAM6 `mo_lrtm_gas_optics.f90`, `fracs =
  #   fracrefa(ig, jpl) + fpl * (...)`). RRTMGP interpolates them with the same
  #   weights as `kmajor` (`compute_Planck_source` in
  #   `mo_gas_optics_rrtmgp_kernels.F90`), unfloored, which keeps the band sum
  #   but not the sign; its inputs never leave the table because the frontend
  #   rejects them (`check_values`).
  #
  # Inside the table the clamp is the identity, so in-range fractions are
  # unchanged.
  temperature_interpolant = optics_utils.create_linear_interpolant(
      temperature, lookup.t_ref, out_of_range=optics_utils.CLAMP
  )
  pressure_interpolant = _pressure_interpolant(
      p, lookup.p_ref, tropo_idx, out_of_range=optics_utils.CLAMP
  )
  ibnd = lookup.g_point_to_bnd[igpt]

  def mix_interpolant_fn(t: IndexAndWeight) -> Interpolant:
    """Relative abundance interpolant function that depends on `temperature`."""
    return _compute_relative_abundance_interpolant(
        lookup, vmr_lib, tropo_idx, t.idx, ibnd, False, vmr_fields
    )

  interpolants_fns = collections.OrderedDict((
      ('t', lambda: temperature_interpolant),
      ('p', lambda: pressure_interpolant),
      ('m', mix_interpolant_fn),
  ))

  # 3-D interpolation of the Planck fraction.
  return optics_utils.interpolate(
      lookup.planck_fraction[..., igpt], interpolants_fns
  )


def compute_planck_sources(
    lookup: LookupGasOpticsLongwave,
    planck_fraction: Array,
    temperature: Array,
    igpt: Array,
) -> Array:
  """Computes the Planck source for the longwave problem.

  Args:
    lookup: An `LookupGasOpticsLongwave` object containing a RRTMGP index for
      all relevant gases and a lookup table for the Planck source.
    planck_fraction: The Planck fraction that scales the Planck source.
    temperature: The temperature [K] for which the Planck source will be
      computed.
    igpt: The absorption rank (g-point) index for which the optical depth will
      be computed.

  Returns:
    The planck source emanating from the points with given `temperature` [W/m²].
  """
  ibnd = lookup.g_point_to_bnd[igpt]

  # 1-D interpolation of the Planck source. Outside the table the source is
  # extrapolated along the end interval, as RRTMG's `planckFunction` does
  # (ECHAM6 `mo_lrtm_driver.f90`) and as RRTMGP's `interpolate1D` does within
  # one table step of either end. The linear extension of the band-integrated
  # Planck function below 160 K reaches zero somewhere between roughly 110 and
  # 155 K depending on the band, so the source is floored there.
  interpolant = optics_utils.create_linear_interpolant(
      temperature, lookup.t_planck
  )
  return planck_fraction * optics_utils.floor_at_zero(
      optics_utils.interpolate(
          lookup.totplnk[ibnd, :],
          collections.OrderedDict({'t': lambda: interpolant}),
      )
  )


# ---------------------------------------------------------------------------
# All-g-point gas optics.
#
# The functions above evaluate the gas optics of ONE g-point, and the spectral
# loop calls them once per g-point (or per chunk of g-points under `vmap`).
# Everything below evaluates ALL g-points of the spectrum in one pass, laid out
# the way the RRTMGP GPU kernels lay it out
# (`rrtmgp/kernels/accel/mo_gas_optics_rrtmgp_kernels.F90`: `interpolation`
# once, then `gas_optical_depths_major` / `_minor` with the g-point innermost):
#
# * Everything that does not depend on the g-point -- the temperature, pressure
#   and mixing-fraction indices and weights, the troposphere mask, the minor
#   absorbers' scalings -- is computed once per cell and shared by every
#   g-point.
# * Every table read is a ROW gather: the table is indexed on its leading
#   temperature / pressure / mixing-fraction axes with the cell's corner
#   indices and a full slice on the g-point axis, so one gather index yields a
#   contiguous run of g-points (`slice_sizes=(1, 1, 1, n)`) instead of a single
#   scalar. The per-g-point path issues one scalar gather per (cell, corner,
#   g-point), which on a GPU is bound by gather throughput rather than by
#   memory bandwidth, and that count is what this layout divides by the row
#   width.
#
# The mixing-fraction interpolant of a cell depends on the band's two major
# species (`key_species`), so rows can only be shared by g-points whose bands
# have the same species pair. Bands are therefore grouped by their `key_species`
# signature (ten groups for each shipped table, against 14-16 bands): the
# interpolants are computed once per group and the group's g-points, which are
# not necessarily contiguous, are gathered together as one row.
#
# The arithmetic of every g-point is that of the single-g-point functions,
# term for term and in the same order (the corner order and the left-to-right
# sum of `optics_utils.interpolate_orig`), so the results are identical to the
# per-g-point path. Both use the direct-indexing (gather) lookup backend.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _BandGroup:
  """Bands that share a major-species pair, hence their cell interpolants."""

  # Band indices, ascending.
  bands: tuple[int, ...]
  # The group's g-points: the bands' g-point ranges concatenated in band order.
  gpts: np.ndarray


def _band_gpt_range(lookup: AbstractLookupGasOptics, ibnd: int) -> np.ndarray:
  """The g-points of band `ibnd`, as a static integer array."""
  lims = np.asarray(lookup.bnd_lims_gpt)
  return np.arange(int(lims[ibnd, 0]), int(lims[ibnd, 1]) + 1)


def band_groups(lookup: AbstractLookupGasOptics) -> list[_BandGroup]:
  """Groups the bands of a lookup by their pair of major species.

  `key_species[ibnd]` names the two major species (lower and upper atmosphere)
  that set a band's relative-abundance interpolant, which is the only part of
  the cell interpolation that depends on the band. Bands with the same
  signature have identical interpolants.

  Args:
    lookup: The gas-optics lookup.

  Returns:
    The groups, in order of their first band.
  """
  key_species = np.asarray(lookup.key_species)
  by_signature: dict[tuple[int, ...], list[int]] = {}
  for ibnd in range(key_species.shape[0]):
    by_signature.setdefault(tuple(key_species[ibnd].ravel().tolist()), []).append(
        ibnd
    )
  return [
      _BandGroup(
          bands=tuple(bands),
          gpts=np.concatenate([_band_gpt_range(lookup, b) for b in bands]),
      )
      for bands in by_signature.values()
  ]


def _columns(table: Array, gpts: np.ndarray) -> Array:
  """The last-axis (g-point) entries `gpts` of a table, as a static selection.

  A contiguous run is taken as a slice; anything else as a take with a static
  index array. Either way the selection is a small constant-index copy of the
  table (a few hundred KB), not part of the per-cell work.
  """
  if gpts.size and np.all(np.diff(gpts) == 1):
    return table[..., int(gpts[0]) : int(gpts[-1]) + 1]
  return table[..., gpts]


class CellInterpolation:
  """The g-point-independent interpolation state of one set of cells.

  Built once per call and shared by the major, minor, Planck and Rayleigh
  evaluations of every band group.
  """

  def __init__(
      self,
      lookup: AbstractLookupGasOptics,
      vmr_lib: LookupVolumeMixingRatio,
      temperature: Array,
      p: Array,
      vmr_fields: dict[int, Array] | None,
  ):
    self.lookup = lookup
    self.vmr_lib = vmr_lib
    self.vmr_fields = vmr_fields
    self.p = p
    self.temperature = temperature
    # 1 for levels above the troposphere limit and 0 otherwise.
    self.tropo_idx = jnp.where(p <= lookup.p_ref_tropo, 1, 0)
    self.t_interp = optics_utils.create_linear_interpolant(
        temperature, lookup.t_ref
    )
    self.p_interp = _pressure_interpolant(
        p=p, p_ref=lookup.p_ref, troposphere_offset=self.tropo_idx
    )
    self._clamped = None
    self._mix = {}
    self._key_species = np.asarray(lookup.key_species)

  def clamped(self) -> tuple[Interpolant, Interpolant]:
    """Temperature and pressure interpolants clamped at the table ends.

    These are the Planck fraction's (see `compute_planck_fraction`).
    """
    if self._clamped is None:
      self._clamped = (
          optics_utils.create_linear_interpolant(
              self.temperature,
              self.lookup.t_ref,
              out_of_range=optics_utils.CLAMP,
          ),
          _pressure_interpolant(
              self.p,
              self.lookup.p_ref,
              self.tropo_idx,
              out_of_range=optics_utils.CLAMP,
          ),
      )
    return self._clamped

  def mix(
      self, ibnd: int, t_side: str, scale_by_mixture: bool
  ) -> Interpolant:
    """Relative-abundance interpolant of band `ibnd` at a temperature corner.

    The interpolant depends on the temperature *index* (through the reference
    mixing ratios), so there is one per temperature corner. Bands with the same
    major-species pair share it.

    Args:
      ibnd: The band.
      t_side: `'high'` or `'low'`, the temperature corner it belongs to.
      scale_by_mixture: Whether the weights carry the major species' combined
        mixing ratio (the major absorption coefficient's, not the Planck
        fraction's or the minor and Rayleigh coefficients').
    """
    key = (
        tuple(self._key_species[ibnd].ravel().tolist()),
        t_side,
        scale_by_mixture,
    )
    if key not in self._mix:
      t = getattr(self.t_interp, f'interp_{t_side}')
      self._mix[key] = _compute_relative_abundance_interpolant(
          self.lookup,
          self.vmr_lib,
          self.tropo_idx,
          t.idx,
          ibnd,
          scale_by_mixture=scale_by_mixture,
          vmr_fields=self.vmr_fields,
      )
    return self._mix[key]


_SIDES = ('high', 'low')


def _interpolate_rows_tpm(
    table: Array,
    t_interp: Interpolant,
    p_interp: Interpolant,
    mix_for_t: dict[str, Interpolant],
) -> Array:
  """Eight-corner interpolation of rows of a `(t, p, eta, n)` table.

  The corner order, the weight product `(w_t * w_p) * w_eta` and the
  left-to-right sum are those of `optics_utils.interpolate_orig` for the
  `t -> p -> eta` interpolation (the mixing-fraction interpolant is chosen by
  the temperature corner), so each element equals the single-g-point result.

  Args:
    table: A `(n_t, n_p, n_eta, n)` table; the last axis is read as one row.
    t_interp: The temperature interpolant, of cell shape.
    p_interp: The pressure interpolant, of cell shape.
    mix_for_t: The mixing-fraction interpolant of each temperature corner.

  Returns:
    An array of shape `(*cell_shape, n)`.
  """
  total = None
  for eta_side in _SIDES:
    for p_side in _SIDES:
      for t_side in _SIDES:
        ti = getattr(t_interp, f'interp_{t_side}')
        pi = getattr(p_interp, f'interp_{p_side}')
        mi = getattr(mix_for_t[t_side], f'interp_{eta_side}')
        weight = (ti.weight * pi.weight) * mi.weight
        term = table[ti.idx, pi.idx, mi.idx] * weight[..., None]
        total = term if total is None else total + term
  return total


def _interpolate_rows_te(
    table: Array,
    t_interp: Interpolant,
    mix_for_t: dict[str, Interpolant],
    leading_idx: Array | None = None,
) -> Array:
  """Four-corner interpolation of rows of a `(t, eta, n)` table.

  Same convention as `_interpolate_rows_tpm`, for the minor and Rayleigh
  tables. With `leading_idx`, the table has one more leading axis that each
  cell indexes with `leading_idx` (the Rayleigh table's lower/upper half).
  """
  total = None
  for eta_side in _SIDES:
    for t_side in _SIDES:
      ti = getattr(t_interp, f'interp_{t_side}')
      mi = getattr(mix_for_t[t_side], f'interp_{eta_side}')
      weight = ti.weight * mi.weight
      if leading_idx is None:
        rows = table[ti.idx, mi.idx]
      else:
        rows = table[leading_idx, ti.idx, mi.idx]
      term = rows * weight[..., None]
      total = term if total is None else total + term
  return total


def _minor_slots(
    lookup: AbstractLookupGasOptics, ibnd: int
) -> list[tuple[bool, int]]:
  """The minor-absorber intervals of band `ibnd` as `(is_lower, interval)`.

  Lower-atmosphere intervals first, then upper, each ascending: the order in
  which the per-g-point path sums its slots.
  """
  slots = []
  for is_lower in (True, False):
    if is_lower:
      start = int(np.asarray(lookup.minor_lower_bnd_start)[ibnd])
      end = int(np.asarray(lookup.minor_lower_bnd_end)[ibnd])
      n_side = lookup.n_minor_absrb_lower
    else:
      start = int(np.asarray(lookup.minor_upper_bnd_start)[ibnd])
      end = int(np.asarray(lookup.minor_upper_bnd_end)[ibnd])
      n_side = lookup.n_minor_absrb_upper
    # A band with no minor absorbers on this side has its start (and end) at
    # `n_side`, one past the last interval.
    if 0 <= start < n_side:
      slots.extend((is_lower, i) for i in range(start, min(end, n_side - 1) + 1))
  return slots


def _minor_optical_depth_band(
    lookup: AbstractLookupGasOptics,
    interp: CellInterpolation,
    molecules: Array,
    ibnd: int,
    dry_factor: Array | float,
) -> Array | None:
  """Minor-absorber optical depth of every g-point of band `ibnd`.

  The physics is that of `compute_minor_optical_depth` (merged lower/upper
  intervals, a cell using the half its pressure selects, the density and
  scaling-gas scalings); only the g-point axis differs. Each interval covers
  the whole band in the shipped tables, so its contribution to the band's
  g-points is one contiguous run of `n_gpt_in_band` contributors, gathered as a
  row. The intervals are accumulated one at a time, so no
  `(n_intervals, *cells, n_gpt)` array exists.

  Returns:
    `(*cell_shape, n_gpt_in_band)`, or `None` for a band with no minor
    absorbers.
  """
  slots = _minor_slots(lookup, ibnd)
  if not slots:
    return None
  gpts = _band_gpt_range(lookup, ibnd)
  width = gpts.size
  is_lower_atmos = interp.p > lookup.p_ref_tropo
  mix_for_t = {s: interp.mix(ibnd, s, False) for s in _SIDES}
  total = None
  for is_lower, i in slots:
    if is_lower:
      table = lookup.kminor_lower
      meta = (
          lookup.idx_minor_gases_lower,
          lookup.idx_scaling_gases_lower,
          lookup.minor_lower_scales_with_density,
          lookup.lower_scale_by_complement,
          lookup.minor_lower_gpt_shift,
      )
    else:
      table = lookup.kminor_upper
      meta = (
          lookup.idx_minor_gases_upper,
          lookup.idx_scaling_gases_upper,
          lookup.minor_upper_scales_with_density,
          lookup.upper_scale_by_complement,
          lookup.minor_upper_gpt_shift,
      )
    gas, scaling_gas, scales_with_density, by_complement, shift = (
        int(np.asarray(m)[i]) for m in meta
    )
    vmr_minor = get_vmr(lookup, interp.vmr_lib, gas, interp.vmr_fields)
    scaling = vmr_minor * molecules / _M2_TO_CM2_FACTOR
    # Scaling by density and by a scaling gas. These flags are properties of
    # the interval, not of the cell, so they are static selections here.
    gas_scaling = 1.0
    if scaling_gas > 0:
      scaling_vmr = get_vmr(
          lookup, interp.vmr_lib, scaling_gas, interp.vmr_fields
      )
      gas_scaling = (
          1.0 - scaling_vmr * dry_factor
          if by_complement == 1
          else scaling_vmr * dry_factor
      )
    if scales_with_density == 1:
      density_scaling = (
          _PASCAL_TO_HPASCAL_FACTOR * interp.p / interp.temperature * gas_scaling
      )
      scaling = scaling * density_scaling
    rows = table[..., shift : shift + width]
    # Floored like the major coefficient: temperature outside the table is
    # extrapolated along its end interval.
    contribution = (
        optics_utils.floor_at_zero(
            _interpolate_rows_te(rows, interp.t_interp, mix_for_t)
        )
        * scaling[..., None]
    )
    side_mask = (is_lower_atmos if is_lower else ~is_lower_atmos)[..., None]
    term = jnp.where(side_mask, contribution, 0.0)
    total = term if total is None else total + term
  return total


def _dry_factor(
    lookup: AbstractLookupGasOptics, vmr_fields: dict[int, Array] | None
) -> Array | float:
  if vmr_fields is not None and lookup.idx_h2o in vmr_fields:
    return 1.0 / (1.0 + vmr_fields[lookup.idx_h2o])
  return 1.0


def _assemble_gpts(
    pieces: list[Array], groups: list[_BandGroup]
) -> Array:
  """Concatenates per-group `(*cells, n)` pieces into `(n_gpt, *cells)`.

  The pieces are in group order; the result is in g-point order, with the
  g-point axis leading so that the spectral loop takes one g-point (or one
  chunk of them) as a contiguous slab.
  """
  order = np.concatenate([g.gpts for g in groups])
  stacked = jnp.moveaxis(jnp.concatenate(pieces, axis=-1), -1, 0)
  if np.array_equal(order, np.arange(order.size)):
    return stacked
  return stacked[np.argsort(order)]


def compute_optical_depth_all_gpts(
    lookup: AbstractLookupGasOptics,
    vmr_lib: LookupVolumeMixingRatio,
    molecules: Array,
    temperature: Array,
    p: Array,
    vmr_fields: dict[int, Array] | None = None,
    interp: CellInterpolation | None = None,
) -> Array:
  """Gas absorption optical depth of every g-point: major plus minor.

  Equal to `compute_minor_optical_depth + compute_major_optical_depth` at each
  g-point (the same sum order).

  Args:
    lookup: The gas-optics lookup.
    vmr_lib: The volume-mixing-ratio library.
    molecules: Molecules per area of each cell [molecules/m^2].
    temperature: Temperature [K].
    p: Pressure [Pa].
    vmr_fields: Optional per-gas volume mixing ratio fields, keyed by gas index.
    interp: The cell interpolation state, if the caller already built it.

  Returns:
    `(n_gpt, *cell_shape)`.
  """
  if interp is None:
    interp = CellInterpolation(lookup, vmr_lib, temperature, p, vmr_fields)
  groups = band_groups(lookup)
  dry_factor = _dry_factor(lookup, vmr_fields)
  pieces = []
  for group in groups:
    ibnd0 = group.bands[0]
    mix_for_t = {s: interp.mix(ibnd0, s, True) for s in _SIDES}
    # Temperature and pressure outside the table are extrapolated along its end
    # interval, which can go negative for a large enough excursion; an
    # absorption coefficient cannot.
    major = (molecules / _M2_TO_CM2_FACTOR)[..., None] * (
        optics_utils.floor_at_zero(
            _interpolate_rows_tpm(
                _columns(lookup.kmajor, group.gpts),
                interp.t_interp,
                interp.p_interp,
                mix_for_t,
            )
        )
    )
    minors = [
        _minor_optical_depth_band(lookup, interp, molecules, b, dry_factor)
        for b in group.bands
    ]
    if all(m is None for m in minors):
      pieces.append(major)
      continue
    minor = jnp.concatenate(
        [
            jnp.zeros(molecules.shape + (_band_gpt_range(lookup, b).size,),
                      major.dtype) if m is None else m
            for b, m in zip(group.bands, minors)
        ],
        axis=-1,
    )
    pieces.append(minor + major)
  return _assemble_gpts(pieces, groups)


def compute_planck_fraction_all_gpts(
    lookup: LookupGasOpticsLongwave,
    vmr_lib: LookupVolumeMixingRatio,
    p: Array,
    temperature: Array,
    vmr_fields: dict[int, Array] | None = None,
    interp: CellInterpolation | None = None,
) -> Array:
  """Planck fraction of every g-point; see `compute_planck_fraction`.

  Returns:
    `(n_gpt, *cell_shape)`.
  """
  if interp is None:
    interp = CellInterpolation(lookup, vmr_lib, temperature, p, vmr_fields)
  # Clamped, not extrapolated, at the table ends: see `compute_planck_fraction`.
  t_clamped, p_clamped = interp.clamped()
  groups = band_groups(lookup)
  pieces = []
  for group in groups:
    ibnd0 = group.bands[0]
    mix_for_t = {s: interp.mix(ibnd0, s, False) for s in _SIDES}
    pieces.append(
        _interpolate_rows_tpm(
            _columns(lookup.planck_fraction, group.gpts),
            t_clamped,
            p_clamped,
            mix_for_t,
        )
    )
  return _assemble_gpts(pieces, groups)


def compute_rayleigh_optical_depth_all_gpts(
    lkp: LookupGasOpticsShortwave,
    vmr_lib: LookupVolumeMixingRatio,
    molecules: Array,
    temperature: Array,
    p: Array,
    vmr_fields: dict[int, Array] | None = None,
    interp: CellInterpolation | None = None,
) -> Array:
  """Rayleigh optical depth of every g-point; see `compute_rayleigh_optical_depth`.

  Returns:
    `(n_gpt, *cell_shape)`.
  """
  if interp is None:
    interp = CellInterpolation(lkp, vmr_lib, temperature, p, vmr_fields)
  if vmr_fields is not None and lkp.idx_h2o in vmr_fields:
    factor = 1.0 + vmr_fields[lkp.idx_h2o]
  else:
    factor = 1.0
  groups = band_groups(lkp)
  pieces = []
  for group in groups:
    ibnd0 = group.bands[0]
    mix_for_t = {s: interp.mix(ibnd0, s, False) for s in _SIDES}
    # The lower and upper tables are stacked so a cell reads the half its
    # pressure selects (the per-g-point path reads both and selects).
    table = jnp.stack([
        _columns(lkp.rayl_lower, group.gpts),
        _columns(lkp.rayl_upper, group.gpts),
    ])
    rayl = optics_utils.floor_at_zero(
        _interpolate_rows_te(
            table, interp.t_interp, mix_for_t, leading_idx=interp.tropo_idx
        )
    )
    pieces.append((factor * molecules / _M2_TO_CM2_FACTOR)[..., None] * rayl)
  return _assemble_gpts(pieces, groups)
