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
    p: Array, p_ref: Array, troposphere_offset: Array | None = None
) -> Interpolant:
  """Create a pressure interpolant based on reference pressure values."""
  log_p = jnp.log(p)
  log_p_ref = jnp.log(p_ref)
  return optics_utils.create_linear_interpolant(
      log_p, log_p_ref, offset=troposphere_offset
  )


def _mixing_fraction_interpolant(
    f: Array, n_mixing_fraction: int
) -> Interpolant:
  """Create a mixing fraction interpolant based on desired number of points."""
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
  return (
      molecules
      / _M2_TO_CM2_FACTOR
      * optics_utils.interpolate(
          lookup_gas_optics.kmajor[..., igpt],
          interpolant_fns=interpolant_fn_dict,
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
  contribution = (
      optics_utils.interpolate(
          coeffs,
          collections.OrderedDict((
              ('t', lambda: temperature_interpolant),
              ('m', mix_interpolant_fn),
              ('i', lambda: optics_utils.exact_index(slot_idx, coeffs.dtype)),
          )),
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
  rayl_tau_lower = optics_utils.interpolate(
      lkp.rayl_lower[..., igpt], interpolant_fns
  )
  rayl_tau_upper = optics_utils.interpolate(
      lkp.rayl_upper[..., igpt], interpolant_fns
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
  temperature_interpolant = optics_utils.create_linear_interpolant(
      temperature, lookup.t_ref
  )
  pressure_interpolant = _pressure_interpolant(
      p, lookup.p_ref, tropo_idx
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

  # 1-D interpolation of the Planck source.
  interpolant = optics_utils.create_linear_interpolant(
      temperature, lookup.t_planck
  )
  return planck_fraction * optics_utils.interpolate(
      lookup.totplnk[ibnd, :],
      collections.OrderedDict({'t': lambda: interpolant}),
  )
