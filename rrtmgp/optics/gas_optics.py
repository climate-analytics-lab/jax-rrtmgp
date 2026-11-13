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
import math
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

# Capability flag advertising native TPU MXU `all_gpts` table contractions so
# downstream callers can skip runtime compatibility shims.
_SUPPORTS_TPU_MXU_ALL_GPTS = True


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
      f, jnp.linspace(0.0, 1.0, n_mixing_fraction, dtype=float)
  )


def _get_vmr_static(
    lookup_gas_optics: AbstractLookupGasOptics,
    vmr_lib: LookupVolumeMixingRatio,
    species_idx: int,
    vmr_fields: dict[int, Array] | None = None,
) -> Array:
  """Return the VMR for a compile-time constant integer `species_idx`."""
  if vmr_fields is not None and species_idx in vmr_fields:
    return vmr_fields[species_idx]
  idx_gases = lookup_gas_optics.idx_gases
  vmr_gm = [0.0] * len(idx_gases)
  for k, v in vmr_lib.global_means.items():
    vmr_gm[idx_gases[k]] = v
  return jnp.asarray(vmr_gm[species_idx], dtype=lookup_gas_optics.vmr_ref.dtype)


def get_vmr(
    lookup_gas_optics: AbstractLookupGasOptics,
    vmr_lib: LookupVolumeMixingRatio,
    species_idx: Array | int,
    vmr_fields: dict[int, Array] | None = None,
) -> Array:
  """Get the volume mixing ratio, given major gas species index and pressure.

  Args:
    lookup_gas_optics: An `AbstractLookupGasOptics` object containing an index
      for the gas species.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    species_idx: An `Array` or static integer containing indices of gas species
      whose VMR will be computed.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    An `Array` of the same shape as `species_idx` or `pressure_idx` containing
    the pointwise volume mixing ratios of the corresponding gas species at that
    pressure level.
  """
  if jax.default_backend() == 'tpu' and isinstance(
      species_idx, (int, np.integer)
  ):
    return _get_vmr_static(
        lookup_gas_optics, vmr_lib, int(species_idx), vmr_fields
    )
  idx_gases = lookup_gas_optics.idx_gases
  # Indices of background gases for which a global mean VMR is available.
  vmr_gm = [0.0] * len(idx_gases)
  # Map the gas names in `vmr_lib.global_means` dict to indices consistent with
  # the RRTMGP `key_species` table.
  for k, v in vmr_lib.global_means.items():
    vmr_gm[idx_gases[k]] = v

  vmr = optics_utils.lookup_values(
      jnp.stack(vmr_gm, dtype=float), (species_idx,)
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


def _compute_relative_abundance_interpolant_mxu(
    lookup_gas_optics: AbstractLookupGasOptics,
    vmr_lib: LookupVolumeMixingRatio,
    troposphere_idx: Array,
    temperature_idx: Array,
    ibnd: int,
    scale_by_mixture: bool,
    vmr_fields: dict[int, Array] | None = None,
    *,
    oh_t: Array | None = None,
    return_combined_vmr: bool = False,
) -> Interpolant | tuple[Interpolant, Array]:
  """MXU-backed relative-abundance interpolant for a static band `ibnd`.

  Avoids 1D and 3D dynamic gathers on TPU by selecting the lower- and upper-
  atmosphere species with `jnp.where` and projecting `vmr_ref` against a
  `(n_t, ncells)` temperature one-hot matrix at IEEE float32 (`HIGHEST`)
  precision, keeping `ncells` on the minor axis so 128-lane tiles are filled
  without padding.
  """
  ks = np.asarray(lookup_gas_optics.key_species)[int(ibnd)]
  n_t = lookup_gas_optics.vmr_ref.shape[0]
  dtype = lookup_gas_optics.vmr_ref.dtype
  cell_shape = temperature_idx.shape
  ncells = math.prod(cell_shape)
  if oh_t is None:
    oh_t = jax.nn.one_hot(
        temperature_idx.reshape(ncells), n_t, dtype=dtype, axis=0
    )
  is_tropo0 = troposphere_idx == 0
  sp00, sp10 = int(ks[0, 0]), int(ks[1, 0])
  sp01, sp11 = int(ks[0, 1]), int(ks[1, 1])
  vmr_for_interp = [
      jnp.where(
          is_tropo0,
          _get_vmr_static(lookup_gas_optics, vmr_lib, sp00, vmr_fields),
          _get_vmr_static(lookup_gas_optics, vmr_lib, sp10, vmr_fields),
      ),
      jnp.where(
          is_tropo0,
          _get_vmr_static(lookup_gas_optics, vmr_lib, sp01, vmr_fields),
          _get_vmr_static(lookup_gas_optics, vmr_lib, sp11, vmr_fields),
      ),
  ]
  ref_rows = jnp.stack(
      [
          lookup_gas_optics.vmr_ref[:, sp00, 0],
          lookup_gas_optics.vmr_ref[:, sp10, 1],
          lookup_gas_optics.vmr_ref[:, sp01, 0],
          lookup_gas_optics.vmr_ref[:, sp11, 1],
      ],
      axis=0,
  )
  ref_all = jnp.matmul(
      ref_rows, oh_t, precision=jax.lax.Precision.HIGHEST
  ).reshape(4, *cell_shape)
  vmr_ref_0 = jnp.where(is_tropo0, ref_all[0], ref_all[1])
  vmr_ref_1 = jnp.where(is_tropo0, ref_all[2], ref_all[3])
  vmr_ref_ratio = vmr_ref_0 / vmr_ref_1
  combined_vmr = vmr_for_interp[0] + vmr_ref_ratio * vmr_for_interp[1]
  is_present = combined_vmr > 0
  safe_combined_vmr = jnp.where(is_present, combined_vmr, 1.0)
  relative_abundance = jnp.where(
      is_present, vmr_for_interp[0] / safe_combined_vmr, 0.5
  )
  interpolant = _mixing_fraction_interpolant(
      relative_abundance, lookup_gas_optics.n_mixing_fraction
  )
  if scale_by_mixture:
    interpolant = Interpolant(
        interp_low=IndexAndWeight(
            idx=interpolant.interp_low.idx,
            weight=interpolant.interp_low.weight * combined_vmr,
        ),
        interp_high=IndexAndWeight(
            idx=interpolant.interp_high.idx,
            weight=interpolant.interp_high.weight * combined_vmr,
        ),
    )
  if return_combined_vmr:
    return interpolant, combined_vmr
  return interpolant


def _compute_relative_abundance_interpolant(
    lookup_gas_optics: AbstractLookupGasOptics,
    vmr_lib: LookupVolumeMixingRatio,
    troposphere_idx: Array,
    temperature_idx: Array,
    ibnd: Array | int,
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
  if (
      jax.default_backend() == 'tpu'
      and isinstance(ibnd, (int, np.integer))
      and temperature_idx.ndim > 0
  ):
    return _compute_relative_abundance_interpolant_mxu(
        lookup_gas_optics,
        vmr_lib,
        troposphere_idx,
        temperature_idx,
        int(ibnd),
        scale_by_mixture,
        vmr_fields,
    )
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


def _group_columns(
    lookup: AbstractLookupGasOptics, table: Array, group: _BandGroup
) -> Array:
  """Select a band group's g-point columns via contiguous per-band slices."""
  if len(group.bands) == 1 or (
      group.gpts.size and np.all(np.diff(group.gpts) == 1)
  ):
    return _columns(table, group.gpts)
  lims = np.asarray(lookup.bnd_lims_gpt)
  return jnp.concatenate(
      [table[..., int(lims[b, 0]) : int(lims[b, 1]) + 1] for b in group.bands],
      axis=-1,
  )


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
    self._oh_t: dict[str, Array] = {}
    self._unscaled_mix: dict[
        tuple[tuple[int, ...], str], tuple[Interpolant, Array]
    ] = {}
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
    sig = tuple(self._key_species[ibnd].ravel().tolist())
    key = (sig, t_side, scale_by_mixture)
    if key not in self._mix:
      t = getattr(self.t_interp, f'interp_{t_side}')
      if jax.default_backend() == 'tpu' and t.idx.ndim > 0:
        if t_side not in self._oh_t:
          n_t = self.lookup.vmr_ref.shape[0]
          dtype = self.lookup.vmr_ref.dtype
          self._oh_t[t_side] = jax.nn.one_hot(
              t.idx.reshape(-1), n_t, dtype=dtype, axis=0
          )
        base_key = (sig, t_side)
        if base_key not in self._unscaled_mix:
          res = _compute_relative_abundance_interpolant_mxu(
              self.lookup,
              self.vmr_lib,
              self.tropo_idx,
              t.idx,
              int(ibnd),
              False,
              self.vmr_fields,
              oh_t=self._oh_t[t_side],
              return_combined_vmr=True,
          )
          assert isinstance(res, tuple)
          self._unscaled_mix[base_key] = res
        base_interp, combined_vmr = self._unscaled_mix[base_key]
        self._mix[(sig, t_side, False)] = base_interp
        self._mix[(sig, t_side, True)] = Interpolant(
            interp_low=IndexAndWeight(
                idx=base_interp.interp_low.idx,
                weight=base_interp.interp_low.weight * combined_vmr,
            ),
            interp_high=IndexAndWeight(
                idx=base_interp.interp_high.idx,
                weight=base_interp.interp_high.weight * combined_vmr,
            ),
        )
      else:
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

  def build_te_weights_mxu(
      self, ibnd: int, scale_by_mixture: bool, dtype: jnp.dtype
  ) -> Array:
    """Build `(n_t * n_eta, ncells)` four-corner MXU weight matrix for `ibnd`."""
    n_t = self.lookup.t_ref.shape[0]
    n_eta = self.lookup.n_mixing_fraction
    ncells = self.temperature.size
    mix_for_t = {s: self.mix(ibnd, s, scale_by_mixture) for s in _SIDES}
    w_oh = None
    for eta_side in _SIDES:
      for t_side in _SIDES:
        ti = getattr(self.t_interp, f'interp_{t_side}')
        mi = getattr(mix_for_t[t_side], f'interp_{eta_side}')
        idx = (ti.idx * n_eta + mi.idx).reshape(ncells)
        w = (ti.weight * mi.weight).reshape(1, ncells)
        oh = w * jax.nn.one_hot(idx, n_t * n_eta, dtype=dtype, axis=0)
        w_oh = oh if w_oh is None else w_oh + oh
    assert w_oh is not None
    return w_oh


_SIDES = ('high', 'low')


def _tpm_block_plan(ncells: int) -> tuple[int, int]:
  """Choose `(n_blocks, block_cells)` for cell-minor MXU `tpm` contraction."""
  if ncells <= 4096:
    return 1, ncells
  if ncells % 128 == 0:
    for n_blocks in range(1, 65):
      if ncells % n_blocks != 0:
        continue
      block_cells = ncells // n_blocks
      if block_cells % 128 == 0 and 12288 <= block_cells <= 32768:
        return n_blocks, block_cells
  block_cells = 16384
  n_blocks = (ncells + block_cells - 1) // block_cells
  return n_blocks, block_cells


def _interpolate_rows_tpm_mxu(
    table: Array,
    t_interp: Interpolant,
    p_interp: Interpolant,
    mix_for_t: dict[str, Interpolant],
) -> Array:
  """Factored cell-minor MXU eight-corner interpolation for `(t, p, eta, n)`.

  Contracts the `(n_t * n_eta = 126)` corners first and the `(n_p = 60)`
  pressure nodes second at IEEE float32 (`HIGHEST`) precision in 128-lane-
  aligned cell blocks with `block_cells` on the minor axis, avoiding both 5D
  whole-grid intermediates (issue #6) and minor-axis tile padding on `n`.
  """
  cell_shape = t_interp.interp_high.idx.shape
  ncells = math.prod(cell_shape)
  n_t, n_p, n_eta, width = table.shape
  n_te = n_t * n_eta
  table_pteg = jnp.transpose(table, (1, 0, 2, 3)).reshape(n_p, n_te, width)
  n_blocks, block_cells = _tpm_block_plan(ncells)
  padded_cells = n_blocks * block_cells
  pad = padded_cells - ncells

  def _pack_idx(arr: Array) -> Array:
    flat = arr.reshape(ncells)
    if pad:
      flat = jnp.pad(flat, (0, pad))
    return flat.reshape(n_blocks, block_cells)

  def _pack_w(arr: Array) -> Array:
    flat = arr.reshape(ncells)
    if pad:
      flat = jnp.pad(flat, (0, pad))
    return flat.reshape(n_blocks, 1, block_cells)

  p_lo_i_b = _pack_idx(p_interp.interp_low.idx)
  p_lo_w_b = _pack_w(p_interp.interp_low.weight)
  p_hi_i_b = _pack_idx(p_interp.interp_high.idx)
  p_hi_w_b = _pack_w(p_interp.interp_high.weight)

  te_idx_w = []
  for eta_side in _SIDES:
    for t_side in _SIDES:
      ti = getattr(t_interp, f'interp_{t_side}')
      mi = getattr(mix_for_t[t_side], f'interp_{eta_side}')
      te_idx_w.append((
          _pack_idx(ti.idx * n_eta + mi.idx),
          _pack_w(ti.weight * mi.weight),
      ))

  def _one_block(inputs):
    p_lo_i, p_lo_w, p_hi_i, p_hi_w, corners = inputs
    wp = p_lo_w * jax.nn.one_hot(
        p_lo_i, n_p, dtype=table.dtype, axis=0
    ) + p_hi_w * jax.nn.one_hot(p_hi_i, n_p, dtype=table.dtype, axis=0)
    wte = None
    for idx_c, w_c in corners:
      oh = w_c * jax.nn.one_hot(idx_c, n_te, dtype=table.dtype, axis=0)
      wte = oh if wte is None else wte + oh
    mid = jnp.einsum(
        'pmg,mc->pgc',
        table_pteg,
        wte,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.einsum(
        'pc,pgc->gc',
        wp,
        mid,
        precision=jax.lax.Precision.HIGHEST,
    )

  if n_blocks == 1:
    first_corners = [(idx_b[0], w_b[0]) for idx_b, w_b in te_idx_w]
    out_gc = _one_block((
        p_lo_i_b[0], p_lo_w_b[0], p_hi_i_b[0], p_hi_w_b[0], first_corners
    ))
    return out_gc.T[:ncells].reshape(*cell_shape, width)

  out_bgc = jax.lax.map(
      _one_block, (p_lo_i_b, p_lo_w_b, p_hi_i_b, p_hi_w_b, te_idx_w)
  )
  out_flat = jnp.moveaxis(out_bgc, 1, 2).reshape(padded_cells, width)
  if pad:
    out_flat = out_flat[:ncells]
  return out_flat.reshape(*cell_shape, width)


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
  if jax.default_backend() == 'tpu' and t_interp.interp_high.idx.ndim > 0:
    return _interpolate_rows_tpm_mxu(table, t_interp, p_interp, mix_for_t)
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


def _interpolate_rows_te_mxu(
    table: Array,
    t_interp: Interpolant,
    mix_for_t: dict[str, Interpolant],
    leading_idx: Array | None = None,
) -> Array:
  """Cell-minor MXU four-corner interpolation of rows of a `(t, eta, n)` table."""
  cell_shape = t_interp.interp_high.idx.shape
  ncells = math.prod(cell_shape)
  if leading_idx is None:
    n_t, n_eta, width = table.shape
    n_te = n_t * n_eta
    flat_table_t = table.reshape(n_te, width).T
    w_oh = None
    for eta_side in _SIDES:
      for t_side in _SIDES:
        ti = getattr(t_interp, f'interp_{t_side}')
        mi = getattr(mix_for_t[t_side], f'interp_{eta_side}')
        idx = (ti.idx * n_eta + mi.idx).reshape(ncells)
        w = (ti.weight * mi.weight).reshape(1, ncells)
        oh = w * jax.nn.one_hot(idx, n_te, dtype=table.dtype, axis=0)
        w_oh = oh if w_oh is None else w_oh + oh
    return jnp.matmul(
        flat_table_t, w_oh, precision=jax.lax.Precision.HIGHEST
    ).T.reshape(*cell_shape, width)

  n_lead, n_t, n_eta, width = table.shape
  n_lte = n_lead * n_t * n_eta
  flat_table_t = table.reshape(n_lte, width).T
  lead_flat = leading_idx.reshape(ncells)
  w_oh = None
  for eta_side in _SIDES:
    for t_side in _SIDES:
      ti = getattr(t_interp, f'interp_{t_side}')
      mi = getattr(mix_for_t[t_side], f'interp_{eta_side}')
      idx = (
          (lead_flat * n_t + ti.idx.reshape(ncells)) * n_eta
          + mi.idx.reshape(ncells)
      )
      w = (ti.weight * mi.weight).reshape(1, ncells)
      oh = w * jax.nn.one_hot(idx, n_lte, dtype=table.dtype, axis=0)
      w_oh = oh if w_oh is None else w_oh + oh
  return jnp.matmul(
      flat_table_t, w_oh, precision=jax.lax.Precision.HIGHEST
  ).T.reshape(*cell_shape, width)


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
  if jax.default_backend() == 'tpu' and t_interp.interp_high.idx.ndim > 0:
    return _interpolate_rows_te_mxu(
        table, t_interp, mix_for_t, leading_idx=leading_idx
    )
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


def _interpolate_groups_tpm_mxu(
    lookup: AbstractLookupGasOptics,
    table: Array,
    t_interp: Interpolant,
    p_interp: Interpolant,
    interp: CellInterpolation,
    scale_by_mixture: bool,
) -> Array:
  """Batched cell-minor MXU `tpm` interpolation across all band groups.

  Evaluates every `_BandGroup` inside a single `jax.lax.map` over 128-lane-
  aligned cell blocks, sharing the `(n_p, block_cells)` pressure one-hot matrix
  across all groups, keeping `block_cells` on the minor axis (`1.00x` tile
  fill), and reassembling bands in ascending band order without a full-spectrum
  gather. Returns `(n_gpt, ncells)`.
  """
  ncells = t_interp.interp_high.idx.size
  n_t, n_p, n_eta, _ = table.shape
  n_te = n_t * n_eta
  groups = band_groups(lookup)
  band_widths = {
      b: _band_gpt_range(lookup, b).size for g in groups for b in g.bands
  }
  sorted_bands = sorted(band_widths)
  band_order = np.concatenate(
      [_band_gpt_range(lookup, b) for b in sorted_bands]
  )
  tables_pteg = [
      jnp.transpose(_group_columns(lookup, table, g), (1, 0, 2, 3)).reshape(
          n_p, n_te, g.gpts.size
      )
      for g in groups
  ]

  n_blocks, block_cells = _tpm_block_plan(ncells)
  padded_cells = n_blocks * block_cells
  pad = padded_cells - ncells

  def _pack_idx(arr: Array) -> Array:
    flat = arr.reshape(ncells)
    if pad:
      flat = jnp.pad(flat, (0, pad))
    return flat.reshape(n_blocks, block_cells)

  def _pack_w(arr: Array) -> Array:
    flat = arr.reshape(ncells)
    if pad:
      flat = jnp.pad(flat, (0, pad))
    return flat.reshape(n_blocks, 1, block_cells)

  p_lo_i_b = _pack_idx(p_interp.interp_low.idx)
  p_lo_w_b = _pack_w(p_interp.interp_low.weight)
  p_hi_i_b = _pack_idx(p_interp.interp_high.idx)
  p_hi_w_b = _pack_w(p_interp.interp_high.weight)

  te_by_group = []
  for g in groups:
    ibnd0 = g.bands[0]
    mix_for_t = {s: interp.mix(ibnd0, s, scale_by_mixture) for s in _SIDES}
    corners = []
    for eta_side in _SIDES:
      for t_side in _SIDES:
        ti = getattr(t_interp, f'interp_{t_side}')
        mi = getattr(mix_for_t[t_side], f'interp_{eta_side}')
        corners.append((
            _pack_idx(ti.idx * n_eta + mi.idx),
            _pack_w(ti.weight * mi.weight),
        ))
    te_by_group.append(tuple(corners))
  te_by_group_t = tuple(te_by_group)

  def _one_block(inputs):
    p_lo_i, p_lo_w, p_hi_i, p_hi_w, group_corners = inputs
    wp = p_lo_w * jax.nn.one_hot(
        p_lo_i, n_p, dtype=table.dtype, axis=0
    ) + p_hi_w * jax.nn.one_hot(p_hi_i, n_p, dtype=table.dtype, axis=0)
    by_band: dict[int, Array] = {}
    for g_idx, g in enumerate(groups):
      wte = None
      for idx_c, w_c in group_corners[g_idx]:
        oh = w_c * jax.nn.one_hot(idx_c, n_te, dtype=table.dtype, axis=0)
        wte = oh if wte is None else wte + oh
      mid = jnp.einsum(
          'pmg,mc->pgc',
          tables_pteg[g_idx],
          wte,
          precision=jax.lax.Precision.HIGHEST,
      )
      out_gc = jnp.einsum(
          'pc,pgc->gc',
          wp,
          mid,
          precision=jax.lax.Precision.HIGHEST,
      )
      off = 0
      for b in g.bands:
        wb = band_widths[b]
        by_band[b] = out_gc[off : off + wb]
        off += wb
    return jnp.concatenate([by_band[b] for b in sorted_bands], axis=0)

  if n_blocks == 1:
    first_groups = tuple(
        tuple((idx_b[0], w_b[0]) for idx_b, w_b in corners)
        for corners in te_by_group_t
    )
    out_gc = _one_block(
        (p_lo_i_b[0], p_lo_w_b[0], p_hi_i_b[0], p_hi_w_b[0], first_groups)
    )
    stacked = out_gc[:, :ncells]
  else:
    out_bgc = jax.lax.map(
        _one_block, (p_lo_i_b, p_lo_w_b, p_hi_i_b, p_hi_w_b, te_by_group_t)
    )
    stacked = jnp.moveaxis(out_bgc, 0, 1).reshape(-1, padded_cells)[:, :ncells]

  if not np.array_equal(band_order, np.arange(band_order.size)):
    stacked = stacked[np.argsort(band_order)]
  return stacked


def _minor_optical_depth_all_gpts_mxu(
    lookup: AbstractLookupGasOptics,
    interp: CellInterpolation,
    molecules: Array,
    dry_factor: Array | float,
) -> Array | None:
  """Cell-minor MXU minor-absorber optical depth `(n_gpt, ncells)` across groups."""
  ncells = molecules.size
  n_te = lookup.t_ref.shape[0] * lookup.n_mixing_fraction
  dtype = lookup.kminor_lower.dtype
  groups = band_groups(lookup)
  band_widths = {
      b: _band_gpt_range(lookup, b).size for g in groups for b in g.bands
  }
  sorted_bands = sorted(band_widths)
  band_order = np.concatenate(
      [_band_gpt_range(lookup, b) for b in sorted_bands]
  )

  molecules_flat = molecules.reshape(ncells)
  p_flat = interp.p.reshape(ncells)
  temp_flat = interp.temperature.reshape(ncells)
  is_lower_flat = p_flat > lookup.p_ref_tropo
  dry_flat = (
      dry_factor.reshape(ncells) if jnp.ndim(dry_factor) > 0 else dry_factor
  )
  vmr_flat = (
      None
      if interp.vmr_fields is None
      else {
          k: v.reshape(ncells) if jnp.ndim(v) > 0 else v
          for k, v in interp.vmr_fields.items()
      }
  )

  band_totals: dict[int, Array] = {}
  for g in groups:
    slot_specs = []
    slot_rows_t = []
    for b in g.bands:
      wb = band_widths[b]
      for is_lower, i in _minor_slots(lookup, b):
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
        vmr_minor = _get_vmr_static(lookup, interp.vmr_lib, gas, vmr_flat)
        scaling = vmr_minor * molecules_flat / _M2_TO_CM2_FACTOR
        gas_scaling = 1.0
        if scaling_gas > 0:
          scaling_vmr = _get_vmr_static(
              lookup, interp.vmr_lib, scaling_gas, vmr_flat
          )
          gas_scaling = (
              1.0 - scaling_vmr * dry_flat
              if by_complement == 1
              else scaling_vmr * dry_flat
          )
        if scales_with_density == 1:
          density_scaling = (
              _PASCAL_TO_HPASCAL_FACTOR * p_flat / temp_flat * gas_scaling
          )
          scaling = scaling * density_scaling
        rows_t = table[..., shift : shift + wb].reshape(n_te, wb).T
        slot_rows_t.append(rows_t)
        slot_specs.append((b, wb, is_lower, scaling))

    if not slot_specs:
      continue
    w_oh = interp.build_te_weights_mxu(g.bands[0], False, dtype)
    raw_all = jnp.matmul(
        jnp.concatenate(slot_rows_t, axis=0),
        w_oh,
        precision=jax.lax.Precision.HIGHEST,
    )
    off = 0
    for b, wb, is_lower, scaling in slot_specs:
      contrib = (
          optics_utils.floor_at_zero(raw_all[off : off + wb])
          * scaling[jnp.newaxis, :]
      )
      off += wb
      mask = (is_lower_flat if is_lower else ~is_lower_flat)[jnp.newaxis, :]
      term = jnp.where(mask, contrib, 0.0)
      band_totals[b] = term if b not in band_totals else band_totals[b] + term

  if not band_totals:
    return None
  stacked = jnp.concatenate(
      [
          band_totals[b]
          if b in band_totals
          else jnp.zeros((band_widths[b], ncells), dtype=dtype)
          for b in sorted_bands
      ],
      axis=0,
  )
  if not np.array_equal(band_order, np.arange(band_order.size)):
    stacked = stacked[np.argsort(band_order)]
  return stacked


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
  dry_factor = _dry_factor(lookup, vmr_fields)
  if jax.default_backend() == 'tpu' and temperature.ndim > 0 and temperature.size > 0:
    cell_shape = temperature.shape
    ncells = temperature.size
    mol_scale = (molecules.reshape(ncells) / _M2_TO_CM2_FACTOR)[jnp.newaxis, :]
    major_gn = mol_scale * optics_utils.floor_at_zero(
        _interpolate_groups_tpm_mxu(
            lookup,
            lookup.kmajor,
            interp.t_interp,
            interp.p_interp,
            interp,
            True,
        )
    )
    minor_gn = _minor_optical_depth_all_gpts_mxu(
        lookup, interp, molecules, dry_factor
    )
    total_gn = major_gn if minor_gn is None else minor_gn + major_gn
    return total_gn.reshape(total_gn.shape[0], *cell_shape)

  groups = band_groups(lookup)
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
                _group_columns(lookup, lookup.kmajor, group),
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
  if jax.default_backend() == 'tpu' and temperature.ndim > 0 and temperature.size > 0:
    pfrac_gn = _interpolate_groups_tpm_mxu(
        lookup,
        lookup.planck_fraction,
        t_clamped,
        p_clamped,
        interp,
        False,
    )
    return pfrac_gn.reshape(pfrac_gn.shape[0], *temperature.shape)

  groups = band_groups(lookup)
  pieces = []
  for group in groups:
    ibnd0 = group.bands[0]
    mix_for_t = {s: interp.mix(ibnd0, s, False) for s in _SIDES}
    pieces.append(
        _interpolate_rows_tpm(
            _group_columns(lookup, lookup.planck_fraction, group),
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
  if jax.default_backend() == 'tpu' and temperature.ndim > 0 and temperature.size > 0:
    cell_shape = temperature.shape
    ncells = temperature.size
    n_te = lkp.t_ref.shape[0] * lkp.n_mixing_fraction
    dtype = lkp.rayl_lower.dtype
    groups = band_groups(lkp)
    band_widths = {
        b: _band_gpt_range(lkp, b).size for g in groups for b in g.bands
    }
    sorted_bands = sorted(band_widths)
    band_order = np.concatenate(
        [_band_gpt_range(lkp, b) for b in sorted_bands]
    )
    scale_flat = (
        (factor * molecules).reshape(ncells) / _M2_TO_CM2_FACTOR
    )[jnp.newaxis, :]
    is_tropo0 = (interp.tropo_idx.reshape(ncells) == 0)[jnp.newaxis, :]
    by_band: dict[int, Array] = {}
    for g in groups:
      wg = g.gpts.size
      w_oh = interp.build_te_weights_mxu(g.bands[0], False, dtype)
      lo_t = _group_columns(lkp, lkp.rayl_lower, g).reshape(n_te, wg).T
      hi_t = _group_columns(lkp, lkp.rayl_upper, g).reshape(n_te, wg).T
      both = jnp.matmul(
          jnp.concatenate([lo_t, hi_t], axis=0),
          w_oh,
          precision=jax.lax.Precision.HIGHEST,
      )
      rayl_g = (
          optics_utils.floor_at_zero(
              jnp.where(is_tropo0, both[:wg], both[wg:])
          )
          * scale_flat
      )
      off = 0
      for b in g.bands:
        wb = band_widths[b]
        by_band[b] = rayl_g[off : off + wb]
        off += wb
    stacked = jnp.concatenate([by_band[b] for b in sorted_bands], axis=0)
    if not np.array_equal(band_order, np.arange(band_order.size)):
      stacked = stacked[np.argsort(band_order)]
    return stacked.reshape(stacked.shape[0], *cell_shape)

  groups = band_groups(lkp)
  pieces = []
  for group in groups:
    ibnd0 = group.bands[0]
    mix_for_t = {s: interp.mix(ibnd0, s, False) for s in _SIDES}
    # The lower and upper tables are stacked so a cell reads the half its
    # pressure selects (the per-g-point path reads both and selects).
    table = jnp.stack([
        _group_columns(lkp, lkp.rayl_lower, group),
        _group_columns(lkp, lkp.rayl_upper, group),
    ])
    rayl = optics_utils.floor_at_zero(
        _interpolate_rows_te(
            table, interp.t_interp, mix_for_t, leading_idx=interp.tropo_idx
        )
    )
    pieces.append((factor * molecules / _M2_TO_CM2_FACTOR)[..., None] * rayl)
  return _assemble_gpts(pieces, groups)


def compute_sw_optical_and_rayleigh_all_gpts_mxu(
    lkp: LookupGasOpticsShortwave,
    vmr_lib: LookupVolumeMixingRatio,
    molecules: Array,
    temperature: Array,
    p: Array,
    vmr_fields: dict[int, Array] | None = None,
    interp: CellInterpolation | None = None,
) -> tuple[Array, Array]:
  """Cell-minor MXU SW absorption + Rayleigh depths `(n_gpt, *cell_shape)`.

  Fuses the minor-absorber and Rayleigh `(t, eta)` contractions of each band
  group into a single MXU matmul against one transient `(126, ncells)` weight
  matrix so `w_oh` is neither recomputed nor retained across groups.
  """
  if interp is None:
    interp = CellInterpolation(lkp, vmr_lib, temperature, p, vmr_fields)
  cell_shape = temperature.shape
  ncells = temperature.size
  n_te = lkp.t_ref.shape[0] * lkp.n_mixing_fraction
  dtype = lkp.rayl_lower.dtype
  dry_factor = _dry_factor(lkp, vmr_fields)
  rayl_factor = (
      1.0 + vmr_fields[lkp.idx_h2o]
      if (vmr_fields is not None and lkp.idx_h2o in vmr_fields)
      else 1.0
  )

  mol_scale = (molecules.reshape(ncells) / _M2_TO_CM2_FACTOR)[jnp.newaxis, :]
  major_gn = mol_scale * optics_utils.floor_at_zero(
      _interpolate_groups_tpm_mxu(
          lkp, lkp.kmajor, interp.t_interp, interp.p_interp, interp, True
      )
  )

  groups = band_groups(lkp)
  band_widths = {
      b: _band_gpt_range(lkp, b).size for g in groups for b in g.bands
  }
  sorted_bands = sorted(band_widths)
  band_order = np.concatenate([_band_gpt_range(lkp, b) for b in sorted_bands])

  molecules_flat = molecules.reshape(ncells)
  p_flat = interp.p.reshape(ncells)
  temp_flat = interp.temperature.reshape(ncells)
  is_lower_flat = p_flat > lkp.p_ref_tropo
  is_tropo0 = (interp.tropo_idx.reshape(ncells) == 0)[jnp.newaxis, :]
  dry_flat = (
      dry_factor.reshape(ncells) if jnp.ndim(dry_factor) > 0 else dry_factor
  )
  rayl_scale_flat = (
      (rayl_factor * molecules).reshape(ncells) / _M2_TO_CM2_FACTOR
  )[jnp.newaxis, :]
  vmr_flat = (
      None
      if interp.vmr_fields is None
      else {
          k: v.reshape(ncells) if jnp.ndim(v) > 0 else v
          for k, v in interp.vmr_fields.items()
      }
  )

  minor_totals: dict[int, Array] = {}
  rayl_by_band: dict[int, Array] = {}
  for g in groups:
    wg = g.gpts.size
    slot_specs = []
    slot_rows_t = []
    for b in g.bands:
      wb = band_widths[b]
      for is_lower, i in _minor_slots(lkp, b):
        if is_lower:
          table = lkp.kminor_lower
          meta = (
              lkp.idx_minor_gases_lower,
              lkp.idx_scaling_gases_lower,
              lkp.minor_lower_scales_with_density,
              lkp.lower_scale_by_complement,
              lkp.minor_lower_gpt_shift,
          )
        else:
          table = lkp.kminor_upper
          meta = (
              lkp.idx_minor_gases_upper,
              lkp.idx_scaling_gases_upper,
              lkp.minor_upper_scales_with_density,
              lkp.upper_scale_by_complement,
              lkp.minor_upper_gpt_shift,
          )
        gas, scaling_gas, scales_with_density, by_complement, shift = (
            int(np.asarray(m)[i]) for m in meta
        )
        vmr_minor = _get_vmr_static(lkp, interp.vmr_lib, gas, vmr_flat)
        scaling = vmr_minor * molecules_flat / _M2_TO_CM2_FACTOR
        gas_scaling = 1.0
        if scaling_gas > 0:
          scaling_vmr = _get_vmr_static(
              lkp, interp.vmr_lib, scaling_gas, vmr_flat
          )
          gas_scaling = (
              1.0 - scaling_vmr * dry_flat
              if by_complement == 1
              else scaling_vmr * dry_flat
          )
        if scales_with_density == 1:
          density_scaling = (
              _PASCAL_TO_HPASCAL_FACTOR * p_flat / temp_flat * gas_scaling
          )
          scaling = scaling * density_scaling
        slot_rows_t.append(table[..., shift : shift + wb].reshape(n_te, wb).T)
        slot_specs.append((b, wb, is_lower, scaling))

    lo_t = _group_columns(lkp, lkp.rayl_lower, g).reshape(n_te, wg).T
    hi_t = _group_columns(lkp, lkp.rayl_upper, g).reshape(n_te, wg).T
    w_oh = interp.build_te_weights_mxu(g.bands[0], False, dtype)
    all_rows_t = jnp.concatenate([*slot_rows_t, lo_t, hi_t], axis=0)
    raw_all = jnp.matmul(all_rows_t, w_oh, precision=jax.lax.Precision.HIGHEST)

    off = 0
    for b, wb, is_lower, scaling in slot_specs:
      contrib = (
          optics_utils.floor_at_zero(raw_all[off : off + wb])
          * scaling[jnp.newaxis, :]
      )
      off += wb
      mask = (is_lower_flat if is_lower else ~is_lower_flat)[jnp.newaxis, :]
      term = jnp.where(mask, contrib, 0.0)
      minor_totals[b] = (
          term if b not in minor_totals else minor_totals[b] + term
      )

    both_rayl = raw_all[off:]
    rayl_g = (
        optics_utils.floor_at_zero(
            jnp.where(is_tropo0, both_rayl[:wg], both_rayl[wg:])
        )
        * rayl_scale_flat
    )
    roff = 0
    for b in g.bands:
      wb = band_widths[b]
      rayl_by_band[b] = rayl_g[roff : roff + wb]
      roff += wb

  rayl_gn = jnp.concatenate([rayl_by_band[b] for b in sorted_bands], axis=0)
  if minor_totals:
    minor_gn = jnp.concatenate(
        [
            minor_totals[b]
            if b in minor_totals
            else jnp.zeros((band_widths[b], ncells), dtype=dtype)
            for b in sorted_bands
        ],
        axis=0,
    )
    if not np.array_equal(band_order, np.arange(band_order.size)):
      idx_sort = np.argsort(band_order)
      minor_gn = minor_gn[idx_sort]
      rayl_gn = rayl_gn[idx_sort]
    tau_gn = minor_gn + major_gn
  else:
    if not np.array_equal(band_order, np.arange(band_order.size)):
      rayl_gn = rayl_gn[np.argsort(band_order)]
    tau_gn = major_gn

  return (
      tau_gn.reshape(tau_gn.shape[0], *cell_shape),
      rayl_gn.reshape(rayl_gn.shape[0], *cell_shape),
  )
