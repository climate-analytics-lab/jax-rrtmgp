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

"""Utility functions for computing the optical properties of clouds."""

import collections
from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
from rrtmgp import smooth_ops
from rrtmgp.optics import lookup_cloud_optics
from rrtmgp.optics import optics_utils

Array: TypeAlias = jax.Array
Interpolant: TypeAlias = optics_utils.Interpolant
IndexAndWeight: TypeAlias = optics_utils.IndexAndWeight

_EPSILON = 1e-6
_KG_TO_G_FACTOR = 1e3
_M_TO_MICRONS_FACTOR = 1e6


class PhaseProperties(NamedTuple):
  """The Mie properties of one condensate phase, read from the tables."""

  # Extinction coefficient [m^2/g].
  ext: Array
  # Single-scattering albedo.
  ssa: Array
  # Asymmetry factor.
  asy: Array


def _particle_sizes(
    lookup: lookup_cloud_optics.LookupCloudOptics,
    radius_eff_liq: Array,
    radius_eff_ice: Array,
) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
  """The particle sizes the tables are indexed by, and the tables' size nodes.

  The tables are indexed by the droplet radius and the ice-particle diameter,
  in microns. The sizes are clamped to the table range, so a particle size
  outside the table gets the optical properties of the nearest end of it:
  RRTMGP rejects such sizes outright (`check_values` in
  `mo_cloud_optics_rrtmgp.F90`), which a jitted function cannot do, and
  extrapolating the Mie tables could push the single-scattering albedo or
  asymmetry factor outside [0, 1]. The clamp also gives the cloud-free cells,
  whose radius is typically zero, a valid size; their contribution is
  eliminated by the cloud mask in `combine_phases`.

  Returns:
    `(size, nodes)` for the liquid and for the ice phase.
  """
  size_liq = jnp.clip(
      _M_TO_MICRONS_FACTOR * radius_eff_liq,
      lookup.radius_liq_lower,
      lookup.radius_liq_upper,
  )
  size_ice = jnp.clip(
      _M_TO_MICRONS_FACTOR * 2.0 * radius_eff_ice,
      lookup.diameter_ice_lower,
      lookup.diameter_ice_upper,
  )
  nodes_liq = jnp.linspace(
      lookup.radius_liq_lower, lookup.radius_liq_upper, lookup.n_size_liq
  )
  nodes_ice = jnp.linspace(
      lookup.diameter_ice_lower, lookup.diameter_ice_upper, lookup.n_size_ice
  )
  return (size_liq, nodes_liq), (size_ice, nodes_ice)


def phase_properties(
    lookup: lookup_cloud_optics.LookupCloudOptics,
    radius_eff_liq: Array,
    radius_eff_ice: Array,
    ibnd: Array,
) -> tuple[PhaseProperties, PhaseProperties]:
  """The liquid and ice Mie properties of band `ibnd` at each cell's size.

  Args:
    lookup: The cloud-optics lookup tables.
    radius_eff_liq: The effective radius of cloud droplets [m].
    radius_eff_ice: The effective radius of cloud ice particles [m].
    ibnd: The spectral band index.

  Returns:
    The liquid and the ice `PhaseProperties`, each of the cells' shape.
  """
  sizes = _particle_sizes(lookup, radius_eff_liq, radius_eff_ice)
  roughness = lookup.ice_roughness.value
  tables = (
      (lookup.ext_liq[ibnd, :], lookup.ssa_liq[ibnd, :],
       lookup.asy_liq[ibnd, :]),
      (lookup.ext_ice[roughness, ibnd, :], lookup.ssa_ice[roughness, ibnd, :],
       lookup.asy_ice[roughness, ibnd, :]),
  )
  phases = []
  for (size, nodes), phase_tables in zip(sizes, tables):
    interp = optics_utils.create_linear_interpolant(size, nodes)
    fns = collections.OrderedDict({'r': lambda interp=interp: interp})
    phases.append(
        PhaseProperties(
            *(optics_utils.interpolate(t, fns) for t in phase_tables)
        )
    )
  return phases[0], phases[1]


def phase_properties_all_bands(
    lookup: lookup_cloud_optics.LookupCloudOptics,
    radius_eff_liq: Array | None,
    radius_eff_ice: Array | None,
) -> tuple[PhaseProperties | None, PhaseProperties | None]:
  """`phase_properties` of every band at once.

  The Mie tables of every band are interpolated in one pass per phase
  (`optics_utils.interpolate_rows_1d`), so a spectral loop that needs each
  band's properties reads them from the result instead of looking them up.

  Args:
    lookup: The cloud-optics lookup tables.
    radius_eff_liq: The effective radius of cloud droplets [m], or `None` to
      skip the liquid phase.
    radius_eff_ice: The effective radius of cloud ice particles [m], or `None`
      to skip the ice phase.

  Returns:
    The liquid and the ice `PhaseProperties` (or `None` where skipped), each
    field `(n_bnd, *cell_shape)`.
  """
  shape_ref = radius_eff_liq if radius_eff_liq is not None else radius_eff_ice
  if shape_ref is None:
    return None, None
  zeros = jnp.zeros_like(shape_ref)
  sizes = _particle_sizes(
      lookup,
      zeros if radius_eff_liq is None else radius_eff_liq,
      zeros if radius_eff_ice is None else radius_eff_ice,
  )
  roughness = lookup.ice_roughness.value
  tables = (
      (lookup.ext_liq, lookup.ssa_liq, lookup.asy_liq),
      (lookup.ext_ice[roughness], lookup.ssa_ice[roughness],
       lookup.asy_ice[roughness]),
  )
  phases = []
  for radius, (size, nodes), phase_tables in zip(
      (radius_eff_liq, radius_eff_ice), sizes, tables
  ):
    if radius is None:
      phases.append(None)
      continue
    n_bnd = phase_tables[0].shape[0]
    rows = optics_utils.interpolate_rows_1d(
        jnp.concatenate(phase_tables, axis=0),
        optics_utils.create_linear_interpolant(
            size, nodes.astype(phase_tables[0].dtype)
        ),
    )
    phases.append(
        PhaseProperties(rows[:n_bnd], rows[n_bnd : 2 * n_bnd], rows[2 * n_bnd :])
    )
  return phases[0], phases[1]


def combine_phases(
    liq: PhaseProperties | None,
    ice: PhaseProperties | None,
    cloud_path_liq: Array | None,
    cloud_path_ice: Array | None,
    tau_scale_liq: Array | float | None = None,
    tau_scale_ice: Array | float | None = None,
) -> dict[str, Array]:
  """The cloud optical properties of the liquid and ice phases together.

  The optical depth is the sum of the two phases' (each the extinction times
  the condensate path); the single-scattering albedo is the optical-depth
  weighted mean of the phases'; the asymmetry factor is the
  scattering-optical-depth weighted mean.

  `tau_scale_liq` and `tau_scale_ice` scale the optical depth of each phase
  after the table lookup, e.g. by a sub-grid inhomogeneity factor. As in
  ECHAM's `mo_cloud_optics.f90` (`ztau = ztol*zinhoml + ztoi*zinhomi`), only
  the optical depth is scaled: the ssa and asymmetry factor are still weighted
  by the physical, unscaled optical depth of each phase. Scaling the condensate
  paths instead is equivalent only when the two factors are equal; with
  unequal factors it would also re-weight the combined ssa and asymmetry
  factor towards the less-reduced phase.

  Args:
    liq: The liquid phase's Mie properties, or `None` for no liquid.
    ice: The ice phase's Mie properties, or `None` for no ice.
    cloud_path_liq: The cloud liquid water path [kg/m²], broadcastable against
      the properties; `None` when `liq` is.
    cloud_path_ice: Same as above, for the ice water path.
    tau_scale_liq: Optional multiplier of the liquid cloud optical depth, a
      scalar or an array broadcastable against the cloud path. When omitted
      the optical depth is left unscaled, bit for bit.
    tau_scale_ice: Same as above, for the ice cloud optical depth.

  Returns:
    'optical_depth': The combined optical depth of the condensate.
    'ssa': The combined single-scattering albedo.
    'asymmetry_factor': The combined asymmetry factor.
  """
  per_phase = []
  for phase, path in ((liq, cloud_path_liq), (ice, cloud_path_ice)):
    if phase is None:
      per_phase.append(None)
      continue
    # Convert the cloud path to g/m² to conform to the lookup tables.
    path_g = path * _KG_TO_G_FACTOR
    # Presence mask that zeroes the contribution of a cell with a negligible
    # cloud path. The hard step is replaced by a narrow smoothstep ramp so a
    # cloud parameter driven toward zero during calibration turns the cloud off
    # continuously instead of discontinuously. The ramp saturates to 1 far
    # below any physical cloud path (at ``2 * _EPSILON`` g/m^2), so a real
    # cloud is unaffected.
    mask = smooth_ops.smooth_gate(path_g, 0.0, 2.0 * _EPSILON)
    tau = phase.ext * path_g * mask
    # Weight the ssa by the optical depth, and the asymmetry factor by the
    # scattering optical depth.
    tau_ssa = phase.ssa * tau
    tau_ssa_g = phase.asy * tau_ssa
    per_phase.append((tau, tau_ssa, tau_ssa_g))
  present = [p for p in per_phase if p is not None]
  if not present:
    raise ValueError('combine_phases needs at least one phase.')
  tau, tau_ssa, tau_ssa_g = (
      present[0] if len(present) == 1 else jax.tree.map(jnp.add, *present)
  )

  # The single-scattering albedo and asymmetry factor un-weight the optical
  # depth (and the ssa-weighted optical depth). A cloud-free cell -- or any cell
  # whose cloud parameter is driven to zero during calibration -- makes those
  # denominators vanish. The `jnp.where` already returns 0 there in the forward
  # pass, but the unselected ``a / 0`` branch would still leave a NaN cotangent
  # in reverse mode, so the division is fed a safe (unit) denominator on the
  # masked branch. Forward values are unchanged.
  tau_is_nonzero = tau != 0
  tau_ssa_is_nonzero = tau_ssa != 0
  safe_tau = jnp.where(tau_is_nonzero, tau, 1.0)
  safe_tau_ssa = jnp.where(tau_ssa_is_nonzero, tau_ssa, 1.0)

  # Per-phase optical-depth scaling. The weights above stay the physical
  # per-phase optical depths, so only the extinction is scaled; this adds no
  # division, and hence no new reverse-mode hazard. Without either factor the
  # optical depth is the unscaled sum, bit for bit.
  optical_depth = tau
  if tau_scale_liq is not None or tau_scale_ice is not None:
    scaled = []
    for p, scale in zip(per_phase, (tau_scale_liq, tau_scale_ice)):
      if p is not None:
        scaled.append(p[0] if scale is None else scale * p[0])
    optical_depth = scaled[0] if len(scaled) == 1 else scaled[0] + scaled[1]

  return {
      'optical_depth': optical_depth,
      'ssa': jnp.where(tau_is_nonzero, tau_ssa / safe_tau, 0.0),
      'asymmetry_factor': jnp.where(
          tau_ssa_is_nonzero, tau_ssa_g / safe_tau_ssa, 0.0
      ),
  }


def compute_optical_properties(
    lookup: lookup_cloud_optics.LookupCloudOptics,
    cloud_path_liq: Array,
    cloud_path_ice: Array,
    radius_eff_liq: Array,
    radius_eff_ice: Array,
    ibnd: Array,
    tau_scale_liq: Array | float | None = None,
    tau_scale_ice: Array | float | None = None,
) -> dict[str, Array]:
  """Computes the optical properties of clouds from lookup tables.

  The table lookup of `phase_properties` and the combination of the two
  phases of `combine_phases`, which documents the physics (including the
  per-phase optical-depth scaling).

  Args:
    lookup: A `LookupCloudOptics` instance containing lookup tables for the
      extinction coefficient, the single-scattering albedo, and the asymmetry
      factor of condensates indexed by spectral band and effective radius of
      cloud particles.
    cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
      [kg/m²].
    cloud_path_ice: The cloud ice water path in each atmospheric grid cell
      [kg/m²].
    radius_eff_liq: The effective radius of cloud droplets in each atmospheric
      grid cell [m].
    radius_eff_ice: The effective radius of cloud ice particles in each
      atmospheric grid cell [m].
    ibnd: The spectral band index.
    tau_scale_liq: Optional multiplier of the liquid cloud optical depth, a
      scalar or an array broadcastable against the cloud path. When omitted
      the optical depth is left unscaled, bit for bit.
    tau_scale_ice: Same as above, for the ice cloud optical depth.

  Returns:
  A dictionary containing:
    'optical_depth': The combined optical depth contribution from condensates.
    'ssa': The combined single-scattering albedo contribution from condensates.
    'asymmetry_factor': The combined asymmetry factor contribution from
      condensates.
  """
  liq, ice = phase_properties(lookup, radius_eff_liq, radius_eff_ice, ibnd)
  return combine_phases(
      liq, ice, cloud_path_liq, cloud_path_ice, tau_scale_liq, tau_scale_ice
  )
