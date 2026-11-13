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

"""Abstract base class defining the interface of an optics scheme."""

import abc
from collections.abc import Mapping
from typing import TypeAlias

import jax
import jax.numpy as jnp
from rrtmgp import interpolation
from rrtmgp import kernel_ops

Array: TypeAlias = jax.Array


def _shift_up(f: Array) -> Array:
  """output_i = f_{i-1}."""
  return kernel_ops.shift_from_minus(f, 2)


def _shift_down(f: Array) -> Array:
  """output_i = f_{i+1}."""
  return kernel_ops.shift_from_plus(f, 2)


def reconstruct_face_values(f: Array, f_lower_bc: Array) -> tuple[Array, Array]:
  """Reconstruct the face values using a high-order scheme.

  This function performs an interpolation from nodes to faces.

  Args:
    f: The cell-center values that will be interpolated.
    f_lower_bc: The boundary condition for f on the lower face (wall).

  Returns:
    A tuple with the reconstructed temperature at the bottom and top face,
    respectively.
  """
  # f is f_ccc.
  # Use WENO5.

  if jax.default_backend() == 'tpu':
    # Set lower halo to the BC wall value and upper halo to Neumann BC via
    # concatenation rather than `.at[...].set(...)` so `vmap` never lowers halo
    # updates into indexed scatters on TPU.
    f = jnp.concatenate(
        [f_lower_bc[..., jnp.newaxis], f[:, :, 1:-1], f[:, :, -2:-1]],
        axis=2,
    )
  else:
    # Set halo value to the BC wall value.
    f = f.at[:, :, 0].set(f_lower_bc)

    # Enforce Neumann BC on the upper halo value.
    f = f.at[:, :, -1].set(f[:, :, -2])

  f_face_plus, f_face_minus = interpolation.weno5_node_to_face_for_rrtmgp(
      f, dim=2, f_lower_bc=f_lower_bc, neumann_upper_bc=True
  )
  # To get the final interpolation, just take the average of the plus and
  # minus reconstructions.
  f_bottom_ccf = 0.5 * (f_face_plus + f_face_minus)

  # Use centered interpolation: UNSTABLE.
  # f_bottom_ccf = interpolation.z_c_to_f(f)

  if jax.default_backend() == 'tpu':
    f_top = jnp.concatenate(
        [f_bottom_ccf[:, :, 1:], f_bottom_ccf[:, :, -1:]],
        axis=2,
    )
  else:
    f_top = _shift_down(f_bottom_ccf)
    # Update the f_top face (wall) value.
    # Look into what is the appropriate BC here.
    f_top = f_top.at[:, :, -1].set(f_top[:, :, -2])
  return f_bottom_ccf, f_top


class OpticsScheme(abc.ABC):
  """Abstract base class for optics scheme."""

  _EPSILON = 1e-6

  def __init__(
      self,
  ):
    self._halo_width = 1  # halos in z
    # self._face_interp_scheme_order = params.face_interp_scheme_order

    self.cloud_optics_lw = None
    self.cloud_optics_sw = None
    self.gas_optics_lw = None
    self.gas_optics_sw = None

  @abc.abstractmethod
  def compute_lw_optical_properties(
      self,
      pressure: Array,
      temperature: Array,
      molecules: Array,
      igpt: Array,
      vmr_fields: dict[int, Array] | None = None,
      cloud_r_eff_liq: Array | None = None,
      cloud_path_liq: Array | None = None,
      cloud_r_eff_ice: Array | None = None,
      cloud_path_ice: Array | None = None,
      cloud_tau_scale_liq: Array | float | None = None,
      cloud_tau_scale_ice: Array | float | None = None,
  ) -> dict[str, Array]:
    """Computes the monochromatic longwave optical properties.

    Args:
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules/m²].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
        [kg/m²].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path in each atmospheric grid cell
        [kg/m²].
      cloud_tau_scale_liq: Optional multiplier of the liquid cloud optical
        depth, applied after the table lookup; the single-scattering albedo
        and asymmetry factor keep the unscaled per-phase weighting.
      cloud_tau_scale_ice: Same as above, for the ice cloud optical depth.

    Returns:
      A dictionary containing (for a single g-point):
        'optical_depth': The longwave optical depth.
        'ssa': The longwave single-scattering albedo.
        'asymmetry_factor': The longwave asymmetry factor.
    """

  @abc.abstractmethod
  def compute_sw_optical_properties(
      self,
      pressure: Array,
      temperature: Array,
      molecules: Array,
      igpt: Array,
      vmr_fields: dict[int, Array] | None = None,
      cloud_r_eff_liq: Array | None = None,
      cloud_path_liq: Array | None = None,
      cloud_r_eff_ice: Array | None = None,
      cloud_path_ice: Array | None = None,
      cloud_tau_scale_liq: Array | float | None = None,
      cloud_tau_scale_ice: Array | float | None = None,
  ) -> dict[str, Array]:
    """Computes the monochromatic shortwave optical properties.

    Args:
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules/m²].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
        [kg/m²].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path in each atmospheric grid cell
        [kg/m²].
      cloud_tau_scale_liq: Optional multiplier of the liquid cloud optical
        depth, applied after the table lookup; the single-scattering albedo
        and asymmetry factor keep the unscaled per-phase weighting.
      cloud_tau_scale_ice: Same as above, for the ice cloud optical depth.

    Returns:
      A dictionary containing (for a single g-point):
        'optical_depth': The shortwave optical depth.
        'ssa': The shortwave single-scattering albedo.
        'asymmetry_factor': The shortwave asymmetry factor.
    """

  # The two methods below split `compute_{lw,sw}_optical_properties` into its
  # cloud-independent part (the gas optics, which dominates the cost of a
  # g-point) and the cloud mix-in, so a solver that needs both an all-sky and a
  # clear-sky answer can evaluate the gas optics once and form both optical
  # property sets from it (`two_stream.solve_{lw,sw}(with_clear_sky=True)`).
  # They are not abstract: a scheme that does not provide them keeps working
  # everywhere except under `with_clear_sky`.
  def compute_gas_optical_properties(
      self,
      pressure: Array,
      temperature: Array,
      molecules: Array,
      igpt: Array,
      is_lw: bool,
      vmr_fields: dict[int, Array] | None = None,
  ) -> dict[str, Array]:
    """Computes the cloud-free monochromatic optical properties of the gas.

    This is exactly `compute_{lw,sw}_optical_properties` called without any
    cloud condensate. The default implementation says just that, so it is
    correct for any scheme; a scheme overrides it only to expose the cloud
    mix-in separately through `add_cloud_optical_properties`.

    Args:
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules/m²].
      igpt: The spectral interval index, or g-point.
      is_lw: `True` for the longwave problem, `False` for the shortwave one.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.

    Returns:
      A dictionary with `'optical_depth'`, `'ssa'` and `'asymmetry_factor'`.
    """
    compute = (
        self.compute_lw_optical_properties
        if is_lw
        else self.compute_sw_optical_properties
    )
    return compute(pressure, temperature, molecules, igpt, vmr_fields)

  def add_cloud_optical_properties(
      self,
      igpt: Array,
      gas_optical_props: Mapping[str, Array],
      is_lw: bool,
      cloud_r_eff_liq: Array | None = None,
      cloud_path_liq: Array | None = None,
      cloud_r_eff_ice: Array | None = None,
      cloud_path_ice: Array | None = None,
      cloud_tau_scale_liq: Array | float | None = None,
      cloud_tau_scale_ice: Array | float | None = None,
  ) -> dict[str, Array]:
    """Mixes cloud optics into already-computed gas optical properties.

    Applied to the result of `compute_gas_optical_properties` it reproduces
    `compute_{lw,sw}_optical_properties` with the same cloud arguments.

    Args:
      igpt: The spectral interval index, or g-point.
      gas_optical_props: The output of `compute_gas_optical_properties`.
      is_lw: `True` for the longwave problem, `False` for the shortwave one.
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
        [kg/m²].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path in each atmospheric grid cell
        [kg/m²].
      cloud_tau_scale_liq: Optional multiplier of the liquid cloud optical
        depth; see `compute_lw_optical_properties`.
      cloud_tau_scale_ice: Same as above, for the ice cloud optical depth.

    Returns:
      A dictionary with `'optical_depth'`, `'ssa'` and `'asymmetry_factor'`.
    """
    raise NotImplementedError(
        f'{type(self).__name__} does not separate its cloud optics from its'
        ' gas optics, which `with_clear_sky=True` needs.'
    )

  @abc.abstractmethod
  def compute_planck_sources(
      self,
      pressure: Array,
      temperature: Array,
      igpt: Array,
      vmr_fields: dict[int, Array] | None = None,
      sfc_temperature: Array | None = None,
  ) -> dict[str, Array]:
    """Computes the Planck sources used in the longwave problem.

    Args:
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      sfc_temperature: An optional 2D plane for the surface temperature [K].

    Returns:
      A dictionary containing the Planck source at the cell center
      (`planck_src`), the top cell boundary (`planck_src_top`), and the bottom
      cell boundary (`planck_src_bottom`).
    """

  @property
  @abc.abstractmethod
  def n_gpt_lw(self) -> int:
    """The number of g-points in the longwave bands."""

  @property
  @abc.abstractmethod
  def n_gpt_sw(self) -> int:
    """The number of g-points in the shortwave bands."""

  @property
  @abc.abstractmethod
  def solar_fraction_by_gpt(self) -> Array:
    """Mapping from g-point to the fraction of total solar radiation."""

  def combine_optical_properties(
      self,
      optical_props_1: Mapping[str, Array],
      optical_props_2: Mapping[str, Array],
  ) -> dict[str, Array]:
    """Combines the optical properties from two separate parameterizations."""
    tau1 = optical_props_1['optical_depth']
    tau2 = optical_props_2['optical_depth']
    ssa1 = optical_props_1['ssa']
    ssa2 = optical_props_2['ssa']
    g1 = optical_props_1['asymmetry_factor']
    g2 = optical_props_2['asymmetry_factor']

    # Combine optical depths
    tau = tau1 + tau2

    # Combine single-scattering albedos.
    ssa_unnormalized = tau1 * ssa1 + tau2 * ssa2

    # Combine asymmetry factors.
    g = (tau1 * ssa1 * g1 + tau2 * ssa2 * g2) / jnp.maximum(
        ssa_unnormalized, self._EPSILON
    )

    return {
        'optical_depth': tau,
        'ssa': ssa_unnormalized / jnp.maximum(tau, self._EPSILON),
        'asymmetry_factor': g,
    }
