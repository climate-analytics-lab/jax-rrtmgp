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

"""Tests whether the shortwavewave optics data for atmospheric gases are loaded properly."""

import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from rrtmgp.optics import cloud_optics
from rrtmgp.optics import lookup_cloud_optics

_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/cloudysky_lw.nc'

root = Path()
_LW_LOOKUP_TABLE_FILEPATH = root / _LW_LOOKUP_TABLE_FILENAME


class CloudOpticsTest(unittest.TestCase):

  def test_compute_optical_properties(self):
    """Checks the cloud optics calculations of optical depth, `ssa`, and `g`."""
    cloud_optics_lw = lookup_cloud_optics.from_nc_file(
        _LW_LOOKUP_TABLE_FILEPATH
    )

    ones_2d = jnp.ones((2, 2), dtype=jnp.float_)
    # Fixed spectral band index.
    ibnd = 5
    # Roughness index for medium roughness.
    rgh_idx = 1
    # Pick effective droplet radius halfway between the reference values at
    # index 8 (10.5e-6 m) and index 9 (11.5e-6 m).
    radius_eff_liq = 11e-6 * ones_2d
    # Pick effective ice particle radius halfway between the reference values at
    # index 9 (100e-6 m) and index 10 (110e-6 m).
    radius_eff_ice = 105e-6 / 2 * ones_2d
    # Cloud liquid path for an atmospheric grid cell in kg/m².
    cld_path_liq = 6e-4 * ones_2d
    # Cloud ice path for an atmospheric grid cell.
    cld_path_ice = 1.2e-3 * ones_2d

    # Lookup tables.
    ext_liq = cloud_optics_lw.ext_liq
    ssa_liq = cloud_optics_lw.ssa_liq
    asy_liq = cloud_optics_lw.asy_liq
    ext_ice = cloud_optics_lw.ext_ice
    ssa_ice = cloud_optics_lw.ssa_ice
    asy_ice = cloud_optics_lw.asy_ice

    # Compute arithmetic mean of liquid lookup values at indices 8 and 9 of
    # effective radius dimension.
    interpolated_ext = (ext_liq[ibnd, 8] + ext_liq[ibnd, 9]) / 2.0
    interpolated_ssa = (ssa_liq[ibnd, 8] + ssa_liq[ibnd, 9]) / 2.0
    interpolated_g = (asy_liq[ibnd, 8] + asy_liq[ibnd, 9]) / 2.0

    # Expected values for liquid only. Use the cloud path in g//m² to scale the
    # table coefficients, which are in units of m²/g.
    expected_optical_depth_liq = (
        1000.0 * cld_path_liq * interpolated_ext * ones_2d
    )
    expected_ssa_liq = interpolated_ssa * ones_2d
    expected_g_liq = interpolated_g * ones_2d

    # Compute arithmetic mean of ice lookup values at indices 9 and 10 of
    # effective radius dimension.
    interpolated_ext = (
        ext_ice[rgh_idx, ibnd, 9] + ext_ice[rgh_idx, ibnd, 10]
    ) / 2.0
    interpolated_ssa = (
        ssa_ice[rgh_idx, ibnd, 9] + ssa_ice[rgh_idx, ibnd, 10]
    ) / 2.0
    interpolated_g = (
        asy_ice[rgh_idx, ibnd, 9] + asy_ice[rgh_idx, ibnd, 10]
    ) / 2.0

    # Expected values for ice only. Use the cloud path in g//m² to scale the
    # table coefficients, which are in units of m²/g.
    expected_optical_depth_ice = (
        1000.0 * cld_path_ice * interpolated_ext * ones_2d
    )
    expected_ssa_ice = interpolated_ssa * ones_2d
    expected_g_ice = interpolated_g * ones_2d

    # Cloud path that is too small to be considered. Note this is in SI units,
    # and only cloud paths that are greater than 1e-6 g/m² are accounted for.
    small_cld_path = 9.9e-10 * jnp.ones_like(cld_path_liq)

    rtol = 1e-5
    atol = 0

    with self.subTest('NonzeroLiquidCloudPath'):
      optical_props = cloud_optics.compute_optical_properties(
          cloud_optics_lw,
          cld_path_liq,
          small_cld_path,
          radius_eff_liq,
          radius_eff_ice,
          ibnd=ibnd,
      )

      np.testing.assert_allclose(
          expected_optical_depth_liq, optical_props['optical_depth'], rtol, atol
      )
      np.testing.assert_allclose(
          expected_ssa_liq, optical_props['ssa'], rtol, atol
      )
      np.testing.assert_allclose(
          expected_g_liq, optical_props['asymmetry_factor'], rtol, atol
      )

    with self.subTest('NonzeroIceCloudPath'):
      optical_props = cloud_optics.compute_optical_properties(
          cloud_optics_lw,
          small_cld_path,
          cld_path_ice,
          radius_eff_liq,
          radius_eff_ice,
          ibnd=ibnd,
      )

      np.testing.assert_allclose(
          expected_optical_depth_ice, optical_props['optical_depth'], rtol, atol
      )
      np.testing.assert_allclose(
          expected_ssa_ice, optical_props['ssa'], rtol, atol
      )
      np.testing.assert_allclose(
          expected_g_ice, optical_props['asymmetry_factor'], rtol, atol
      )

    with self.subTest('NonzeroTwoPhaseCloudPath'):
      optical_props = cloud_optics.compute_optical_properties(
          cloud_optics_lw,
          cld_path_liq,
          cld_path_ice,
          radius_eff_liq,
          radius_eff_ice,
          ibnd=ibnd,
      )

      expected_optical_depth = (
          expected_optical_depth_liq + expected_optical_depth_ice
      )
      weighted_ssa = (
          expected_optical_depth_liq * expected_ssa_liq
          + expected_optical_depth_ice * expected_ssa_ice
      )
      expected_ssa = weighted_ssa / expected_optical_depth
      expected_g = (
          expected_optical_depth_liq * expected_ssa_liq * expected_g_liq
          + expected_optical_depth_ice * expected_ssa_ice * expected_g_ice
      ) / weighted_ssa

      np.testing.assert_allclose(
          expected_optical_depth * ones_2d,
          optical_props['optical_depth'],
          rtol,
          atol,
      )
      np.testing.assert_allclose(
          expected_ssa * ones_2d, optical_props['ssa'], rtol, atol
      )
      np.testing.assert_allclose(
          expected_g * ones_2d, optical_props['asymmetry_factor'], rtol, atol
      )


class CloudOpticalDepthScalingTest(unittest.TestCase):
  """Per-phase optical-depth scaling for sub-grid cloud inhomogeneity (#37).

  ECHAM scales each phase's optical depth by its own inhomogeneity factor
  (`ztau = ztol*zinhoml + ztoi*zinhomi` in `mo_cloud_optics.f90`) and keeps the
  single-scattering albedo and asymmetry factor weighted by the physical,
  unscaled optical depths. Scaling the condensate paths instead only agrees
  with that when the two factors are equal.
  """

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.lookup = lookup_cloud_optics.from_nc_file(_LW_LOOKUP_TABLE_FILEPATH)

  def setUp(self):
    super().setUp()
    # Two-phase cloud with the phase mix varying across the cells, plus one
    # cloud-free cell, so the ssa/g weighting depends on the phase partition.
    self.path_liq = jnp.array([[6e-4, 1e-3], [0.0, 2e-4]], dtype=jnp.float_)
    self.path_ice = jnp.array([[1.2e-3, 1e-4], [0.0, 3e-3]], dtype=jnp.float_)
    self.r_liq = 11e-6 * jnp.ones_like(self.path_liq)
    self.r_ice = 52.5e-6 * jnp.ones_like(self.path_ice)
    self.ibnd = 5

  def _props(self, path_liq=None, path_ice=None, **kwargs):
    return cloud_optics.compute_optical_properties(
        self.lookup,
        self.path_liq if path_liq is None else path_liq,
        self.path_ice if path_ice is None else path_ice,
        self.r_liq,
        self.r_ice,
        ibnd=self.ibnd,
        **kwargs,
    )

  def test_unit_factors_are_bit_for_bit(self):
    """Explicit factors of one reproduce the default exactly."""
    default = self._props()
    unit = self._props(tau_scale_liq=1.0, tau_scale_ice=1.0)
    explicit_none = self._props(tau_scale_liq=None, tau_scale_ice=None)
    for key in ('optical_depth', 'ssa', 'asymmetry_factor'):
      np.testing.assert_array_equal(unit[key], default[key], err_msg=key)
      np.testing.assert_array_equal(explicit_none[key], default[key], key)

  def test_equal_factors_match_scaled_cloud_paths(self):
    """With one factor for both phases, scaling tau is scaling the paths."""
    factor = 0.7
    scaled_tau = self._props(tau_scale_liq=factor, tau_scale_ice=factor)
    scaled_path = self._props(
        path_liq=factor * self.path_liq, path_ice=factor * self.path_ice
    )
    for key in ('optical_depth', 'ssa', 'asymmetry_factor'):
      np.testing.assert_allclose(
          scaled_tau[key], scaled_path[key], rtol=1e-6, atol=0, err_msg=key
      )

  def test_unequal_factors_scale_tau_but_not_ssa_or_asymmetry(self):
    """ECHAM's rule: tau is the scaled sum, ssa and g keep physical weights."""
    f_liq, f_ice = 0.4, 0.85
    unscaled = self._props()
    scaled = self._props(tau_scale_liq=f_liq, tau_scale_ice=f_ice)
    zeros = jnp.zeros_like(self.path_liq)
    tau_liq = self._props(path_ice=zeros)['optical_depth']
    tau_ice = self._props(path_liq=zeros)['optical_depth']

    np.testing.assert_allclose(
        scaled['optical_depth'],
        f_liq * tau_liq + f_ice * tau_ice,
        rtol=1e-6,
        atol=0,
    )
    # The weighting is computed from exactly the same numbers as without
    # scaling, so these agree bit for bit.
    np.testing.assert_array_equal(scaled['ssa'], unscaled['ssa'])
    np.testing.assert_array_equal(
        scaled['asymmetry_factor'], unscaled['asymmetry_factor']
    )
    # Scaling the paths with the same unequal factors would have re-weighted
    # the ssa towards the ice, which is what this option exists to avoid.
    path_scaled = self._props(
        path_liq=f_liq * self.path_liq, path_ice=f_ice * self.path_ice
    )
    two_phase = (self.path_liq > 0) & (self.path_ice > 0)
    self.assertTrue(
        np.all(
            np.abs(path_scaled['ssa'] - unscaled['ssa'])[two_phase] > 1e-6
        )
    )

  def test_per_cell_and_single_phase_factors(self):
    """Factors broadcast per cell, and one phase may be scaled on its own."""
    f_ice = jnp.array([[0.5, 1.0], [0.25, 0.85]], dtype=jnp.float_)
    zeros = jnp.zeros_like(self.path_liq)
    tau_liq = self._props(path_ice=zeros)['optical_depth']
    tau_ice = self._props(path_liq=zeros)['optical_depth']

    scaled = self._props(tau_scale_ice=f_ice)
    np.testing.assert_allclose(
        scaled['optical_depth'], tau_liq + f_ice * tau_ice, rtol=1e-6, atol=0
    )
    np.testing.assert_array_equal(scaled['ssa'], self._props()['ssa'])

  def test_gradients_are_finite(self):
    """Reverse mode stays finite, including in the cloud-free cell."""

    def loss(path_liq, path_ice, f_liq, f_ice):
      props = self._props(
          path_liq=path_liq,
          path_ice=path_ice,
          tau_scale_liq=f_liq,
          tau_scale_ice=f_ice,
      )
      return sum(jnp.sum(v) for v in props.values())

    # A zero factor as well as a cloud-free cell: neither may poison the
    # cotangents.
    f_liq = jnp.array([[0.4, 0.0], [0.7, 1.0]], dtype=jnp.float_)
    f_ice = jnp.array([[0.85, 1.0], [0.0, 0.5]], dtype=jnp.float_)
    grads = jax.grad(loss, argnums=(0, 1, 2, 3))(
        self.path_liq, self.path_ice, f_liq, f_ice
    )
    for name, grad in zip(('path_liq', 'path_ice', 'f_liq', 'f_ice'), grads):
      self.assertTrue(
          np.all(np.isfinite(grad)), msg=f'd/d{name} is not finite: {grad}'
      )
    # The factors multiply the per-phase optical depth, so their gradient is
    # that optical depth wherever the phase is present.
    zeros = jnp.zeros_like(self.path_liq)
    np.testing.assert_allclose(
        grads[2], self._props(path_ice=zeros)['optical_depth'], rtol=1e-6
    )


if __name__ == '__main__':
  unittest.main()
