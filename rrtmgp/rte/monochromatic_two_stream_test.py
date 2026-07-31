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

import functools
from typing import TypeAlias

import unittest
from parameterized import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from rrtmgp import test_util
from rrtmgp.rte import monochromatic_two_stream

Array: TypeAlias = jax.Array


def _strip_top(f: Array) -> Array:
  """Remove the topmost plane."""
  return f[:, :, :-1]


def _strip_bottom(f: Array) -> Array:
  """Remove the bottommost plane."""
  return f[:, :, 1:]


def _remove_halos(f: Array) -> Array:
  """Remove the halos from the output."""
  return f[:, :, 1:-1]


class MonochromaticTwoStreamTest(unittest.TestCase):

  def test_lw_combine_sources(self):
    # SETUP
    n = 16
    dx = 0.1
    planck_src_top = 0.2 + dx * jnp.arange(n, dtype=jnp.float_)
    planck_src_bottom = 0.5 + dx * jnp.arange(n, dtype=jnp.float_)

    # Convert from 1D to 3D arrays.
    convert_to_3d = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=n
    )
    planck_src_top = convert_to_3d(planck_src_top)
    planck_src_bottom = convert_to_3d(planck_src_bottom)

    expected_src = jnp.sqrt(
        _strip_top(planck_src_top) * _strip_bottom(planck_src_bottom)
    )
    planck_srcs = {
        'planck_src_top': planck_src_top,
        'planck_src_bottom': planck_src_bottom,
    }

    # ACTION
    output = monochromatic_two_stream.lw_combine_sources(planck_srcs)

    # VERIFICATION
    # Remove the halos from the output.
    src_top_output = _strip_top(output['planck_src_top'])
    src_bottom_output = _strip_bottom(output['planck_src_bottom'])
    np.testing.assert_allclose(src_top_output, expected_src, rtol=1e-5, atol=0)
    np.testing.assert_allclose(
        src_bottom_output, expected_src, rtol=1e-5, atol=0
    )

  def test_lw_cell_source_and_properties(self):
    """Test the longwave sources are appropriately combined at cell faces."""
    # SETUP
    n = 4
    dx = 0.1
    planck_src_top = 0.2 + dx * jnp.arange(n, dtype=jnp.float_)

    # Convert from 1D to 3D arrays.
    convert_to_3d = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=n
    )
    planck_src_top = convert_to_3d(planck_src_top)

    # Shift up to obtain the bottom face.
    planck_src_bottom = planck_src_top - dx

    optical_depth = 0.03 * jnp.ones_like(planck_src_top, dtype=jnp.float_)
    ssa = 0.01 * jnp.ones_like(planck_src_top, dtype=jnp.float_)
    asymmetry_factor = 0.5 * jnp.ones_like(planck_src_top, dtype=jnp.float_)

    # ACTION
    output = monochromatic_two_stream.lw_cell_source_and_properties(
        optical_depth, ssa, planck_src_bottom, planck_src_top, asymmetry_factor
    )

    # VERIFICATION
    # Expected output evaluated in float128 from the analytic two-stream
    # solution (Meador & Weaver 1980 eqs 25-26 + the Toon et al. 1989
    # linear-in-tau sources). The original CliMA-derived values agreed only
    # to ~2e-5 (float32 noise of the cancelling formulation they were
    # generated with); these are exact to well beyond the tolerance.
    expected_t = 0.95177512 * jnp.ones_like(planck_src_top, dtype=jnp.float_)
    expected_r = 1.1854426e-4 * jnp.ones_like(planck_src_top, dtype=jnp.float_)
    expected_src_up = convert_to_3d(
        jnp.array([0.022731973, 0.037845025, 0.052958076, 0.068071128])
    )
    expected_src_down = convert_to_3d(
        jnp.array([0.022607182, 0.037720234, 0.052833285, 0.067946337])
    )

    np.testing.assert_allclose(output['t_diff'], expected_t, rtol=1e-5, atol=0)
    np.testing.assert_allclose(output['r_diff'], expected_r, rtol=1e-5, atol=0)
    np.testing.assert_allclose(
        output['src_up'], expected_src_up, rtol=1e-5, atol=0)
    np.testing.assert_allclose(
        output['src_down'], expected_src_down, rtol=1e-5, atol=0
    )

  def test_sw_cell_properties(self):
    """Check the shortwave transmittance and reflectance are correct."""
    n = 4
    # Inputs in the regime |1 - k_mu^2| > EPSILON.
    # SETUP
    optical_depth = 0.03 * jnp.ones((n, n, n), dtype=jnp.float_)
    asymmetry_factor = 0.5 * jnp.ones_like(optical_depth, dtype=jnp.float_)
    ssa = 0.01 * jnp.ones_like(optical_depth, dtype=jnp.float_)
    zenith = 1.57  # Zenith angle is slightly below pi/2.

    # ACTION
    output = monochromatic_two_stream.sw_cell_properties(
        zenith, optical_depth, ssa, asymmetry_factor
    )

    # VERIFICATION
    # Expected output obtained by running the original RRTMGP code on the input
    # above.
    expected_t_diff = 0.9422237 * jnp.ones_like(optical_depth, dtype=jnp.float_)
    expected_r_diff = 1.06062755e-4 * jnp.ones_like(optical_depth, dtype=jnp.float_)
    expected_t_dir = 4.7214041e-3 * jnp.ones_like(optical_depth, dtype=jnp.float_)
    expected_r_dir = 4.9896492e-3 * jnp.ones_like(optical_depth, dtype=jnp.float_)

    np.testing.assert_allclose(
        output['t_diff'], expected_t_diff, rtol=1e-5, atol=0
    )
    np.testing.assert_allclose(
        output['r_diff'], expected_r_diff, rtol=1e-5, atol=0
    )
    np.testing.assert_allclose(
        output['t_dir'], expected_t_dir, rtol=1e-5, atol=0
    )
    np.testing.assert_allclose(
        output['r_dir'], expected_r_dir, rtol=1e-5, atol=0
    )

    # Inputs in the regime |1 - k_mu^2| < EPSILON.
    # SETUP
    asymmetry_factor = 0.16116116 * jnp.ones_like(optical_depth, dtype=jnp.float_)
    ssa = 0.2722723 * jnp.ones_like(optical_depth, dtype=jnp.float_)
    zenith = 0.90439791

    # ACTION
    output = monochromatic_two_stream.sw_cell_properties(
        zenith, optical_depth, ssa, asymmetry_factor
    )

    # VERIFICATION
    # Expected diffuse values obtained by running the original RRTMGP code on
    # the input above. The direct reflectance is instead the float128 limit of
    # the analytic expression: this input sits at the removable singularity
    # k*mu0 = 1 (|1 - k^2 mu0^2| ~ 8.5e-7), where the original code's
    # fixed-epsilon denominator clamp produced 0.0012536654 -- an artifact
    # ~4x below the true value. The reformulated solver evaluates the limit
    # exactly, so the reference is the true value.
    expected_t_diff = 0.9523814 * jnp.ones_like(optical_depth, dtype=jnp.float_)
    expected_r_diff = 0.0048960485 * jnp.ones_like(optical_depth, dtype=jnp.float_)
    expected_r_dir = 0.0053730679 * jnp.ones_like(optical_depth, dtype=jnp.float_)

    np.testing.assert_allclose(
        output['t_diff'], expected_t_diff, rtol=1e-5, atol=0
    )
    np.testing.assert_allclose(
        output['r_diff'], expected_r_diff, rtol=1e-5, atol=0
    )
    # Note: the result in `t_dir` cannot be properly checked because it involves
    # subtraction of two quantities that are almost identical; there is no way
    # to calculate it accurately in single precision as every digit of precision
    # is lost.
    np.testing.assert_allclose(
        output['r_dir'], expected_r_dir, rtol=1e-5, atol=0
    )

  @parameterized.expand([(True,), (False,)])
  def test_sw_cell_source(self, use_scan: bool):
    # SETUP
    n = 8
    nz = 18  # 16 layers plus halo_width of 1.

    # Constant cell properties
    t_dir = 0.97 * jnp.ones((n, n, nz), dtype=jnp.float_)
    r_dir = 0.03 * jnp.ones_like(t_dir, dtype=jnp.float_)
    optical_depth = 1.5e-3 * jnp.ones_like(t_dir, dtype=jnp.float_)

    # Incident flux at the top of the atmosphere.
    toa_flux_down = 0.8 * jnp.ones((n, n), dtype=jnp.float_)

    # Albedo of direct radiation at the surface
    sfc_albedo_direct = 0.2 * jnp.ones((n, n), dtype=jnp.float_)

    zenith = 2.0  # Zenith angle in radians.

    # ACTION
    output = monochromatic_two_stream.sw_cell_source(
        t_dir,
        r_dir,
        optical_depth,
        toa_flux_down,
        sfc_albedo_direct,
        zenith,
        use_scan,
    )

    # VERIFICATION
    # pyformat: disable
    # Expected output obtained by running the original RRTMGP code on the input
    # above.
    expected_sw_src_up = jnp.array([
        -0.010542383, -0.0105044525, -0.010466657, -0.010428998, -0.010391475,
        -0.010354087, -0.010316832, -0.010279712, -0.010242727, -0.010205874,
        -0.010169154, -0.010132565, -0.010096109, -0.010059784, -0.010023589,
        -0.009987524,
    ])
    expected_sw_src_down = jnp.array([
        -0.3408704, -0.33964396, -0.33842194, -0.33720428, -0.33599102,
        -0.33478215, -0.3335776, -0.33237737, -0.3311815, -0.32998994,
        -0.32880265, -0.32761964, -0.32644087, -0.32526636, -0.32409605,
        -0.32292998,
    ])
    expected_sw_flux_down_direct = jnp.array([
        -0.35268173, -0.35141277, -0.3501484, -0.34888858, -0.34763327,
        -0.3463825, -0.34513623, -0.34389442, -0.3426571, -0.34142423,
        -0.3401958, -0.3389718, -0.3377522, -0.33653697, -0.33532614,
        -0.33411965,
    ])
    expected_sfc_src = 0.2 * expected_sw_flux_down_direct[0]
    # pyformat: enable
    convert_to_3d = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=n
    )
    expected_sw_src_up = convert_to_3d(expected_sw_src_up)
    expected_sw_src_down = convert_to_3d(expected_sw_src_down)
    expected_sw_flux_down_direct = convert_to_3d(expected_sw_flux_down_direct)
    expected_sfc_src = expected_sfc_src * jnp.ones((n, n))

    # Remove halos from the output.
    sw_src_up = _remove_halos(output['src_up'])
    sw_src_down = _remove_halos(output['src_down'])
    sw_flux_down_direct = _remove_halos(output['flux_down_dir'])

    np.testing.assert_allclose(sw_src_up, expected_sw_src_up, rtol=1e-5, atol=0)
    np.testing.assert_allclose(
        sw_src_down, expected_sw_src_down, rtol=1e-5, atol=0
    )
    np.testing.assert_allclose(
        sw_flux_down_direct, expected_sw_flux_down_direct, rtol=1e-5, atol=0
    )

    # Check the surface source.
    np.testing.assert_allclose(
        output['sfc_src'], expected_sfc_src, rtol=1e-5, atol=0
    )

  @parameterized.expand([(True,), (False,)])
  def test_lw_transport(self, use_scan: bool):
    """Check the longwave radiative transfer equation is solved correctly."""
    # SETUP
    n = 8
    nz = 18  # 16 layers plus halo_width of 1.

    # Create vertically variable upward and downward emission sources. The zeros
    # are values in the halos.
    # pyformat: disable
    src_up = jnp.array([
        0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 0,
    ])
    # pyformat: enable
    convert_to_3d = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=n
    )
    src_up = convert_to_3d(src_up)
    src_down = src_up + 0.05

    # Constant cell properties
    t_diff = 0.97 * jnp.ones((n, n, nz))
    r_diff = 0.03 * jnp.ones_like(t_diff)

    # Boundary conditions.
    toa_flux_down = 0.0 * jnp.ones((n, n))
    # Surface source.
    sfc_src = 0.02 * jnp.ones((n, n))
    # Surface emissivity.
    sfc_emiss = 0.8 * jnp.ones((n, n))

    # ACTION
    output = monochromatic_two_stream.lw_transport(
        t_diff,
        r_diff,
        src_up,
        src_down,
        toa_flux_down,
        sfc_src,
        sfc_emiss,
        use_scan,
    )

    # VERIFICATION
    # pyformat: disable
    # Expected fluxes from running the original RRTMGP code.
    expected_lw_flux_up = jnp.array([
        3.152796, 3.727328, 4.384849, 5.119175, 5.924118, 6.793495, 7.721119,
        8.700806, 9.726369, 10.791622, 11.89038, 13.016458, 14.16367, 15.32583,
        16.496754, 17.670252,
    ])
    expected_lw_flux_down = jnp.array([
        15.512652, 15.637185, 15.644706, 15.529032, 15.283975, 14.90335,
        14.380974, 13.710661, 12.886224, 11.901477, 10.750236, 9.426313,
        7.923525, 6.2356844, 4.356607, 2.2801077,
    ])
    # pyformat: enable
    expected_lw_flux_up = convert_to_3d(expected_lw_flux_up)
    expected_lw_flux_down = convert_to_3d(expected_lw_flux_down)
    expected_lw_flux_net = expected_lw_flux_up - expected_lw_flux_down

    lw_flux_up = _remove_halos(output['flux_up'])
    lw_flux_down = _remove_halos(output['flux_down'])
    lw_flux_net = _remove_halos(output['flux_net'])

    np.testing.assert_allclose(
        lw_flux_up, expected_lw_flux_up, rtol=1e-5, atol=0
    )
    np.testing.assert_allclose(
        lw_flux_down, expected_lw_flux_down, rtol=1e-5, atol=0
    )
    np.testing.assert_allclose(
        lw_flux_net, expected_lw_flux_net, rtol=1e-5, atol=0
    )

  @parameterized.expand([(True,), (False,)])
  def test_sw_transport(self, use_scan: bool):
    """Check the shortwave radiative transfer equation is solved correctly."""
    # SETUP
    n = 8
    nz = 18  # 16 layers plus halo_width of 1.

    # Sources and direct-beam flux.
    # Create vertically variable upward and downward emission sources. The zeros
    # are values in the halos.
    # pyformat: disable
    src_up = jnp.array([
        0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 0,
    ])
    # pyformat: enable
    convert_to_3d = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=n
    )
    src_up = convert_to_3d(src_up)
    src_down = src_up + 0.05
    flux_down_dir = 0.8 * jnp.ones_like(src_up, dtype=jnp.float_)

    # Constant cell properties
    t_diff = 0.97 * jnp.ones((n, n, nz), dtype=jnp.float_)
    r_diff = 0.03 * jnp.ones_like(t_diff, dtype=jnp.float_)

    # Boundary conditions: surface source and albedo.
    sfc_src = 0.02 * jnp.ones((n, n), dtype=jnp.float_)
    sfc_albedo = 0.5 * jnp.ones((n, n), dtype=jnp.float_)

    # ACTION
    output = monochromatic_two_stream.sw_transport(
        t_diff,
        r_diff,
        src_up,
        src_down,
        sfc_src,
        sfc_albedo,
        flux_down_dir,
        use_scan,
    )

    # VERIFICATION
    # pyformat: disable
    # Expected fluxes from running the original RRTMGP code.
    expected_sw_flux_up = jnp.array([
        8.693392, 9.15329, 9.6961775, 10.31587, 11.006181, 11.760924,
        12.5739155, 13.438969, 14.349897, 15.300518, 16.284643, 17.296087,
        18.328667, 19.376192, 20.432482, 21.49135,  # 22.54661,
    ])
    expected_sw_flux_down = jnp.array([
        18.146782, 18.156681, 18.049568, 17.81926, 17.45957, 16.964314,
        16.327305, 15.542358, 14.603287, 13.503907, 12.238033, 10.799479,
        9.182056, 7.3795834, 5.3858733, 3.1947403,  # 0.8
    ])
    # pyformat: enable
    expected_sw_flux_up = convert_to_3d(expected_sw_flux_up)
    expected_sw_flux_down = convert_to_3d(expected_sw_flux_down)
    expected_sw_flux_net = expected_sw_flux_up - expected_sw_flux_down

    sw_flux_up = _remove_halos(output['flux_up'])
    sw_flux_down = _remove_halos(output['flux_down'])
    sw_flux_net = _remove_halos(output['flux_net'])

    np.testing.assert_allclose(
        sw_flux_up, expected_sw_flux_up, rtol=1e-5, atol=0
    )
    np.testing.assert_allclose(
        sw_flux_down, expected_sw_flux_down, rtol=1e-5, atol=0
    )
    np.testing.assert_allclose(
        sw_flux_net, expected_sw_flux_net, rtol=1.2e-5, atol=0
    )


if __name__ == '__main__':
  unittest.main()
