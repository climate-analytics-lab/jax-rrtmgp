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

import collections
from typing import TypeAlias

import unittest
from parameterized import parameterized
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from rrtmgp.optics import lookup_gas_optics_longwave
from rrtmgp.optics import optics_utils

IndexAndWeight: TypeAlias = optics_utils.IndexAndWeight
Interpolant: TypeAlias = optics_utils.Interpolant
OrderedDict: TypeAlias = collections.OrderedDict

_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g256.nc'

root = Path()
_LW_LOOKUP_TABLE_FILEPATH = root / _LW_LOOKUP_TABLE_FILENAME


def assert_interpolant_allclose(i1: Interpolant, i2: Interpolant):
  rtol = 1e-5
  atol = 0
  np.testing.assert_allclose(i1.interp_low.idx, i2.interp_low.idx, rtol, atol)
  np.testing.assert_allclose(
      i1.interp_low.weight, i2.interp_low.weight, rtol, atol
  )
  np.testing.assert_allclose(i1.interp_high.idx, i2.interp_high.idx, rtol, atol)
  np.testing.assert_allclose(
      i1.interp_high.weight, i2.interp_high.weight, rtol, atol
  )


class OpticsUtilsTest(unittest.TestCase):

  @parameterized.expand([
    # name, use_direct, coeffs, idxs, expected
    ("1D_1D", True,
     jnp.array([1,2,3,4,5,6,7,8,9], dtype=jnp.float_),
     ([8,7,6,5,4,3,2,1,0],),
     np.array([9,8,7,6,5,4,3,2,1])),
    ("1D_2D", True,
     jnp.array([1,2,3,4,5,6,7,8,9], dtype=jnp.float_),
     ([[8,7,6],[5,4,3],[2,1,0]],),
     np.array([[9,8,7],[6,5,4],[3,2,1]])),
    ("2D_1D", True,
     jnp.array([[1,2,3],[4,5,6],[7,8,9]], dtype=jnp.float_),
     ([2,2,2,1,1,1,0,0,0],[2,1,0,2,1,0,2,1,0]),
     np.array([9,8,7,6,5,4,3,2,1])),
    ("2D_2D", True,
     jnp.array([[1,2,3],[4,5,6],[7,8,9]], dtype=jnp.float_),
     ([[2,2,2],[1,1,1],[0,0,0]],[[2,1,0],[2,1,0],[2,1,0]]),
     np.array([[9,8,7],[6,5,4],[3,2,1]])),
    ("3D_3D", True,
     jnp.array([[[1,2,3],[4,5,6],[7,8,9]],
                [[10,20,30],[40,50,60],[70,80,90]]], dtype=jnp.float_),
     ([[0,1,0],[1,0,1],[0,1,0]],
      [[2,2,2],[1,1,1],[0,0,0]],
      [[2,1,0],[2,1,0],[2,1,0]]),
     np.array([[9,80,7],[60,5,40],[3,20,1]])),
    ("1D_1D_einsum", False,
     jnp.array([1,2,3,4,5,6,7,8,9], dtype=jnp.float_),
     ([8,7,6,5,4,3,2,1,0],),
     np.array([9,8,7,6,5,4,3,2,1]))
  ])
  
  def test_lookup_values(self, name, use_direct_indexing, coeffs, idxs, expected):
      lookup_fn = (
          optics_utils.lookup_values_direct_indexing
          if use_direct_indexing else optics_utils.lookup_values
      )
      result = lookup_fn(coeffs, idxs)
      np.testing.assert_equal(
          result, expected, f"{name} failed for " +
          ("direct" if use_direct_indexing else "einsum")
      )
      
  def test_evaluate_weighted_lookup(self):
    """Test whether the `weighted_lookup` yields the correct scaled values."""
    coeffs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    idx = jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    weight = jnp.array(
        [[1.0, 1 / 2, 1 / 3], [1 / 4, 1 / 5, 1 / 6], [1 / 7, 1 / 8, 1 / 9]]
    )
    result = optics_utils.evaluate_weighted_lookup(
        coeffs, [IndexAndWeight(idx, weight)]
    )
    expected = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    np.testing.assert_equal(result, expected)

  def test_floor_idx(self):
    """Tests whether `floor_idx` returns the floor index of given values."""
    ref_vals = jnp.array((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0))
    vals = jnp.array((1.0, 1.5, 5.6, 7.8, 9.0, 10.01))
    expected_floor_idx = np.array((0, 0, 4, 6, 8, 9))
    floor_idx = optics_utils.floor_idx(vals, ref_vals)
    np.testing.assert_equal(floor_idx, expected_floor_idx)

  def test_create_linear_interpolant(self):
    """Tests the creation of a linear interpolant and out of range exception."""
    ref_vals = jnp.array((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0))
    vals = jnp.array((1.0, 1.5, 5.6, 7.8, 9.0, 10.0))
    idx_low = jnp.array((0, 0, 4, 6, 8, 9))
    idx_high = jnp.array((1, 1, 5, 7, 9, 9))
    weight_low = jnp.array((1.0, 0.5, 0.4, 0.2, 1.0, 1.0))
    weight_high = jnp.array((0.0, 0.5, 0.6, 0.8, 0.0, 0.0))
    idx_and_weight_low = IndexAndWeight(idx_low, weight_low)
    idx_and_weight_high = IndexAndWeight(idx_high, weight_high)
    expected_interpolant = Interpolant(idx_and_weight_low, idx_and_weight_high)

    with self.subTest('ValuesWithinReferenceRange'):
      interpolant = optics_utils.create_linear_interpolant(vals, ref_vals)
      assert_interpolant_allclose(interpolant, expected_interpolant)

    with self.subTest('WithOffset'):
      offset = jnp.array((1, 0, 1, 0, 1, 0))
      interpolant = optics_utils.create_linear_interpolant(
          vals, ref_vals, offset
      )
      idx_and_weight_low_offset = IndexAndWeight(idx_low + offset, weight_low)
      idx_and_weight_high_offset = IndexAndWeight(
          idx_high + offset, weight_high
      )
      expected_interpolant_offset = Interpolant(
          idx_and_weight_low_offset, idx_and_weight_high_offset
      )
      assert_interpolant_allclose(interpolant, expected_interpolant_offset)

  @parameterized.expand([(True,), (False,)])
  def test_interpolate(self, use_optimized_interpolation: bool):
    """Tests `interpolate` on a given lookup array and list of interpolants."""
    # SETUP
    if use_optimized_interpolation:
      interpolate_fn = optics_utils.interpolate_optimized
    else:
      interpolate_fn = optics_utils.interpolate_orig

    coeffs = jnp.array([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
    ])
    idx1_low = [[0, 1], [0, 1]]
    idx1_low_weight = 0.2 * jnp.ones((2, 2), dtype=jnp.float_)

    idx1_high = [[1, 1], [1, 1]]
    idx1_high_weight = 0.8 * jnp.ones((2, 2), dtype=jnp.float_)

    idx2_low = [[1, 2], [0, 2]]
    idx_2_low_weight = 0.4 * jnp.ones((2, 2), dtype=jnp.float_)

    idx2_high = [[2, 2], [1, 2]]
    idx_2_high_weight = 0.6 * jnp.ones((2, 2), dtype=jnp.float_)

    idx3_low = [[1, 2], [2, 0]]
    idx_3_low_weight = 0.9 * jnp.ones((2, 2), dtype=jnp.float_)

    idx3_high = [[2, 2], [2, 1]]
    idx_3_high_weight = 0.1 * jnp.ones((2, 2), dtype=jnp.float_)

    element00 = (0.2 * 0.4 * 0.9 * 5.0 +  # low, low, low
                 0.8 * 0.4 * 0.9 * 50.0 +  # high, low, low
                 0.2 * 0.6 * 0.9 * 8.0 +  # low, high, low
                 0.2 * 0.4 * 0.1 * 6.0 +  # low, low, high
                 0.8 * 0.6 * 0.9 * 80.0 +  # high, high, low
                 0.8 * 0.4 * 0.1 * 60.0 +  # high, low, high
                 0.2 * 0.6 * 0.1 * 9.0 +  # low, high, high
                 0.8 * 0.6 * 0.1 * 90.0)  # high, high, high
    idx1_weight_low = IndexAndWeight(idx1_low, idx1_low_weight)
    idx1_weight_high = IndexAndWeight(idx1_high, idx1_high_weight)
    interpolant1 = Interpolant(idx1_weight_low, idx1_weight_high)

    idx2_weight_low = IndexAndWeight(idx2_low, idx_2_low_weight)
    idx2_weight_high = IndexAndWeight(idx2_high, idx_2_high_weight)
    interpolant2 = Interpolant(idx2_weight_low, idx2_weight_high)

    idx3_weight_low = IndexAndWeight(idx3_low, idx_3_low_weight)
    idx3_weight_high = IndexAndWeight(idx3_high, idx_3_high_weight)
    interpolant3 = Interpolant(idx3_weight_low, idx3_weight_high)

    interpolant_fns = OrderedDict((
        ('x', lambda: interpolant1),
        ('y', lambda: interpolant2),
        ('z', lambda: interpolant3),
    ))

    # ACTION
    interpolated_values = interpolate_fn(coeffs, interpolant_fns)

    # VERIFICATION
    self.assertEqual(interpolated_values.shape, (2, 2))
    self.assertEqual(np.float32(interpolated_values[0, 0]), np.float32(element00))

  @parameterized.expand([(True,), (False,)])
  def test_interpolate_with_dependency(self, use_optimized_interpolation: bool):
    """Tests `interpolate` on a given lookup tensor and list of interpolants."""
    # SETUP
    if use_optimized_interpolation:
      interpolate_fn = optics_utils.interpolate_optimized
    else:
      interpolate_fn = optics_utils.interpolate_orig

    coeffs = jnp.array([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
    ])
    idx1_low = [[0, 1], [0, 1]]
    idx1_low_weight = 0.2 * jnp.ones((2, 2), dtype=jnp.float_)

    idx1_high = [[1, 1], [1, 1]]
    idx1_high_weight = 0.8 * jnp.ones((2, 2), dtype=jnp.float_)

    idx2_low = [[1, 2], [0, 2]]
    idx_2_low_weight = 0.4 * jnp.ones((2, 2), dtype=jnp.float_)

    idx2_high = [[2, 2], [1, 2]]
    idx_2_high_weight = 0.6 * jnp.ones((2, 2), dtype=jnp.float_)

    idx3_low = [[1, 2], [2, 0]]
    idx_3_low_weight = 0.9 * jnp.ones((2, 2), dtype=jnp.float_)

    idx3_high = [[2, 2], [2, 1]]
    idx_3_high_weight = 0.1 * jnp.ones((2, 2), dtype=jnp.float_)

    idx1_weight_low = IndexAndWeight(idx1_low, idx1_low_weight)
    idx1_weight_high = IndexAndWeight(idx1_high, idx1_high_weight)
    interpolant1 = Interpolant(idx1_weight_low, idx1_weight_high)

    idx2_weight_low = IndexAndWeight(idx2_low, idx_2_low_weight)
    idx2_weight_high = IndexAndWeight(idx2_high, idx_2_high_weight)
    interpolant2 = Interpolant(idx2_weight_low, idx2_weight_high)

    # Create a dependent interpolant for the 3-rd axis that adds the `x` weights
    # to the lower index weight and adds the `y` weights to the upper index
    # weights.
    def interpolant_3_fn(x: IndexAndWeight, y: IndexAndWeight) -> Interpolant:
      idx3_weight_low = IndexAndWeight(
          idx3_low, x.weight + idx_3_low_weight
      )
      idx3_weight_high = IndexAndWeight(
          idx3_high, y.weight + idx_3_high_weight
      )
      return Interpolant(idx3_weight_low, idx3_weight_high)

    interpolant_fns = OrderedDict((
        ('x', lambda: interpolant1),
        ('y', lambda: interpolant2),
        ('z', interpolant_3_fn),
    ))

    # ACTION
    interpolated_values = interpolate_fn(coeffs, interpolant_fns)

    # VERIFICATION
    expected_element00 = (
        0.2 * 0.4 * (0.2 + 0.9) * 5.0 +  # low, low, low
        0.8 * 0.4 * (0.8 + 0.9) * 50.0 +  # high, low, low
        0.2 * 0.6 * (0.2 + 0.9) * 8.0 +  # low, high, low
        0.2 * 0.4 * (0.4 + 0.1) * 6.0 +  # low, low, high
        0.8 * 0.6 * (0.8 + 0.9) * 80.0 +  # high, high, low
        0.8 * 0.4 * (0.4 + 0.1) * 60.0 +  # high, low, high
        0.2 * 0.6 * (0.6 + 0.1) * 9.0 +  # low, high, high
        0.8 * 0.6 * (0.6 + 0.1) * 90.0)  # high, high, high
    self.assertEqual(interpolated_values.shape, (2, 2))
    self.assertAlmostEqual(
        interpolated_values[0, 0], expected_element00, delta=1e-4
    )

  @parameterized.expand([
      (True,  'IncreasingOrder'),
      (False, 'DecreasingOrder'),
  ])
  def test_recover_original_values_via_interpolation(
      self, use_optimized_interpolation, case_name
  ):
      gas_optics = lookup_gas_optics_longwave.from_nc_file(
          _LW_LOOKUP_TABLE_FILEPATH
      )
      t_ref = gas_optics.t_ref
      data_key = {'IncreasingOrder': t_ref,
                  'DecreasingOrder': jnp.flip(t_ref, 0)}[case_name]
      key = jax.random.PRNGKey(42)
      t = jax.random.uniform(
          key, t_ref.shape, minval=jnp.min(t_ref), maxval=jnp.max(t_ref)
      )
      interpolate_fn = (
          optics_utils.interpolate_optimized
          if use_optimized_interpolation else optics_utils.interpolate_orig
      )
      interpolant = optics_utils.create_linear_interpolant(t, data_key)
      interpolant_fns = OrderedDict({'x': lambda: interpolant})
      interpolated = interpolate_fn(data_key, interpolant_fns)
      np.testing.assert_allclose(
          interpolated, t, rtol=2e-7,
          err_msg=f"{case_name} failed for " +
                  ("optimized" if use_optimized_interpolation else "orig")
      )

  @parameterized.expand([(True,), (False,)])
  def test_exact_index_axis_matches_slicing_that_axis(
      self, use_optimized_interpolation
  ):
    """An exactly-indexed axis selects, rather than interpolating, an entry.

    This is what lets a caller evaluate a whole batch of table entries in one
    lookup instead of slicing the table once per entry, so the property that
    matters is that the batched result equals the per-entry one.
    """
    key = jax.random.split(jax.random.PRNGKey(7), 3)
    n_t, n_m, n_entry = 6, 5, 4
    t_ref = jnp.linspace(180.0, 320.0, n_t)
    m_ref = jnp.linspace(0.0, 1.0, n_m)
    table = jax.random.normal(key[0], (n_t, n_m, n_entry))
    field = (3, 2)
    t = jax.random.uniform(key[1], field, minval=185.0, maxval=315.0)
    m = jax.random.uniform(key[2], field, minval=0.05, maxval=0.95)

    interpolate_fn = (
        optics_utils.interpolate_optimized
        if use_optimized_interpolation else optics_utils.interpolate_orig
    )
    t_interp = optics_utils.create_linear_interpolant(t, t_ref)
    m_interp = optics_utils.create_linear_interpolant(m, m_ref)

    # One lookup over every entry: the entry index carries a leading batch
    # axis that broadcasts against the field's axes.
    entry = jnp.arange(n_entry).reshape((n_entry, 1, 1))
    batched = interpolate_fn(
        table,
        OrderedDict((
            ('t', lambda: t_interp),
            ('m', lambda: m_interp),
            ('e', lambda: optics_utils.exact_index(entry, table.dtype)),
        )),
    )
    self.assertEqual(batched.shape, (n_entry,) + field)

    # The same thing one entry at a time, with the entry axis sliced away.
    for i in range(n_entry):
      one = interpolate_fn(
          table[..., i],
          OrderedDict((('t', lambda: t_interp), ('m', lambda: m_interp))),
      )
      np.testing.assert_allclose(batched[i], one, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
  unittest.main()
