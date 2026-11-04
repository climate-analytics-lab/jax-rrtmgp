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

from typing import TypeAlias

import unittest
from parameterized import parameterized
from itertools import product
import jax
import jax.numpy as jnp
import numpy as np
from rrtmgp.rte import rte_utils

Array: TypeAlias = jax.Array


class RteUtilsTest(unittest.TestCase):

  @parameterized.expand([
    (fwd, use_scan)
    for fwd, use_scan in product([True, False], [True, False])
  ])
  def test_recurrent_op_1d(self, forward: bool, use_scan: bool):
    # SETUP
    n = 8
    dx = 0.1
    expected_x = 0.2 + dx * jnp.arange(n, dtype=jnp.float_)
    # expected_x = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if not forward:
      expected_x = expected_x[::-1]

    # Create some arbitrary b
    b = expected_x + jnp.log(expected_x)
    # Create a w array so the recurrence formula x[i] = w[i] * x[i-1] + b[i]
    # yields the expected x.
    w = (expected_x - b) / (expected_x - dx)

    inputs = {'w': w, 'b': b}
    init = 0.1

    def f(carry, w, b):
      return w * carry + b, w * carry + b

    # ACTION
    if not use_scan:
      _, output = rte_utils.recurrent_op_1d(f, init, inputs, forward)
    else:
      _, output = rte_utils.recurrent_op_1d_scan(f, init, inputs, forward)

    # VERIFICATION
    np.testing.assert_allclose(output, expected_x, rtol=1e-5, atol=1e-5)

  @parameterized.expand([
    (fwd, use_scan)
    for fwd, use_scan in product([True, False], [True, False])
  ])
  def test_recurrent_op(self, forward: bool, use_scan: bool):
    # SETUP
    n = 8
    dx = 0.1
    expected_x_1d = 0.2 + dx * jnp.arange(n, dtype=jnp.float_)
    # expected_x = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if not forward:
      expected_x_1d = expected_x_1d[::-1]

    expected_x = jnp.tile(
        expected_x_1d[jnp.newaxis, jnp.newaxis, :], reps=(2, 3, 1)
    )

    # Create some arbitrary b
    b = expected_x + jnp.log(expected_x)
    # Create a w array so the recurrence formula x[i] = w[i] * x[i-1] + b[i]
    # yields the expected x.
    w = (expected_x - b) / (expected_x - dx)

    inputs = {'w': w, 'b': b}
    nx, ny, _ = expected_x.shape
    init = 0.1 * jnp.ones((nx, ny), dtype=jnp.float_)

    def f(carry, w, b):
      return w * carry + b, w * carry + b

    # ACTION
    if not use_scan:
      _, output = rte_utils.recurrent_op(f, init, inputs, forward)
    else:
      _, output = rte_utils.recurrent_op_scan(f, init, inputs, forward)

    # VERIFICATION
    np.testing.assert_allclose(output, expected_x, rtol=1e-5, atol=1e-5)


  @parameterized.expand(list(product([True, False], [True, False])))
  def test_fence_chain_changes_no_values_and_adds_one_barrier(
      self, forward: bool, with_halos: bool
  ):
    """`fence_chain` is a fusion hint only: values agree, one barrier is added."""
    nx, ny, nz = 2, 3, 6
    rng = np.random.default_rng(0)
    w = jnp.asarray(rng.uniform(0.2, 0.9, (nx, ny, nz + 2 * with_halos)))
    b = jnp.asarray(rng.uniform(0.0, 1.0, (nx, ny, nz + 2 * with_halos)))
    init = jnp.asarray(rng.uniform(0.0, 1.0, (nx, ny)))

    def f(carry, w, b):
      return w * carry + b, w * carry + b

    def run(fence_chain):
      if with_halos:
        return rte_utils.recurrent_op_with_halos(
            f, init, {'w': w, 'b': b}, forward, fence_chain=fence_chain
        )
      return rte_utils.recurrent_op(
          f, init, {'w': w, 'b': b}, forward, fence_chain=fence_chain
      )

    def n_barriers(fence_chain):
      text = jax.jit(lambda: run(fence_chain)).lower().as_text()
      return text.count('optimization_barrier')

    carry_off, out_off = run(False)
    carry_on, out_on = run(True)
    np.testing.assert_array_equal(out_on, out_off)
    np.testing.assert_array_equal(carry_on, carry_off)
    # `recurrent_op_with_halos` always fences its result; `fence_chain` adds the
    # inner fence to either entry point.
    self.assertEqual(n_barriers(False), 1 if with_halos else 0)
    self.assertEqual(n_barriers(True), 2 if with_halos else 1)


if __name__ == '__main__':
  unittest.main()
