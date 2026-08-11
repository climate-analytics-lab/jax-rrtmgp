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

"""Tests for the jaxpr cost model.

This module exists to be trusted by performance guards, so its own failure
modes are the quiet kind: a cost model that reports a plausible number while
being blind to the work that actually grew. Each test below pins one such
blind spot.
"""

import unittest

import jax
import jax.numpy as jnp

from rrtmgp import jaxpr_cost


class JaxprCostTest(unittest.TestCase):

    def test_scales_with_scan_trip_count(self):
        """The blind spot this module exists for.

        `cost_analysis()` reports a loop body, so it returns the same number
        for both of these. Anything built on it cannot see a loop grow.
        """
        def solver(length):
            def fn(x):
                def body(carry, i):
                    return carry + jnp.sin(x * i).sum(), None
                return jax.lax.scan(body, 0.0, jnp.arange(length))[0]
            return fn

        x = jnp.ones(1000)
        short = jaxpr_cost.op_counts(solver(10), x)
        long = jaxpr_cost.op_counts(solver(100), x)

        self.assertEqual(short.scan_lengths, [10])
        self.assertEqual(long.scan_lengths, [100])
        self.assertAlmostEqual(
            long.transcendentals / short.transcendentals, 10.0, places=6
        )

    def test_reductions_scale_with_input_not_output(self):
        """A reduction's output is tiny; its work is not.

        Weighting by output size alone would score a sum over a million
        elements the same as a sum over ten.
        """
        small = jaxpr_cost.op_counts(jnp.sum, jnp.ones(10))
        large = jaxpr_cost.op_counts(jnp.sum, jnp.ones(1_000_000))
        self.assertGreater(large.total, 100.0 * small.total)

    def test_contraction_scales_with_contracted_dimension(self):
        """A matmul's cost is the contraction volume, not its output size."""
        n = 64
        counts = jaxpr_cost.op_counts(
            lambda a, b: a @ b, jnp.ones((n, n)), jnp.ones((n, n))
        )
        # Output is n*n elements, each accumulating over n positions.
        self.assertGreaterEqual(counts.total, float(n ** 3))

    def test_integer_power_is_not_a_transcendental(self):
        """`x ** 2` lowers to multiplication, not to a special-function unit.

        Counting it as transcendental would let a cost-neutral `x * x` ->
        `x ** 2` refactor fail a transcendental budget for no reason.
        """
        squared_by_mul = jaxpr_cost.op_counts(lambda x: x * x, jnp.ones(1000))
        squared_by_pow = jaxpr_cost.op_counts(lambda x: x ** 2, jnp.ones(1000))
        self.assertEqual(squared_by_mul.transcendentals, 0.0)
        self.assertEqual(squared_by_pow.transcendentals, 0.0)

    def test_real_transcendentals_are_still_counted(self):
        """The converse: a genuine special function must not be missed."""
        for fn in (jnp.exp, jnp.log, jnp.sqrt, lambda x: x ** 0.5):
            with self.subTest(fn=getattr(fn, '__name__', 'pow')):
                counts = jaxpr_cost.op_counts(fn, jnp.ones(1000))
                self.assertEqual(counts.transcendentals, 1000.0)

    def test_both_arms_of_a_where_are_counted(self):
        """`jnp.where` is not a branch -- both arms are evaluated.

        A guard that hides an `exp` behind a `where` pays for it on every
        element, and the cost model has to say so.
        """
        guarded = jaxpr_cost.op_counts(
            lambda x: jnp.where(x < 0.5, x, jnp.exp(x)), jnp.ones(1000)
        )
        self.assertEqual(guarded.transcendentals, 1000.0)

    def test_unknown_trip_loops_are_reported(self):
        """A `while_loop` trip count cannot be known statically; say so."""
        def fn(x):
            return jax.lax.while_loop(
                lambda s: s[0] < 5, lambda s: (s[0] + 1, s[1] + jnp.sin(x).sum()),
                (0, 0.0),
            )[1]

        counts = jaxpr_cost.op_counts(fn, jnp.ones(100))
        self.assertGreater(counts.unknown_trip_loops, 0)


if __name__ == '__main__':
    unittest.main()
