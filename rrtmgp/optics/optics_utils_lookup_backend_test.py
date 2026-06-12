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

"""The 'matmul' and 'gather' table-lookup backends must agree (#6)."""

import collections
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from rrtmgp.optics import optics_utils


def _both(fn):
  """Run `fn` under each lookup backend, restoring the prior selection."""
  prev = optics_utils.get_lookup_impl()
  try:
    optics_utils.set_lookup_impl("matmul")
    out_mm = fn()
    optics_utils.set_lookup_impl("gather")
    out_ga = fn()
  finally:
    optics_utils.set_lookup_impl(prev)
  return out_mm, out_ga


class LookupBackendEquivalenceTest(unittest.TestCase):
  """`gather` (direct indexing) must match `matmul` (one-hot/einsum)."""

  def setUp(self):
    super().setUp()
    # Points carry a (bands, gpts) batch — the column-vmap regime where the
    # backends diverge in memory/throughput but must agree in value.
    batch = (14, 112)
    self.t_ref = jnp.linspace(160.0, 355.0, 14)
    self.p_ref = jnp.linspace(0.0, 18.0, 60)
    self.m_ref = jnp.linspace(0.0, 1.0, 9)
    k = jax.random.split(jax.random.PRNGKey(0), 4)
    self.t = jax.random.uniform(k[0], batch, minval=165.0, maxval=350.0)
    self.p = jax.random.uniform(k[1], batch, minval=0.5, maxval=17.5)
    self.table2 = jax.random.normal(k[2], (self.t_ref.size, self.p_ref.size))
    self.table3 = jax.random.normal(
        k[3], (self.t_ref.size, self.p_ref.size, self.m_ref.size)
    )

  def _fns_independent(self):
    ti = optics_utils.create_linear_interpolant(self.t, self.t_ref)
    pi = optics_utils.create_linear_interpolant(self.p, self.p_ref)
    return collections.OrderedDict((('t', lambda: ti), ('p', lambda: pi)))

  def _fns_dependent(self):
    ti = optics_utils.create_linear_interpolant(self.t, self.t_ref)
    pi = optics_utils.create_linear_interpolant(self.p, self.p_ref)

    def m_fn(t):  # relative-abundance interpolant depends on temperature index
      frac = (jnp.sin(t.idx.astype(jnp.float32) * 0.3) + 1.0) * 0.5
      return optics_utils.create_linear_interpolant(frac, self.m_ref)

    return collections.OrderedDict(
        (('t', lambda: ti), ('p', lambda: pi), ('m', m_fn))
    )

  def test_lookup_values_agree(self):
    idx = [
        jnp.clip((self.t * 0).astype(jnp.int32) + 3, 0, self.t_ref.size - 1),
        jnp.clip((self.p * 0).astype(jnp.int32) + 5, 0, self.p_ref.size - 1),
    ]
    mm, ga = _both(lambda: optics_utils.lookup_values(self.table2, idx))
    np.testing.assert_allclose(np.asarray(mm), np.asarray(ga), rtol=1e-6,
                               atol=1e-6)

  def test_interpolate_independent_agree(self):
    fns = self._fns_independent()
    mm, ga = _both(lambda: optics_utils.interpolate(self.table2, fns))
    np.testing.assert_allclose(np.asarray(mm), np.asarray(ga), rtol=1e-5,
                               atol=1e-5)

  def test_interpolate_dependent_agree(self):
    fns = self._fns_dependent()
    mm, ga = _both(lambda: optics_utils.interpolate(self.table3, fns))
    np.testing.assert_allclose(np.asarray(mm), np.asarray(ga), rtol=1e-5,
                               atol=1e-5)

  def test_grad_agree(self):
    fns = self._fns_dependent()

    def loss(table):
      return jnp.sum(optics_utils.interpolate(table, fns) ** 2)

    g_mm, g_ga = _both(lambda: jax.grad(loss)(self.table3))
    np.testing.assert_allclose(np.asarray(g_mm), np.asarray(g_ga), rtol=1e-4,
                               atol=1e-4)


if __name__ == "__main__":
  unittest.main()
