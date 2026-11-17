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

"""The pieces of the column-minor solve against their level-minor originals.

Each column-minor (`(n_levels, ..., n_columns)`, no halos) building block of
`two_stream.solve_lw_zc` / `solve_sw_zc` is checked against the level-minor
(`(nx, ny, nz + 2)`, with halos) function it restates, with the CPU/GPU and
with the TPU kernels (`rrtmgp.backend`), which step the vertical scans a block
of levels at a time.
"""

from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized
from rrtmgp import backend
from rrtmgp.optics import cloud_optics
from rrtmgp.optics import lookup_cloud_optics
from rrtmgp.optics import optics_base
from rrtmgp.rte import longwave_noscat
from rrtmgp.rte import monochromatic_two_stream
from rrtmgp.rte import rte_utils
from rrtmgp.rte import two_stream

_CLD_LW_LOOKUP_TABLE_FILEPATH = (
    Path() / 'rrtmgp/optics/rrtmgp_data/cloudysky_lw.nc'
)
_KERNELS = [('cpu_gpu_kernels', False), ('tpu_kernels', True)]


def _to_zc(f: np.ndarray) -> jax.Array:
  """`(nx, ny, nz + 2)` with halos -> `(nz, nx * ny)` interior."""
  nx, ny, nzh = f.shape
  return jnp.asarray(np.moveaxis(f[..., 1:-1], 2, 0).reshape(nzh - 2, nx * ny))


def _faces_to_zc(f: jax.Array) -> np.ndarray:
  """Level-minor face fluxes `(nx, ny, nz + 2)` -> `(nz + 1, nx * ny)`.

  Entry `k` of the level-minor layout is the bottom face of cell `k`, so the
  surface is entry 1 and the top of the atmosphere entry `nz + 1`.
  """
  f = np.asarray(f)
  nx, ny, nzh = f.shape
  return np.moveaxis(f[..., 1:], 2, 0).reshape(nzh - 1, nx * ny)


class ColumnMinorTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(42)
    self.shape = (3, 4, 49)  # 47 interior levels: not a multiple of 8.

  def _uniform(self, lo, hi, shape=None):
    return self.rng.uniform(lo, hi, self.shape if shape is None else shape)

  @parameterized.expand([(n, rev) for n in (5, 16, 47) for rev in (0, 1)])
  def test_block_stepped_scan_is_the_scan(self, n_levels, reverse):
    """With a vacuum layer the TPU's blocked scan equals a plain scan."""
    xs = (
        jnp.asarray(self._uniform(0.1, 1.0, (n_levels, 7))),
        jnp.asarray(self._uniform(0.0, 0.5, (n_levels, 7))),
    )
    init = jnp.asarray(self._uniform(0.0, 1.0, (7,)))
    want = rte_utils.scan_levels(
        monochromatic_two_stream._propagate_step_zc, init, xs,  # pylint: disable=protected-access
        reverse=bool(reverse), vacuum=(1.0, 0.0),
    )
    with backend.tpu_kernels(True):
      got = rte_utils.scan_levels(
          monochromatic_two_stream._propagate_step_zc, init, xs,  # pylint: disable=protected-access
          reverse=bool(reverse), vacuum=(1.0, 0.0),
      )
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
      np.testing.assert_allclose(g, w, rtol=1e-6)

  def test_face_values_of_both_layouts_agree(self):
    temperature = self._uniform(200.0, 300.0)
    sfc = self._uniform(280.0, 300.0, self.shape[:2])
    want_bottom, want_top = optics_base.reconstruct_face_values(
        jnp.asarray(temperature), jnp.asarray(sfc), axis=2
    )
    nx, ny, nzh = self.shape
    got_bottom, got_top = optics_base.reconstruct_face_values(
        jnp.asarray(np.moveaxis(temperature, 2, 0).reshape(nzh, nx * ny)),
        jnp.asarray(sfc.reshape(nx * ny)),
        axis=0,
    )
    for got, want in ((got_bottom, want_bottom), (got_top, want_top)):
      np.testing.assert_allclose(
          got, np.moveaxis(np.asarray(want), 2, 0).reshape(nzh, nx * ny),
          rtol=1e-6,
      )

  @parameterized.expand(_KERNELS)
  def test_two_stream_transport(self, _, tpu):
    t_diff = self._uniform(0.2, 1.0)
    r_diff = self._uniform(0.0, 0.5) * (1.0 - t_diff)
    src_up = self._uniform(0.0, 5.0)
    src_down = self._uniform(0.0, 5.0)
    toa = self._uniform(0.0, 2.0, self.shape[:2])
    sfc_emission = self._uniform(100.0, 400.0, self.shape[:2])
    sfc_reflectance = self._uniform(0.0, 0.3, self.shape[:2])
    want = monochromatic_two_stream._solve_rte_2stream(  # pylint: disable=protected-access
        *(jnp.asarray(a) for a in (
            t_diff, r_diff, src_up, src_down, toa, sfc_emission,
            sfc_reflectance,
        )),
        use_scan=True,
    )
    flat = lambda a: jnp.asarray(a.reshape(-1))
    with backend.tpu_kernels(tpu):
      got = monochromatic_two_stream.solve_rte_2stream_zc(
          _to_zc(t_diff), _to_zc(r_diff), _to_zc(src_up), _to_zc(src_down),
          flat(toa), flat(sfc_emission), flat(sfc_reflectance),
      )
    for key in ('flux_up', 'flux_down'):
      np.testing.assert_allclose(
          got[key], _faces_to_zc(want[key]), rtol=2e-6, atol=1e-5,
          err_msg=key,
      )

  @parameterized.expand(_KERNELS)
  def test_absorbing_transport_is_the_two_stream_without_reflection(
      self, _, tpu
  ):
    trans = jnp.asarray(self._uniform(0.0, 1.0, (47, 5)))
    src_up = jnp.asarray(self._uniform(0.0, 5.0, (47, 5)))
    src_down = jnp.asarray(self._uniform(0.0, 5.0, (47, 5)))
    toa, emission, refl = (
        jnp.asarray(self._uniform(0.0, 1.0, (5,))) for _ in range(3)
    )
    with backend.tpu_kernels(tpu):
      want = monochromatic_two_stream.solve_rte_2stream_zc(
          trans, jnp.zeros_like(trans), src_up, src_down, toa, emission, refl
      )
      got = monochromatic_two_stream.solve_rte_absorbing_zc(
          trans, src_up, src_down, toa, emission, refl
      )
    for key in want:
      np.testing.assert_allclose(got[key], want[key], rtol=1e-6, atol=1e-6)

  @parameterized.expand(_KERNELS)
  def test_noscat_transport(self, _, tpu):
    tau = self._uniform(0.0, 3.0)
    ssa = self._uniform(0.0, 0.5)
    src = self._uniform(10.0, 100.0)
    src_top = self._uniform(10.0, 100.0)
    src_bottom = self._uniform(10.0, 100.0)
    toa = self._uniform(0.0, 1.0, self.shape[:2])
    sfc_src = self._uniform(50.0, 150.0, self.shape[:2])
    emis = self._uniform(0.9, 1.0, self.shape[:2])
    want = longwave_noscat.lw_noscat_transport(
        *(jnp.asarray(a) for a in (
            tau, ssa, src, src_top, src_bottom, toa, sfc_src, emis
        )),
        use_scan=True,
    )
    flat = lambda a: jnp.asarray(a.reshape(-1))
    with backend.tpu_kernels(tpu):
      got = longwave_noscat.lw_noscat_transport_zc(
          _to_zc(tau), _to_zc(ssa), _to_zc(src), _to_zc(src_top),
          _to_zc(src_bottom), flat(toa), flat(sfc_src), flat(emis),
      )
    for key in ('flux_up', 'flux_down'):
      np.testing.assert_allclose(
          got[key], _faces_to_zc(want[key]), rtol=2e-6, atol=1e-4,
          err_msg=key,
      )

  @parameterized.expand(_KERNELS)
  def test_direct_beam(self, _, tpu):
    tau = self._uniform(0.0, 0.5)
    mu = 0.6
    toa = self._uniform(100.0, 1000.0, self.shape[:2])
    zeros = np.zeros(self.shape)
    want = monochromatic_two_stream.sw_cell_source(
        jnp.asarray(zeros), jnp.asarray(zeros), jnp.asarray(tau),
        jnp.asarray(toa), jnp.asarray(np.zeros(self.shape[:2])),
        float(np.arccos(mu)), use_scan=True,
    )['flux_down_dir']
    with backend.tpu_kernels(tpu):
      flux_sfc, flux_bottom, flux_top = (
          monochromatic_two_stream.sw_direct_beam_zc(
              jnp.exp(-_to_zc(tau) / mu),
              jnp.asarray(toa.reshape(-1) * mu),
          )
      )
    want = _faces_to_zc(want)
    np.testing.assert_allclose(flux_bottom, want[:-1], rtol=1e-5)
    np.testing.assert_allclose(flux_top, want[1:], rtol=1e-5)
    np.testing.assert_allclose(flux_sfc, want[0], rtol=1e-5)

  @parameterized.expand(_KERNELS)
  def test_cloud_optics_of_every_band(self, _, tpu):
    """`phase_properties_all_bands` + `combine_phases` per band."""
    lookup = lookup_cloud_optics.from_nc_file(_CLD_LW_LOOKUP_TABLE_FILEPATH)
    shape = (6, 50)
    r_liq = jnp.asarray(self._uniform(1e-6, 25e-6, shape))
    r_ice = jnp.asarray(self._uniform(5e-6, 90e-6, shape))
    path_liq = jnp.asarray(self._uniform(0.0, 0.1, shape))
    path_ice = jnp.asarray(self._uniform(0.0, 0.05, shape))
    path_liq = path_liq.at[0].set(0.0)  # Cloud-free cells.
    with backend.tpu_kernels(tpu):
      liq, ice = cloud_optics.phase_properties_all_bands(lookup, r_liq, r_ice)
      got = cloud_optics.combine_phases(
          liq, ice, path_liq[jnp.newaxis], path_ice[jnp.newaxis], 0.7, None
      )
    for ibnd in (0, 7, lookup.ext_liq.shape[0] - 1):
      want = cloud_optics.compute_optical_properties(
          lookup, path_liq, path_ice, r_liq, r_ice, ibnd, 0.7, None
      )
      for key, value in want.items():
        np.testing.assert_allclose(
            got[key][ibnd], value, rtol=2e-6, atol=1e-7, err_msg=key
        )

  def test_band_projection_of_both_kernels(self):
    g2b = np.repeat(np.arange(4), [3, 5, 2, 6])
    field = jnp.asarray(self._uniform(0.0, 1.0, (7, 4, 9)))
    with backend.tpu_kernels(False):
      index = two_stream._band_selector(g2b, 4, 2, 8, jnp.float32)  # pylint: disable=protected-access
    with backend.tpu_kernels(True):
      one_hot = two_stream._band_selector(g2b, 4, 2, 8, jnp.float32)  # pylint: disable=protected-access
    for chunk in range(2):
      want = np.asarray(field)[:, g2b[np.minimum(np.arange(8) + 8 * chunk, 15)]]
      for sel in (index[chunk], one_hot[chunk]):
        np.testing.assert_allclose(
            two_stream._by_gpt(sel, field), want, rtol=1e-6  # pylint: disable=protected-access
        )
        np.testing.assert_allclose(
            two_stream._by_gpt(sel, field[0]), want[0], rtol=1e-6  # pylint: disable=protected-access
        )

  def test_chunk_plan(self):
    plan = two_stream._column_minor_chunk_plan  # pylint: disable=protected-access
    with backend.tpu_kernels(False):
      # Evened out to two chunks of 56, not 64 and a padded 48.
      self.assertEqual(plan(112, 64, 47, 9216), (56, 2, False))
      self.assertEqual(plan(128, 64, 47, 9216), (64, 2, False))
      self.assertEqual(plan(128, 48, 47, 9216), (48, 3, True))
    with backend.tpu_kernels(True):
      # A (47, chunk, 9216) float32 array fits 32 MiB for chunk <= 16.
      self.assertEqual(plan(128, 64, 47, 9216), (16, 8, False))
      self.assertEqual(plan(112, 64, 47, 18432), (8, 14, False))
      self.assertEqual(plan(112, 16, 47, 9216), (16, 7, False))


if __name__ == '__main__':
  unittest.main()
