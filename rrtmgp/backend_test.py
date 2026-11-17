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

import os
import unittest
from unittest import mock

import jax
from rrtmgp import backend


class BackendTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._env = mock.patch.dict(os.environ)
    self._env.start()
    os.environ.pop(backend.ENV_VAR, None)

  def tearDown(self):
    self._env.stop()
    super().tearDown()

  def test_follows_the_default_backend(self):
    self.assertEqual(
        backend.use_tpu_kernels(), jax.default_backend() == 'tpu'
    )
    with mock.patch.object(jax, 'default_backend', return_value='tpu'):
      self.assertTrue(backend.use_tpu_kernels())

  def test_an_enclosing_default_device_wins(self):
    with mock.patch.object(jax, 'default_backend', return_value='tpu'):
      with jax.default_device(jax.devices('cpu')[0]):
        self.assertFalse(backend.use_tpu_kernels())
      with jax.default_device('cpu'):
        self.assertFalse(backend.use_tpu_kernels())

  def test_environment_variable(self):
    for value, want in (('1', True), ('on', True), ('0', False), ('off', False)):
      os.environ[backend.ENV_VAR] = value
      self.assertEqual(backend.use_tpu_kernels(), want, value)
    os.environ[backend.ENV_VAR] = 'auto'
    self.assertEqual(
        backend.use_tpu_kernels(), jax.default_backend() == 'tpu'
    )

  def test_context_overrides_and_restores(self):
    os.environ[backend.ENV_VAR] = '0'
    with backend.tpu_kernels(True):
      self.assertTrue(backend.use_tpu_kernels())
      with backend.tpu_kernels(False):
        self.assertFalse(backend.use_tpu_kernels())
      with backend.tpu_kernels(None):
        self.assertFalse(backend.use_tpu_kernels())  # The environment again.
      self.assertTrue(backend.use_tpu_kernels())
    self.assertFalse(backend.use_tpu_kernels())


if __name__ == '__main__':
  unittest.main()
