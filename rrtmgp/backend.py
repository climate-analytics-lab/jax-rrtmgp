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

"""The one switch between the TPU kernels and the CPU/GPU kernels.

Several kernels have a second form written for the TPU, where a gather runs
serially on the vector unit and the matrix unit (MXU) is idle: table lookups
become one-hot contractions on the MXU, vertical scans are stepped a block of
levels at a time, and halo updates are concatenations rather than scatters.
The values are the same to float32 rounding; the programs are not. Which form
is traced is decided here and nowhere else, at trace time.

`use_tpu_kernels()` follows, in order:

  1. the innermost `tpu_kernels(enabled)` context, if any;
  2. the `RRTMGP_TPU_KERNELS` environment variable, if set to `1`/`on`/`true`
     or `0`/`off`/`false` (anything else, or unset, means automatic);
  3. automatically, the platform of the default device: that of an enclosing
     `jax.default_device(...)` context when there is one, else
     `jax.default_backend()`.

The automatic choice cannot see the device a computation is finally placed on
when that is not the default one -- `jax.jit(..., backend=...)`, a
`device_put` to another platform, or lowering / exporting for a TPU from a CPU
host. Those callers select the kernels explicitly, with `tpu_kernels(...)` or
the environment variable. Either form is correct on any platform; the choice
only decides which one is fast.
"""

from collections.abc import Iterator
import contextlib
import contextvars
import os

import jax

ENV_VAR = 'RRTMGP_TPU_KERNELS'
_TRUE = ('1', 'on', 'true', 'yes')
_FALSE = ('0', 'off', 'false', 'no')

_override: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    'rrtmgp_tpu_kernels', default=None
)


def _default_platform() -> str:
  """The platform computations are placed on by default."""
  device = jax.config.jax_default_device
  if device is not None:
    # `jax.default_device` accepts a `Device` or a platform name.
    return device if isinstance(device, str) else device.platform
  return jax.default_backend()


def use_tpu_kernels() -> bool:
  """Whether to trace the TPU forms of the kernels that have one."""
  override = _override.get()
  if override is not None:
    return override
  env = os.environ.get(ENV_VAR, '').strip().lower()
  if env in _TRUE:
    return True
  if env in _FALSE:
    return False
  return _default_platform() == 'tpu'


@contextlib.contextmanager
def tpu_kernels(enabled: bool | None) -> Iterator[None]:
  """Select the TPU (`True`) or CPU/GPU (`False`) kernels inside the block.

  `None` restores the automatic choice. The selection applies to functions
  traced inside the block; a function compiled earlier keeps its program.

  Args:
    enabled: Whether to trace the TPU kernels, or `None` for automatic.

  Yields:
    Nothing; the selection is active until the block exits.
  """
  token = _override.set(enabled)
  try:
    yield
  finally:
    _override.reset(token)
