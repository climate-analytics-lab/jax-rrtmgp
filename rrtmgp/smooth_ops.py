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

"""Smooth (C1+) replacements for hard clamp / threshold operations.

Gradient-based calibration through the radiative-transfer scheme benefits from a
continuously differentiable forward map. The physical parameterisation enforces
bounds and thresholds with hard ``jnp.maximum`` / ``jnp.minimum`` / ``jnp.clip``
/ ``jnp.where`` operations, each of which puts a derivative discontinuity (a
"kink") into the reverse-mode gradient wherever it saturates -- a cell that
crosses the threshold as the state or a model parameter is perturbed sees the
gradient jump.

These helpers replace those hard operations with smooth transitions whose width
is controlled by a sharpness parameter. The sharpness values used by the callers
are chosen tight enough that the forward result is unchanged to well within the
scheme's accuracy on the cells that stay away from the threshold -- which is the
overwhelming majority, since the guarded operations are typically inactive there
-- while a cell that does cross a threshold picks up a smooth ramp instead of a
step. As sharpness -> 0 every function here recovers its hard counterpart
exactly.
"""

from typing import TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array


def smooth_minimum(
    x: Array, upper: Array | float, sharpness: float
) -> Array:
  """C-infinity upper bound; ``-> jnp.minimum(x, upper)`` as ``sharpness -> 0``.

  Uses the hyperbolic smoothing ``0.5 * (x + c - sqrt((x - c)**2 + s**2))``,
  which is monotonically increasing in ``x``, exact far from the corner, and
  undershoots the corner value by at most ``sharpness / 2``. The ``sqrt``
  argument is bounded below by ``sharpness**2 > 0``, so the derivative is finite
  everywhere (no ``jnp.minimum`` sub-gradient ambiguity).

  Args:
    x: The value to bound from above.
    upper: The upper bound.
    sharpness: Transition half-width; smaller is closer to the hard minimum.

  Returns:
    The smoothly upper-bounded value.
  """
  d = x - upper
  return 0.5 * (x + upper - jnp.sqrt(d * d + sharpness * sharpness))


def smooth_gate(x: Array, lo: Array | float, hi: Array | float) -> Array:
  """C1 ramp: ``0`` for ``x <= lo``, ``1`` for ``x >= hi``, smoothstep between.

  The cubic smoothstep ``3 s**2 - 2 s**3`` has zero derivative at both ends, so
  the gate is C1-continuous across the clamp boundaries -- there is no gradient
  jump where the ramp meets the flat ``0`` and ``1`` regions. Used to replace a
  hard ``jnp.where(x > threshold, ...)`` mask with a narrow smooth ramp centred
  on the threshold.

  Args:
    x: The quantity being thresholded.
    lo: Lower edge of the ramp (gate is 0 at and below this).
    hi: Upper edge of the ramp (gate is 1 at and above this).

  Returns:
    A weight in ``[0, 1]``.
  """
  s = jnp.clip((x - lo) / (hi - lo), 0.0, 1.0)
  return s * s * (3.0 - 2.0 * s)
