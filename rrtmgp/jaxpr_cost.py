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

"""Trip-count-aware cost inspection of a traced JAX program.

Why this exists: XLA's `cost_analysis()` reports the cost of a loop *body*, not
of the loop. A `scan` of length 10 and a `scan` of length 100 are
indistinguishable to it. That blind spot is not academic -- it is exactly how
the issue #22 regression reached production: a scan started walking a whole
table instead of the handful of entries the physics needs, multiplying the work
in the hottest loop of the model, and every flops-based check stayed green.

This module walks the jaxpr instead and multiplies each operation by the trip
counts of the loops enclosing it, giving a number that moves when a loop gets
longer. It is deterministic (no wall-clock), so it is usable as a CI assertion.

Typical use, as a performance guard:

    counts = op_counts(my_solver, arg0, arg1)
    assert counts.transcendentals < BUDGET

or, to catch a loop that grew:

    assert max(scan_lengths(my_solver, arg0)) <= expected

Caveats worth knowing before trusting a number from here:

  * `while_loop` trip counts are genuinely unknowable statically, so a
    `while` body is counted **once** and the fact is reported in
    `unknown_trip_loops`. `fori_loop` over a static range usually lowers to
    `scan` (counted properly) but can lower to `while` depending on context
    -- check `unknown_trip_loops` rather than assuming.
  * This counts *operations issued*, weighted by array size. It is not a
    runtime model: it knows nothing about memory bandwidth, fusion, or how
    much cheaper an `add` is than an `exp`. Treat it as a regression detector,
    not a predictor of seconds.
  * **The jaxpr is pre-optimization.** XLA still has to run CSE, algebraic
    simplification and fusion over it, so these counts are an upper bound on
    what actually executes. In particular, calling two helpers that recompute
    the same subexpression from the same inputs looks like duplicated work
    here and is usually eliminated by the compiler. Redundancy that survives
    is redundancy the compiler *cannot* see through -- different operands,
    different guard thresholds, or a division the compiler may not turn into a
    reciprocal because that would change rounding. Confirm a suspected win
    against `cost_analysis()` (post-optimization) or wall-clock before
    believing it.
  * Both arms of a `jnp.where`/`select_n` are counted, because both are
    genuinely evaluated -- that is the point. A high `select_n` count next to a
    high `sqrt`/`exp` count is the signature of guards paying full price.
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any, Callable, Iterable, TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array

# Primitives that map to a hardware transcendental / special-function unit.
# Counted separately because they dominate the radiative transfer inner loop
# and are what regressed in issue #22.
TRANSCENDENTAL_PRIMITIVES = frozenset({
    'exp', 'exp2', 'expm1', 'log', 'log1p', 'pow', 'integer_pow',
    'sqrt', 'rsqrt', 'cbrt', 'logistic',
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
    'erf', 'erfc', 'erf_inv',
})

# Primitives that are pure bookkeeping: they move or relabel data rather than
# computing with it, so counting them as "work" mostly adds noise.
_STRUCTURAL_PRIMITIVES = frozenset({
    'squeeze', 'reshape', 'broadcast_in_dim', 'transpose', 'convert_element_type',
    'copy', 'slice', 'concatenate', 'pjit', 'closed_call', 'custom_jvp_call',
    'custom_vjp_call', 'custom_vjp_call_jaxpr', 'scan', 'while', 'cond',
})


@dataclasses.dataclass
class OpCounts:
  """Operation counts weighted by array size and enclosing loop trip counts."""

  by_primitive: dict[str, float]
  scan_lengths: list[int]
  unknown_trip_loops: int

  @property
  def total(self) -> float:
    """All non-structural element-operations issued."""
    return sum(
        v for k, v in self.by_primitive.items()
        if k not in _STRUCTURAL_PRIMITIVES
    )

  @property
  def transcendentals(self) -> float:
    """Element-operations issued to transcendental units."""
    return sum(
        v for k, v in self.by_primitive.items()
        if k in TRANSCENDENTAL_PRIMITIVES
    )

  def top(self, n: int = 15) -> list[tuple[str, float]]:
    """The `n` costliest primitives, for eyeballing where the work is."""
    items = [
        (k, v) for k, v in self.by_primitive.items()
        if k not in _STRUCTURAL_PRIMITIVES
    ]
    return sorted(items, key=lambda kv: -kv[1])[:n]

  def __str__(self) -> str:
    lines = [
        f'total={self.total:,.0f}  transcendentals={self.transcendentals:,.0f}',
        f'scan lengths={sorted(set(self.scan_lengths))}  '
        f'unknown-trip loops={self.unknown_trip_loops}',
    ]
    lines += [f'  {k:<24s} {v:>16,.0f}' for k, v in self.top()]
    return '\n'.join(lines)


def _sub_jaxprs(params: dict[str, Any]) -> Iterable[Any]:
  """Every nested jaxpr reachable from an equation's params."""
  for value in params.values():
    candidates = value if isinstance(value, (tuple, list)) else (value,)
    for candidate in candidates:
      inner = getattr(candidate, 'jaxpr', candidate)
      if hasattr(inner, 'eqns'):
        yield inner


def _accumulate(jaxpr, multiplier: float, counts: dict[str, float],
                lengths: list[int], unknown: list[int]) -> None:
  """Walk `jaxpr`, weighting each operation by `multiplier`."""
  for eqn in jaxpr.eqns:
    name = eqn.primitive.name

    if name == 'scan':
      length = int(eqn.params['length'])
      lengths.append(length)
      for sub in _sub_jaxprs(eqn.params):
        _accumulate(sub, multiplier * length, counts, lengths, unknown)
      continue

    if name == 'while':
      # Trip count is not statically known. Count the body once and surface
      # the fact, rather than silently pretending it runs once.
      unknown[0] += 1
      for sub in _sub_jaxprs(eqn.params):
        _accumulate(sub, multiplier, counts, lengths, unknown)
      continue

    nested = list(_sub_jaxprs(eqn.params))
    if nested:
      # `cond` evaluates one branch at runtime, but under `vmap` with a batched
      # predicate it becomes a select over both. Count every branch: the
      # pessimistic reading is the one that matters for a guard.
      for sub in nested:
        _accumulate(sub, multiplier, counts, lengths, unknown)
      continue

    size = sum(
        int(getattr(v.aval, 'size', 1)) for v in eqn.outvars
        if hasattr(v, 'aval')
    )
    counts[name] = counts.get(name, 0.0) + multiplier * max(size, 1)


def op_counts(fn: Callable[..., Any], *args, **kwargs) -> OpCounts:
  """Trip-count-aware operation counts for `fn(*args, **kwargs)`.

  Args:
    fn: Function to trace. Not executed -- only traced.
    *args: Arguments to trace with; only their shapes and dtypes matter.
    **kwargs: Keyword arguments to trace with.

  Returns:
    An `OpCounts` with per-primitive weighted counts, the scan trip counts
    encountered, and how many loops had a statically unknown trip count.
  """
  closed = jax.make_jaxpr(fn)(*args, **kwargs)
  counts: dict[str, float] = {}
  lengths: list[int] = []
  unknown = [0]
  _accumulate(closed.jaxpr, 1.0, counts, lengths, unknown)
  return OpCounts(by_primitive=counts, scan_lengths=lengths,
                  unknown_trip_loops=unknown[0])


def scan_lengths(fn: Callable[..., Any], *args, **kwargs) -> list[int]:
  """Every `scan` trip count in the traced program, outermost first."""
  return op_counts(fn, *args, **kwargs).scan_lengths
