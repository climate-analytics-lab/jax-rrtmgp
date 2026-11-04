"""Pytest hooks shared by the whole suite."""

import gc

import jax
import pytest


@pytest.fixture(autouse=True, scope='class')
def _drop_compiled_programs_after_each_class():
  """Release the compiled programs of a test class once it has finished.

  The radiation solves compile to large executables (every g-point program, for
  each solver and k-distribution), and the suite runs many of them in one
  process; left cached they accumulate to more memory than a CI runner has.
  Clearing per class bounds that, and costs only the recompilation of programs
  a later class would have shared, which the classes do not.
  """
  yield
  jax.clear_caches()
  gc.collect()
