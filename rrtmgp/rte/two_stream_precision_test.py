"""Precision tests for the reformulated monochromatic two-stream solver.

The solver is written to be accurate in float32 across the regimes where the
textbook Meador & Weaver (1980) / Toon et al. (1989) expressions suffer
catastrophic cancellation:

* thin layers (``tau -> 0``): reflectance/transmittance/sources are O(tau)
  residuals of O(1) terms;
* conservative scattering (``k -> 0``);
* the direct-beam removable singularity ``k * mu0 = 1``.

Every test here compares the float32 JAX implementation against a
``numpy.float128`` evaluation of the *analytic* two-stream solution (the
plain, cancelling formulas — harmless at ~1e-19 eps), so the asserted bounds
measure true forward error, not agreement with another float32 code.
"""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from rrtmgp.rte import monochromatic_two_stream

_Q = np.float128
_LW_DIFFUSIVE_FACTOR = 1.66


def _diffuse_ref(gamma1, gamma2, tau):
  """Float128 diffuse R/T (MW80 eqs 25-26, plain formulation)."""
  k = np.sqrt(np.maximum((gamma1 + gamma2) * (gamma1 - gamma2), _Q(1e-300)))
  e2 = np.exp(-2 * k * tau)
  denom = k * (1 + e2) + gamma1 * (1 - e2)
  r = gamma2 * (1 - e2) / denom
  t = 2 * k * np.exp(-k * tau) / denom
  return r, t, k, denom


def _lw_ref(tau, ssa, s_bottom, s_top, g):
  """Float128 LW cell quantities (Toon et al. 1989 linear-in-tau sources)."""
  gamma1 = _Q(_LW_DIFFUSIVE_FACTOR) * (1 - _Q(0.5) * ssa * (1 + g))
  gamma2 = _Q(_LW_DIFFUSIVE_FACTOR) * _Q(0.5) * ssa * (1 - g)
  r, t, _, _ = _diffuse_ref(gamma1, gamma2, tau)
  big_g = (1 + r - t) / (tau * (gamma1 + gamma2))
  ds = s_bottom - s_top
  pi = _Q(math.pi)
  src_up = pi * (s_top * (1 - r - t) + ds * (big_g - t))
  src_down = pi * (s_bottom * (1 - r - t) - ds * (big_g - t))
  return {'r_diff': r, 't_diff': t, 'src_up': src_up, 'src_down': src_down}


def _sw_ref(zenith, tau, ssa, g):
  """Float128 SW cell quantities (MW80 eqs 14-15/25-26, Zdunkowski gammas)."""
  mu0 = np.cos(_Q(zenith))
  gamma1 = _Q(0.25) * (8 - ssa * (5 + 3 * g))
  gamma2 = _Q(0.25) * 3 * ssa * (1 - g)
  gamma3 = _Q(0.25) * (2 - 3 * mu0 * g)
  gamma4 = 1 - gamma3
  alpha1 = gamma1 * gamma4 + gamma2 * gamma3
  alpha2 = gamma1 * gamma3 + gamma2 * gamma4
  r_diff, t_diff, k, denom = _diffuse_ref(gamma1, gamma2, tau)
  k_mu = k * mu0
  eta = 1 - k_mu * k_mu
  t0 = np.exp(-tau / mu0)
  e1 = np.exp(-k * tau)
  e2 = np.exp(-2 * k * tau)
  r_dir = ssa * (
      (1 - k_mu) * (alpha2 + k * gamma3)
      - (1 + k_mu) * (alpha2 - k * gamma3) * e2
      - 2 * (k * gamma3 - alpha2 * k_mu) * e1 * t0
  ) / (denom * eta)
  t_dir = -ssa * (
      (1 + k_mu) * (alpha1 + k * gamma4) * t0
      - (1 - k_mu) * (alpha1 - k * gamma4) * e2 * t0
      - 2 * (k * gamma4 + alpha1 * k_mu) * e1
  ) / (denom * eta)
  return {'r_diff': r_diff, 't_diff': t_diff, 'r_dir': r_dir, 't_dir': t_dir}


def _max_rel(actual, ref, significance=1e-6):
  """Max relative error over elements that are significant vs the field max."""
  ref = np.asarray(ref, np.float64)
  actual = np.asarray(actual, np.float64)
  sig = np.abs(ref) > significance * np.max(np.abs(ref))
  rel = np.abs(actual - ref) / np.maximum(np.abs(ref), 1e-300)
  return rel[sig].max()


class TwoStreamFloat32PrecisionTest(unittest.TestCase):
  """Float32 forward error against a float128 analytic reference."""

  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(2026)
    n = 20000
    self.tau = 10.0 ** rng.uniform(-8, 2, n)
    self.ssa = rng.uniform(0.0, 0.999, n)
    self.g = rng.uniform(0.0, 0.85, n)
    self.s_bottom = rng.uniform(5.0, 40.0, n)
    self.s_top = self.s_bottom * rng.uniform(0.9, 1.1, n)

  def test_lw_forward_error(self):
    out = monochromatic_two_stream.lw_cell_source_and_properties(
        jnp.asarray(self.tau, jnp.float32),
        jnp.asarray(self.ssa, jnp.float32),
        jnp.asarray(self.s_bottom, jnp.float32),
        jnp.asarray(self.s_top, jnp.float32),
        jnp.asarray(self.g, jnp.float32),
    )
    ref = _lw_ref(self.tau.astype(_Q), self.ssa.astype(_Q),
                  self.s_bottom.astype(_Q), self.s_top.astype(_Q),
                  self.g.astype(_Q))
    for key, bound in [('r_diff', 1e-4), ('t_diff', 1e-4),
                       ('src_up', 5e-4), ('src_down', 5e-4)]:
      err = _max_rel(out[key], ref[key])
      self.assertFalse(np.isnan(np.asarray(out[key])).any(), key)
      self.assertLess(err, bound, f'{key}: float32 max rel error {err:.3e}')

  def test_sw_forward_error(self):
    zenith = 0.3
    out = monochromatic_two_stream.sw_cell_properties(
        jnp.float32(zenith),
        jnp.asarray(self.tau, jnp.float32),
        jnp.asarray(self.ssa, jnp.float32),
        jnp.asarray(self.g, jnp.float32),
    )
    ref = _sw_ref(zenith, self.tau.astype(_Q), self.ssa.astype(_Q),
                  self.g.astype(_Q))
    for key in ('r_diff', 't_diff'):
      actual = np.asarray(out[key])
      self.assertFalse(np.isnan(actual).any(), key)
      err = _max_rel(actual, np.asarray(ref[key], np.float64))
      self.assertLess(err, 1e-4, f'{key}: float32 max rel error {err:.3e}')
    # The public r_dir/t_dir additionally pass through the smooth
    # energy-conservation cap, which by design perturbs values at its
    # absolute sharpness scale (1e-5) — precision of the cap-free solver is
    # asserted in test_sw_direct_solver_forward_error; here check the public
    # outputs are finite and physical.
    for key in ('r_dir', 't_dir'):
      actual = np.asarray(out[key])
      self.assertFalse(np.isnan(actual).any(), key)
      self.assertGreaterEqual(actual.min(), 0.0, key)
      self.assertLessEqual(actual.max(), 1.0, key)

  def test_sw_direct_solver_forward_error(self):
    """Direct-beam R/T precision, upstream of the energy-conservation cap.

    This is the regime the reformulation targets: the pre-refactor
    Meador-Weaver evaluation had unbounded relative error in float32 for
    thin layers and near the k*mu0 = 1 resonance (worst observed ~6e3).
    """
    zenith = 0.3
    mu0 = np.cos(zenith)
    gamma1 = 0.25 * (8 - self.ssa * (5 + 3 * self.g))
    gamma2 = 0.25 * 3 * self.ssa * (1 - self.g)
    gamma3 = 0.25 * (2 - 3 * mu0 * self.g)
    gamma4 = 1 - gamma3
    alpha1 = gamma1 * gamma4 + gamma2 * gamma3
    alpha2 = gamma1 * gamma3 + gamma2 * gamma4
    out = monochromatic_two_stream._direct_quantities(
        *(jnp.asarray(v, jnp.float32) for v in
          (gamma1, gamma2, gamma3, gamma4, alpha1, alpha2, self.tau,
           self.ssa)),
        jnp.float32(zenith),
    )
    ref = _sw_ref(zenith, self.tau.astype(_Q), self.ssa.astype(_Q),
                  self.g.astype(_Q))
    for key in ('r_dir', 't_dir'):
      actual = np.asarray(out[key])
      self.assertFalse(np.isnan(actual).any(), key)
      err = _max_rel(actual, np.asarray(ref[key], np.float64))
      self.assertLess(err, 1e-3, f'{key}: float32 max rel error {err:.3e}')

  def test_lw_thin_layer_source_matches_linear_limit(self):
    """Thin layers must emit their linear-limit amount, not zero.

    The pre-refactor code zeroed the LW sources below a fixed tau = 1e-4;
    the closed-form sources instead approach
    ``pi * S * (gamma1 + gamma2 - 2 gamma2) * tau`` smoothly. Verify the
    O(tau) limit at tau values far below the old cutoff.
    """
    tau = np.array([1e-7, 1e-6, 1e-5], np.float32)
    ssa, g = np.float32(0.3), np.float32(0.4)
    s = np.float32(20.0)
    out = monochromatic_two_stream.lw_cell_source_and_properties(
        jnp.asarray(tau), jnp.full_like(tau, ssa), jnp.full_like(tau, s),
        jnp.full_like(tau, s), jnp.full_like(tau, g),
    )
    gamma1 = _LW_DIFFUSIVE_FACTOR * (1 - 0.5 * ssa * (1 + g))
    gamma2 = _LW_DIFFUSIVE_FACTOR * 0.5 * ssa * (1 - g)
    # 1 - R - T -> (gamma1 - gamma2) tau;  G - T -> 0 for a uniform source.
    expected = math.pi * s * (gamma1 - gamma2) * tau
    np.testing.assert_allclose(out['src_up'], expected, rtol=1e-4)
    np.testing.assert_allclose(out['src_down'], expected, rtol=1e-4)
    # And exactly zero at tau = 0 (no cutoff discontinuity to hide it).
    zero = monochromatic_two_stream.lw_cell_source_and_properties(
        jnp.zeros(1), jnp.full((1,), ssa), jnp.full((1,), s),
        jnp.full((1,), s), jnp.full((1,), g),
    )
    self.assertEqual(float(zero['src_up'][0]), 0.0)
    self.assertEqual(float(zero['src_down'][0]), 0.0)

  def test_pure_absorption_identity(self):
    """ssa = 0: R = 0 exactly and T = exp(-gamma1 tau) (LW gamma1 = 1.66)."""
    tau = np.array([1e-6, 1e-2, 1.0, 30.0], np.float32)
    out = monochromatic_two_stream.lw_cell_source_and_properties(
        jnp.asarray(tau), jnp.zeros_like(tau), jnp.ones_like(tau),
        jnp.ones_like(tau), jnp.zeros_like(tau),
    )
    np.testing.assert_allclose(out['r_diff'], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        out['t_diff'], np.exp(-_LW_DIFFUSIVE_FACTOR * tau), rtol=1e-5
    )
    # SW at ssa = 0: the direct-beam diffuse scattering vanishes exactly.
    sw = monochromatic_two_stream.sw_cell_properties(
        0.3, jnp.asarray(tau), jnp.zeros_like(tau), jnp.zeros_like(tau)
    )
    np.testing.assert_array_equal(np.asarray(sw['r_dir']), 0.0)
    np.testing.assert_array_equal(np.asarray(sw['t_dir']), 0.0)

  def test_direct_beam_continuity_through_resonance(self):
    """r_dir/t_dir are continuous in mu0 across the k*mu0 = 1 resonance."""
    tau, ssa, g = 0.5, 0.5373, 0.679  # k ~ 1.0 for these gammas
    gamma1 = 0.25 * (8 - ssa * (5 + 3 * g))
    gamma2 = 0.25 * 3 * ssa * (1 - g)
    k = math.sqrt((gamma1 + gamma2) * (gamma1 - gamma2))
    mu_res = 1.0 / k
    self.assertLess(mu_res, 1.0)  # resonance reachable for this profile
    # Dense sweep of mu0 through the resonance (a few float32 ulps apart
    # near it, wider outside).
    mu = np.unique(np.concatenate([
        np.linspace(mu_res - 1e-3, mu_res + 1e-3, 2001),
        mu_res + np.array([-3e-7, -1e-7, 0.0, 1e-7, 3e-7]),
    ])).astype(np.float32)
    zen = np.arccos(mu)
    ones = jnp.ones_like(jnp.asarray(mu))
    out = monochromatic_two_stream.sw_cell_properties(
        jnp.asarray(zen), tau * ones, ssa * ones, g * ones
    )
    for key in ('r_dir', 't_dir'):
      vals = np.asarray(out[key], np.float64)
      self.assertFalse(np.isnan(vals).any(), key)
      # Continuity: adjacent samples differ by no more than the local slope
      # allows (the function is smooth; catch any clamp-style jump).
      jumps = np.abs(np.diff(vals))
      scale = np.maximum(np.abs(vals[:-1]), 1e-3)
      self.assertLess((jumps / scale).max(), 1e-3, key)

  def test_gradients_finite_at_degenerate_inputs(self):
    """Reverse-mode gradients stay finite at tau = 0, ssa = 0, resonance."""

    def lw_sum(tau, ssa):
      out = monochromatic_two_stream.lw_cell_source_and_properties(
          tau, ssa, jnp.full_like(tau, 20.0), jnp.full_like(tau, 19.0),
          jnp.full_like(tau, 0.5),
      )
      return sum(jnp.sum(v) for v in out.values())

    def sw_sum(tau, ssa):
      out = monochromatic_two_stream.sw_cell_properties(
          0.90439791, tau, ssa, jnp.full_like(tau, 0.16116116)
      )
      return sum(jnp.sum(v) for v in out.values())

    # tau = 0, thin tau, thick tau; ssa = 0, near-conservative, and
    # EXACTLY conservative (ssa = 1 -> k^2 = 0: an f32 Rayleigh-only
    # g-point rounds there, and a sqrt(k2) on the reverse path NaN'd every
    # clear-sky SW temperature gradient before the x2-parameterized
    # helpers); the SW profile also sits at the k*mu0 = 1 resonance of the
    # KGO test point.
    tau = jnp.asarray([0.0, 1e-8, 1e-4, 0.03, 5.0, 100.0, 0.1, 10.0])
    ssa = jnp.asarray([0.0, 0.999, 0.2722723, 0.2722723, 0.5, 0.0, 1.0, 1.0])
    for fn in (lw_sum, sw_sum):
      grads = jax.grad(fn, argnums=(0, 1))(tau, ssa)
      for grad in grads:
        self.assertTrue(bool(jnp.isfinite(grad).all()), fn.__name__)


if __name__ == '__main__':
  unittest.main()
