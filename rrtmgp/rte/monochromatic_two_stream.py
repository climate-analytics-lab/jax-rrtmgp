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

"""The radiative transfer equation solver.

Common symbols:
ssa: single-scattering albedo;
tau: optical depth;
g: asymmetry factor;
sw: shortwave;
lw: longwave;
gamma: exchange rate coefficient in the radiative transfer equation;
zenith: the zenith angle of collimated solar radiation.
"""

import math
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from rrtmgp import kernel_ops
from rrtmgp import smooth_ops
from rrtmgp.rte import rte_utils

Array: TypeAlias = jax.Array
StatesMap: TypeAlias = dict[str, Array]

# Secant of the longwave diffusivity angle per Fu et al. (1997).
_LW_DIFFUSIVE_FACTOR = 1.66

# Switch point between the hyperbolic (small k*tau) and exp-scaled (large
# k*tau) evaluations of the diffuse two-stream quantities. Both forms are
# algebraically exact; at k*tau = 1 both are well-conditioned in float32, so
# the switch itself introduces no discontinuity beyond roundoff.
_KTAU_SWITCH = 1.0

# Absolute transition half-width for the differentiable replacement of the hard
# upper (energy-conservation) clamp on the shortwave direct-beam reflectance /
# transmittance (see `rrtmgp.smooth_ops.smooth_minimum`), on the [0, 1] scale of
# those quantities. Small enough that a value well below the cap is offset only
# negligibly, while the reverse-mode gradient stays continuous where a strongly
# scattering (cloudy) layer drives the direct beam onto the cap.
_SW_CLIP_SHARPNESS = 1e-5


def _shift_up(f: Array) -> Array:
  """output_i = f_{i-1}."""
  return kernel_ops.shift_from_minus(f, 2)


def _shift_down(f: Array) -> Array:
  """output_i = f_{i+1}."""
  return kernel_ops.shift_from_plus(f, 2)


def lw_combine_sources(planck_srcs: StatesMap) -> StatesMap:
  """Combine the longwave source functions at each cell face.

  RRTMGP provides two source functions at each cell interface using the
  spectral mapping of each adjacent layer. These source functions are combined
  here via a geometric mean, and the result can be used for two-stream
  calculations.

  Args:
    planck_srcs: A dictionary containing the longwave Planck sources at the cell
      faces [ccf]. The `planck_src_top` 3D variable contains the Planck source
      at the top cell face derived from the cell center's spectral mapping while
      the `planck_src_bottom` 3D variable contains the Planck source at the
      bottom cell face.

  Returns:
    A map of 3D variables for the combined Planck sources at the top face and
    the bottom cell face, respectively with the same keys as the ones in the
    input `planck_srcs`.
  """
  planck_src_top = planck_srcs['planck_src_top']
  planck_src_bottom = planck_srcs['planck_src_bottom']
  # Geometric mean of the two adjacent-layer sources. `jnp.sqrt` has an infinite
  # derivative at zero, so a cell where the product vanishes (e.g. a halo/edge
  # layer with a zero Planck source) yields a finite forward value but a NaN
  # cotangent in reverse mode. Guard the argument with the standard double-`where`
  # so the sqrt never sees a non-positive value on the differentiated path while
  # the forward result (`sqrt(0) = 0`) is preserved.
  planck_product = planck_src_top * _shift_down(planck_src_bottom)
  product_is_positive = planck_product > 0.0
  combined_src_top = jnp.where(
      product_is_positive,
      jnp.sqrt(jnp.where(product_is_positive, planck_product, 1.0)),
      0.0,
  )
  combined_src_bottom = _shift_up(combined_src_top)
  return {
      'planck_src_top': combined_src_top,
      'planck_src_bottom': combined_src_bottom,
  }


def _eps_of(x: Array) -> float:
  """Machine epsilon of ``x``'s dtype.

  Every numerical guard in this module scales with it, so the same logic is
  correct at float32 and float64 (fixed guards like the former ``1e-2`` k
  floor mean very different things at different precisions and biased the
  conservative-scattering limit at float32).
  """
  return float(jnp.finfo(jnp.result_type(x)).eps)


def _sinhc(x: Array) -> Array:
  """``sinh(x)/x`` with the exact limit 1 at ``x = 0``.

  For ``x`` below an eps-scaled threshold the direct quotient loses relative
  accuracy (and is 0/0 at exactly 0), so the truncated series
  ``1 + x^2/6 + x^4/120`` is used; its truncation error ``~x^6/5040`` is below
  machine eps at the switch point ``(5040*eps)**(1/6)``. Safe operands sit
  inside both ``where`` branches so reverse-mode gradients never see 0/0.
  """
  x0 = (5040.0 * _eps_of(x)) ** (1.0 / 6.0)
  small = jnp.abs(x) < x0
  x_safe = jnp.where(small, 1.0, x)
  x2 = jnp.square(jnp.where(small, x, 0.0))
  return jnp.where(small, 1.0 + x2 / 6.0 * (1.0 + x2 / 20.0),
                   jnp.sinh(x_safe) / x_safe)


def _sinhc_m1(x: Array) -> Array:
  """``sinh(x)/x - 1`` without cancellation near ``x = 0``.

  The direct form subtracts two numbers that agree to ``O(x^2)``; the series
  ``x^2/6*(1 + x^2/20)`` is exact to machine eps for
  ``x < (840*eps)**(1/4)`` (next term ``x^6/5040`` relative ``~x^4/840``).
  """
  x0 = (840.0 * _eps_of(x)) ** 0.25
  small = jnp.abs(x) < x0
  x_safe = jnp.where(small, 1.0, x)
  x2 = jnp.square(jnp.where(small, x, 0.0))
  return jnp.where(small, x2 / 6.0 * (1.0 + x2 / 20.0),
                   jnp.sinh(x_safe) / x_safe - 1.0)


def _k_squared(gamma1: Array, gamma2: Array) -> Array:
  """``k^2 = (gamma1+gamma2)(gamma1-gamma2)`` (Meador & Weaver 1980).

  Non-negative for physical inputs (``gamma1 >= gamma2`` whenever
  ``ssa <= 1``); clamped at zero against roundoff. Kept as ``k^2`` — the
  small-``k*tau`` diffuse branch is written so ``k`` itself cancels, making
  the conservative-scattering limit ``k -> 0`` exact with no floor at all.
  """
  return jnp.maximum((gamma1 + gamma2) * (gamma1 - gamma2), 0.0)


def _k_fn(gamma1: Array, gamma2: Array) -> Array:
  """``k`` with an eps-scaled floor (used by the exp-branch / direct beam).

  The floor is ``sqrt(eps)`` of the working dtype — ``~3e-4`` at float32,
  ``~1.5e-8`` at float64 — replacing the former fixed ``1e-2``, which biased
  the conservative-scattering limit identically at both precisions. It is
  kept hard (not smoothed): ``k`` feeds the removable-singularity handling
  at ``k cos(zenith) = 1`` in ``_direct_quantities``, and the floor only
  engages as ssa -> 1 where the small-``k*tau`` branch (which needs no
  ``k``) carries the diffuse quantities anyway.
  """
  return jnp.sqrt(jnp.maximum(_k_squared(gamma1, gamma2), _eps_of(gamma1)))


def _expm1_over_x(x: Array) -> Array:
  """``expm1(x) / x``, exact-in-the-limit at ``x = 0`` (value 1).

  ``expm1`` itself is accurate for all arguments; only the literal ``0/0`` at
  ``x = 0`` needs the series fallback, so the switch threshold is machine eps
  (below it ``1 + x/2`` is the correctly rounded value). Safe operands inside
  the ``where`` keep reverse-mode gradients NaN-free.
  """
  near_zero = jnp.abs(x) < _eps_of(x)
  x_safe = jnp.where(near_zero, 1.0, x)
  return jnp.where(near_zero, 1.0 + 0.5 * x, jnp.expm1(x_safe) / x_safe)


def _direct_quantities(
    gamma1: Array,
    gamma2: Array,
    gamma3: Array,
    gamma4: Array,
    alpha1: Array,
    alpha2: Array,
    tau: Array,
    ssa: Array,
    zenith: float | Array,
) -> StatesMap:
  """Direct-beam reflectance and transmittance (MW80 eqs 14-15), f32-stable.

  Meador and Weaver's eqs 14-15 are catastrophically ill-conditioned in
  float32 in two regimes:

  * **thin layers** — the numerators are differences of O(1) terms whose
    residual is O(tau) (100% relative error observed below tau ~ 1e-5);
  * **the resonance ``k mu0 = 1``** — numerator and denominator both vanish
    (a removable singularity; Clough et al. 1992, Toon et al. 1989), which
    the old code papered over with a fixed-epsilon clamp on
    ``1 - k^2 mu0^2``.

  Both are cured by the same algebraic step: subtract the (exactly zero)
  ``tau = 0`` value of each numerator, which turns every exponential into an
  ``expm1``, and then factor the resonance ``eta = 1 - k^2 mu0^2`` out of
  the numerator *analytically*. With ``a2 = alpha2 - k gamma3``,
  ``c = alpha1 + k gamma4``, ``d = alpha1 - k gamma4``,
  ``tau' = tau / (mu0 (1 + k mu0))`` and the diffuse denominator ``D``:

      R = (ssa/D) [ -a2 (expm1(-2 k tau)/(1 + k mu0) + 2 k mu0 P_r)
                    - 2 k gamma3 expm1(-(k + 1/mu0) tau)/(1 + k mu0) ]
      T = (ssa/D) [ -c (1 + k mu0) P_t
                    + d e^{-k tau} expm1(-(k + 1/mu0) tau)/(1 + k mu0) ]

  where the resonance pairs

      P_r = (e^{-2 k tau} - e^{-(k + 1/mu0) tau}) / eta
      P_t = (e^{-tau/mu0} - e^{-k tau}) / eta

  are evaluated, for ``|eta tau'| < 1``, in the exactly-equivalent factored
  forms ``P_r = tau' e^{-(k + 1/mu0) tau} phi(eta tau')`` and
  ``P_t = -e^{-k tau} tau' phi(-eta tau')`` with ``phi(x) = expm1(x)/x`` —
  no division by ``eta`` anywhere, finite and smooth *through* the
  resonance. For ``|eta tau'| >= 1`` the literal quotients are used (their
  relative cancellation error is ~eps/|eta tau'| <= eps there). Thin-layer
  limits are reproduced exactly: R -> ssa gamma3 tau/mu0,
  T -> ssa gamma4 tau/mu0, and both vanish at tau = 0 and at ssa = 0 (the
  factor ssa now *multiplies*, so no safe-divide masking is needed).

  Note eq 15's full transmittance includes the unscattered beam
  ``exp(-tau/mu0)``; as before, only the diffusely-transmitted part is
  returned (the direct beam is handled separately in ``sw_cell_source``).
  """
  mu0 = jnp.cos(zenith)
  k = _k_fn(gamma1, gamma2)
  k_mu = k * mu0
  one_plus_kmu = 1.0 + k_mu

  e2m1 = jnp.expm1(-2.0 * k * tau)                # exp(-2 k tau) - 1
  denom = k * (2.0 + e2m1) - gamma1 * e2m1        # diffuse denominator D
  etm1 = jnp.expm1(-(k + 1.0 / mu0) * tau)        # exp(-(k + 1/mu0) tau) - 1
  e_ktmu = 1.0 + etm1                             # exp(-(k + 1/mu0) tau)
  e_kt = jnp.exp(-k * tau)
  t0 = jnp.exp(-tau / mu0)

  eta = 1.0 - k_mu * k_mu
  tau_p = tau / (mu0 * one_plus_kmu)
  x = eta * tau_p
  factored = jnp.abs(x) < 1.0
  # Safe operands: clip the factored-form argument (its phi/tau' pieces are
  # only well-scaled where selected) and the quotient-form divisor.
  x_f = jnp.where(factored, x, 0.0)
  eta_safe = jnp.where(factored, 1.0, eta)
  pair_r = jnp.where(
      factored,
      tau_p * e_ktmu * _expm1_over_x(x_f),
      ((1.0 + e2m1) - e_ktmu) / eta_safe,
  )
  pair_t = jnp.where(
      factored,
      -e_kt * tau_p * _expm1_over_x(-x_f),
      (t0 - e_kt) / eta_safe,
  )

  a2 = alpha2 - k * gamma3
  r_dir = (ssa / denom) * (
      -a2 * (e2m1 / one_plus_kmu + 2.0 * k_mu * pair_r)
      - 2.0 * k * gamma3 * etm1 / one_plus_kmu
  )
  c = alpha1 + k * gamma4
  d = alpha1 - k * gamma4
  t_dir = (ssa / denom) * (
      -c * one_plus_kmu * pair_t + d * e_kt * etm1 / one_plus_kmu
  )
  return {'r_dir': r_dir, 't_dir': t_dir}


def _diffuse_quantities(gamma1: Array, gamma2: Array, tau: Array) -> StatesMap:
  """Diffuse two-stream quantities (MW80 eqs 25-26), float32-stable.

  Two algebraically exact evaluations are blended at ``k*tau = 1``:

  * ``k*tau < 1`` — hyperbolic, k-free form. With the identities
    ``1 - exp(-2x) = 2 exp(-x) sinh(x)`` and
    ``1 + exp(-2x) = 2 exp(-x) cosh(x)``, MW80's refactored expressions
    reduce to ``R = gamma2 sinh(k tau)/Dh``, ``T = k/Dh`` with
    ``Dh = k cosh(k tau) + gamma1 sinh(k tau)``. Factoring one ``k`` out of
    numerator and denominator (``sinh(k tau) = k tau sinhc(k tau)``) leaves

        C = cosh(k tau) + gamma1 tau sinhc(k tau)
        R = gamma2 tau sinhc(k tau) / C,      T = 1 / C

    which involves only ``k^2 = (g1+g2)(g1-g2)`` (via cosh/sinhc arguments),
    is exact at ``k = 0`` (conservative scattering — the old fixed ``1e-2``
    floor is gone) and cancellation-free for thin layers.
  * ``k*tau >= 1`` — the exp-scaled original (rte-rrtmgp) form with
    ``-expm1(-2 k tau)`` replacing ``1 - exp(-2 k tau)``; ``cosh`` would
    overflow float32 beyond ``k tau ~ 88``, and this form is
    well-conditioned for thick layers.

  Also returns the two non-cancelling combinations the longwave linear-in-tau
  source needs (Toon et al. 1989 eqs 26-27; see
  ``lw_cell_source_and_properties``):

      one_minus_r_minus_t = 1 - R - T                       (O(tau) small)
      g_minus_t           = (1 + R - T)/(tau (g1+g2)) - T   (O(tau) small)

  Assembling these from ``R`` and ``T`` would subtract numbers agreeing to
  ``O(tau)`` — exactly the float32 cancellation this refactor removes; the
  closed forms below are derived from the same hyperbolic identities
  (``cosh(x) - 1 = 2 sinh^2(x/2)``, computed as squares — never by
  subtraction) and, on the thick branch, from
  ``k (1 - exp(-k tau))^2 = k expm1(-k tau)^2``.
  """
  eps = _eps_of(gamma1)
  k2 = _k_squared(gamma1, gamma2)
  k = _k_fn(gamma1, gamma2)
  ktau = jnp.sqrt(k2) * tau
  small = ktau < _KTAU_SWITCH
  gsum = jnp.maximum(gamma1 + gamma2, eps)

  # --- small-k*tau branch (safe operands: argument clipped to the switch
  # point inside the branch so sinh/cosh never overflow when unselected).
  x = jnp.where(small, ktau, _KTAU_SWITCH)
  shc = _sinhc(x)
  ch = jnp.cosh(x)
  c_small = ch + gamma1 * tau * shc
  r_small = gamma2 * tau * shc / c_small
  t_small = 1.0 / c_small
  # 1 - R - T = tau*(k^2 tau/2 * sinhc^2(x/2) + (g1-g2) sinhc(x)) / C
  shc_half = _sinhc(0.5 * x)
  omrt_small = tau * (0.5 * k2 * tau * jnp.square(shc_half)
                      + (gamma1 - gamma2) * shc) / c_small
  # G - T = (k^2 tau/(2 gsum) * sinhc^2(x/2) + sinhc_m1(x)) / C
  gmt_small = (0.5 * k2 * tau * jnp.square(shc_half) / gsum
               + _sinhc_m1(x)) / c_small

  # --- large-k*tau branch (exp-scaled; safe at any tau).
  ktau_big = jnp.where(small, _KTAU_SWITCH, k * tau)
  em1 = jnp.expm1(-ktau_big)           # in [-1, 0)
  a = -jnp.expm1(-2.0 * ktau_big)      # 1 - exp(-2 k tau), no cancellation
  e1 = jnp.exp(-ktau_big)
  d_big = k * (1.0 + jnp.exp(-2.0 * ktau_big)) + gamma1 * a
  r_big = gamma2 * a / d_big
  t_big = 2.0 * k * e1 / d_big
  # D - 2k e^{-ktau} -/+ gamma2*(1-e^{-2ktau}) = k(1-e^{-ktau})^2 + (g1-/+g2)a
  omrt_big = (k * jnp.square(em1) + (gamma1 - gamma2) * a) / d_big
  # Safe operand: on the small branch (unselected here) tau can be 0;
  # ktau >= 1 on the selected branch guarantees tau > 0.
  tau_big = jnp.where(small, 1.0, tau)
  g_big = (k * jnp.square(em1) + gsum * a) / (tau_big * gsum * d_big)
  gmt_big = g_big - t_big

  return {
      'r_diff': jnp.where(small, r_small, r_big),
      't_diff': jnp.where(small, t_small, t_big),
      'one_minus_r_minus_t': jnp.where(small, omrt_small, omrt_big),
      'g_minus_t': jnp.where(small, gmt_small, gmt_big),
  }



def lw_cell_source_and_properties(
    optical_depth: Array,
    ssa: Array,
    level_src_bottom: Array,
    level_src_top: Array,
    asymmetry_factor: Array,
) -> StatesMap:
  """Compute the longwave two-stream reflectance, transmittance, and sources.

  The upwelling and downwelling Planck functions and the optical properties
  (transmission and reflectance) are calculated at the cell centers. Equations
  are developed in Meador and Weaver (1980) and Toon et al. (1989).

  Args:
    optical_depth: The pointwise optical depth.
    ssa: The pointwise single-scattering albedo.
    level_src_bottom: The Planck source at the bottom cell face [W / m^2 / sr].
    level_src_top: The Planck source at the top cell face [W / m^2 / sr].
    asymmetry_factor: The pointwise asymmetry factor.

  Returns:
    A dictionary containing the following items:
      'r_diff': A 3D variable containing the pointwise reflectance.
      't_diff': A 3D variable containing the pointwise transmittance.
      'src_up': A 3D variable containing the pointwise upwelling Planck source.
      'src_down': A 3D variable with the pointwise downwelling Planck source.
  """
  # Optical depth is physically non-negative, but halo/edge cells can carry
  # negative values (an artifact of forming molecular column amounts from
  # finite-differenced halo pressures). A negative optical depth makes the
  # `exp(-tau * k)` terms in the diffuse reflectance/transmittance overflow to
  # `+inf`, so their shared denominator evaluates to `inf - inf = NaN`. These
  # halo cells are stripped before the flux recurrence, leaving the forward
  # result unchanged, but the NaN would otherwise poison the reverse-mode
  # cotangents (a masked-away `0 * NaN`). Clamp to the physical range so both
  # the forward and backward passes stay finite; interior cells (optical depth
  # already positive) are untouched.
  optical_depth = jnp.maximum(optical_depth, 0.0)

  # The coefficient of the parallel irradiance in the 2-stream RTE.
  gamma1 = _LW_DIFFUSIVE_FACTOR * (1 - 0.5 * ssa * (1 + asymmetry_factor))
  # The coefficient of the antiparallel irradiance in the 2-stream RTE.
  gamma2 = _LW_DIFFUSIVE_FACTOR * 0.5 * ssa * (1 - asymmetry_factor)

  diffuse = _diffuse_quantities(gamma1, gamma2, optical_depth)
  r_diff = diffuse['r_diff']
  t_diff = diffuse['t_diff']

  # Cell-center sources for the linear-in-tau Planck profile (Toon et al.
  # 1989 eqs 26-27). Writing S_t/S_b for the top/bottom face Planck sources,
  # dS = S_b - S_t and G = (1 + R - T) / (tau (gamma1 + gamma2)), the exact
  # cell-center sources are
  #
  #     src_up   = pi * (S_t (1 - R - T) + dS (G - T))
  #     src_dn   = pi * (S_b (1 - R - T) - dS (G - T))
  #
  # (algebraically identical to the b_1/residual formulation this replaces).
  # Both (1 - R - T) and (G - T) are O(tau) small for thin layers, so
  # assembling them from separately computed R, T, G loses all significant
  # digits in float32 — the reason the old code zeroed the sources below a
  # fixed tau = 1e-4, biasing thin-layer emission. `_diffuse_quantities`
  # returns them in closed, non-cancelling form instead: they go to zero
  # smoothly and *exactly* at tau = 0, so no cutoff (and no gradient kink at
  # it) is needed, and thin layers emit their correct linear-limit amount.
  delta_src = level_src_bottom - level_src_top
  src_up = math.pi * (
      level_src_top * diffuse['one_minus_r_minus_t']
      + delta_src * diffuse['g_minus_t']
  )
  src_down = math.pi * (
      level_src_bottom * diffuse['one_minus_r_minus_t']
      - delta_src * diffuse['g_minus_t']
  )
  return {
      't_diff': t_diff,
      'r_diff': r_diff,
      'src_up': src_up,
      'src_down': src_down,
  }


def sw_cell_properties(
    zenith: float | Array, optical_depth: Array, ssa: Array, asymmetry_factor: Array
) -> StatesMap:
  """Compute shortwave reflectance and transmittance.

  Two-stream solutions to direct and diffuse reflectance and transmittance as
  a function of optical depth, single-scattering albedo, and asymmetry factor.
  Equations are developed in Meador and Weaver (1980).

  Args:
    zenith: The zenith angle of the shortwave collimated radiation.
    optical_depth: A 3D variable containing the pointwise optical depth.
    ssa: A 3D variable containing the pointwise single-scattering albedo.
    asymmetry_factor: A 3D variable containing the pointwise asymmetry factor.

  Returns:
    A dictionary containing the following items:
    't_diff': A 3D variable containing the diffuse transmittance.
    'r_diff': A 3D variable containing the diffuse reflectance.
    't_dir': A 3D variable containing the direct transmittance.
    'r_dir': A 3D variable containing the direct reflectance.
  """
  # Clamp to the physical range: negative halo/edge optical depths would make
  # the `exp(-tau * k)` and `exp(-tau / cos(zenith))` terms overflow to `+inf`
  # and produce NaN denominators, which -- although masked out of the forward
  # fluxes -- poison the reverse-mode cotangents. See the longwave counterpart
  # in `lw_cell_source_and_properties` for details.
  optical_depth = jnp.maximum(optical_depth, 0.0)

  # Exchange rate coefficients from Zdunkowski et al. (1980).
  g = asymmetry_factor
  gamma1 = 0.25 * (8 - ssa * (5 + 3 * g))
  gamma2 = 0.25 * 3 * ssa * (1 - g)
  gamma3 = 0.25 * (2 - 3 * jnp.cos(zenith) * g)
  gamma4 = 1 - gamma3
  alpha1 = gamma1 * gamma4 + gamma2 * gamma3
  alpha2 = gamma1 * gamma3 + gamma2 * gamma4

  # Diffuse reflectance and transmittance.
  diffuse = _diffuse_quantities(gamma1, gamma2, optical_depth)
  r_diff = diffuse['r_diff']
  t_diff = diffuse['t_diff']

  # Direct reflectance and transmittance. The reformulated expressions in
  # `_direct_quantities` *multiply* by the single-scattering albedo (rather
  # than dividing by it inside a shared denominator), so a non-scattering
  # cell (`ssa == 0`, e.g. a pure-absorption g-point or a halo layer) yields
  # an exact 0 with a finite adjoint — no safe-divide masking is needed.
  direct = _direct_quantities(
      gamma1, gamma2, gamma3, gamma4, alpha1, alpha2, optical_depth, ssa,
      zenith,
  )
  r_dir_unconstrained = direct['r_dir']
  t_dir_unconstrained = direct['t_dir']

  # Constrain reflectance and transmittance to be positive and to not go above
  # physical limits by enforcing the constraint that the direct beam can
  # either be reflected, penetrate unscattetered to the bottom of the grid
  # cell, or penetrate through but be scattered on the way.
  #
  # Only the upper (energy-conservation) bound is smoothed: that is the bound a
  # strongly scattering (cloudy) layer actually saturates, so smoothing it makes
  # the gradient continuous in exactly that regime, while a value well below the
  # cap -- the common case -- is left unchanged to within O(sharpness**2). A
  # smooth *lower* clamp would instead offset every small positive value by
  # ~sharpness, so the lower positivity bound is kept as a hard `jnp.maximum`.
  #
  # The hard positivity floor is applied *after* (outside) the smooth cap. This
  # matters when the cap itself collapses toward zero -- a transparent /
  # non-scattering cell has `1 - t0 -> 0`, and `smooth_minimum(x, 0, s)`
  # undershoots to `~ -s/2`; the outer `jnp.maximum(..., 0)` clamps that back to
  # zero so the direct reflectance/transmittance can never go negative (which
  # would otherwise inject a spurious negative diffuse source in
  # `sw_cell_source`). Away from the degenerate cap the floor is inactive, so
  # the upper-cap smoothing is preserved.

  # Direct transmittance.
  t0 = jnp.exp(-optical_depth / jnp.cos(zenith))
  r_dir = jnp.maximum(
      smooth_ops.smooth_minimum(r_dir_unconstrained, 1 - t0, _SW_CLIP_SHARPNESS),
      0.0,
  )
  t_dir = jnp.maximum(
      smooth_ops.smooth_minimum(
          t_dir_unconstrained, 1 - t0 - r_dir, _SW_CLIP_SHARPNESS
      ),
      0.0,
  )

  return {
      't_diff': t_diff,
      'r_diff': r_diff,
      't_dir': t_dir,
      'r_dir': r_dir,
  }


def sw_cell_source(
    t_dir: Array,
    r_dir: Array,
    optical_depth: Array,
    toa_flux: Array,
    sfc_albedo_direct: Array,
    zenith: float | Array,
    use_scan: bool = False,
) -> StatesMap:
  """Compute the monochromatic shortwave direct-beam flux and diffuse source.

  Args:
    t_dir: Direct-beam transmittance, a 3D field.
    r_dir: Direct-beam reflectance, a 3D field.
    optical_depth: Optical depth, a 3D field.
    toa_flux: The top-of-atmosphere incoming flux, a 2D field.
    sfc_albedo_direct: The surface albedo with respect to direct radiation, a 2D
      field.
    zenith: The solar zenith angle.
    use_scan: Whether to use scan or for loops for the recurrent operation.

  Returns:
    A dictionary containing the following items:
      'src_up': A 3D field for the cell center upward source.
      'src_down': A 3D field for the cell center downward source.
      'flux_down_dir': A 3D field for the solved downwelling direct-beam
        radiative flux at the bottom cell face.
      'sfc_src': A 2D field for the shortwave source emanating from the surface.
  """
  # Clamp to the physical range so a negative halo/edge optical depth cannot
  # overflow the direct-beam transmittance to `+inf` (which would poison the
  # reverse-mode cotangents). Interior cells are untouched.
  optical_depth = jnp.maximum(optical_depth, 0.0)

  # Transmittance of direct, unscattered beam.
  t_noscat = jnp.exp(-optical_depth / jnp.cos(zenith))
  mu = jnp.cos(zenith)

  # The vertical component of incident flux at the top boundary.
  flux_down_direct_bc = toa_flux * mu

  # Global recurrent accumulation for the direct-beam downward flux at the
  # bottom cell face unraveling from the top of the atmosphere down to the
  # surface. The recurrence follows the simple relation:
  # flux_down_direct[i] = T_no_scatter[i] * flux_down_direct[i + 1]
  def op(carry, w):
    return w * carry, w * carry
  init = flux_down_direct_bc
  inputs = {'w': t_noscat}
  _, flux_down_direct = rte_utils.recurrent_op_with_halos(
      op, init, inputs, forward=False, use_scan=use_scan
  )

  # Upward source from direct-beam reflection at the cell center.
  src_up = r_dir * _shift_down(flux_down_direct)

  # Downward source from direct-beam transmittance at the cell center.
  src_down = t_dir * _shift_down(flux_down_direct)

  # Direct-beam flux incident on the surface.
  halo_width = 1
  flux_down_sfc = flux_down_direct[:, :, halo_width]

  # The surface source is the direct-beam downard flux that is reflected from
  # the surface.
  sfc_src = sfc_albedo_direct * flux_down_sfc

  srcs_primary = {
      'src_up': src_up,
      'src_down': src_down,
      'flux_down_dir': flux_down_direct,
      'sfc_src': sfc_src,
  }
  return srcs_primary


def _solve_rte_2stream(
    t_diff: Array,
    r_diff: Array,
    src_up: Array,
    src_down: Array,
    toa_flux_down: Array,
    sfc_emission: Array,
    sfc_reflectance: Array,
    use_scan: bool = False,
) -> StatesMap:
  """Solves the monochromatic two-stream radiative transfer equation.

  Given boundary conditions for the downward flux at the top of the atmosphere
  (`toa_flux_down`) and the upward surface emission (`sfc_emission`), this
  computes the two-stream approximation of the upwelling and downwelling
  radiative fluxes at the cell faces based on the equations of Shonk and Hogan
  (2008).  All the computations here assume a single absorption interval (or 'g'
  interval in RRTM nomenclature).  This function needs to be applied to each 'g'
  interval separately.

  Args:
    t_diff: Cell center transmittance, 3D field
    r_diff: Cell center reflectance, 3D field.
    src_up: Cell center upward emission, 3D field.
    src_down: Cell center downward emission, 3D field.
    toa_flux_down: The downward component of the incoming flux at the top
      boundary of the atmosphere.  This corresponds to the downward flux at the
      wall of the domain (e.g., the face above the last interior node).  2D
      field.
    sfc_emission: The upward surface emission.  This corresponds to the wall of
      the domain (e.g., the face below the first interior node).  2D field.
    sfc_reflectance: The surface reflectance.
    use_scan: Whether to use scan or for loops for the recurrent operation.

  Returns:
    A dictionary containing fluxes at the bottom cell face:
      'flux_up': The upwelling radiative flux.
      'flux_down': The downwelling radiative flux.
  """
  # Global recurrent accumulation for the albedo of the atmosphere below a
  # certain level, computed from the surface to the top boundary.  The
  # recurrence relation for albedo is taken from Shonk and Hogan, Eq. 9.

  def albedo_op(
      albedo_below: Array, r_diff: Array, t_diff: Array
  ) -> tuple[Array, Array]:
    """Recurrent formula for albedo solution, starting from the surface."""
    # Geometric series solution accounting for infinite reflection events.
    beta = 1 / (1 - r_diff * albedo_below)
    out = r_diff + t_diff**2 * beta * albedo_below
    return out, out  # Carry and output are the same.

  init = sfc_reflectance
  albedo_inputs = {'r_diff': r_diff, 't_diff': t_diff}
  _, albedo = rte_utils.recurrent_op_with_halos(
      albedo_op, init, albedo_inputs, forward=True, use_scan=use_scan
  )

  # Global recurrent accumulation for the aggregate upwelling source emission
  # computed from the surface to the top of the atmosphere.  The coefficient and
  # bias terms of the recurrent relation for emission are taken from Shonk and
  # Hogan, Eq. 11.  The upward emission is a combination of 1) the upward source
  # from the grid cell center, 2) aggregate emission from the atmosphere below,
  # transmitted through the cell, and 3) the downward source from the grid cell
  # center that is reflected from the atmosphere below and transmitted up
  # through the layer.

  def upward_emission_op(
      emission_from_below: Array,
      src_up: Array,
      src_down: Array,
      t_diff: Array,
      r_diff: Array,
      albedo: Array,
  ) -> tuple[Array, Array]:
    """Recurrent formula for upward emission, starting from the surface."""
    # Geometric series solution accounting for infinite reflection events.
    beta = 1 / (1 - r_diff * albedo)
    out = src_up + t_diff * beta * (emission_from_below + src_down * albedo)
    return out, out  # Carry and output are the same.

  init = sfc_emission
  emission_inputs = {
      'src_up': src_up,
      'src_down': src_down,
      't_diff': t_diff,
      'r_diff': r_diff,
      'albedo': _shift_up(albedo),
  }
  _, emission_up = rte_utils.recurrent_op_with_halos(
      upward_emission_op, init, emission_inputs, forward=True, use_scan=use_scan
  )

  # Global recurrent accumulation for the downwelling radiative flux solution at
  # the bottom face, unravelling from the top of the atmosphere down to the
  # surface.  The coefficient and bias terms are taken from Shonk and Hogan,
  # Eq. 13.  The downward flux at the bottom face is a combination of 1) the
  # downward source emitted from the grid cell, 2) the downward flux from the
  # face above transmitted through the cell, and 3) the aggregate upward
  # emissions from the atmosphere below that are reflected from the cell.

  def flux_down_op(
      flux_down_from_above: Array,
      emiss_up: Array,
      src_down: Array,
      t_diff: Array,
      r_diff: Array,
      albedo: Array,
  ) -> tuple[Array, Array]:
    """Recurrent formula for downwelling flux initiating at top boundar."""
    # Geometric series solution accounting for infinite reflection events.
    beta = 1 / (1 - r_diff * albedo)
    out = (t_diff * flux_down_from_above + r_diff * emiss_up + src_down) * beta
    return out, out  # Carry and output are the same.

  init = toa_flux_down
  flux_down_inputs = {
      'emiss_up': _shift_up(emission_up),
      'src_down': src_down,
      't_diff': t_diff,
      'r_diff': r_diff,
      'albedo': _shift_up(albedo),
  }
  _, flux_down = rte_utils.recurrent_op_with_halos(
      flux_down_op, init, flux_down_inputs, forward=False, use_scan=use_scan
  )

  # The upwelling radiative flux at the bottom face can now be computed
  # directly from the cumulative upward emissions, the cumulative albedo of
  # the atmosphere below, and the downwelling radiative flux at the same face.
  flux_up = flux_down * _shift_up(albedo) + _shift_up(emission_up)
  fluxes = {
      'flux_up': flux_up,
      'flux_down': flux_down,
  }
  return fluxes


def lw_transport(
    t_diff: Array,
    r_diff: Array,
    src_up: Array,
    src_down: Array,
    toa_flux_down: Array,
    sfc_src: Array,
    sfc_emissivity: Array,
    use_scan: bool = False,
) -> StatesMap:
  """Compute the monochromatic longwave diffusive flux of the atmosphere.

  The upwelling and downwelling fluxes are computed from the equations of Shonk
  and Hogan.  The net flux is also computed at every face.  Note that the net
  flux is computed only at cell faces and does not correspond to the net flux
  into or out of the grid cell.  For the overall grid cell net flux, one must
  take the difference of the net fluxes of the upper and bottom faces.

  Args:
    t_diff: Cell center transmittance, 3D field
    r_diff: Cell center reflectance, 3D field.
    src_up: Cell center Planck upward emission, 3D field.
    src_down: Cell center Planck downward emission, 3D field.
    toa_flux_down: The downward flux at the top boundary of the atmosphere, 2D
      field.
    sfc_src: The surface Planck source, 2D field.
    sfc_emissivity: The surface emissivity, 2D field.
    use_scan: Whether to use scan or for loops for the recurrent operation.

  Returns:
    A dictionary containing fluxes at the cell faces [W/m^2]:
      'flux_up': The upwelling radiative flux.
      'flux_down': The downwelling radiative flux.
      'net_flux': The net radiative flux.
  """
  # The source of diffuse radiation is the surface emission.
  sfc_emission = np.pi * sfc_emissivity * sfc_src
  # The surface reflectance is just the complement of the surface emissivity.
  sfc_reflectance = 1 - sfc_emissivity
  fluxes = _solve_rte_2stream(
      t_diff,
      r_diff,
      src_up,
      src_down,
      toa_flux_down,
      sfc_emission,
      sfc_reflectance,
      use_scan,
  )
  fluxes['flux_net'] = fluxes['flux_up'] - fluxes['flux_down']
  return fluxes


def sw_transport(
    t_diff: Array,
    r_diff: Array,
    src_up: Array,
    src_down: Array,
    sfc_src: Array,
    sfc_albedo: Array,
    flux_down_dir: Array,
    use_scan: bool = False,
) -> StatesMap:
  """Compute the monochromatic shortwave fluxes in a layered atmosphere.

  The direct-beam downward flux `flux_down_dir` is added to the downwelling
  diffuse flux in the final solution.

  The upwelling and downwelling diffuse fluxes are computed from the equations
  of Shonk and Hogan.  The net flux is also computed at every face.  Note that
  the net flux is computed only at cell interfaces and does not correspond to
  the net flux into or out of the cell.  For the overall grid cell net flux, one
  must take the difference of the net fluxes of the upper and bottom faces.

  Args:
    t_diff: Cell center transmittance, 3D field
    r_diff: Cell center reflectance, 3D field.
    src_up: Cell center upward source, 3D field.
    src_down: Cell center downward source, 3D field.
    sfc_src: Direct-beam shortwave radiation reflected upward from the surface,
      2D field.
    sfc_albedo: The surface albedo, 2D field.
    flux_down_dir: The downwelling direct-beam radiative flux at the cell faces,
      3D field.
    use_scan: Whether to use scan or for loops for the recurrent operation.

  Returns:
    A dictionary containing fluxes at the z faces:
      'flux_up': The upwelling radiative flux.
      'flux_down': The downwelling radiative flux.
      'net_flux': The net radiative flux.
  """
  fluxes = _solve_rte_2stream(
      t_diff,
      r_diff,
      src_up,
      src_down,
      jnp.zeros_like(sfc_src),
      sfc_src,
      sfc_albedo,
      use_scan,
  )

  # Add the direct-beam contribution to the downwelling flux.
  fluxes['flux_down'] = fluxes['flux_down'] + flux_down_dir

  # The net flux computed at cell faces.
  fluxes['flux_net'] = fluxes['flux_up'] - fluxes['flux_down']
  return fluxes
