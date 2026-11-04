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

"""The longwave solver for an absorbing, non-scattering atmosphere.

This is the "no-scattering" longwave solver of RTE+RRTMGP, the alternative to
the two-stream solver in `monochromatic_two_stream`. It transcribes the
single-angle path of upstream RTE,

  rte/kernels/mo_rte_solver_kernels.F90
    lw_solver_noscat_oneangle   (driver: boundaries and radiance -> flux)
    lw_source_noscat            (layer emission, linear-in-tau Planck function)
    lw_transport_noscat_dn / _up (the two recurrences),

which is also what ECHAM6's `mo_lrtm_solver.f90` ("IR (no scattering)") does.
Radiation travels at one fixed diffusivity angle with secant `D`, so each layer
transmits `exp(-D tau)` and the downward and upward intensities obey two
independent first-order recurrences: no albedo / reflectance recurrence and no
coupling between the two streams, unlike the two-stream solver.

Conventions (all established against the two-stream solver, see
`longwave_noscat_test.py`):

* The Planck sources handed in are radiances (W/m^2/sr), the same quantities the
  two-stream path multiplies by pi. Upstream carries radiances through the
  recurrences and converts to flux with `pi * weight` at the end, with the
  prescribed incident flux divided by `pi * weight` on the way in. Both
  recurrences are linear in their sources and boundary values, so this
  implementation scales the sources and the surface emission by `pi * weight`
  instead and carries *fluxes*: the same result, with the prescribed incident
  flux entering unchanged (the top face returns exactly what was prescribed).
* The Planck source handed to each layer's two faces is one source per level
  (see `lw_transport_noscat_*`'s caller, `two_stream._lw_planck_sources`), as in
  current upstream RTE.
* Level layout is that of `monochromatic_two_stream.lw_transport`: surface
  first, one halo cell at each end of z, face `k` is the *bottom* face of layer
  `k`, the top-of-atmosphere face is the last index, and the halo entries are
  zero.
"""

from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from rrtmgp.rte import rte_utils

Array: TypeAlias = jax.Array
StatesMap: TypeAlias = dict[str, Array]

# Secant of the single quadrature angle: the one-point "Gauss-Jacobi-5" rule of
# Hogan (2023, doi:10.1002/qj.4598, Table 1), cos(theta) = 0.6096748751, which
# is the default of upstream `rte_lw` (`gauss_Ds(1, 1)`). Its weight is 1.
LW_NOSCAT_DEFAULT_SECANT = 1.0 / 0.6096748751
LW_NOSCAT_DEFAULT_WEIGHT = 1.0


def _shift_toward_toa(f: Array) -> Array:
  """`output_k = f_{k-1}` along z, with `output_0 = 0`.

  A slice and a zero pad, where `jnp.roll` would emit a concatenate that XLA
  fuses into the recurrences feeding it. The wrapped-around entry `f_{-1}` is
  the halo entry that `recurrent_op_with_halos` pads with zero, so the two
  agree exactly.
  """
  return jnp.pad(f[:, :, :-1], ((0, 0), (0, 0), (1, 0)))


def lw_noscat_layer_sources(
    tau: Array,
    planck_src: Array,
    planck_src_top: Array,
    planck_src_bottom: Array,
) -> tuple[Array, Array, Array]:
  """Transmittance and emitted upward / downward radiance of each layer.

  Transcribes `lw_source_noscat` (Clough et al. 1992, doi:10.1029/92JD01419,
  eq. 13), the exact solution for a Planck function that is linear in the
  optical depth, anchored on the layer-centre value `planck_src` and the two
  face values:

      source_up   = (1 - T) S_top    + 2 f (S_layer - S_top)
      source_down = (1 - T) S_bottom + 2 f (S_layer - S_bottom)
      f           = (1 - T) / tau - T

  `f` is a difference of nearly equal numbers for thin layers (it is O(tau)
  while each term is O(1)), so below `tau_thresh = eps**(1/4)`, where the
  rounding error of the closed form (~eps / tau) reaches the truncation error
  of the series, upstream switches to the 3rd-order series
  `tau (1/2 - tau/3 + tau^2/8)`. Both branches are written as a double
  `jnp.where` with a safe operand on the branch not taken: the closed form is
  divided by `tau`, which is zero in a clear, gas-free cell, and a masked
  `0 / 0` would otherwise reach the cotangents as `NaN`.

  Args:
    tau: Optical path along the propagation angle, `D * tau_absorption`, `>= 0`.
    planck_src: Planck radiance at the layer centre [W/m^2/sr].
    planck_src_top: Planck radiance at the top face of the layer.
    planck_src_bottom: Planck radiance at the bottom face of the layer.

  Returns:
    The transmittance `exp(-tau)`, the radiance emitted out of the top face of
    the layer (`source_up`) and out of its bottom face (`source_down`), in the
    units of the Planck sources.
  """
  trans = jnp.exp(-tau)
  tau_thresh = float(jnp.finfo(jnp.result_type(tau)).eps) ** 0.25
  thick = tau > tau_thresh

  # Only the division is singular (`trans` is finite for every `tau >= 0`), so
  # `trans` itself needs no safe operand.
  safe_tau = jnp.where(thick, tau, 1.0)
  fact_thick = (1.0 - trans) / safe_tau - trans
  small_tau = jnp.where(thick, 0.0, tau)
  fact_thin = small_tau * (
      0.5 + small_tau * (-1.0 / 3.0 + small_tau * (1.0 / 8.0))
  )
  fact = jnp.where(thick, fact_thick, fact_thin)

  one_minus_trans = 1.0 - trans
  source_up = one_minus_trans * planck_src_top + 2.0 * fact * (
      planck_src - planck_src_top
  )
  source_down = one_minus_trans * planck_src_bottom + 2.0 * fact * (
      planck_src - planck_src_bottom
  )
  return trans, source_up, source_down


def lw_noscat_transport(
    optical_depth: Array,
    ssa: Array,
    planck_src: Array,
    planck_src_top: Array,
    planck_src_bottom: Array,
    toa_flux_down: Array,
    sfc_src: Array,
    sfc_emissivity: Array,
    use_scan: bool = False,
    secant: float | Array = LW_NOSCAT_DEFAULT_SECANT,
    weight: float = LW_NOSCAT_DEFAULT_WEIGHT,
    fence_chain: bool = True,
) -> StatesMap:
  """Monochromatic longwave fluxes for an absorbing, non-scattering atmosphere.

  Transcribes `lw_solver_noscat_oneangle` for one g-point. The solver sees only
  absorption: the optical depth is `tau (1 - ssa)` of the combined gas, cloud
  and aerosol properties, the `ty_optical_props_1scl` convention of RTE and what
  ECHAM6's `mo_lrtm_solver.f90` does (no longwave scattering). Scattering is
  neither represented nor rescaled (upstream's optional Tang et al. 2018
  rescaling, `do_rescaling`, is not implemented).

  Downward, from the top of the atmosphere, then at the surface

      F_dn(bottom of layer) = T F_dn(top of layer) + source_down
      F_up(surface)         = (1 - eps) F_dn(surface) + eps pi w S_sfc

  and upward from there `F_up(top of layer) = T F_up(bottom of layer) +
  source_up`, with `T = exp(-D tau_abs)` and the layer sources of
  `lw_noscat_layer_sources` scaled by `pi * weight` (radiance to flux for an
  azimuthally isotropic field, as in upstream).

  Args:
    optical_depth: Layer optical depth of the combined properties, `[nx, ny,
      nz]` with a halo cell at each end of z.
    ssa: Single-scattering albedo of the combined properties, same shape.
    planck_src: Planck radiance at the layer centres [W/m^2/sr], same shape.
    planck_src_top: Planck radiance at the top face of each layer.
    planck_src_bottom: Planck radiance at the bottom face of each layer. Both
      face values are the level sources shared with the neighbouring layer (the
      geometric mean of the two adjacent layers' spectral mappings, as
      `monochromatic_two_stream.lw_combine_sources` forms them), which is what
      current upstream RTE passes; `solve_lw` supplies them. The solver itself
      accepts any face values, e.g. each layer's own, which is the RRTMG / RTE
      <= 1.6 / ECHAM6 `rtrnmc` convention.
    toa_flux_down: The prescribed downward flux at the top of the atmosphere
      [W/m^2], `[nx, ny]`.
    sfc_src: The surface Planck radiance, `[nx, ny]`.
    sfc_emissivity: The surface emissivity, `[nx, ny]`.
    use_scan: Whether to use scan or for loops for the recurrences.
    secant: Secant of the propagation angle. The default is the one-point
      quadrature of upstream RTE (`LW_NOSCAT_DEFAULT_SECANT`).
    weight: The quadrature weight of that angle (1 for the one-point rule).
    fence_chain: Whether the unrolled recurrences put an optimization barrier on
      their dynamic-update-slice chain (`rte_utils.recurrent_op`). The
      per-level work here is a multiply and an add, so the chain itself is the
      cost, and whether the barrier pays depends on what is fused around the
      solve (T63L47 ECHAM-RRTMGP, A100, `use_scan=False`, `gpt_chunk=4`): with
      the gas optics inside the g-point loop (`gas_optics_mode='per_gpt'`) the
      fence takes the no-scattering solve from 3138 to 502 ms/call; with it
      hoisted (`'all_gpts'`) the fence costs, 210 ms/call against 163.5 without.
      `two_stream.solve_lw` chooses accordingly; the default is the fenced
      form. No value changes. Has no effect with `use_scan=True`.

  Returns:
    A dictionary of fluxes [W/m^2] at the cell faces, laid out like
    `monochromatic_two_stream.lw_transport`: 'flux_up', 'flux_down', 'flux_net'.
  """
  # Optical depth is physically non-negative, but halo cells can carry negative
  # values (an artifact of forming molecular column amounts from
  # finite-differenced halo pressures); `exp(-D tau)` of those overflows. The
  # halos are stripped before the recurrences, so clamping changes nothing
  # forward and keeps the cotangents finite.
  tau_abs = jnp.maximum(optical_depth * (1.0 - ssa), 0.0)
  trans, source_up, source_down = lw_noscat_layer_sources(
      secant * tau_abs, planck_src, planck_src_top, planck_src_bottom
  )
  flux_scale = np.pi * weight
  source_up = flux_scale * source_up
  source_down = flux_scale * source_down

  def flux_down_op(
      flux_down_from_above: Array, trans: Array, source_down: Array
  ) -> tuple[Array, Array]:
    out = trans * flux_down_from_above + source_down
    return out, out  # Carry and output are the same.

  # `fence_chain` is passed to both recurrences. With the gas optics inside the
  # g-point loop XLA otherwise fuses the zero-pad onto the unrolled
  # dynamic-update-slice chain and re-evaluates the chain per output element
  # (3138 vs 502 ms per radiation call on an A100), and the two-stream solvers
  # measure the other way, so the fence is a per-solver, per-mode choice; see
  # the `fence_chain` argument.
  _, flux_down = rte_utils.recurrent_op_with_halos(
      flux_down_op,
      toa_flux_down,
      {'trans': trans, 'source_down': source_down},
      forward=False,
      use_scan=use_scan,
      fence_chain=fence_chain,
  )

  # The downward flux at the surface face (the bottom face of the first
  # interior layer) is reflected, and the surface adds its own emission.
  sfc_flux_down = flux_down[:, :, 1]
  sfc_flux_up = (1.0 - sfc_emissivity) * sfc_flux_down + (
      sfc_emissivity * flux_scale * sfc_src
  )

  def flux_up_op(
      flux_up_from_below: Array, trans: Array, source_up: Array
  ) -> tuple[Array, Array]:
    out = trans * flux_up_from_below + source_up
    return out, out

  _, flux_up_top = rte_utils.recurrent_op_with_halos(
      flux_up_op,
      sfc_flux_up,
      {'trans': trans, 'source_up': source_up},
      forward=True,
      use_scan=use_scan,
      fence_chain=fence_chain,
  )
  # The recurrence stores each layer's *top*-face flux in that layer's slot (and
  # the surface flux in the bottom halo slot); face `k` of the output layout is
  # the bottom face of layer `k`, which is the top face of layer `k - 1`.
  flux_up = _shift_toward_toa(flux_up_top)

  return {
      'flux_up': flux_up,
      'flux_down': flux_down,
      'flux_net': flux_up - flux_down,
  }
