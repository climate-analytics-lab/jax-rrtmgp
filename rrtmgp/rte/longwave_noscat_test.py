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

"""Tests of the no-scattering longwave solver (`rrtmgp.rte.longwave_noscat`).

The first class tests the transport kernel on synthetic columns (no lookup
tables): the conventions it shares with the two-stream solver -- radiance
versus flux, the halo layout, the surface boundary -- and the numerics of its
thin-layer branch. The second runs the whole `solve_lw(lw_solver='noscat')`
path: against the upstream RTE+RRTMGP RFMIP clear-sky reference (which that
project computes with this solver) at the 256-point k-distribution, against
the two-stream solver on the RFMIP clear-sky atmospheres, with
the shared clear-sky solve, g-point chunking, McICA and aerosol, and under
reverse-mode differentiation.
"""

import functools
from typing import TypeAlias
from unittest import mock

import unittest
from parameterized import parameterized
import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
from rrtmgp import test_util
from rrtmgp.optics import atmospheric_state
from rrtmgp.optics import optics
from rrtmgp.rte import all_sky_test
from rrtmgp.rte import clear_sky_test
from rrtmgp.rte import longwave_noscat
from rrtmgp.rte import monochromatic_two_stream
from rrtmgp.rte import two_stream

Array: TypeAlias = jax.Array

_SIGMA_OVER_PI = 5.670374419e-8 / np.pi


def _planck(temperature):
  """Broadband Planck radiance [W/m^2/sr]."""
  return _SIGMA_OVER_PI * temperature**4


def _synthetic_column(nlev: int = 12, n_horiz: int = 2, seed: int = 0):
  """A temperature profile with its face / centre sources, halos included.

  Layout as the solvers see it: `[nx, ny, nz]` with `nz = nlev + 2`, surface
  first, one halo cell at each end. Sources are the layer's own face values
  and centre value, the inputs of `lw_noscat_transport`.
  """
  rng = np.random.default_rng(seed)
  nz = nlev + 2
  t_face = 300.0 - 6.0 * np.arange(nz + 1) + rng.normal(0, 0.5, nz + 1)
  t_layer = 0.5 * (t_face[:-1] + t_face[1:])
  shape = (n_horiz, n_horiz, nz)

  def tile(v):
    return jnp.asarray(np.broadcast_to(v, shape), jnp.float32)

  return {
      'planck_src': tile(_planck(t_layer)),
      'planck_src_bottom': tile(_planck(t_face[:-1])),
      'planck_src_top': tile(_planck(t_face[1:])),
      'sfc_src': jnp.full((n_horiz, n_horiz), _planck(t_face[1] + 2.0),
                          jnp.float32),
      'shape': shape,
  }


class NoScatKernelTest(unittest.TestCase):

  def test_isothermal_equilibrium_fluxes_are_uniform(self):
    """An isothermal column over a surface at its temperature is in equilibrium.

    Every face carries `pi B` whatever the optical depth, so this pins the
    radiance-to-flux factor `pi`, the surface boundary, and the recurrence
    layout at once. The optical depths span zero (the series branch), thin and
    optically thick.
    """
    nlev = 10
    shape = (2, 2, nlev + 2)
    b = _planck(260.0)

    def full(v):
      return jnp.full(shape, v, jnp.float32)

    tau = np.zeros(shape, np.float32)
    tau[..., 1:-1] = np.logspace(-6, 1.5, nlev)[::-1]
    tau[..., 3] = 0.0
    for emissivity in (1.0, 0.9):
      with self.subTest(emissivity=emissivity):
        fluxes = longwave_noscat.lw_noscat_transport(
            jnp.asarray(tau), jnp.zeros(shape, jnp.float32),
            full(b), full(b), full(b),
            jnp.full(shape[:2], np.pi * b, jnp.float32),
            jnp.full(shape[:2], b, jnp.float32),
            jnp.full(shape[:2], emissivity, jnp.float32),
        )
        # Faces 1 .. nlev + 1 are the surface .. top of atmosphere.
        for key in ('flux_up', 'flux_down'):
          np.testing.assert_allclose(
              np.asarray(fluxes[key])[..., 1:], np.pi * b, rtol=2e-6,
              err_msg=key,
          )
        np.testing.assert_allclose(
            np.asarray(fluxes['flux_net'])[..., 1:], 0.0, atol=2e-5 * np.pi * b
        )

  def test_transparent_atmosphere(self):
    """No absorber: the incident flux reaches the surface, and the surface
    returns its reflection plus its own emission to space."""
    col = _synthetic_column()
    shape = col['shape']
    toa, emissivity = 7.0, 0.8
    fluxes = longwave_noscat.lw_noscat_transport(
        jnp.zeros(shape), jnp.zeros(shape), col['planck_src'],
        col['planck_src_top'], col['planck_src_bottom'],
        jnp.full(shape[:2], toa), col['sfc_src'],
        jnp.full(shape[:2], emissivity),
    )
    flux_down = np.asarray(fluxes['flux_down'])
    flux_up = np.asarray(fluxes['flux_up'])
    np.testing.assert_allclose(flux_down[..., 1:], toa, rtol=1e-6)
    expected_up = (1 - emissivity) * toa + emissivity * np.pi * np.asarray(
        col['sfc_src']
    )
    np.testing.assert_allclose(
        flux_up[..., 1:], expected_up[..., None] * np.ones(shape[2] - 1),
        rtol=1e-6,
    )

  def test_layout_matches_two_stream_transport(self):
    """Halo entries are zero and the TOA face is the last index, as in
    `monochromatic_two_stream.lw_transport`."""
    col = _synthetic_column()
    shape = col['shape']
    tau = jnp.full(shape, 0.3)
    toa = 5.0
    fluxes = longwave_noscat.lw_noscat_transport(
        tau, jnp.zeros(shape), col['planck_src'], col['planck_src_top'],
        col['planck_src_bottom'], jnp.full(shape[:2], toa), col['sfc_src'],
        jnp.full(shape[:2], 0.95),
    )
    self.assertEqual(set(fluxes), {'flux_up', 'flux_down', 'flux_net'})
    for key in ('flux_up', 'flux_down', 'flux_net'):
      self.assertEqual(fluxes[key].shape, shape)
    np.testing.assert_array_equal(np.asarray(fluxes['flux_up'])[..., 0], 0.0)
    np.testing.assert_array_equal(np.asarray(fluxes['flux_down'])[..., 0], 0.0)
    # The prescribed boundary value sits at the last face, exactly.
    np.testing.assert_array_equal(
        np.asarray(fluxes['flux_down'])[..., -1], toa
    )
    np.testing.assert_allclose(
        np.asarray(fluxes['flux_net']),
        np.asarray(fluxes['flux_up'] - fluxes['flux_down']),
    )

  def test_matches_two_stream_without_scattering(self):
    """With ssa = 0 and the two-stream solver's own inputs, the two agree.

    The two-stream solver at ssa = 0 is absorption with diffusivity 1.66 and a
    Planck function linear in optical depth between the two face sources. The
    no-scattering transcription reduces to exactly that when its layer-centre
    source is the mean of the face sources (the `2 f (S_layer - S_face)` term
    then becomes the linear-in-tau slope) and the face sources are the same
    two-stream ones. This is the check on the radiance/flux factor, the
    layout, the surface boundary and the `fact` weighting taken together; it
    does not hold for a centre value that differs from the face mean, which is
    the (intended) difference between the schemes.
    """
    col = _synthetic_column(nlev=20, seed=3)
    shape = col['shape']
    rng = np.random.default_rng(1)
    tau = jnp.asarray(10.0 ** rng.uniform(-5, 0.7, shape), jnp.float32)
    ssa = jnp.zeros(shape, jnp.float32)
    combined = monochromatic_two_stream.lw_combine_sources({
        'planck_src_top': col['planck_src_top'],
        'planck_src_bottom': col['planck_src_bottom'],
    })
    top, bottom = combined['planck_src_top'], combined['planck_src_bottom']
    toa = jnp.full(shape[:2], 4.0)
    emissivity = jnp.full(shape[:2], 0.93)

    cell = monochromatic_two_stream.lw_cell_source_and_properties(
        tau, ssa, bottom, top, ssa
    )
    want = monochromatic_two_stream.lw_transport(
        cell['t_diff'], cell['r_diff'], cell['src_up'], cell['src_down'], toa,
        col['sfc_src'], emissivity,
    )
    got = longwave_noscat.lw_noscat_transport(
        tau, ssa, 0.5 * (top + bottom), top, bottom, toa, col['sfc_src'],
        emissivity, secant=1.66,
    )
    for key in ('flux_up', 'flux_down', 'flux_net'):
      np.testing.assert_allclose(
          np.asarray(got[key]), np.asarray(want[key]), rtol=3e-4, atol=2e-3,
          err_msg=key,
      )

  def test_layer_sources_match_float64_reference(self):
    """The layer sources over the full range of optical path, thin to thick.

    Straddles `tau_thresh = eps**(1/4)` (0.0186 in float32), where the closed
    form `(1 - T)/tau - T` hands over to the series: both sides must be
    accurate and the handover continuous.
    """
    # The reference sees exactly the float32 optical paths the solver does.
    thresh = float(np.finfo(np.float32).eps) ** 0.25
    tau32 = np.concatenate([
        [0.0], np.logspace(-9, 2, 400),
        # Either side of the handover to the series, a hair apart.
        thresh * np.array([0.9999, 1.0001]),
    ]).astype(np.float32)
    tau64 = tau32.astype(np.float64)
    s_layer, s_top, s_bottom = 1.0, 0.85, 1.2
    t64 = np.exp(-tau64)
    with np.errstate(invalid='ignore', divide='ignore'):
      fact = np.where(tau64 > 0, -np.expm1(-tau64) / tau64 - t64, 0.0)
    want_up = (-np.expm1(-tau64)) * s_top + 2 * fact * (s_layer - s_top)
    want_down = (-np.expm1(-tau64)) * s_bottom + 2 * fact * (s_layer - s_bottom)

    one = jnp.ones_like(tau64, dtype=jnp.float32)
    trans, up, down = longwave_noscat.lw_noscat_layer_sources(
        jnp.asarray(tau32), s_layer * one, s_top * one,
        s_bottom * one,
    )
    np.testing.assert_allclose(np.asarray(trans), t64, rtol=1e-6, atol=1e-30)
    # Relative error of the source, with an absolute floor of a few float32
    # ulp of the Planck scale: `1 - T` is exactly 0 below tau ~ 6e-8, so the
    # leading O(tau) term of a vanishingly thin layer's emission is below
    # float32 resolution of the flux it is added to (as upstream's, which
    # forms `1 - T` the same way).
    for got, want in ((up, want_up), (down, want_down)):
      err = np.abs(np.asarray(got) - want)
      self.assertTrue(np.all(err <= 2e-3 * np.abs(want) + 1e-7),
                      msg=f'worst: {err.max()}')

  def test_scan_matches_unrolled(self):
    col = _synthetic_column()
    shape = col['shape']
    args = (
        jnp.full(shape, 0.2), jnp.zeros(shape), col['planck_src'],
        col['planck_src_top'], col['planck_src_bottom'],
        jnp.full(shape[:2], 3.0), col['sfc_src'], jnp.full(shape[:2], 0.97),
    )
    unrolled = longwave_noscat.lw_noscat_transport(*args, use_scan=False)
    scanned = longwave_noscat.lw_noscat_transport(*args, use_scan=True)
    for key in unrolled:
      np.testing.assert_allclose(
          np.asarray(scanned[key]), np.asarray(unrolled[key]), rtol=1e-6,
          atol=1e-4,
      )

  def test_fence_chain_changes_the_program_not_the_values(self):
    """`fence_chain` adds one optimization barrier per unrolled recurrence (the
    downward and the upward one) on top of the barrier every halo recurrence
    ends with, and no value changes."""
    col = _synthetic_column()
    shape = col['shape']
    args = (
        jnp.full(shape, 0.2), jnp.zeros(shape), col['planck_src'],
        col['planck_src_top'], col['planck_src_bottom'],
        jnp.full(shape[:2], 3.0), col['sfc_src'], jnp.full(shape[:2], 0.97),
    )

    def n_barriers(fence_chain, use_scan=False):
      jaxpr = jax.make_jaxpr(
          lambda *a: longwave_noscat.lw_noscat_transport(
              *a, use_scan=use_scan, fence_chain=fence_chain
          )
      )(*args)
      return str(jaxpr).count('optimization_barrier')

    self.assertEqual(n_barriers(True), n_barriers(False) + 2)
    # The scanned recurrence has no update chain to fence.
    self.assertEqual(n_barriers(True, True), n_barriers(False, True))
    fenced = longwave_noscat.lw_noscat_transport(*args, fence_chain=True)
    unfenced = longwave_noscat.lw_noscat_transport(*args, fence_chain=False)
    for key in fenced:
      np.testing.assert_array_equal(
          np.asarray(fenced[key]), np.asarray(unfenced[key])
      )

  def test_gradients_are_finite_where_the_forward_pass_is_singular(self):
    """Zero optical depth (the series branch), `ssa = 1` exactly (so the
    absorption optical depth is zero), and negative halo optical depth must not
    put a NaN or infinity on the reverse path."""
    col = _synthetic_column()
    shape = col['shape']
    tau = np.full(shape, 0.4, np.float32)
    ssa = np.zeros(shape, np.float32)
    tau[..., 2] = 0.0          # clear, gas-free layer
    tau[..., 4] = 1e-12        # below the series threshold
    ssa[..., 5] = 1.0          # conservative scatterer: no absorption
    tau[..., 0] = -3.0         # a halo cell can carry a negative optical depth
    tau[..., -1] = -1.0

    def loss(tau, ssa, planck, top, bottom, sfc_src):
      f = longwave_noscat.lw_noscat_transport(
          tau, ssa, planck, top, bottom, jnp.full(shape[:2], 2.0), sfc_src,
          jnp.full(shape[:2], 0.97),
      )
      return jnp.sum(f['flux_net'][..., 1:]) + jnp.sum(f['flux_up'][..., -1])

    args = (jnp.asarray(tau), jnp.asarray(ssa), col['planck_src'],
            col['planck_src_top'], col['planck_src_bottom'], col['sfc_src'])
    for use_jit in (False, True):
      fn = jax.jit(jax.grad(loss, argnums=tuple(range(6)))) if use_jit else (
          jax.grad(loss, argnums=tuple(range(6))))
      grads = fn(*args)
      for name, g in zip(('tau', 'ssa', 'planck', 'top', 'bottom', 'sfc'),
                         grads):
        self.assertTrue(np.all(np.isfinite(np.asarray(g))), msg=name)

  def test_gradient_matches_finite_difference(self):
    """Reverse mode against a central difference, in float64."""
    with jax.enable_x64(True):
      col = _synthetic_column(nlev=8, seed=5)
      shape = col['shape']

      def cast(a):
        return jnp.asarray(a, jnp.float64)

      planck, top, bottom, sfc = (cast(col[k]) for k in (
          'planck_src', 'planck_src_top', 'planck_src_bottom', 'sfc_src'))
      rng = np.random.default_rng(2)
      tau0 = cast(10.0 ** rng.uniform(-3, 0.3, shape))
      ssa = cast(np.zeros(shape))

      def loss(scale):
        f = longwave_noscat.lw_noscat_transport(
            scale * tau0, ssa, planck, top, bottom,
            jnp.full(shape[:2], 1.0, jnp.float64), sfc,
            jnp.full(shape[:2], 0.9, jnp.float64),
        )
        return jnp.sum(f['flux_net'][..., 1:] ** 2) + jnp.sum(
            f['flux_up'][..., -1])

      h = 1e-6
      fd = (loss(1.0 + h) - loss(1.0 - h)) / (2 * h)
      ad = jax.grad(loss)(1.0)
      np.testing.assert_allclose(float(ad), float(fd), rtol=1e-6)


class NoScatSolveTest(unittest.TestCase):
  """`solve_lw(lw_solver='noscat')` end to end."""

  _fixture = None

  @classmethod
  def _cloudy(cls):
    if cls._fixture is None:
      cls._fixture = all_sky_test._compact_cloudy_setup()
    return cls._fixture

  @staticmethod
  def _rfmip_site(site: int, compact: bool = True):
    """One RFMIP clear-sky column batch, as `clear_sky_test` builds it.

    `compact` selects the 128 / 112-point k-distributions, otherwise the full
    256 / 224-point ones.
    """
    (ds, pressure, pressure_level, temperature, _, h2o, o3, sfc_t) = (
        clear_sky_test._setup_atmospheric_profiles()
    )
    params = clear_sky_test._setup_radiation_params(site, compact, ds)
    atmos_state = atmospheric_state.from_config(params.atmospheric_state_cfg)
    optics_lib = optics.optics_factory(params.optics, atmos_state.vmr)
    conv = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=2
    )
    h2 = conv(h2o[0, site])
    return dict(
        pressure=conv(pressure[site]),
        temperature=conv(temperature[0, site]),
        molecules=clear_sky_test._air_molecules_per_area(
            conv(pressure_level[site]), h2
        ),
        optics_lib=optics_lib,
        atmos_state=atmos_state,
        vmr_fields={'h2o': h2, 'o3': conv(o3[0, site])},
        sfc_temperature=sfc_t[0, site] * jnp.ones((2, 2), jnp.float32),
    )

  def _solve(self, fx, **kwargs):
    return two_stream.solve_lw(
        fx['pressure'], fx['temperature'], fx['molecules'], fx['optics_lib'],
        fx['atmos_state'], fx['vmr_fields'], fx['sfc_temperature'], **kwargs
    )

  # The clear-sky RFMIP reference is what upstream RTE+RRTMGP computes with this
  # very solver (its `rte_lw` uses the no-scattering transport, the one-point
  # Gauss-Jacobi-5 quadrature, and the g256 k-distribution), shipped as
  # `clearsky_lw_flux_{up,dn}_NoScat.nc` (rrtmgp-data `examples/rfmip-clear-sky/
  # reference/rl{u,d}_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc`, unchanged).
  # The largest deviations are in the downward flux at the surface of the
  # moistest columns (1.7 W/m^2 at site 7, 0.6 W/m^2 in the upward flux), so the
  # all-face tolerance is set by those, and the top-of-atmosphere upward flux,
  # which is the OLR, is pinned separately and much tighter.
  @parameterized.expand([(site,) for site in range(10)])
  def test_clear_sky_matches_the_upstream_rte_reference(self, site):
    fx = self._rfmip_site(site, compact=False)
    got = self._solve(fx, lw_solver='noscat')

    def reference(filename, variable):
      ref = nc.Dataset(f'rrtmgp/optics/test_data/{filename}')[variable][:].data
      # Surface-first, with the bottom halo and the TOA face, like the solve.
      return np.pad(np.flip(ref, axis=-1), ((0, 0), (0, 0), (1, 0)))[0, site]

    ref_up = reference('clearsky_lw_flux_up_NoScat.nc', 'rlu')[1:]
    ref_dn = reference('clearsky_lw_flux_dn_NoScat.nc', 'rld')[1:]
    up = np.asarray(got['flux_up'])[0, 0, 1:]
    dn = np.asarray(got['flux_down'])[0, 0, 1:]
    np.testing.assert_allclose(up, ref_up, rtol=2.5e-3, atol=0.1)
    np.testing.assert_allclose(dn, ref_dn, rtol=6e-3, atol=0.1)
    np.testing.assert_allclose(up[-1], ref_up[-1], rtol=0, atol=0.05)

  # Site 0 is a typical column; site 6 has a 7.5 K inversion across one layer.
  @parameterized.expand([(0,), (6,)])
  def test_gas_only_column_matches_two_stream_with_matching_sources(self, site):
    """Clear sky, secant 1.66: the no-scattering solver is the two-stream one.

    The two-stream solver of an absorbing column is no-scattering transport
    with diffusivity 1.66, provided the Planck function of a layer is the same
    in both. The two-stream solver only sees the two face sources of each
    layer (combined across the face), so this test hands the no-scattering
    solver the same ones: the layer-centre value is the mean of the combined
    faces. Every face then agrees to rounding -- this is the end-to-end check
    of the unit and layout conventions.
    """
    fx = self._rfmip_site(site)
    want = self._solve(fx)

    original = type(fx['optics_lib']).compute_planck_sources

    def two_stream_sources(self, *args, **kwargs):
      src = dict(original(self, *args, **kwargs))
      combined = monochromatic_two_stream.lw_combine_sources(src)
      src['planck_src_top'] = combined['planck_src_top']
      src['planck_src_bottom'] = combined['planck_src_bottom']
      src['planck_src'] = 0.5 * (src['planck_src_top']
                                 + src['planck_src_bottom'])
      return src

    with mock.patch.object(
        type(fx['optics_lib']), 'compute_planck_sources', two_stream_sources
    ):
      got = self._solve(fx, lw_solver='noscat', lw_secant=1.66)
    for key in ('flux_up', 'flux_down', 'flux_net'):
      np.testing.assert_allclose(
          np.asarray(got[key])[:, :, 1:], np.asarray(want[key])[:, :, 1:],
          atol=0.02, err_msg=key,
      )

  @parameterized.expand([(0, 0.7), (6, 2.6)])
  def test_gas_only_column_is_close_to_two_stream_as_implemented(
      self, site, tolerance
  ):
    """With the layer-centre Planck source the solver is meant to use, the
    residual to the two-stream solver is the temperature curvature inside each
    layer (`B(T_layer)` against the mean of its face values), which the
    two-stream source function does not see. It is largest where the profile
    kinks, which is what site 6's inversion layer does (a localised ~2 W/m^2
    shift that then rides up the column)."""
    fx = self._rfmip_site(site)
    want = self._solve(fx)
    got = self._solve(fx, lw_solver='noscat', lw_secant=1.66)
    for key in ('flux_up', 'flux_down'):
      diff = np.abs(np.asarray(got[key]) - np.asarray(want[key]))[:, :, 1:]
      self.assertLess(diff.max(), tolerance, msg=key)

  @parameterized.expand([('per_gpt', True), ('all_gpts', False)])
  def test_solve_fences_the_update_chain_only_with_in_loop_gas_optics(
      self, gas_optics_mode, expect_fence
  ):
    """The caller picks the fence from the gas-optics mode (see
    `lw_noscat_transport`): fenced with the gas optics inside the g-point loop,
    unfenced with it hoisted. Both give the same fluxes."""
    fx = self._rfmip_site(0)
    real = longwave_noscat.lw_noscat_transport
    with mock.patch.object(
        longwave_noscat, 'lw_noscat_transport', wraps=real
    ) as spy:
      got = self._solve(
          fx, lw_solver='noscat', gas_optics_mode=gas_optics_mode,
          gpt_chunk=4,
      )
    self.assertGreater(spy.call_count, 0)
    for call in spy.call_args_list:
      self.assertEqual(call.kwargs['fence_chain'], expect_fence)
    # Same values as the other mode.
    other = 'all_gpts' if gas_optics_mode == 'per_gpt' else 'per_gpt'
    want = self._solve(
        fx, lw_solver='noscat', gas_optics_mode=other, gpt_chunk=4
    )
    for key in ('flux_up', 'flux_down', 'flux_net'):
      np.testing.assert_allclose(
          np.asarray(got[key]), np.asarray(want[key]), rtol=1e-4, atol=1e-3,
          err_msg=key,
      )

  def test_default_solver_is_two_stream(self):
    fx = self._rfmip_site(0)
    default = self._solve(fx)
    explicit = self._solve(fx, lw_solver='two_stream')
    for key in default:
      np.testing.assert_array_equal(
          np.asarray(default[key]), np.asarray(explicit[key])
      )

  def test_unknown_solver_is_rejected(self):
    fx = self._rfmip_site(0)
    with self.assertRaisesRegex(ValueError, 'lw_solver'):
      self._solve(fx, lw_solver='delta_eddington')

  def test_secant_is_a_parameter(self):
    """A longer slant path absorbs more: the surface-down flux rises."""
    fx = self._rfmip_site(0)
    short = self._solve(fx, lw_solver='noscat', lw_secant=1.4)
    long_ = self._solve(fx, lw_solver='noscat', lw_secant=1.8)
    self.assertGreater(
        float(long_['flux_down'][0, 0, 1]), float(short['flux_down'][0, 0, 1])
    )
    self.assertLess(
        float(long_['flux_up'][0, 0, -1]), float(short['flux_up'][0, 0, -1])
    )

  def _cloud_kwargs(self, with_mcica, with_aerosol):
    (optics_lib, _, _, temperature, *_, cloud) = self._cloudy()
    helper = all_sky_test.GPointChunkingTest()
    helper._cached = self._cloudy()
    extra = helper._mcica_and_aerosol(optics_lib, 'lw', cloud, temperature.shape)
    kwargs = dict(cloud)
    if with_mcica:
      kwargs['cloud_path_liq_per_gpt'] = extra['cloud_path_liq_per_gpt']
      kwargs['cloud_path_ice_per_gpt'] = extra['cloud_path_ice_per_gpt']
    if with_aerosol:
      kwargs['aerosol_optics'] = extra['aerosol_optics']
    no_cloud = {k: v for k, v in kwargs.items() if 'cloud' not in k}
    return kwargs, no_cloud

  def _cloudy_solve(self, **kwargs):
    (optics_lib, atmos_state, p, temperature, molecules, vmr_fields,
     sfc_temperature, _) = self._cloudy()
    temperature = kwargs.pop('temperature', temperature)
    return two_stream.solve_lw(
        p, temperature, molecules, optics_lib, atmos_state, vmr_fields,
        sfc_temperature, lw_solver='noscat', **kwargs
    )

  @parameterized.expand([
      (chunk, with_mcica, with_aerosol)
      for chunk, with_mcica, with_aerosol in (
          (1, False, False), (4, False, True), (1, True, True), (4, True, False)
      )
  ])
  def test_clear_sky_is_shared_and_both_transports_are_noscat(
      self, chunk, with_mcica, with_aerosol
  ):
    """`with_clear_sky=True` returns what a cloud-free call returns, and leaves
    the all-sky fluxes as they were, for every cloud / aerosol path."""
    kwargs, no_cloud = self._cloud_kwargs(with_mcica, with_aerosol)
    shared = self._cloudy_solve(gpt_chunk=chunk, with_clear_sky=True, **kwargs)
    all_sky_shared, clear_shared = two_stream.split_clear_sky_fluxes(shared)
    all_sky = self._cloudy_solve(gpt_chunk=chunk, **kwargs)
    clear = self._cloudy_solve(gpt_chunk=chunk, **no_cloud)
    for name, got, want in (('all-sky', all_sky_shared, all_sky),
                            ('clear-sky', clear_shared, clear)):
      for key in want:
        np.testing.assert_allclose(
            np.asarray(got[key]), np.asarray(want[key]), rtol=2e-6, atol=1e-4,
            err_msg=f'{name} {key}',
        )
    # The cloud must matter or this proves nothing about the clear branch.
    diff = np.abs(np.asarray(all_sky['flux_up']) - np.asarray(clear['flux_up']))
    self.assertGreater(diff.max(), 1.0)

  @parameterized.expand([(4, False), (4, True), (3, True)])
  def test_gpt_chunk_matches_unchunked(self, chunk, with_mcica):
    """Chunking regroups independent g-points; 3 does not divide the g-point
    count, so it exercises the padded tail."""
    kwargs, _ = self._cloud_kwargs(with_mcica, True)
    ref = self._cloudy_solve(gpt_chunk=1, **kwargs)
    got = self._cloudy_solve(gpt_chunk=chunk, **kwargs)
    for key in ref:
      np.testing.assert_allclose(
          np.asarray(got[key]), np.asarray(ref[key]), rtol=1e-5, atol=1e-4,
          err_msg=key,
      )

  def test_scattering_is_dropped(self):
    """The solver sees the absorption optical depth `tau (1 - ssa)` only.

    The combined optical properties mix additively in `tau` and `tau * ssa`,
    so an aerosol with `ssa = 1` adds no absorption however thick it is and
    must leave the fluxes unchanged; the same optical depth as a pure absorber
    (`ssa = 0`) must not. (The two-stream solver reflects with the scatterer.)
    """
    (optics_lib, *_) = self._cloudy()
    shape = self._cloudy()[3].shape
    n_bnd = optics_lib.gas_optics_lw.n_bnd
    scatterer = {
        'optical_depth': jnp.full((n_bnd,) + shape, 0.5, jnp.float32),
        'ssa': jnp.ones((n_bnd,) + shape, jnp.float32),
        'asymmetry_factor': jnp.full((n_bnd,) + shape, 0.5, jnp.float32),
    }
    absorber = dict(scatterer, ssa=jnp.zeros_like(scatterer['ssa']))
    bare = self._cloudy_solve(gpt_chunk=4)
    with_scatterer = self._cloudy_solve(gpt_chunk=4, aerosol_optics=scatterer)
    with_absorber = self._cloudy_solve(gpt_chunk=4, aerosol_optics=absorber)
    for key in ('flux_up', 'flux_down'):
      np.testing.assert_allclose(
          np.asarray(with_scatterer[key]), np.asarray(bare[key]), rtol=1e-5,
          atol=1e-3, err_msg=key,
      )
    self.assertGreater(
        np.abs(np.asarray(with_absorber['flux_up'])
               - np.asarray(bare['flux_up'])).max(), 1.0
    )

  @parameterized.expand([(1,), (4,)])
  def test_gradients_are_finite_with_cloud_and_aerosol(self, chunk):
    """Reverse mode through the whole solve, w.r.t. temperature, for a scalar
    of `flux_net` and one of the TOA `flux_up`, with McICA clouds and aerosol."""
    kwargs, _ = self._cloud_kwargs(True, True)
    temperature = self._cloudy()[3]

    def net(temp):
      out = self._cloudy_solve(temperature=temp, gpt_chunk=chunk, **kwargs)
      return jnp.sum(out['flux_net'][:, :, 1:] ** 2) * 1e-4

    def toa_up(temp):
      out = self._cloudy_solve(temperature=temp, gpt_chunk=chunk, **kwargs)
      return jnp.sum(out['flux_up'][:, :, -1])

    for name, fn in (('flux_net', net), ('toa flux_up', toa_up)):
      grad = np.asarray(jax.grad(fn)(temperature))
      self.assertTrue(np.all(np.isfinite(grad)), msg=name)
      self.assertGreater(np.abs(grad).max(), 0.0, msg=name)


if __name__ == '__main__':
  unittest.main()
