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

"""End-to-end tests for `RRTMGP.compute_heating_rate`."""

import dataclasses
import functools
from pathlib import Path
from typing import TypeAlias

import unittest
import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
from parameterized import parameterized
from rrtmgp import backend
from rrtmgp import constants
from rrtmgp import rrtmgp
from rrtmgp import rrtmgp_common
from rrtmgp import test_util
from rrtmgp.config import radiative_transfer
from rrtmgp.optics import lookup_volume_mixing_ratio
from rrtmgp.rte import two_stream

Array: TypeAlias = jax.Array

_VMR_GLOBAL_MEAN_FILENAME = 'rrtmgp/optics/test_data/rcemip_global_mean_vmr.json'
_VMR_SOUNDING_FILENAME = 'rrtmgp/optics/test_data/rcemip_vmr_sounding.csv'
_ATMOSPHERIC_STATE_FILENAME = 'rrtmgp/optics/test_data/cloudysky_as.nc'
_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g128.nc'
_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-sw-g112.nc'
_CLD_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/cloudysky_lw.nc'
_CLD_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/cloudysky_sw.nc'

root = Path()


def _build_radiative_transfer_cfg() -> radiative_transfer.RadiativeTransfer:
  """Build a `RadiativeTransfer` config wired to the RCEMIP test data."""
  return radiative_transfer.RadiativeTransfer(
      optics=radiative_transfer.OpticsParameters(
          optics=radiative_transfer.RRTMOptics(
              longwave_nc_filepath=root / _LW_LOOKUP_TABLE_FILENAME,
              shortwave_nc_filepath=root / _SW_LOOKUP_TABLE_FILENAME,
              cloud_longwave_nc_filepath=root / _CLD_LW_LOOKUP_TABLE_FILENAME,
              cloud_shortwave_nc_filepath=root / _CLD_SW_LOOKUP_TABLE_FILENAME,
          )
      ),
      atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
          sfc_emis=0.98,
          sfc_alb=0.06,
          zenith=0.535526654,
          irrad=1360.8585174,
          toa_flux_lw=0.0,
          vmr_global_mean_filepath=root / _VMR_GLOBAL_MEAN_FILENAME,
          vmr_sounding_filepath=root / _VMR_SOUNDING_FILENAME,
      ),
  )


def _build_inputs(
    site: int = 0, n_horiz: int = 2
) -> dict[str, Array]:
  """Build a 3D input bundle for `RRTMGP.compute_heating_rate` from
  `cloudysky_as.nc`. Cloud condensate is zeroed so the test isolates the
  gas-VMR pathway.
  """
  ds = nc.Dataset(root / _ATMOSPHERIC_STATE_FILENAME, 'r')

  halo_width = 1
  paddings_2d = ((0, 0), (halo_width, halo_width))
  pressure = np.pad(
      np.transpose(ds['p_lay'][:].data), paddings_2d, mode='edge'
  )

  temp_internal = np.transpose(ds['t_lay'][:].data)
  temp_level = np.transpose(ds['t_lev'][:].data)
  nx, nz = temp_internal.shape
  nz_with_halos = nz + 2 * halo_width
  temperature = np.zeros((nx, nz_with_halos), dtype=jnp.float_)
  temperature[:, halo_width:-halo_width] = temp_internal
  temperature[:, 0] = 2 * temp_level[:, 0] - temp_internal[:, 0]
  temperature[:, -1] = 2 * temp_level[:, -1] - temp_internal[:, -1]

  vmr_h2o_profile = np.pad(
      np.transpose(ds['vmr_h2o'][:].data), paddings_2d, mode='edge'
  )
  sfc_temp_1d = ds['t_sfc'][:].data

  convert_to_3d = functools.partial(
      test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=n_horiz
  )
  p_ref_xxc = convert_to_3d(pressure[site, :])
  temperature_3d = convert_to_3d(temperature[site, :])
  vmr_h2o = convert_to_3d(vmr_h2o_profile[site, :])

  # Invert _humidity_to_volume_mixing_ratio to back out a consistent q_t (with
  # q_c = 0 since this is a clear-sky setup). vmr_h2o = mol_ratio * q_v / (1 -
  # q_t) with q_v = q_t, so q_t = vmr_h2o / (vmr_h2o + mol_ratio).
  mol_ratio = constants.R_V / constants.R_D
  q_t = vmr_h2o / (vmr_h2o + mol_ratio)
  zeros = jnp.zeros_like(q_t)

  # Density via ideal gas law, used for the cloud-path scaling (zeros here, so
  # the actual value is irrelevant — keep it physically sensible).
  rho = p_ref_xxc / (constants.R_D * temperature_3d)

  sfc_temperature = sfc_temp_1d[site] * jnp.ones(
      (n_horiz, n_horiz), dtype=jnp.float_
  )
  return {
      'rho_xxc': rho,
      'q_t': q_t,
      'q_liq': zeros,
      'q_ice': zeros,
      'q_c': zeros,
      'cloud_r_eff_liq': zeros,
      'cloud_r_eff_ice': zeros,
      'temperature': temperature_3d,
      'sfc_temperature': sfc_temperature,
      'p_ref_xxc': p_ref_xxc,
      'sg_map': {},
  }


class AirMoleculesPerAreaTest(unittest.TestCase):
  """Pins the production molecular-burden routine to the hydrostatic column.

  Regression test for climate-analytics-lab/jax-rrtmgp#26: the routine used
  `centered_difference`, which spans two layers, so every layer's molecular
  burden -- and hence every gas optical depth, in both streams -- came out
  twice too large. The clear-sky and all-sky tests never caught it because
  they compute molecules with their own interface-based helper rather than
  calling `_air_molecules_per_area`.
  """

  def test_column_burden_matches_hydrostatic_value(self):
    # Dry isothermal column on log-spaced cell-center pressures; the summed
    # burden must equal the hydrostatic p_sfc / (g * m_air).
    p_centers = np.geomspace(1.0e5, 1.0e2, 64)
    p_xxc = p_centers[np.newaxis, np.newaxis, :]
    vmr_h2o_xxc = np.zeros_like(p_xxc)

    molecules = np.asarray(
        rrtmgp._air_molecules_per_area(  # pylint: disable=protected-access
            jnp.asarray(p_xxc), jnp.asarray(vmr_h2o_xxc)
        )
    )

    # The centered difference is undefined on the end cells; over the interior
    # the half-thicknesses telescope to half the pressure drop across the two
    # cells at each end. A missing factor of 0.5 shows up here as a ratio of 2.
    interior = molecules[0, 0, 1:-1]
    expected = (
        0.5
        * (p_centers[0] + p_centers[1] - p_centers[-2] - p_centers[-1])
        / constants.G
        * constants.AVOGADRO
        / constants.DRY_AIR_MOL_MASS
    )
    self.assertAlmostEqual(interior.sum() / expected, 1.0, places=6)


class RRTMGPVMROverrideTest(unittest.TestCase):
  """Regression test for the `vmr_fields` override kwarg.

  Mirrors the failure mode reported in climate-analytics-lab/jax-gcm#483:
  before this kwarg existed, an upstream-supplied per-cell ozone profile was
  silently ignored and the library's pressure-reconstructed default was used
  instead. This test perturbs ozone by 10x via the new kwarg and asserts that
  the shortwave heating rate actually responds.
  """

  def test_ozone_override_changes_shortwave_heating(self):
    cfg = _build_radiative_transfer_cfg()
    rt = rrtmgp.RRTMGP(cfg, dz=500.0)
    inputs = _build_inputs()

    baseline = rt.compute_heating_rate(**inputs)

    o3_default = (
        lookup_volume_mixing_ratio.reconstruct_vmr_fields_from_pressure(
            rt.atmospheric_state.vmr, inputs['p_ref_xxc']
        )['o3']
    )
    perturbed = rt.compute_heating_rate(
        **inputs, vmr_fields={'o3': 10.0 * o3_default}
    )

    baseline_hr = baseline[rrtmgp_common.KEY_STORED_RADIATION]
    perturbed_hr = perturbed[rrtmgp_common.KEY_STORED_RADIATION]

    # Strip halos (top + bottom) before comparing, matching how downstream
    # consumers use the field.
    diff = np.asarray(perturbed_hr[:, :, 1:-1] - baseline_hr[:, :, 1:-1])
    max_abs_diff = np.max(np.abs(diff))
    # Pre-fix, the override was a no-op and this difference was bit-exact zero.
    # 1e-6 K/s ~ 0.09 K/day, well above noise but well below stratospheric SW
    # ozone-heating scales.
    self.assertGreater(
        max_abs_diff,
        1e-6,
        msg=(
            'Scaling ozone 10x via vmr_fields did not change the heating '
            f'rate (max |diff| = {max_abs_diff:.3e} K/s). The vmr_fields '
            'override is not being threaded through compute_heating_rate.'
        ),
    )

  def test_matching_ozone_override_matches_baseline(self):
    """Passing the reconstructed o3 as an override is a no-op."""
    cfg = _build_radiative_transfer_cfg()
    rt = rrtmgp.RRTMGP(cfg, dz=500.0)
    inputs = _build_inputs()

    baseline = rt.compute_heating_rate(**inputs)

    o3_default = (
        lookup_volume_mixing_ratio.reconstruct_vmr_fields_from_pressure(
            rt.atmospheric_state.vmr, inputs['p_ref_xxc']
        )['o3']
    )
    echoed = rt.compute_heating_rate(**inputs, vmr_fields={'o3': o3_default})

    np.testing.assert_array_equal(
        np.asarray(echoed[rrtmgp_common.KEY_STORED_RADIATION]),
        np.asarray(baseline[rrtmgp_common.KEY_STORED_RADIATION]),
    )


class RRTMGPCloudTauScaleTest(unittest.TestCase):
  """The `cloud_tau_scale_{liq,ice}` kwargs reach both streams (#37)."""

  def tearDown(self):
    # Each eager solve compiles and caches its own executable, tens of MB
    # apiece, and the whole suite runs in one process. Drop them so this
    # class does not add its solves to the suite's peak memory.
    jax.clear_caches()
    super().tearDown()

  def _cloudy_inputs(self):
    inputs = _build_inputs()
    p = inputs['p_ref_xxc']
    t = inputs['temperature']
    in_cloud = jnp.logical_and(p > 30000, p < 90000)
    ones = jnp.ones_like(p)
    # Mixed-phase layers between 263 K and 273 K. `q_c` is left at zero so the
    # radiative water vapour is identical when the condensate is rescaled.
    inputs['q_liq'] = jnp.where(in_cloud & (t > 263), 1e-4 * ones, 0.0)
    inputs['q_ice'] = jnp.where(in_cloud & (t < 273), 5e-5 * ones, 0.0)
    inputs['cloud_r_eff_liq'] = 1.2e-5 * ones
    inputs['cloud_r_eff_ice'] = 4e-5 * ones
    return inputs

  def test_equal_factors_match_scaled_condensate(self):
    cfg = _build_radiative_transfer_cfg()
    rt = rrtmgp.RRTMGP(cfg, dz=500.0)
    inputs = self._cloudy_inputs()
    factor = 0.5

    baseline = rt.compute_heating_rate(**inputs)
    unit = rt.compute_heating_rate(
        **inputs, cloud_tau_scale_liq=1.0, cloud_tau_scale_ice=1.0
    )
    scaled_tau = rt.compute_heating_rate(
        **inputs, cloud_tau_scale_liq=factor, cloud_tau_scale_ice=factor
    )
    scaled_q = dict(inputs)
    scaled_q['q_liq'] = factor * inputs['q_liq']
    scaled_q['q_ice'] = factor * inputs['q_ice']
    scaled_q = rt.compute_heating_rate(**scaled_q)

    for key in ('sw_flux_up_full', 'lw_flux_up_full',
                rrtmgp_common.KEY_STORED_RADIATION):
      np.testing.assert_array_equal(
          np.asarray(unit[key]), np.asarray(baseline[key]), err_msg=key
      )
      np.testing.assert_allclose(
          np.asarray(scaled_tau[key]),
          np.asarray(scaled_q[key]),
          rtol=1e-5,
          atol=1e-6 if key == rrtmgp_common.KEY_STORED_RADIATION else 1e-3,
          err_msg=key,
      )
    # Both streams respond to the factors.
    for key in ('sw_flux_up_full', 'lw_flux_up_full'):
      diff = np.abs(np.asarray(scaled_tau[key] - baseline[key]))
      self.assertGreater(np.max(diff), 0.1, msg=key)


class RRTMGPSurfaceAlbedoTest(unittest.TestCase):
  """The `sfc_alb_{dir,dif}` kwargs and the direct surface flux output (#38)."""

  def tearDown(self):
    # Each eager solve compiles and caches its own executable, tens of MB
    # apiece, and the whole suite runs in one process. Drop them so this
    # class does not add its solves to the suite's peak memory.
    jax.clear_caches()
    super().tearDown()

  def test_direct_and_diffuse_albedos(self):
    cfg = dataclasses.replace(
        _build_radiative_transfer_cfg(),
        do_clear_sky=True,
        save_lw_sw_heating_rates=True,
    )
    diagnostics = (
        'surf_sw_flux_down_dir_2d_xy',
        'surf_sw_flux_down_dir_clearsky_2d_xy',
    )
    rt = rrtmgp.RRTMGP(cfg, dz=500.0, diagnostic_fields=diagnostics)
    inputs = _build_inputs()
    sfc_alb = cfg.atmospheric_state_cfg.sfc_alb
    n_bnd_sw = rt.optics_lib.gas_optics_sw.n_bnd

    baseline = rt.compute_heating_rate(**inputs)
    matching = rt.compute_heating_rate(
        **inputs,
        sfc_alb_dir=sfc_alb,
        sfc_alb_dif=jnp.full((2, 2, n_bnd_sw), sfc_alb, jnp.float_),
    )
    for key, val in baseline.items():
      np.testing.assert_array_equal(
          np.asarray(matching[key]), np.asarray(val), err_msg=key
      )

    # The direct surface flux is always returned, and is part of the total.
    direct = np.asarray(baseline['sw_flux_down_dir_sfc'])
    down = np.asarray(baseline['sw_flux_down_full'][:, :, 0])
    self.assertEqual(direct.shape, down.shape)
    self.assertTrue(np.all(direct > 0.0) and np.all(direct <= down))
    np.testing.assert_array_equal(
        np.asarray(baseline['surf_sw_flux_down_dir_2d_xy']), direct
    )
    # This column is cloud free, so the clear-sky solve is the same one.
    np.testing.assert_array_equal(
        np.asarray(baseline['sw_flux_down_dir_clearsky_sfc']), direct
    )
    np.testing.assert_array_equal(
        np.asarray(baseline['surf_sw_flux_down_dir_clearsky_2d_xy']), direct
    )

    # Distinct albedos: the surface reflects each part with its own albedo,
    # in the all-sky and the clear-sky solve alike.
    a_dir, a_dif = 0.3, 0.05
    split = rt.compute_heating_rate(
        **inputs, sfc_alb_dir=a_dir, sfc_alb_dif=a_dif
    )
    for suffix in ('', '_clearsky'):
      up = np.asarray(split[f'sw_flux_up{suffix}_full'][:, :, 0])
      down = np.asarray(split[f'sw_flux_down{suffix}_full'][:, :, 0])
      direct = np.asarray(split[f'sw_flux_down_dir{suffix}_sfc'])
      np.testing.assert_allclose(
          up, a_dir * direct + a_dif * (down - direct), rtol=1e-5, err_msg=suffix
      )
    self.assertFalse(
        np.allclose(
            np.asarray(split[rrtmgp_common.KEY_STORED_RADIATION]),
            np.asarray(baseline[rrtmgp_common.KEY_STORED_RADIATION]),
        )
    )


class RRTMGPClearSkyTest(unittest.TestCase):
  """The clear-sky diagnostics share the all-sky solve's gas optics."""

  def tearDown(self):
    jax.clear_caches()
    super().tearDown()

  def _cloudy_inputs(self):
    return RRTMGPCloudTauScaleTest._cloudy_inputs(self)

  def _rt(self, **cfg_kwargs):
    cfg = dataclasses.replace(
        _build_radiative_transfer_cfg(),
        save_lw_sw_heating_rates=True,
        **cfg_kwargs,
    )
    diagnostics = (
        'toa_sw_flux_outgoing_2d_xy',
        'toa_lw_flux_outgoing_2d_xy',
        'toa_sw_flux_outgoing_clearsky_2d_xy',
        'toa_lw_flux_outgoing_clearsky_2d_xy',
        'surf_sw_flux_down_clearsky_2d_xy',
        'surf_lw_flux_down_clearsky_2d_xy',
        'rad_heat_lw_clearsky_1d_z',
    )
    return rrtmgp.RRTMGP(cfg, dz=500.0, diagnostic_fields=diagnostics)

  def test_per_call_flag_is_the_config_flag(self):
    """`clear_sky_fluxes=True` returns exactly what `do_clear_sky` does."""
    inputs = self._cloudy_inputs()
    per_call = self._rt().compute_heating_rate(**inputs, clear_sky_fluxes=True)
    configured = self._rt(do_clear_sky=True).compute_heating_rate(**inputs)
    self.assertEqual(set(per_call), set(configured))
    for key, val in configured.items():
      np.testing.assert_array_equal(
          np.asarray(per_call[key]), np.asarray(val), err_msg=key
      )

  def test_all_sky_outputs_do_not_depend_on_the_flag(self):
    inputs = self._cloudy_inputs()
    rt = self._rt()
    off = rt.compute_heating_rate(**inputs)
    on = rt.compute_heating_rate(**inputs, clear_sky_fluxes=True)
    self.assertFalse([k for k in off if 'clearsky' in k])
    self.assertTrue(
        {'sw_flux_up_clearsky_full', 'sw_flux_down_dir_clearsky_sfc',
         'rad_heat_sw_clearsky_3d', 'toa_lw_flux_outgoing_clearsky_2d_xy',
         'rad_heat_lw_clearsky_1d_z'} <= set(on)
    )
    for key, val in off.items():
      # Rounding-level: the combined program is fused differently (see
      # `all_sky_test.SharedClearSkyTest`).
      np.testing.assert_allclose(
          np.asarray(on[key]), np.asarray(val), rtol=2e-6, atol=1e-4,
          err_msg=key,
      )

  def test_clear_sky_is_the_all_sky_solve_without_condensate(self):
    """The clear-sky fluxes of a cloudy column are those of its cloud-free twin."""
    cloudy = self._cloudy_inputs()
    cloud_free = _build_inputs()
    rt = self._rt()
    got = rt.compute_heating_rate(**cloudy, clear_sky_fluxes=True)
    want = rt.compute_heating_rate(**cloud_free, clear_sky_fluxes=True)
    for stem in ('sw_flux_up', 'sw_flux_down', 'lw_flux_up', 'lw_flux_down'):
      np.testing.assert_allclose(
          np.asarray(got[f'{stem}_clearsky_full']),
          np.asarray(want[f'{stem}_full']),
          rtol=2e-6, atol=1e-4, err_msg=stem,
      )
    # ... and the cloud makes a difference to the all-sky ones.
    self.assertGreater(
        np.max(np.abs(np.asarray(got['sw_flux_up_full'])
                      - np.asarray(got['sw_flux_up_clearsky_full']))),
        1.0,
    )


class RRTMGPAerosolFreeTest(unittest.TestCase):
  """`aerosol_free_fluxes=True`: the aerosol-free twins, from the one solve."""

  def tearDown(self):
    jax.clear_caches()
    super().tearDown()

  def _inputs(self):
    """A cloudy column with a visible aerosol in both bands."""
    inputs = RRTMGPCloudTauScaleTest._cloudy_inputs(self)
    shape = inputs['p_ref_xxc'].shape
    rt = self._rt()
    def bundle(n_bnd, tau):
      ones = jnp.ones((n_bnd,) + shape, dtype=jnp.float_)
      return {'optical_depth': tau * ones, 'ssa': 0.9 * ones,
              'asymmetry_factor': 0.7 * ones}
    aerosol = dict(
        aerosol_optics_sw=bundle(rt.optics_lib.gas_optics_sw.n_bnd, 0.05),
        aerosol_optics_lw=bundle(rt.optics_lib.gas_optics_lw.n_bnd, 0.02),
    )
    return inputs, aerosol

  def _rt(self):
    cfg = dataclasses.replace(
        _build_radiative_transfer_cfg(), save_lw_sw_heating_rates=True
    )
    diagnostics = (
        'toa_sw_flux_outgoing_2d_xy',
        'toa_lw_flux_outgoing_2d_xy',
        'toa_sw_flux_outgoing_clearsky_2d_xy',
        'toa_sw_flux_outgoing_aerosolfree_2d_xy',
        'toa_lw_flux_outgoing_aerosolfree_2d_xy',
        'toa_sw_flux_outgoing_aerosolfree_clearsky_2d_xy',
        'toa_lw_flux_outgoing_aerosolfree_clearsky_2d_xy',
        'surf_sw_flux_down_aerosolfree_2d_xy',
        'rad_heat_lw_aerosolfree_1d_z',
    )
    return rrtmgp.RRTMGP(cfg, dz=500.0, diagnostic_fields=diagnostics)

  _STEMS = ('sw_flux_up', 'sw_flux_down', 'lw_flux_up', 'lw_flux_down')

  @parameterized.expand([
      ('per_gpt', dict(), None),
      # jax-gcm's production placement: hoisted gas optics, scanned
      # recurrences, 64 g-points per chunk, solved column-minor; and the
      # level-minor solve of the same placement.
      ('all_gpts', dict(gas_optics_mode='all_gpts', gpt_chunk=64,
                        use_scan=True), None),
      ('all_gpts_level_minor', dict(gas_optics_mode='all_gpts', gpt_chunk=64,
                                    use_scan=True), False),
  ])
  def test_twins_are_the_solve_without_aerosol(
      self, _, placement, column_minor
  ):
    """Each twin equals the same call with the aerosol arguments left out."""
    inputs, aerosol = self._inputs()
    rt = self._rt()
    with two_stream._column_minor(column_minor):  # pylint: disable=protected-access
      got = rt.compute_heating_rate(
          **inputs, **aerosol, clear_sky_fluxes=True,
          aerosol_free_fluxes=True, **placement,
      )
      want = rt.compute_heating_rate(
          **inputs, clear_sky_fluxes=True, **placement
      )
    for stem in self._STEMS:
      for tag, want_tag in (('aerosolfree', ''),
                            ('aerosolfree_clearsky', 'clearsky_')):
        np.testing.assert_allclose(
            np.asarray(got[f'{stem}_{tag}_full']),
            np.asarray(want[f'{stem}_{want_tag}full']),
            rtol=2e-6, atol=1e-4, err_msg=f'{stem} {tag}',
        )
    np.testing.assert_allclose(
        np.asarray(got['toa_sw_flux_outgoing_aerosolfree_2d_xy']),
        np.asarray(want['toa_sw_flux_outgoing_2d_xy']), rtol=2e-6, atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(got['toa_lw_flux_outgoing_aerosolfree_clearsky_2d_xy']),
        np.asarray(want['lw_flux_up_clearsky_full'][:, :, -1]),
        rtol=2e-6, atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(got['rad_heat_lw_aerosolfree_3d']),
        np.asarray(want['rad_heat_lw_3d']), rtol=2e-6, atol=1e-9,
    )
    self.assertIn('rad_heat_lw_aerosolfree_1d_z', got)
    self.assertIn('surf_sw_flux_down_aerosolfree_2d_xy', got)
    self.assertIn('sw_flux_down_dir_aerosolfree_clearsky_sfc', got)
    # ... and the aerosol makes a difference to the problems that keep it.
    for stem in ('sw_flux_down', 'lw_flux_up'):
      self.assertGreater(
          np.max(np.abs(np.asarray(got[f'{stem}_full'])
                        - np.asarray(got[f'{stem}_aerosolfree_full']))),
          0.1, stem,
      )

  def test_other_outputs_do_not_depend_on_the_flag(self):
    inputs, aerosol = self._inputs()
    rt = self._rt()
    off = rt.compute_heating_rate(**inputs, **aerosol, clear_sky_fluxes=True)
    on = rt.compute_heating_rate(
        **inputs, **aerosol, clear_sky_fluxes=True, aerosol_free_fluxes=True
    )
    self.assertFalse([k for k in off if 'aerosolfree' in k])
    self.assertTrue(set(off) < set(on))
    for key, val in off.items():
      # Rounding-level: the combined program is fused differently (see
      # `all_sky_test.SharedAerosolFreeTest`).
      np.testing.assert_allclose(
          np.asarray(on[key]), np.asarray(val), rtol=2e-6, atol=1e-4,
          err_msg=key,
      )

  def test_without_aerosol_the_twins_are_bit_identical(self):
    inputs, _ = self._inputs()
    rt = self._rt()
    out = rt.compute_heating_rate(**inputs, aerosol_free_fluxes=True)
    self.assertFalse([k for k in out if 'clearsky' in k])
    for stem in self._STEMS:
      np.testing.assert_array_equal(
          np.asarray(out[f'{stem}_aerosolfree_full']),
          np.asarray(out[f'{stem}_full']), err_msg=stem,
      )
    np.testing.assert_array_equal(
        np.asarray(out['rad_heat_sw_aerosolfree_3d']),
        np.asarray(out['rad_heat_sw_3d']),
    )


class RRTMGPAerosolTest(unittest.TestCase):
  """Regression tests for the per-band aerosol_optics_{lw,sw} kwargs."""

  def _make_zero_aerosol(self, n_bnd: int, shape: tuple[int, int, int]):
    z = jnp.zeros((n_bnd,) + shape, dtype=jnp.float_)
    return {'optical_depth': z, 'ssa': z, 'asymmetry_factor': z}

  def test_zero_aerosol_matches_baseline(self):
    """Passing an all-zero aerosol bundle is a no-op vs. not passing one."""
    cfg = _build_radiative_transfer_cfg()
    rt = rrtmgp.RRTMGP(cfg, dz=500.0)
    inputs = _build_inputs()

    baseline = rt.compute_heating_rate(**inputs)

    shape = inputs['p_ref_xxc'].shape
    aer_lw = self._make_zero_aerosol(rt.optics_lib.gas_optics_lw.n_bnd, shape)
    aer_sw = self._make_zero_aerosol(rt.optics_lib.gas_optics_sw.n_bnd, shape)
    zeroed = rt.compute_heating_rate(
        **inputs, aerosol_optics_lw=aer_lw, aerosol_optics_sw=aer_sw
    )

    # Zero tau contributes nothing in combine_optical_properties — but the
    # extra combination step touches ssa via a `max(tau, EPSILON)` guard, so
    # the two paths agree only to floating-point precision, not bit-exact.
    # rtol=1e-2 / atol=1e-7 covers the platform-dependent rounding from the
    # EPSILON guard at float32 (CI x86_64 vs local arm64 differ at ~0.3%);
    # still tight enough to catch any real wiring regression — the SW and LW
    # aerosol tests below confirm a real perturbation moves heating rates by
    # orders of magnitude more than this.
    np.testing.assert_allclose(
        np.asarray(zeroed[rrtmgp_common.KEY_STORED_RADIATION]),
        np.asarray(baseline[rrtmgp_common.KEY_STORED_RADIATION]),
        rtol=1e-2,
        atol=1e-7,
    )

  def test_sw_aerosol_reduces_surface_flux(self):
    """Increasing SW aerosol optical depth must reduce the surface
    down-flux (scattering + absorption attenuates the direct + diffuse beam).
    """
    cfg = dataclasses.replace(
        _build_radiative_transfer_cfg(),
        save_lw_sw_heating_rates=True,
    )
    rt = rrtmgp.RRTMGP(cfg, dz=500.0)
    inputs = _build_inputs()

    baseline = rt.compute_heating_rate(**inputs)

    shape = inputs['p_ref_xxc'].shape
    n_bnd_sw = rt.optics_lib.gas_optics_sw.n_bnd
    # A modestly absorbing/scattering aerosol uniformly distributed in the
    # column: tau=0.05 per layer per band, ssa=0.9, g=0.7. This is sulfate-ish
    # and well above floating-point noise.
    aer_sw = {
        'optical_depth': 0.05 * jnp.ones((n_bnd_sw,) + shape, dtype=jnp.float_),
        'ssa': 0.9 * jnp.ones((n_bnd_sw,) + shape, dtype=jnp.float_),
        'asymmetry_factor': 0.7 * jnp.ones(
            (n_bnd_sw,) + shape, dtype=jnp.float_
        ),
    }
    perturbed = rt.compute_heating_rate(**inputs, aerosol_optics_sw=aer_sw)

    # Surface SW down flux at the first physical level (above the surface halo)
    # must drop when aerosols are added.
    hw = 1
    baseline_surf = np.asarray(baseline['sw_flux_down_full'][:, :, 0])
    perturbed_surf = np.asarray(perturbed['sw_flux_down_full'][:, :, 0])
    self.assertTrue(
        np.all(perturbed_surf < baseline_surf),
        msg=(
            'SW aerosol did not reduce surface down-flux everywhere. '
            f'min(baseline - perturbed) = {np.min(baseline_surf - perturbed_surf):.3e} W/m²'
        ),
    )
    del hw

  def test_lw_aerosol_changes_toa_outgoing(self):
    """Adding an absorbing LW aerosol must change TOA outgoing LW flux."""
    cfg = _build_radiative_transfer_cfg()
    rt = rrtmgp.RRTMGP(cfg, dz=500.0)
    inputs = _build_inputs()

    baseline = rt.compute_heating_rate(**inputs)

    shape = inputs['p_ref_xxc'].shape
    n_bnd_lw = rt.optics_lib.gas_optics_lw.n_bnd
    # Pure absorber: tau=0.1, ssa=0 (no scattering — typical for LW dust).
    aer_lw = {
        'optical_depth': 0.1 * jnp.ones((n_bnd_lw,) + shape, dtype=jnp.float_),
        'ssa': jnp.zeros((n_bnd_lw,) + shape, dtype=jnp.float_),
        'asymmetry_factor': jnp.zeros(
            (n_bnd_lw,) + shape, dtype=jnp.float_
        ),
    }
    perturbed = rt.compute_heating_rate(**inputs, aerosol_optics_lw=aer_lw)

    baseline_toa = np.asarray(baseline['lw_flux_up_full'][:, :, -1])
    perturbed_toa = np.asarray(perturbed['lw_flux_up_full'][:, :, -1])
    max_diff = np.max(np.abs(baseline_toa - perturbed_toa))
    self.assertGreater(
        max_diff,
        1e-3,
        msg=(
            'LW aerosol left TOA outgoing flux unchanged '
            f'(max |Δ| = {max_diff:.3e} W/m²).'
        ),
    )


def _varied_columns(nx: int = 6, ny: int = 7, seed: int = 1234):
  """An `(nx, ny)` grid of distinct, horizontally varying columns.

  The 42 columns of `cloudysky_as.nc` with the surface pressure scaled by
  0.75-1.02 in a hybrid-sigma-like way (so one model level spans several
  pressure-table nodes across the grid), liquid and ice cloud with random
  effective radii, per-band LW and SW aerosol, per-band direct and diffuse SW
  albedos, a 2D emissivity, surface temperature, ozone and zenith (with a
  horizon and a night column), and per-phase cloud optical-depth scaling.

  Returns:
    The radiative transfer object, the keyword arguments of
    `compute_heating_rate` (3D, `(nx, ny, nz + 2)`) and of
    `compute_heating_rate_zc` (`(nz + 2, nx * ny)`), and the McICA inputs of
    each layout.
  """
  cfg = dataclasses.replace(
      _build_radiative_transfer_cfg(), save_lw_sw_heating_rates=True
  )
  diagnostics = (
      'surf_lw_flux_down_2d_xy', 'surf_lw_flux_up_2d_xy',
      'surf_sw_flux_down_2d_xy', 'surf_sw_flux_up_2d_xy',
      'surf_sw_flux_down_dir_2d_xy',
      'toa_sw_flux_incoming_2d_xy', 'toa_sw_flux_outgoing_2d_xy',
      'toa_lw_flux_outgoing_2d_xy',
      'toa_sw_flux_outgoing_clearsky_2d_xy',
      'toa_lw_flux_outgoing_clearsky_2d_xy',
      'toa_sw_flux_outgoing_aerosolfree_2d_xy',
      'toa_lw_flux_outgoing_aerosolfree_2d_xy',
      'rad_heat_lw_1d_z', 'rad_heat_sw_1d_z',
  )
  rt = rrtmgp.RRTMGP(cfg, dz=500.0, diagnostic_fields=diagnostics)
  olib = rt.optics_lib
  n_bnd_lw, n_bnd_sw = olib.gas_optics_lw.n_bnd, olib.gas_optics_sw.n_bnd
  n_gpt_lw, n_gpt_sw = olib.n_gpt_lw, olib.n_gpt_sw

  ds = nc.Dataset(root / _ATMOSPHERIC_STATE_FILENAME, 'r')
  p_lay = np.asarray(ds['p_lay'][:].data).T
  t_lay = np.asarray(ds['t_lay'][:].data).T
  t_lev = np.asarray(ds['t_lev'][:].data).T
  h2o = np.asarray(ds['vmr_h2o'][:].data).T
  o3 = np.asarray(ds['vmr_o3'][:].data).T
  t_sfc = np.asarray(ds['t_sfc'][:].data)
  ncol, nz = p_lay.shape
  assert ncol == nx * ny
  rng = np.random.default_rng(seed)
  pscale = rng.uniform(0.75, 1.02, ncol)[:, None]
  p_lay = p_lay * (1.0 + (pscale - 1.0) * p_lay / p_lay.max(axis=1, keepdims=True))

  def pad_edge(a):
    return np.concatenate([a[..., :1], a, a[..., -1:]], axis=-1)

  p = pad_edge(p_lay)
  temp = np.zeros((ncol, nz + 2))
  temp[:, 1:-1] = t_lay
  temp[:, 0] = 2 * t_lev[:, 0] - t_lay[:, 0]
  temp[:, -1] = 2 * t_lev[:, -1] - t_lay[:, -1]
  vh2o = pad_edge(h2o)
  q_v = vh2o / (vh2o + constants.R_V / constants.R_D)
  rho = p / (constants.R_D * temp)
  q_liq = np.where((p > 6e4) & (p < 9.5e4), rng.uniform(0, 3e-4, p.shape), 0.0)
  q_ice = np.where((p > 1.5e4) & (p < 4e4), rng.uniform(0, 1e-4, p.shape), 0.0)
  q_liq[:, [0, -1]] = 0
  q_ice[:, [0, -1]] = 0
  q_c = q_liq + q_ice
  r_liq = rng.uniform(6e-6, 14e-6, p.shape)
  r_ice = rng.uniform(20e-6, 70e-6, p.shape)
  zen = rng.uniform(0.0, 0.45 * np.pi, ncol)
  zen[0] = 0.5 * np.pi
  zen[9] = 0.7 * np.pi
  alb_dir = rng.uniform(0.05, 0.4, (ncol, n_bnd_sw))
  alb_dif = rng.uniform(0.05, 0.4, (ncol, n_bnd_sw))
  sfc_emis = rng.uniform(0.92, 0.99, ncol)
  tau_scale_liq = rng.uniform(0.6, 1.0, ncol)

  def aer(n_bnd, scale):
    return {
        'optical_depth': rng.uniform(0, scale, (n_bnd, ncol, nz)),
        'ssa': rng.uniform(0.6, 0.97, (n_bnd, ncol, nz)),
        'asymmetry_factor': rng.uniform(0.5, 0.8, (n_bnd, ncol, nz)),
    }

  aer_lw, aer_sw = aer(n_bnd_lw, 0.02), aer(n_bnd_sw, 0.05)

  # McICA: each g-point sees the cell's cloud or none.
  dz = 500.0
  lwp = rho[:, 1:-1] * q_liq[:, 1:-1] * dz
  iwp = rho[:, 1:-1] * q_ice[:, 1:-1] * dz
  mask_lw = rng.uniform(size=(n_gpt_lw, ncol, nz)) < 0.5
  mask_sw = rng.uniform(size=(n_gpt_sw, ncol, nz)) < 0.5

  f32 = lambda a: jnp.asarray(a, jnp.float32)
  to3 = lambda a: f32(a.reshape(nx, ny, *a.shape[1:]))
  zc = lambda a: f32(np.moveaxis(a, 0, -1))  # (ncol, ...) -> (..., ncol)
  common = dict(
      irrad=jnp.float32(1360.0),
      cloud_tau_scale_ice=jnp.float32(0.8),
  )
  kwargs_3d = dict(
      rho_xxc=to3(rho), q_t=to3(q_v + q_c), q_liq=to3(q_liq), q_ice=to3(q_ice),
      q_c=to3(q_c), cloud_r_eff_liq=to3(r_liq), cloud_r_eff_ice=to3(r_ice),
      temperature=to3(temp), sfc_temperature=to3(t_sfc), p_ref_xxc=to3(p),
      sg_map={}, vmr_fields={'o3': to3(pad_edge(o3))}, zenith=to3(zen),
      sfc_emis=to3(sfc_emis), sfc_alb_dir=to3(alb_dir),
      sfc_alb_dif=to3(alb_dif),
      aerosol_optics_lw={
          k: f32(pad_edge(v).reshape(n_bnd_lw, nx, ny, nz + 2))
          for k, v in aer_lw.items()
      },
      aerosol_optics_sw={
          k: f32(pad_edge(v).reshape(n_bnd_sw, nx, ny, nz + 2))
          for k, v in aer_sw.items()
      },
      cloud_tau_scale_liq=to3(tau_scale_liq[:, None]),
      **common,
  )
  kwargs_zc = dict(
      rho_zc=zc(rho), q_t=zc(q_v + q_c), q_liq=zc(q_liq), q_ice=zc(q_ice),
      q_c=zc(q_c), cloud_r_eff_liq=zc(r_liq), cloud_r_eff_ice=zc(r_ice),
      temperature=zc(temp), sfc_temperature=f32(t_sfc), p_ref_zc=zc(p),
      dz_zc=dz, vmr_fields={'o3': zc(pad_edge(o3))}, zenith=f32(zen),
      sfc_emis=f32(sfc_emis), sfc_alb_dir=f32(alb_dir.T),
      sfc_alb_dif=f32(alb_dif.T),
      aerosol_optics_lw={k: f32(np.moveaxis(v, 1, 2)) for k, v in aer_lw.items()},
      aerosol_optics_sw={k: f32(np.moveaxis(v, 1, 2)) for k, v in aer_sw.items()},
      cloud_tau_scale_liq=f32(tau_scale_liq),
      **common,
  )

  def per_gpt_3d(mask, path):
    a = pad_edge(mask * path[None])
    a[..., 0] = a[..., -1] = 0.0
    return f32(a.reshape(a.shape[0], nx, ny, nz + 2))

  mcica_3d = dict(
      cloud_path_liq_lw_per_gpt=per_gpt_3d(mask_lw, lwp),
      cloud_path_ice_lw_per_gpt=per_gpt_3d(mask_lw, iwp),
      cloud_path_liq_sw_per_gpt=per_gpt_3d(mask_sw, lwp),
      cloud_path_ice_sw_per_gpt=per_gpt_3d(mask_sw, iwp),
  )
  # The column-minor solve takes the same thing as boolean masks of the bulk
  # paths, `(nz, n_gpt, nc)`.
  mcica_zc = dict(
      cloud_path_liq=zc(lwp), cloud_path_ice=zc(iwp),
      cloud_mask_lw_per_gpt=jnp.asarray(np.transpose(mask_lw, (2, 0, 1))),
      cloud_mask_sw_per_gpt=jnp.asarray(np.transpose(mask_sw, (2, 0, 1))),
  )
  return rt, kwargs_3d, kwargs_zc, mcica_3d, mcica_zc


def _as_zc(value: np.ndarray) -> np.ndarray:
  """A 3D-layout output, `(nx, ny, ...)`, in the column-minor layout."""
  value = np.asarray(value)
  if value.ndim == 3:
    return np.moveaxis(value, 2, 0).reshape(value.shape[2], -1)
  if value.ndim == 2:
    return value.reshape(-1)
  return value


# Every extra problem, so that their sharing of the spectral loop is tested.
_FLAGS = dict(clear_sky_fluxes=True, aerosol_free_fluxes=True)

# Agreement of the column-minor and the level-minor solves. They solve the
# same equations with differently fused float32 arithmetic (and, with the TPU
# kernels, `Precision.HIGHEST` contractions); `main`'s own two placements
# (`use_scan` True / False) differ by 2.4e-4 W/m^2 and 0.011 K/day.
_FLUX_ATOL = 1e-2  # W/m^2
_HEATING_ATOL = 0.05 / 86400  # K/s, 0.05 K/day


def _assert_outputs_close(
    test, want, got, axis, rtol=1e-4, flux_atol=_FLUX_ATOL
):
  """`want` and `got` agree; `axis` is their vertical axis."""
  test.assertEqual(set(want), set(got))
  for key in want:
    w, g = np.asarray(want[key]), np.asarray(got[key])
    test.assertEqual(w.shape, g.shape, key)
    heating = key.startswith('rad_heat')
    if heating:
      # The heating rates of the halo levels are not defined.
      ax = 0 if w.ndim == 1 else axis
      interior = np.arange(1, w.shape[ax] - 1)
      w, g = np.take(w, interior, axis=ax), np.take(g, interior, axis=ax)
    np.testing.assert_allclose(
        g, w, rtol=rtol, atol=_HEATING_ATOL if heating else flux_atol,
        err_msg=key,
    )


class RRTMGPColumnMinorTest(unittest.TestCase):
  """The column-minor solve against the level-minor one it restates.

  `compute_heating_rate` with the gas optics hoisted, scanned recurrences and
  `gpt_chunk > 1` is solved column-minor on every backend; the level-minor
  solve of the same placement (forced with `two_stream._column_minor(False)`)
  is the reference. The column-minor solve is checked through both entry
  points and with both kernel sets (`rrtmgp.backend`), on distinct,
  horizontally varying columns, for values and gradients.
  """

  def tearDown(self):
    jax.clear_caches()
    super().tearDown()

  def _solve_both(self, mcica, **options):
    options.update(_FLAGS)
    rt, kw_3d, kw_zc, mcica_3d, mcica_zc = _varied_columns()
    if mcica:
      kw_3d = {**kw_3d, **mcica_3d}
      kw_zc = {**kw_zc, **mcica_zc}
    placement = dict(gas_optics_mode='all_gpts', use_scan=True, gpt_chunk=64)
    with two_stream._column_minor(False):  # pylint: disable=protected-access
      want = jax.jit(
          functools.partial(rt.compute_heating_rate, **placement, **options)
      )(**kw_3d)
    got_3d = jax.jit(
        functools.partial(rt.compute_heating_rate, **placement, **options)
    )(**kw_3d)
    got_zc = {}
    for tpu in (False, True):
      with backend.tpu_kernels(tpu):
        got_zc[tpu] = jax.jit(
            functools.partial(
                rt.compute_heating_rate_zc, gpt_chunk=16, **options
            )
        )(**kw_zc)
    return want, got_3d, got_zc

  @parameterized.expand([
      ('bulk_two_stream', False, 'two_stream'),
      ('mcica_noscat', True, 'noscat'),
  ])
  def test_matches_the_level_minor_solve(self, _, mcica, lw_solver):
    want, got_3d, got_zc = self._solve_both(mcica, lw_solver=lw_solver)
    with self.subTest('3D entry point'):
      _assert_outputs_close(self, want, got_3d, axis=2)
    want_zc = {k: _as_zc(v) for k, v in want.items()}
    for tpu, got in got_zc.items():
      with self.subTest('column-minor entry point', tpu_kernels=tpu):
        _assert_outputs_close(self, want_zc, got, axis=0)
    # Night and horizon columns carry no shortwave flux.
    for key in ('sw_flux_down_full', 'sw_flux_up_full'):
      np.testing.assert_array_equal(np.asarray(got_zc[False][key])[:, [0, 9]], 0)

  def test_gradients_match_the_level_minor_solve(self):
    rt, kw_3d, kw_zc, _, _ = _varied_columns()
    placement = dict(gas_optics_mode='all_gpts', use_scan=True, gpt_chunk=64)

    def loss_3d(t_int):
      t = jnp.concatenate(
          [kw_3d['temperature'][..., :1], t_int,
           kw_3d['temperature'][..., -1:]], axis=-1,
      )
      out = rt.compute_heating_rate(
          **{**kw_3d, 'temperature': t}, **placement, **_FLAGS
      )
      return (
          jnp.sum(out['rad_heat_src'][..., 1:-1] ** 2) * 1e8
          + 1e-3 * jnp.sum(out['toa_lw_flux_outgoing_2d_xy'])
          + 1e-3 * jnp.sum(out['toa_sw_flux_outgoing_2d_xy'])
      )

    def loss_zc(t_int):
      t = jnp.concatenate(
          [kw_zc['temperature'][:1], t_int, kw_zc['temperature'][-1:]], axis=0
      )
      out = rt.compute_heating_rate_zc(
          **{**kw_zc, 'temperature': t}, gpt_chunk=16, **_FLAGS
      )
      return (
          jnp.sum(out['rad_heat_src'][1:-1] ** 2) * 1e8
          + 1e-3 * jnp.sum(out['toa_lw_flux_outgoing_2d_xy'])
          + 1e-3 * jnp.sum(out['toa_sw_flux_outgoing_2d_xy'])
      )

    with two_stream._column_minor(False):  # pylint: disable=protected-access
      want = _as_zc(jax.jit(jax.grad(loss_3d))(kw_3d['temperature'][..., 1:-1]))
    got = {}
    for tpu in (False, True):
      with backend.tpu_kernels(tpu):
        got[tpu] = np.asarray(
            jax.jit(jax.grad(loss_zc))(kw_zc['temperature'][1:-1])
        )
    scale = np.abs(want).max()
    for tpu, g in got.items():
      with self.subTest(tpu_kernels=tpu):
        self.assertTrue(np.isfinite(g).all())
        np.testing.assert_allclose(g, want, rtol=0, atol=1e-3 * scale)

  def test_masks_equal_masked_per_gpt_paths(self):
    """`cloud_mask_*_per_gpt` of the bulk paths equal the masked paths."""
    rt, _, kw_zc, _, mcica_zc = _varied_columns()
    masked_paths = {
        f'cloud_path_{phase}_{band}_per_gpt': jnp.where(
            mcica_zc[f'cloud_mask_{band}_per_gpt'],
            mcica_zc[f'cloud_path_{phase}'][:, jnp.newaxis],
            0.0,
        )
        for phase in ('liq', 'ice')
        for band in ('lw', 'sw')
    }
    base = {k: v for k, v in kw_zc.items() if k not in ('q_liq', 'q_ice')}
    for tpu in (False, True):
      with self.subTest(tpu_kernels=tpu), backend.tpu_kernels(tpu):
        want = rt.compute_heating_rate_zc(
            **base, **masked_paths, gpt_chunk=16, **_FLAGS
        )
        got = rt.compute_heating_rate_zc(
            **base, **mcica_zc, gpt_chunk=16, **_FLAGS
        )
        _assert_outputs_close(
            self, want, got, axis=0, rtol=1e-5, flux_atol=1e-4
        )

  def test_masks_and_per_gpt_paths_are_exclusive(self):
    rt, _, kw_zc, _, mcica_zc = _varied_columns()
    paths = {
        'cloud_path_liq_lw_per_gpt': jnp.zeros_like(
            mcica_zc['cloud_mask_lw_per_gpt'], dtype=jnp.float32
        )
    }
    with self.assertRaisesRegex(ValueError, 'cloud_mask_per_gpt'):
      rt.compute_heating_rate_zc(**kw_zc, **mcica_zc, **paths)

  def test_tpu_program_has_no_gather_scatter_or_barrier(self):
    rt, _, kw_zc, _, mcica_zc = _varied_columns()
    with backend.tpu_kernels(True):
      text = (
          jax.jit(
              functools.partial(
                  rt.compute_heating_rate_zc, gpt_chunk=16, **_FLAGS
              )
          )
          .lower(**kw_zc, **mcica_zc)
          .as_text()
      )
    for op in ('gather', 'scatter', 'optimization_barrier'):
      self.assertEqual(text.count(f'stablehlo.{op}'), 0, op)

  def test_routing(self):
    rt, kw_3d, *_ = _varied_columns()
    olib = rt.optics_lib
    t = kw_3d['temperature']
    route = functools.partial(two_stream._uses_column_minor, olib, t)  # pylint: disable=protected-access
    self.assertTrue(route(True, True, 64))
    self.assertFalse(route(True, False, 64))  # unrolled recurrences
    self.assertFalse(route(True, True, 1))  # one g-point per iteration
    self.assertFalse(route(False, True, 64))  # gas optics in the loop
    self.assertFalse(
        two_stream._uses_column_minor(olib, t[0], True, True, 64)  # pylint: disable=protected-access
    )
    with two_stream._column_minor(False):  # pylint: disable=protected-access
      self.assertFalse(route(True, True, 64))
    with two_stream._column_minor(True):  # pylint: disable=protected-access
      self.assertTrue(route(True, False, 1))


if __name__ == '__main__':
  unittest.main()
