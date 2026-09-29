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

import dataclasses
import functools
from typing import TypeAlias

import unittest
from parameterized import parameterized
from itertools import product
from pathlib import Path
import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
from rrtmgp import constants
from rrtmgp import kernel_ops
from rrtmgp import test_util
from rrtmgp.config import radiative_transfer
from rrtmgp.optics import atmospheric_state
from rrtmgp.optics import optics
from rrtmgp.rte import two_stream

Array: TypeAlias = jax.Array

_VMR_GLOBAL_MEAN_FILENAME = 'rrtmgp/optics/test_data/rcemip_global_mean_vmr.json'
_ATMOSPHERIC_STATE_FILENAME = 'rrtmgp/optics/test_data/cloudysky_as.nc'
_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g256.nc'
_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-sw-g224.nc'
_LW_LOOKUP_TABLE_COMPACT_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g128.nc'
_SW_LOOKUP_TABLE_COMPACT_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-sw-g112.nc'
_CLD_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/cloudysky_lw.nc'
_CLD_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/cloudysky_sw.nc'
_ALL_SKY_REFERENCE_FILENAME = 'rrtmgp/optics/test_data/cloudysky_lut.nc'

root = Path()
_VMR_GLOBAL_MEAN_FILEPATH = root / _VMR_GLOBAL_MEAN_FILENAME
_ATMOSPHERIC_STATE_FILEPATH = root / _ATMOSPHERIC_STATE_FILENAME
_LW_LOOKUP_TABLE_FILEPATH = root / _LW_LOOKUP_TABLE_FILENAME
_SW_LOOKUP_TABLE_FILEPATH = root / _SW_LOOKUP_TABLE_FILENAME
_LW_LOOKUP_TABLE_COMPACT_FILEPATH = root / _LW_LOOKUP_TABLE_COMPACT_FILENAME
_SW_LOOKUP_TABLE_COMPACT_FILEPATH = root / _SW_LOOKUP_TABLE_COMPACT_FILENAME

_CLOUD_LONGWAVE_NC_FILEPATH = root / _CLD_LW_LOOKUP_TABLE_FILENAME
_CLOUD_SHORTWAVE_NC_FILEPATH = root / _CLD_SW_LOOKUP_TABLE_FILENAME
_ALL_SKY_REFERENCE_FILEPATH = root / _ALL_SKY_REFERENCE_FILENAME


def _remove_halos(f: Array) -> Array:
  """Remove the halos from the output."""
  return f[:, :, 1:-1]


def _setup_radiation_params(
    use_compact_lookup: bool,
) -> radiative_transfer.RadiativeTransfer:
  """Create an instance of `RadiativeTransfer`."""
  if use_compact_lookup:
    lw_lookup_table_nc_filepath = _LW_LOOKUP_TABLE_COMPACT_FILEPATH
    sw_lookup_table_nc_filepath = _SW_LOOKUP_TABLE_COMPACT_FILEPATH
  else:
    lw_lookup_table_nc_filepath = _LW_LOOKUP_TABLE_FILEPATH
    sw_lookup_table_nc_filepath = _SW_LOOKUP_TABLE_FILEPATH

  return radiative_transfer.RadiativeTransfer(
      optics=radiative_transfer.OpticsParameters(
          optics=radiative_transfer.RRTMOptics(
              longwave_nc_filepath=lw_lookup_table_nc_filepath,
              shortwave_nc_filepath=sw_lookup_table_nc_filepath,
              cloud_longwave_nc_filepath=_CLOUD_LONGWAVE_NC_FILEPATH,
              cloud_shortwave_nc_filepath=_CLOUD_SHORTWAVE_NC_FILEPATH,
          )
      ),
      atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
          sfc_emis=0.98,
          sfc_alb=0.06,
          zenith=0.535526654,
          irrad=1360.8585174,
          toa_flux_lw=0.0,
          vmr_global_mean_filepath=_VMR_GLOBAL_MEAN_FILEPATH,
      ),
  )


def _setup_atmospheric_profiles() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
]:
  """Load vertical profiles of pressure and temperature from RFMIP."""
  halo_width = 1
  atmos_state_ds = nc.Dataset(_ATMOSPHERIC_STATE_FILEPATH, 'r')

  # Reverse order of the profiles so they correspond to increasing altitude.
  p_internal = atmos_state_ds['p_lay'][:].data
  pres_level = atmos_state_ds['p_lev'][:].data

  # Set the boundary values using linear extrapolation from the outermost
  # levels.
  paddings_2d = ((0, 0), (halo_width, halo_width))
  # Transpose so the vertical profile is stored in the last axis.
  pressure = np.pad(np.transpose(p_internal), paddings_2d, mode='edge')
  pressure_level = np.pad(
      np.transpose(pres_level), ((0, 0), (halo_width, halo_width - 1))
  )
  # After padding, shape should be (42, 42 + 2) = (42, 44).

  temp_internal = np.transpose(atmos_state_ds['t_lay'][:].data)
  temp_level = np.transpose(atmos_state_ds['t_lev'][:].data)

  nx, nz = temp_internal.shape  # 42, 42
  nz_with_halos = nz + 2 * halo_width
  temperature = np.zeros((nx, nz_with_halos), dtype=jnp.float_)
  temperature[:, halo_width:-halo_width] = temp_internal
  # Fill in halos by extrapolating from face (temp_level) and node values.
  temperature[:, 0] = 2 * temp_level[:, 0] - temp_internal[:, 0]
  temperature[:, -1] = 2 * temp_level[:, -1] - temp_internal[:, -1]

  temperature_level = np.zeros_like(temperature)
  temperature_level[:, 1:] = temp_level
  # Fill in halos with same value as in `temperature`.
  temperature_level[:, 0] = temperature[:, 0]

  vmr_profiles = {}
  for k in atmos_state_ds.variables:
    if not k.startswith('vmr_'):
      continue
    chem_formula = k[len('vmr_') :]
    vmr_profiles[chem_formula] = np.pad(
        np.transpose(atmos_state_ds[k][:].data), paddings_2d, mode='edge'
    )

  sfc_temperature = atmos_state_ds['t_sfc'][:].data
  # Here `_level` means these are the values on faces.
  return (
      pressure,
      pressure_level,
      temperature,
      temperature_level,
      vmr_profiles,
      sfc_temperature,
  )


def _load_expected_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
  reference_data = nc.Dataset(_ALL_SKY_REFERENCE_FILEPATH, 'r')

  halo_width = 1
  paddings = ((0, 0), (halo_width, halo_width - 1))

  lw_flux_up = np.pad(
      np.transpose(reference_data['lw_flux_up'][:].data), paddings
  )
  lw_flux_down = np.pad(
      np.transpose(reference_data['lw_flux_dn'][:].data), paddings
  )
  sw_flux_up = np.pad(
      np.transpose(reference_data['sw_flux_up'][:].data), paddings
  )
  sw_flux_down = np.pad(
      np.transpose(reference_data['sw_flux_dn'][:].data), paddings
  )
  sw_flux_dir = np.pad(
      np.transpose(reference_data['sw_flux_dir'][:].data), paddings
  )
  return lw_flux_up, lw_flux_down, sw_flux_up, sw_flux_down, sw_flux_dir


def _air_molecules_per_area(p_bottom: Array, vmr_h2o: Array) -> Array:
  """Compute the number of molecules in an atmospheric grid cell per area."""
  dp = kernel_ops.forward_difference(p_bottom, dim=2)
  mol_m_air = constants.DRY_AIR_MOL_MASS + constants.WATER_MOL_MASS * vmr_h2o
  return -(dp / constants.G) * constants.AVOGADRO / mol_m_air


def _compact_cloudy_setup(n_horiz: int = 2):
  """Cloudy single-profile column batch with the compact lookup tables."""
  site = 0
  (
      pressure_allsites,
      pressure_level_allsites,
      temperature_allsites,
      temperature_level_allsites,
      vmr_profiles_allsites,
      _,
  ) = _setup_atmospheric_profiles()
  # Compact tables (g128 / g112) keep compile time down; they have the same
  # band/g-point structure as the full ones.
  radiation_params = _setup_radiation_params(use_compact_lookup=True)
  atmos_state = atmospheric_state.from_config(
      radiation_params.atmospheric_state_cfg
  )
  optics_lib = optics.optics_factory(radiation_params.optics, atmos_state.vmr)

  convert_to_3d = functools.partial(
      test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=n_horiz
  )
  sfc_temperature = temperature_level_allsites[site, 1] * jnp.ones(
      (n_horiz, n_horiz), dtype=jnp.float_
  )
  vmr_fields = {
      k: convert_to_3d(v[site, :]) for k, v in vmr_profiles_allsites.items()
  }
  p = convert_to_3d(pressure_allsites[site, :])
  pressure_level = convert_to_3d(pressure_level_allsites[site, :])
  temperature = convert_to_3d(temperature_allsites[site, :])
  molecules = _air_molecules_per_area(pressure_level, vmr_fields['h2o'])

  ones = jnp.ones_like(p)
  in_cloud = jnp.logical_and(p > 10000, p < 90000)
  cloud = {
      'cloud_r_eff_liq': jnp.where(
          jnp.logical_and(in_cloud, temperature > 263), 1.2e-5 * ones, 0.0
      ),
      'cloud_path_liq': jnp.where(
          jnp.logical_and(in_cloud, temperature > 263), 1e-2 * ones, 0.0
      ),
      'cloud_r_eff_ice': jnp.where(
          jnp.logical_and(in_cloud, temperature < 273), 4.75e-5 * ones, 0.0
      ),
      'cloud_path_ice': jnp.where(
          jnp.logical_and(in_cloud, temperature < 273), 1e-2 * ones, 0.0
      ),
  }
  return (optics_lib, atmos_state, p, temperature, molecules, vmr_fields,
          sfc_temperature, cloud)


class AllSkyTest(unittest.TestCase):

  @parameterized.expand([
    (use_compact, use_scan)
    for use_compact, use_scan in product([True, False], [True, False])
  ])
  def test_two_stream_solver_with_cloudy_sky(
      self, use_compact_lookup: bool, use_scan: bool
  ):
    # SETUP
    site = 0  # Use data from site 0.

    (
        pressure_allsites,
        pressure_level_allsites,
        temperature_allsites,
        temperature_level_allsites,
        vmr_profiles_allsites,
        _,
    ) = _setup_atmospheric_profiles()
    # pressure, pressure_level, etc. are 2D arrays, where the 2nd dimension is
    # the vertical dimension.  The 1st dimension is a dimension corresponding to
    # the 'site' the data comes from.

    radiation_params = _setup_radiation_params(use_compact_lookup)
    atmos_state = atmospheric_state.from_config(
        radiation_params.atmospheric_state_cfg
    )
    optics_lib = optics.optics_factory(radiation_params.optics, atmos_state.vmr)

    # Model inputs.
    n_horiz = 2
    convert_to_3d = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=n_horiz
    )
    sfc_temperature = temperature_level_allsites[site, 1] * jnp.ones(
        (n_horiz, n_horiz), dtype=jnp.float_
    )
    vmr_fields = {
        k: convert_to_3d(v[site, :]) for k, v in vmr_profiles_allsites.items()
    }
    p = convert_to_3d(pressure_allsites[site, :])
    pressure_level = convert_to_3d(pressure_level_allsites[site, :])
    temperature = convert_to_3d(temperature_allsites[site, :])

    # Effective radius of condensate particles.
    r_liq = 1.2e-5
    # The division by 2 is to enable compatibility with the old RRTMGP
    # tables, after the ice radius -> diameter fix was made. We should remove
    # the division by 2 after the tables with expected values are updated.
    r_ice = 9.5e-5 / 2
    # Cloud path for liquid and ice in kg/kg.
    cld_path = 1e-2
    # Let there be condensates only between 100 hPa and 900 hPa.

    ones = jnp.ones_like(p)
    ind_valid_p_range = jnp.logical_and(p > 10000, p < 90000)
    r_eff_liq = jnp.where(
        jnp.logical_and(ind_valid_p_range, temperature > 263),
        r_liq * ones,
        0.0,
    )
    cld_path_liq = jnp.where(
        jnp.logical_and(ind_valid_p_range, temperature > 263),
        cld_path * ones,
        0.0,
    )
    r_eff_ice = jnp.where(
        jnp.logical_and(ind_valid_p_range, temperature < 273),
        r_ice * ones,
        0.0,
    )
    cld_path_ice = jnp.where(
        jnp.logical_and(ind_valid_p_range, temperature < 273),
        cld_path * ones,
        0.0,
    )

    molecules = _air_molecules_per_area(pressure_level, vmr_fields['h2o'])

    # ACTION
    output_lw = two_stream.solve_lw(
        p,
        temperature,
        molecules,
        optics_lib,
        atmos_state,
        vmr_fields,
        sfc_temperature,
        cloud_r_eff_liq=r_eff_liq,
        cloud_path_liq=cld_path_liq,
        cloud_r_eff_ice=r_eff_ice,
        cloud_path_ice=cld_path_ice,
        use_scan=use_scan,
    )
    output_sw = two_stream.solve_sw(
        p,
        temperature,
        molecules,
        optics_lib,
        atmos_state,
        vmr_fields,
        cloud_r_eff_liq=r_eff_liq,
        cloud_path_liq=cld_path_liq,
        cloud_r_eff_ice=r_eff_ice,
        cloud_path_ice=cld_path_ice,
        use_scan=use_scan,
    )

    # VERIFICATION
    (
        expected_lw_flux_up,
        expected_lw_flux_down,
        expected_sw_flux_up,
        expected_sw_flux_down,
        _,
    ) = _load_expected_data()

    expected_flux_up_lw = convert_to_3d(expected_lw_flux_up[site, :])
    expected_flux_down_lw = convert_to_3d(expected_lw_flux_down[site, :])
    expected_flux_up_sw = convert_to_3d(expected_sw_flux_up[site, :])
    expected_flux_down_sw = convert_to_3d(expected_sw_flux_down[site, :])

    np.testing.assert_allclose(
        _remove_halos(output_lw['flux_down']),
        _remove_halos(expected_flux_down_lw),
        rtol=2e-5,
        atol=0.3 if use_compact_lookup else 0.22,
    )
    np.testing.assert_allclose(
        _remove_halos(output_lw['flux_up']),
        _remove_halos(expected_flux_up_lw),
        rtol=2e-3,
        atol=0.1,
    )
    np.testing.assert_allclose(
        _remove_halos(output_sw['flux_down']),
        _remove_halos(expected_flux_down_sw),
        rtol=1.4e-3 if use_compact_lookup else 1e-3,
        atol=0,
    )
    np.testing.assert_allclose(
        _remove_halos(output_sw['flux_up']),
        _remove_halos(expected_flux_up_sw),
        rtol=3e-3 if use_compact_lookup else 1e-3,
        atol=0,
    )

  def test_per_gpoint_cloud_path_reduces_to_broadcast(self):
    """Per-g-point cloud paths must equal broadcast paths when sub-columns are
    identical across all g-points.

    This is the McICA-stratiform sanity check: if every stochastic sub-column
    is the same as the mean cloud profile, McICA-style per-g-point inputs must
    produce the same fluxes as the column-only API. Use the compact lookup
    tables to keep n_gpt small enough for the broadcast to fit in memory.
    """
    site = 0
    use_compact_lookup = True

    (
        pressure_allsites,
        pressure_level_allsites,
        temperature_allsites,
        temperature_level_allsites,
        vmr_profiles_allsites,
        _,
    ) = _setup_atmospheric_profiles()

    radiation_params = _setup_radiation_params(use_compact_lookup)
    atmos_state = atmospheric_state.from_config(
        radiation_params.atmospheric_state_cfg
    )
    optics_lib = optics.optics_factory(radiation_params.optics, atmos_state.vmr)

    n_horiz = 2
    convert_to_3d = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=n_horiz
    )
    sfc_temperature = temperature_level_allsites[site, 1] * jnp.ones(
        (n_horiz, n_horiz), dtype=jnp.float_
    )
    vmr_fields = {
        k: convert_to_3d(v[site, :]) for k, v in vmr_profiles_allsites.items()
    }
    p = convert_to_3d(pressure_allsites[site, :])
    pressure_level = convert_to_3d(pressure_level_allsites[site, :])
    temperature = convert_to_3d(temperature_allsites[site, :])

    r_liq = 1.2e-5
    r_ice = 9.5e-5 / 2
    cld_path = 1e-2
    ones = jnp.ones_like(p)
    ind_valid_p_range = jnp.logical_and(p > 10000, p < 90000)
    r_eff_liq = jnp.where(
        jnp.logical_and(ind_valid_p_range, temperature > 263), r_liq * ones, 0.0
    )
    cld_path_liq = jnp.where(
        jnp.logical_and(ind_valid_p_range, temperature > 263),
        cld_path * ones,
        0.0,
    )
    r_eff_ice = jnp.where(
        jnp.logical_and(ind_valid_p_range, temperature < 273), r_ice * ones, 0.0
    )
    cld_path_ice = jnp.where(
        jnp.logical_and(ind_valid_p_range, temperature < 273),
        cld_path * ones,
        0.0,
    )

    molecules = _air_molecules_per_area(pressure_level, vmr_fields['h2o'])

    # Reference: column-only (broadcast) cloud paths.
    ref_lw = two_stream.solve_lw(
        p, temperature, molecules, optics_lib, atmos_state, vmr_fields,
        sfc_temperature,
        cloud_r_eff_liq=r_eff_liq, cloud_path_liq=cld_path_liq,
        cloud_r_eff_ice=r_eff_ice, cloud_path_ice=cld_path_ice,
    )
    ref_sw = two_stream.solve_sw(
        p, temperature, molecules, optics_lib, atmos_state, vmr_fields,
        cloud_r_eff_liq=r_eff_liq, cloud_path_liq=cld_path_liq,
        cloud_r_eff_ice=r_eff_ice, cloud_path_ice=cld_path_ice,
    )

    # Build per-g-point arrays by tiling the single profile across n_gpt.
    n_gpt_lw = optics_lib.n_gpt_lw
    n_gpt_sw = optics_lib.n_gpt_sw
    cpl_lw_per_gpt = jnp.broadcast_to(cld_path_liq, (n_gpt_lw,) + cld_path_liq.shape)
    cpi_lw_per_gpt = jnp.broadcast_to(cld_path_ice, (n_gpt_lw,) + cld_path_ice.shape)
    cpl_sw_per_gpt = jnp.broadcast_to(cld_path_liq, (n_gpt_sw,) + cld_path_liq.shape)
    cpi_sw_per_gpt = jnp.broadcast_to(cld_path_ice, (n_gpt_sw,) + cld_path_ice.shape)

    # Per-g-point cloud paths must produce the same fluxes. Pass `None` for
    # the broadcast paths to confirm precedence: the per-g-point input is the
    # sole source of cloud condensate.
    out_lw = two_stream.solve_lw(
        p, temperature, molecules, optics_lib, atmos_state, vmr_fields,
        sfc_temperature,
        cloud_r_eff_liq=r_eff_liq, cloud_path_liq=None,
        cloud_r_eff_ice=r_eff_ice, cloud_path_ice=None,
        cloud_path_liq_per_gpt=cpl_lw_per_gpt,
        cloud_path_ice_per_gpt=cpi_lw_per_gpt,
    )
    out_sw = two_stream.solve_sw(
        p, temperature, molecules, optics_lib, atmos_state, vmr_fields,
        cloud_r_eff_liq=r_eff_liq, cloud_path_liq=None,
        cloud_r_eff_ice=r_eff_ice, cloud_path_ice=None,
        cloud_path_liq_per_gpt=cpl_sw_per_gpt,
        cloud_path_ice_per_gpt=cpi_sw_per_gpt,
    )

    for key in ('flux_up', 'flux_down', 'flux_net'):
      np.testing.assert_allclose(out_lw[key], ref_lw[key], rtol=1e-6, atol=1e-6)
      np.testing.assert_allclose(out_sw[key], ref_sw[key], rtol=1e-6, atol=1e-6)

  def test_solve_sw_vmap_keeps_gas_optics_tables_shared(self):
    """Regression guard for issue #8 (per-column gas-optics table broadcast).

    When `solve_sw` is vmapped over columns with a per-column `zenith` (the GCM
    use case), the loop-invariant gas-optics tables must remain a single shared
    operand -- they must not be broadcast across the column batch. The shortwave
    night handling used to be a `jax.lax.cond(zenith >= pi/2)` wrapping the
    g-point loop; under `vmap` that cond lowered to a `select` that ran both
    branches and broadcast the ~3 MB `kmajor` table across every column
    (~31 GB of scratch at 9216 columns). This test confirms (1) no such
    per-column table broadcast appears in the compiled HLO and (2) the
    night-masking semantics survive `vmap`: columns with the sun at or below the
    horizon return zero fluxes, and daytime columns match a direct single-column
    solve.
    """
    site = 0
    (
        pressure_allsites,
        pressure_level_allsites,
        temperature_allsites,
        _,
        vmr_profiles_allsites,
        _,
    ) = _setup_atmospheric_profiles()
    # Compact lookup keeps n_gpt small so the test compiles quickly.
    radiation_params = _setup_radiation_params(use_compact_lookup=True)
    atmos_state = atmospheric_state.from_config(
        radiation_params.atmospheric_state_cfg
    )
    optics_lib = optics.optics_factory(radiation_params.optics, atmos_state.vmr)

    convert_to_3d = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=1
    )
    vmr_fields = {
        k: convert_to_3d(v[site, :]) for k, v in vmr_profiles_allsites.items()
    }
    p = convert_to_3d(pressure_allsites[site, :])
    pressure_level = convert_to_3d(pressure_level_allsites[site, :])
    temperature = convert_to_3d(temperature_allsites[site, :])
    molecules = _air_molecules_per_area(pressure_level, vmr_fields['h2o'])

    # Per-column zeniths: a mix of daytime (< pi/2) and sun-below-horizon
    # (>= pi/2) columns, so the predicate is genuinely batched under vmap.
    zeniths = jnp.array([0.2, 0.7, 1.3, 1.7, 2.2, 2.9], dtype=jnp.float_)
    n_col = int(zeniths.shape[0])
    is_night = np.asarray(zeniths) >= 0.5 * np.pi

    def run_col(zenith: Array) -> dict[str, Array]:
      st = dataclasses.replace(atmos_state, zenith=zenith)
      return two_stream.solve_sw(
          p, temperature, molecules, optics_lib, st, vmr_fields
      )

    vmapped = jax.vmap(run_col)

    # (1) Memory guard: the bug materialises an [n_col, *kmajor.shape] buffer.
    # Match its leading dims in the optimized HLO; legitimate per-column buffers
    # are shaped [n_col, nx, ny, nz] and never collide with the table dims.
    hlo = jax.jit(vmapped).lower(zeniths).compile().as_text()
    ks = optics_lib.gas_optics_sw.kmajor.shape  # e.g. (14, 60, 9, 112)
    broadcast_needle = f'f32[{n_col},{ks[0]},{ks[1]},{ks[2]}'
    self.assertNotIn(
        broadcast_needle,
        hlo,
        msg=(
            f'gas-optics kmajor table is broadcast per-column '
            f'("{broadcast_needle}" found in compiled HLO); see issue #8.'
        ),
    )

    # (2) Semantics guard under vmap.
    out = vmapped(zeniths)
    for key in ('flux_up', 'flux_down', 'flux_net'):
      vals = np.asarray(out[key])
      self.assertTrue(
          np.all(np.isfinite(vals)), msg=f'{key} has non-finite values'
      )
      # Sun at/below horizon -> exactly zero flux.
      np.testing.assert_array_equal(
          vals[is_night], np.zeros_like(vals[is_night])
      )
      # Daytime columns carry nonzero shortwave flux.
      self.assertTrue(np.any(vals[~is_night] != 0.0))

    # Daytime vmap columns must match a direct single-column solve.
    for i in np.nonzero(~is_night)[0]:
      ref = run_col(zeniths[i])
      for key in ('flux_up', 'flux_down', 'flux_net'):
        np.testing.assert_allclose(
            np.asarray(out[key][i]), np.asarray(ref[key]), rtol=1e-6, atol=1e-6
        )


class GPointChunkingTest(unittest.TestCase):
  """`gpt_chunk` must change only the cost of the solve, never its answer.

  Solving several g-points per iteration of the spectral loop is a pure
  restructuring -- g-points are independent radiative transfer problems -- so
  every chunk size must reproduce the unchunked fluxes to within the reordering
  of one floating-point sum. The cases below deliberately exercise the three
  quantities that are indexed *per g-point* rather than per column, because a
  chunked gather of any of them is where a silent mis-association would live:
  the gas-optics table slices, the McICA per-g-point cloud sub-column, and the
  per-band aerosol lookup (whose band index varies *within* a chunk once the
  chunk is wider than a band's g-point range).
  """

  def _setup(self, n_horiz=2):
    return _compact_cloudy_setup(n_horiz)

  def _mcica_and_aerosol(self, optics_lib, band, cloud, shape):
    """McICA sub-columns and a band-dependent aerosol bundle for `band`."""
    rng = np.random.default_rng(0)
    n_gpt = optics_lib.n_gpt_lw if band == 'lw' else optics_lib.n_gpt_sw
    gas_lookup = (
        optics_lib.gas_optics_lw if band == 'lw' else optics_lib.gas_optics_sw
    )
    n_bnd = gas_lookup.n_bnd

    def subcolumns(path):
      draw = jnp.asarray(rng.random((n_gpt,) + shape) < 0.5, jnp.float_)
      return draw * path

    # The aerosol optical depth is a different multiple of a base value in every
    # band, so a chunk that gathered the wrong band -- or broadcast one band's
    # value across the chunk -- cannot agree with the unchunked solve.
    scale = jnp.arange(1, n_bnd + 1, dtype=jnp.float_).reshape(
        (n_bnd,) + (1,) * len(shape)
    )
    aerosol = {
        'optical_depth': scale * jnp.full((n_bnd,) + shape, 2e-2, jnp.float_),
        'ssa': (scale / n_bnd) * jnp.full((n_bnd,) + shape, 0.8, jnp.float_),
        'asymmetry_factor': jnp.full((n_bnd,) + shape, 0.6, jnp.float_),
    }
    return {
        'cloud_path_liq_per_gpt': subcolumns(cloud['cloud_path_liq']),
        'cloud_path_ice_per_gpt': subcolumns(cloud['cloud_path_ice']),
        'aerosol_optics': aerosol,
    }

  def _solve(self, band, gpt_chunk, extra=None):
    (optics_lib, atmos_state, p, temperature, molecules, vmr_fields,
     sfc_temperature, cloud) = self._cached
    kwargs = dict(cloud)
    if extra is not None:
      kwargs.update(extra)
    if band == 'lw':
      return two_stream.solve_lw(
          p, temperature, molecules, optics_lib, atmos_state, vmr_fields,
          sfc_temperature, gpt_chunk=gpt_chunk, **kwargs
      )
    return two_stream.solve_sw(
        p, temperature, molecules, optics_lib, atmos_state, vmr_fields,
        gpt_chunk=gpt_chunk, **kwargs
    )

  # Loading the profiles and solving the unchunked reference are the same work
  # for every parameterisation below, and an unrolled solve is not cheap to
  # compile, so both are memoised on the class rather than redone per case.
  _fixture = None
  _reference = {}

  def setUp(self):
    super().setUp()
    if GPointChunkingTest._fixture is None:
      GPointChunkingTest._fixture = self._setup()
    self._cached = GPointChunkingTest._fixture

  def _extra_for(self, band, with_mcica):
    if not with_mcica:
      return None
    return self._mcica_and_aerosol(
        self._cached[0], band, self._cached[7], self._cached[3].shape
    )

  def _reference_for(self, band, with_mcica):
    key = (band, with_mcica)
    if key not in GPointChunkingTest._reference:
      GPointChunkingTest._reference[key] = self._solve(
          band, 1, self._extra_for(band, with_mcica)
      )
    return GPointChunkingTest._reference[key]

  @parameterized.expand([
      (band, chunk, with_mcica)
      # 3 is deliberately not a divisor of the g-point count (128 / 112): it
      # exercises the padded tail iteration, whose surplus g-points must be
      # masked out of the spectral sum rather than double-counted.
      for band, chunk, with_mcica in product(
          ('lw', 'sw'), (2, 16, 3), (False, True)
      )
  ])
  def test_gpt_chunk_matches_unchunked(self, band, chunk, with_mcica):
    ref = self._reference_for(band, with_mcica)
    got = self._solve(band, chunk, self._extra_for(band, with_mcica))
    for key in ('flux_up', 'flux_down', 'flux_net'):
      # The only admissible difference is the reordered spectral sum, which at
      # float32 over ~128 g-points is a few ulp of the total.
      np.testing.assert_allclose(
          np.asarray(got[key]), np.asarray(ref[key]), rtol=1e-5, atol=1e-4,
          err_msg=f'{band} {key} changed with gpt_chunk={chunk}',
      )

  def test_gpt_chunk_keeps_gas_optics_tables_shared(self):
    """The chunk must gather table *slices*, not replicate whole tables.

    This is the g-point analogue of the issue #8 per-column blowup guarded in
    `test_solve_sw_vmap_keeps_gas_optics_tables_shared`. The chunk is batched by
    mapping over the g-point index; a loop-invariant operand dragged into that
    map would materialise a `[chunk, *table.shape]` buffer -- for `kmajor`, the
    whole ~3 MB table once per chunk element. That must not happen. What the
    chunk *is* expected to materialise is the per-g-point table slice it already
    took at chunk size one, `chunk` of them: `[chunk, 14, 60, 9]`, a few hundred
    kB, which is the design.

    The second assertion states the same invariant as a number rather than a
    string match, and is the one that actually matters: the whole trade here is
    launches for memory, so memory must grow no faster than the *fields* do. It
    is measured at a column batch large enough for the fields to dominate the
    scratch -- at the two-by-two grid the rest of this class uses, a field is
    176 elements and the per-g-point table slices alone are several times the
    whole rest of the program, so the ratio there measures nothing.
    """
    chunk = 8
    fixture = self._setup(n_horiz=12)
    optics_lib, _, _, temperature = fixture[0], fixture[1], fixture[2], fixture[3]

    def compile_at(gpt_chunk):
      saved, self._cached = self._cached, fixture
      try:
        return jax.jit(
            lambda t: self._solve_with_temperature('sw', gpt_chunk, t)['flux_net']
        ).lower(temperature).compile()
      finally:
        self._cached = saved

    compiled = compile_at(chunk)
    ks = optics_lib.gas_optics_sw.kmajor.shape  # (14, 60, 9, n_gpt)
    needle = 'f32[' + ','.join(str(d) for d in (chunk,) + tuple(ks))
    self.assertNotIn(
        needle, compiled.as_text(),
        msg=(f'the whole gas-optics kmajor table is replicated across the '
             f'g-point chunk ("{needle}" found in compiled HLO); the chunk must '
             f'index the table, not copy it. See issue #8 for the per-column '
             f'version of this failure.'),
    )

    unchunked_bytes = compile_at(1).memory_analysis().temp_size_in_bytes
    chunked_bytes = compiled.memory_analysis().temp_size_in_bytes
    # Linear in the chunk, with 25% of slack for the per-g-point table slices
    # and the chunk's own index bookkeeping, neither of which scales with the
    # fields. A replicated table would be orders of magnitude over this.
    self.assertLessEqual(
        chunked_bytes, 1.25 * chunk * unchunked_bytes,
        msg=(f'gpt_chunk={chunk} needs {chunked_bytes:,} bytes of scratch '
             f'against {unchunked_bytes:,} unchunked -- more than the {chunk}x '
             f'the batched fields account for, so something table-sized is '
             f'being replicated per g-point.'),
    )

  def _solve_with_temperature(self, band, gpt_chunk, temperature):
    """`_solve` with the temperature taken from an argument, for lowering."""
    (optics_lib, atmos_state, p, _, molecules, vmr_fields,
     sfc_temperature, cloud) = self._cached
    if band == 'lw':
      return two_stream.solve_lw(
          p, temperature, molecules, optics_lib, atmos_state, vmr_fields,
          sfc_temperature, gpt_chunk=gpt_chunk, **cloud
      )
    return two_stream.solve_sw(
        p, temperature, molecules, optics_lib, atmos_state, vmr_fields,
        gpt_chunk=gpt_chunk, **cloud
    )

  def test_gradients_stay_finite_under_chunking(self):
    """Reverse mode through the *whole* chunked solve, not a kernel in isolation.

    Differentiating the solve end to end is the point: a NaN in the two-stream
    cell kernels once got past a standalone kernel test and only showed up
    through the full reverse pass. The cloudy column below drives the kernels
    into their awkward regimes -- optically thin layers at cloud edges, and the
    near-conservative-scattering limit inside the cloud -- while staying a
    configuration the model actually produces.

    The aerosol path is exercised separately in
    `test_aerosol_gradients_stay_finite` rather than here, only to keep this
    test's cloud-edge / in-cloud focus uncluttered -- not because it is
    unsafe. (An earlier revision of this comment claimed an `aerosol_optics`
    bundle NaN'd the reverse pass; that was the conservative-scattering
    `k^2 = 0` defect, which was already fixed for gas, cloud *and* aerosol by
    the `x2`-parameterised two-stream helpers -- see issue #30 and the sweep
    in `test_aerosol_gradients_stay_finite`.)
    """
    (optics_lib, atmos_state, p, temperature, molecules, vmr_fields,
     sfc_temperature, cloud) = self._cached

    def loss(temp, gpt_chunk):
      fluxes = two_stream.solve_sw(
          p, temp, molecules, optics_lib, atmos_state, vmr_fields,
          gpt_chunk=gpt_chunk, **cloud
      )
      return jnp.sum(fluxes['flux_net'] ** 2)

    ref = np.asarray(jax.grad(loss)(temperature, 1))
    self.assertTrue(np.all(np.isfinite(ref)), 'unchunked gradient is not finite')
    for chunk in (2, 8):
      got = np.asarray(jax.grad(loss)(temperature, chunk))
      self.assertTrue(
          np.all(np.isfinite(got)),
          msg=f'gradient has non-finite values at gpt_chunk={chunk}',
      )
      scale = max(np.max(np.abs(ref)), 1e-30)
      np.testing.assert_allclose(
          got / scale, ref / scale, rtol=1e-4, atol=1e-5,
          err_msg=f'gradient changed with gpt_chunk={chunk}',
      )

  def test_aerosol_gradients_stay_finite(self):
    """Reverse mode through the *whole* solve *with* an `aerosol_optics` bundle.

    Regression for issue #30. A per-band aerosol bundle is mixed into the
    gas+cloud optics via `combine_optical_properties`, and its single-scattering
    albedo drives the *combined* `ssa` into regimes a gas/cloud-only column may
    never reach on its own -- in particular `ssa = 1` exactly, which lands the
    two-stream at conservative scattering `k^2 = (gamma1+gamma2)(gamma1-gamma2)
    = 0`. That is the same degenerate point at which a `sqrt(k2)` on the reverse
    path once manufactured `0 * inf = NaN` cotangents (fixed for gas, cloud and
    aerosol alike by the `x2`-parameterised hyperbolic helpers in
    `monochromatic_two_stream`). The `combine_optical_properties` normalisations
    themselves divide by `jnp.maximum(x, EPSILON)`, whose clamped-branch adjoint
    is a finite zero, so they add no new poison.

    The sweep includes the corners the issue calls out -- `optical_depth`
    spanning zero / tiny / physical, and `ssa = 1.0` reached *exactly* (not
    merely approached) -- for both the shortwave and longwave solves, since both
    share the mix and the diffuse two-stream code. Forward fluxes were finite
    throughout even when the reverse pass was not, so a finite-forward assertion
    would not have caught the original defect; the cotangents are what matter.
    """
    (optics_lib, atmos_state, p, temperature, molecules, vmr_fields,
     sfc_temperature, cloud) = self._cached

    def band_bundle(n_bnd, optical_depth, ssa, asymmetry):
      shape = (n_bnd,) + temperature.shape
      return {
          'optical_depth': jnp.full(shape, optical_depth, jnp.float_),
          'ssa': jnp.full(shape, ssa, jnp.float_),
          'asymmetry_factor': jnp.full(shape, asymmetry, jnp.float_),
      }

    # One compiled reverse pass per band, reused across every bundle below
    # (the bundle is a traced argument), so the whole sweep costs two solves'
    # worth of compilation rather than one per parameter combination.
    @jax.jit
    def sw_grad(temp, aerosol):
      def loss(t):
        fluxes = two_stream.solve_sw(
            p, t, molecules, optics_lib, atmos_state, vmr_fields,
            aerosol_optics=aerosol, gpt_chunk=1, **cloud
        )
        return jnp.sum(fluxes['flux_net'] ** 2)
      return jax.grad(loss)(temp)

    @jax.jit
    def lw_grad(temp, aerosol):
      def loss(t):
        fluxes = two_stream.solve_lw(
            p, t, molecules, optics_lib, atmos_state, vmr_fields,
            sfc_temperature, aerosol_optics=aerosol, gpt_chunk=1, **cloud
        )
        return jnp.sum(fluxes['flux_net'] ** 2)
      return jax.grad(loss)(temp)

    optical_depths = (0.0, 1e-8, 1e-2)
    ssas = (0.0, 0.999, 1.0)  # 1.0 exactly -> combined ssa can hit k^2 = 0
    asymmetries = (0.0, 0.85)
    cases = [
        (band, grad_fn, n_bnd, od, ssa, g)
        for band, grad_fn, n_bnd in (
            ('sw', sw_grad, optics_lib.gas_optics_sw.n_bnd),
            ('lw', lw_grad, optics_lib.gas_optics_lw.n_bnd),
        )
        for od in optical_depths
        for ssa in ssas
        for g in asymmetries
    ]
    for band, grad_fn, n_bnd, od, ssa, g in cases:
      grad = np.asarray(grad_fn(temperature, band_bundle(n_bnd, od, ssa, g)))
      self.assertTrue(
          np.all(np.isfinite(grad)),
          msg=(f'{band} temperature gradient has non-finite values with '
               f'aerosol optical_depth={od}, ssa={ssa}, asymmetry={g}'),
      )


class CloudTauScaleTest(unittest.TestCase):
  """`cloud_tau_scale_{liq,ice}` through the full solves (issue #37)."""

  _fixture = None

  def setUp(self):
    super().setUp()
    if CloudTauScaleTest._fixture is None:
      CloudTauScaleTest._fixture = _compact_cloudy_setup()
    (self.optics_lib, self.atmos_state, self.p, self.temperature,
     self.molecules, self.vmr_fields, self.sfc_temperature,
     self.cloud) = CloudTauScaleTest._fixture

  def tearDown(self):
    # Each eager solve compiles and caches its own executable, tens of MB
    # apiece, and the whole suite runs in one process. Drop them so this
    # class does not add its solves to the suite's peak memory.
    jax.clear_caches()
    super().tearDown()

  def _solve(self, band, cloud=None, **kwargs):
    cloud = self.cloud if cloud is None else cloud
    if band == 'lw':
      return two_stream.solve_lw(
          self.p, self.temperature, self.molecules, self.optics_lib,
          self.atmos_state, self.vmr_fields, self.sfc_temperature,
          **cloud, **kwargs
      )
    return two_stream.solve_sw(
        self.p, self.temperature, self.molecules, self.optics_lib,
        self.atmos_state, self.vmr_fields, **cloud, **kwargs
    )

  @parameterized.expand([('lw',), ('sw',)])
  def test_unit_factors_are_bit_for_bit(self, band):
    default = self._solve(band)
    unit = self._solve(band, cloud_tau_scale_liq=1.0, cloud_tau_scale_ice=1.0)
    for key in ('flux_up', 'flux_down', 'flux_net'):
      np.testing.assert_array_equal(
          np.asarray(unit[key]), np.asarray(default[key]), err_msg=key
      )

  @parameterized.expand([('lw',), ('sw',)])
  def test_equal_factors_match_scaled_cloud_paths(self, band):
    factor = 0.6
    scaled_tau = self._solve(
        band, cloud_tau_scale_liq=factor, cloud_tau_scale_ice=factor
    )
    cloud = dict(self.cloud)
    cloud['cloud_path_liq'] = factor * cloud['cloud_path_liq']
    cloud['cloud_path_ice'] = factor * cloud['cloud_path_ice']
    scaled_path = self._solve(band, cloud=cloud)
    unscaled = self._solve(band)
    for key in ('flux_up', 'flux_down', 'flux_net'):
      np.testing.assert_allclose(
          np.asarray(scaled_tau[key]), np.asarray(scaled_path[key]),
          rtol=1e-5, atol=1e-4, err_msg=key,
      )
    # And the factor is not a no-op.
    self.assertGreater(
        np.max(np.abs(np.asarray(scaled_tau['flux_net'] - unscaled['flux_net']))),
        1.0,
    )

  def test_unequal_factors_differ_from_scaled_cloud_paths(self):
    """With unequal factors the two are different physics, as intended.

    Scaling the paths re-weights the combined single-scattering albedo and
    asymmetry factor of the mixed-phase layers; scaling the optical depths does
    not. The optical depths agree, so the difference is purely the weighting.
    """
    f_liq, f_ice = 0.4, 0.85
    scaled_tau = self._solve(
        'sw', cloud_tau_scale_liq=f_liq, cloud_tau_scale_ice=f_ice
    )
    cloud = dict(self.cloud)
    cloud['cloud_path_liq'] = f_liq * cloud['cloud_path_liq']
    cloud['cloud_path_ice'] = f_ice * cloud['cloud_path_ice']
    scaled_path = self._solve('sw', cloud=cloud)
    self.assertGreater(
        np.max(np.abs(np.asarray(scaled_tau['flux_up'] - scaled_path['flux_up']))),
        1e-2,
    )

  @parameterized.expand([('lw',), ('sw',)])
  def test_gradients_are_finite(self, band):
    """Reverse mode w.r.t. per-cell factors, zero in some cells, is finite."""
    shape = self.temperature.shape
    f_liq = jnp.full(shape, 0.7, jnp.float_).at[:, :, ::3].set(0.0)
    f_ice = jnp.full(shape, 0.85, jnp.float_).at[:, :, 1::4].set(0.0)

    def loss(f_liq, f_ice):
      fluxes = self._solve(
          band, cloud_tau_scale_liq=f_liq, cloud_tau_scale_ice=f_ice
      )
      return jnp.sum(fluxes['flux_net'] ** 2)

    grads = jax.grad(loss, argnums=(0, 1))(f_liq, f_ice)
    for name, grad in zip(('liq', 'ice'), grads):
      grad = np.asarray(grad)
      self.assertTrue(
          np.all(np.isfinite(grad)), msg=f'{band} d/d(f_{name}) not finite'
      )
      # Cloudy layers respond; cloud-free layers have no optical depth to
      # scale, so their factor has no effect at all.
      self.assertGreater(np.max(np.abs(grad)), 0.0)
      path = np.asarray(self.cloud[f'cloud_path_{name}'])
      np.testing.assert_array_equal(grad[path == 0], 0.0)


class SurfaceAlbedoTest(unittest.TestCase):
  """Separate direct / diffuse and per-band surface albedos (issue #38).

  The two-stream solver already treats the two separately: the direct beam
  reaching the surface is reflected with the direct albedo into the diffuse
  source (`sw_cell_source`), and the diffuse downwelling flux is reflected with
  the diffuse albedo (`sw_transport`). At the surface face that gives the exact
  identity

    flux_up = a_dir * F_dir + a_dif * (flux_down - F_dir),

  with `F_dir` the direct beam incident on the surface, which several tests
  below lean on.
  """

  _fixture = None
  _hw = 1  # Surface face of the flux fields.

  def setUp(self):
    super().setUp()
    if SurfaceAlbedoTest._fixture is None:
      SurfaceAlbedoTest._fixture = _compact_cloudy_setup()
    (self.optics_lib, self.atmos_state, self.p, self.temperature,
     self.molecules, self.vmr_fields, _, self.cloud) = SurfaceAlbedoTest._fixture
    self.n_bnd = self.optics_lib.gas_optics_sw.n_bnd
    self.plane = tuple(self.temperature.shape[:2])

  def tearDown(self):
    # Each eager solve compiles and caches its own executable, tens of MB
    # apiece, and the whole suite runs in one process. Drop them so this
    # class does not add its solves to the suite's peak memory.
    jax.clear_caches()
    super().tearDown()

  def _solve(self, cloudy=False, **kwargs):
    cloud = self.cloud if cloudy else {}
    return two_stream.solve_sw(
        self.p, self.temperature, self.molecules, self.optics_lib,
        self.atmos_state, self.vmr_fields, **cloud, **kwargs
    )

  def _surface(self, fluxes):
    hw = self._hw
    return (np.asarray(fluxes['flux_up'][:, :, hw]),
            np.asarray(fluxes['flux_down'][:, :, hw]),
            np.asarray(fluxes['flux_down_dir_sfc']))

  def test_omitted_or_matching_albedos_are_bit_for_bit(self):
    sfc_alb = self.atmos_state.sfc_alb
    default = self._solve(cloudy=True)
    matching = {
        'scalar': dict(sfc_alb_dir=sfc_alb, sfc_alb_dif=sfc_alb),
        'direct only': dict(sfc_alb_dir=sfc_alb),
        'per column': dict(
            sfc_alb_dir=jnp.full(self.plane, sfc_alb, jnp.float_),
            sfc_alb_dif=jnp.full(self.plane, sfc_alb, jnp.float_),
        ),
        'per band': dict(
            sfc_alb_dir=jnp.full(self.plane + (self.n_bnd,), sfc_alb,
                                 jnp.float_),
            sfc_alb_dif=jnp.full((self.n_bnd,), sfc_alb, jnp.float_)[
                None, None, :],
        ),
    }
    for name, kwargs in matching.items():
      got = self._solve(cloudy=True, **kwargs)
      for key in ('flux_up', 'flux_down', 'flux_net', 'flux_down_dir_sfc'):
        np.testing.assert_array_equal(
            np.asarray(got[key]), np.asarray(default[key]),
            err_msg=f'{name}: {key}',
        )

  @parameterized.expand([('clear', False), ('cloudy', True)])
  def test_surface_fluxes_partition_into_direct_and_diffuse(self, _, cloudy):
    a_dir, a_dif = 0.35, 0.08
    fluxes = self._solve(cloudy=cloudy, sfc_alb_dir=a_dir, sfc_alb_dif=a_dif)
    up, down, direct = self._surface(fluxes)
    self.assertTrue(np.all(direct >= 0.0))
    self.assertTrue(np.all(direct <= down))
    np.testing.assert_allclose(
        up, a_dir * direct + a_dif * (down - direct), rtol=1e-5, atol=1e-4
    )
    # The direct beam never sees the surface on its way down, so it does not
    # depend on either albedo.
    np.testing.assert_array_equal(
        direct, np.asarray(self._solve(cloudy=cloudy)['flux_down_dir_sfc'])
    )
    if not cloudy:
      # A clear sky lets a large share of the sun through unscattered.
      self.assertTrue(np.all(direct > 0.5 * down))

  def test_direct_albedo_is_irrelevant_without_a_direct_beam(self):
    """Under an opaque cloud all surface light is diffuse."""
    cloud = dict(self.cloud)
    cloud['cloud_path_liq'] = 20.0 * cloud['cloud_path_liq']
    base = two_stream.solve_sw(
        self.p, self.temperature, self.molecules, self.optics_lib,
        self.atmos_state, self.vmr_fields, **cloud,
        sfc_alb_dir=0.0, sfc_alb_dif=0.3,
    )
    _, down, direct = self._surface(base)
    self.assertTrue(np.all(down > 1.0))
    np.testing.assert_allclose(direct, 0.0, atol=1e-6)
    bright = two_stream.solve_sw(
        self.p, self.temperature, self.molecules, self.optics_lib,
        self.atmos_state, self.vmr_fields, **cloud,
        sfc_alb_dir=1.0, sfc_alb_dif=0.3,
    )
    for key in ('flux_up', 'flux_down'):
      np.testing.assert_allclose(
          np.asarray(bright[key]), np.asarray(base[key]), rtol=0, atol=1e-5,
          err_msg=key,
      )

  def test_per_band_albedo_uses_each_gpoints_band(self):
    """Band `k`'s albedo acts on band `k`'s light and nothing else.

    An opaque, purely absorbing aerosol in every band but `k` leaves only band
    `k`'s direct beam at the surface. With a zero diffuse albedo the surface
    upward flux is then the direct albedo times that beam, so an albedo that is
    one in band `k` alone must reflect all of it and one that is one in every
    other band must reflect none. The aerosol takes its band from the same
    g-point -> band map, which is tested on its own.
    """
    g_to_b = np.asarray(self.optics_lib.gas_optics_sw.g_point_to_bnd)
    solar = np.asarray(self.optics_lib.solar_fraction_by_gpt)
    band_solar = np.bincount(g_to_b, weights=solar, minlength=self.n_bnd)
    k = int(np.argmax(band_solar))
    only_k = np.arange(self.n_bnd) == k
    shape = (self.n_bnd,) + self.temperature.shape
    tau = np.where(only_k, 0.0, 50.0).reshape((self.n_bnd, 1, 1, 1))
    aerosol = {
        'optical_depth': jnp.asarray(np.broadcast_to(tau, shape), jnp.float_),
        'ssa': jnp.zeros(shape, jnp.float_),
        'asymmetry_factor': jnp.zeros(shape, jnp.float_),
    }

    def surface_up(albedo_by_band):
      albedo = jnp.broadcast_to(
          jnp.asarray(albedo_by_band, jnp.float_), self.plane + (self.n_bnd,)
      )
      fluxes = self._solve(
          aerosol_optics=aerosol, sfc_alb_dir=albedo, sfc_alb_dif=0.0
      )
      up, _, direct = self._surface(fluxes)
      return up, direct

    up_k, direct = surface_up(only_k.astype(float))
    self.assertTrue(np.all(direct > 1.0))
    np.testing.assert_allclose(up_k, direct, rtol=1e-5)
    up_others, _ = surface_up((~only_k).astype(float))
    np.testing.assert_allclose(up_others, 0.0, atol=1e-4)

  def test_per_band_albedo_is_chunk_invariant(self):
    """The band gather is per g-point, also inside a g-point chunk."""
    rng = np.random.default_rng(0)
    alb_dir = jnp.asarray(rng.uniform(0.0, 1.0, self.plane + (self.n_bnd,)),
                          jnp.float_)
    alb_dif = jnp.asarray(rng.uniform(0.0, 1.0, (self.n_bnd,)), jnp.float_)[
        None, None, :]
    ref = self._solve(cloudy=True, sfc_alb_dir=alb_dir, sfc_alb_dif=alb_dif)
    got = self._solve(cloudy=True, sfc_alb_dir=alb_dir, sfc_alb_dif=alb_dif,
                      gpt_chunk=16)
    for key in ('flux_up', 'flux_down', 'flux_net', 'flux_down_dir_sfc'):
      np.testing.assert_allclose(
          np.asarray(got[key]), np.asarray(ref[key]), rtol=1e-5, atol=1e-4,
          err_msg=key,
      )

  def test_albedo_gradients_are_finite(self):
    """Reverse mode w.r.t. per-band albedos, including the 0 and 1 ends."""
    alb_dir = jnp.linspace(0.0, 1.0, self.n_bnd, dtype=jnp.float_)
    alb_dir = jnp.broadcast_to(alb_dir, self.plane + (self.n_bnd,))
    alb_dif = jnp.linspace(1.0, 0.0, self.n_bnd, dtype=jnp.float_)[
        None, None, :]

    def loss(alb_dir, alb_dif):
      fluxes = self._solve(
          cloudy=True, sfc_alb_dir=alb_dir, sfc_alb_dif=alb_dif
      )
      return (jnp.sum(fluxes['flux_net'] ** 2)
              + jnp.sum(fluxes['flux_down_dir_sfc']))

    grads = jax.grad(loss, argnums=(0, 1))(alb_dir, alb_dif)
    for name, grad in zip(('sfc_alb_dir', 'sfc_alb_dif'), grads):
      grad = np.asarray(grad)
      self.assertTrue(np.all(np.isfinite(grad)), msg=f'd/d{name} not finite')
      self.assertGreater(np.max(np.abs(grad)), 0.0, msg=name)

  def test_night_columns_have_no_direct_flux(self):
    night = dataclasses.replace(self.atmos_state, zenith=0.6 * np.pi)
    fluxes = two_stream.solve_sw(
        self.p, self.temperature, self.molecules, self.optics_lib, night,
        self.vmr_fields, sfc_alb_dir=0.2, sfc_alb_dif=0.1,
    )
    np.testing.assert_array_equal(
        np.asarray(fluxes['flux_down_dir_sfc']), 0.0
    )

  def test_albedo_shape_resolution(self):
    resolve = two_stream._sw_sfc_albedo_by_band  # pylint: disable=protected-access
    n = self.n_bnd
    lib = self.optics_lib
    # Per column: anything that broadcasts against the plane.
    for shape in ((), (1,), (2,), (2, 2), (1, 2)):
      self.assertIsNone(resolve(jnp.zeros(shape), (2, 2), lib, 'a'), shape)
    # Per band, laid out band-major for the gather.
    by_band = resolve(
        jnp.arange(n, dtype=jnp.float_)[None, None, :] * jnp.ones((2, 2, 1)),
        (2, 2), lib, 'a',
    )
    self.assertEqual(by_band.shape, (n, 2, 2))
    np.testing.assert_array_equal(by_band[:, 1, 0], np.arange(n))
    # A band vector that cannot be per column is one spectrum for every
    # column; on a single-column plane that is the column-`vmap` case.
    for plane in ((1, 1), (2, 2)):
      self.assertEqual(
          resolve(jnp.zeros((n,)), plane, lib, 'a').shape, (n,) + plane
      )
    for shape in ((3,), (2, 2, n + 1), (3, 2, n), (2, 2, n, 1)):
      with self.assertRaises(ValueError, msg=f'{shape}'):
        resolve(jnp.zeros(shape), (2, 2), lib, 'a')
    gray = optics.GrayAtmosphereOptics(
        radiative_transfer.OpticsParameters(
            optics=radiative_transfer.GrayAtmosphereOptics()
        )
    )
    with self.assertRaises(ValueError):
      resolve(jnp.zeros((2, 2, n)), (2, 2), gray, 'a')


if __name__ == '__main__':
  unittest.main()
