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

    The aerosol path is deliberately *not* differentiated here. Supplying any
    `aerosol_optics` bundle NaNs the reverse pass of `solve_sw` on the RFMIP
    clear-sky profile, at `gpt_chunk=1` and on the parent commit alike, so it is
    a pre-existing defect in the aerosol mix rather than anything chunking does;
    including it would make this test fail for an unrelated reason. Reproduce
    with `solve_sw(..., aerosol_optics={'optical_depth': full(1e-2), 'ssa':
    full(1.0), 'asymmetry_factor': zeros})` under `jax.grad` -- the forward
    fluxes are finite, only the cotangents are not.
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


if __name__ == '__main__':
  unittest.main()
