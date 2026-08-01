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

# filenames
_VMR_GLOBAL_MEAN_FILENAME = 'rrtmgp/optics/test_data/vmr_global_means.json'
_ATMOSPHERIC_STATE_FILENAME = 'rrtmgp/optics/test_data/clearsky_as.nc'
_LW_FLUX_DOWN_REFERENCE_FILENAME = 'rrtmgp/optics/test_data/clearsky_lw_flux_dn_TwoStream.nc'
_LW_FLUX_UP_REFERENCE_FILENAME = 'rrtmgp/optics/test_data/clearsky_lw_flux_up_TwoStream.nc'
_SW_FLUX_DOWN_REFERENCE_FILENAME = 'rrtmgp/optics/test_data/clearsky_sw_flux_dn_TwoStream.nc'
_SW_FLUX_UP_REFERENCE_FILENAME = 'rrtmgp/optics/test_data/clearsky_sw_flux_up_TwoStream.nc'

# root for relative paths
root = Path()
_VMR_GLOBAL_MEAN_FILEPATH = root / _VMR_GLOBAL_MEAN_FILENAME
_ATMOSPHERIC_STATE_FILEPATH = root / _ATMOSPHERIC_STATE_FILENAME
_LW_FLUX_DOWN_REFERENCE_FILEPATH = root / _LW_FLUX_DOWN_REFERENCE_FILENAME
_LW_FLUX_UP_REFERENCE_FILEPATH = root / _LW_FLUX_UP_REFERENCE_FILENAME
_SW_FLUX_DOWN_REFERENCE_FILEPATH = root / _SW_FLUX_DOWN_REFERENCE_FILENAME
_SW_FLUX_UP_REFERENCE_FILEPATH = root / _SW_FLUX_UP_REFERENCE_FILENAME

# open netCDF files once
_atm_ds = nc.Dataset(_ATMOSPHERIC_STATE_FILEPATH, 'r')
_ref_lw_up_ds   = nc.Dataset(_LW_FLUX_UP_REFERENCE_FILEPATH, 'r')
_ref_lw_down_ds = nc.Dataset(_LW_FLUX_DOWN_REFERENCE_FILEPATH, 'r')
_ref_sw_up_ds   = nc.Dataset(_SW_FLUX_UP_REFERENCE_FILEPATH, 'r')
_ref_sw_down_ds = nc.Dataset(_SW_FLUX_DOWN_REFERENCE_FILEPATH, 'r')


def _remove_halos(f: Array) -> Array:
    """Remove the halos from the output."""
    return f[:, :, 1:-1]


def _interior_and_toa(f: Array) -> Array:
    """Drop the bottom halo but keep the top face, which is the TOA level.

    The reference fluxes are padded at the bottom only, so index -1 holds the
    top-of-atmosphere level rather than an unused halo. Comparing only
    `_remove_halos` therefore leaves the TOA -- the level OLR is read from --
    completely unchecked, which is how a spurious extrapolation there went
    unnoticed (issue #19).
    """
    return f[:, :, 1:]


def _setup_radiation_params(
    site: int,
    use_compact_lookup: bool,
    atmos_state_ds: nc.Dataset,
) -> radiative_transfer.RadiativeTransfer:
    """Create an instance of `RadiativeTransfer`."""
    if use_compact_lookup:
        lw_lookup = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g128.nc'
        sw_lookup = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-sw-g112.nc'
    else:
        lw_lookup = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g256.nc'
        sw_lookup = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-sw-g224.nc'

    return radiative_transfer.RadiativeTransfer(
        optics=radiative_transfer.OpticsParameters(
            optics=radiative_transfer.RRTMOptics(
                longwave_nc_filepath=lw_lookup,
                shortwave_nc_filepath=sw_lookup,
                cloud_longwave_nc_filepath='rrtmgp/optics/rrtmgp_data/cloudysky_lw.nc',
                cloud_shortwave_nc_filepath='rrtmgp/optics/rrtmgp_data/cloudysky_sw.nc',
            )
        ),
        atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
            sfc_emis=atmos_state_ds['surface_emissivity'][:].data[site],
            sfc_alb=atmos_state_ds['surface_albedo'][:].data[site],
            zenith=np.radians(atmos_state_ds['solar_zenith_angle'][:].data[site]),
            irrad=atmos_state_ds['total_solar_irradiance'][:].data[site],
            toa_flux_lw=0.0,
            vmr_global_mean_filepath=_VMR_GLOBAL_MEAN_FILEPATH,
        ),
    )


def _setup_atmospheric_profiles() -> tuple[
    nc.Dataset,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load vertical profiles from netCDF."""
    ds = _atm_ds
    halo = 1
    p_int = np.flip(ds['pres_layer'][:].data, axis=-1)
    p_lev = np.flip(ds['pres_level'][:].data, axis=-1)

    pad2 = ((0, 0), (halo, halo))
    pad2_lev = ((0, 0), (halo, halo-1))
    pressure = np.pad(p_int, pad2, mode='edge')
    pressure_level = np.pad(p_lev, pad2_lev, mode='edge')

    t_int = np.flip(ds['temp_layer'][:].data, axis=-1)
    t_lev = np.flip(ds['temp_level'][:].data, axis=-1)
    nx, ny, nz = t_int.shape
    nzh = nz + 2*halo
    temperature = np.zeros((nx, ny, nzh), dtype=jnp.float_)
    temperature[:, :, halo:-halo] = t_int
    temperature[:, :, 0] =        2*t_lev[:, :, 0]  - t_int[:, :, 0]
    temperature[:, :, -1] = 2*t_lev[:, :, -1] - t_int[:, :, -1]

    temp_lev3 = np.zeros_like(temperature)
    temp_lev3[:, :, 1:] = t_lev
    temp_lev3[:, :, 0]  = temperature[:, :, 0]

    pad3 = ((0,0),(0,0),(halo, halo))
    h2o = np.pad(np.flip(ds['water_vapor'][:].data, axis=-1), pad3, mode='edge')
    o3  = np.pad(np.flip(ds['ozone'][:].data, axis=-1), pad3, mode='edge')

    sfc_t = ds['surface_temperature'][:].data
    return (ds, pressure, pressure_level, temperature, temp_lev3, h2o, o3, sfc_t)


def _load_expected_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads expected fluxes from reference files."""
    up_ds = _ref_lw_up_ds
    dn_ds = _ref_lw_down_ds
    su_ds = _ref_sw_up_ds
    sd_ds = _ref_sw_down_ds

    halo = 1
    pads = ((0,0),(0,0),(halo, halo-1))
    lw_up = np.pad(np.flip(up_ds['rlu'][:].data, axis=-1), pads)
    lw_dn = np.pad(np.flip(dn_ds['rld'][:].data, axis=-1), pads)
    sw_up = np.pad(np.flip(su_ds['rsu'][:].data, axis=-1), pads)
    sw_dn = np.pad(np.flip(sd_ds['rsd'][:].data, axis=-1), pads)
    return lw_up, lw_dn, sw_up, sw_dn


def _air_molecules_per_area(p_bottom: Array, vmr_h2o: Array) -> Array:
    dp = kernel_ops.forward_difference(p_bottom, dim=2)
    mol = constants.DRY_AIR_MOL_MASS + constants.WATER_MOL_MASS * vmr_h2o
    return -(dp/constants.G)*constants.AVOGADRO/mol


class ClearSkyTest(unittest.TestCase):

    @parameterized.expand([
        (site, uc, us) for site, uc, us in product(range(10), [True, False], [True, False])
    ])
    def test_two_stream_solver_with_clear_sky(
        self, rfmip_site: int, use_compact_lookup: bool, use_scan: bool
    ):
        rfmip_expt_id = 0
        (
            ds,
            pressure_allsites,
            pressure_level_allsites,
            temperature_allsites,
            _,
            h2o_allsites,
            o3_allsites,
            sfc_t_allsites,
        ) = _setup_atmospheric_profiles()

        radiation_params = _setup_radiation_params(
            rfmip_site, use_compact_lookup, ds
        )
        atmos_state = atmospheric_state.from_config(
            radiation_params.atmospheric_state_cfg
        )
        optics_lib = optics.optics_factory(radiation_params.optics, atmos_state.vmr)

        # prepare inputs
        n_horiz = 2
        conv3d = functools.partial(
            test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=n_horiz
        )
        sfc_tv = sfc_t_allsites[rfmip_expt_id, rfmip_site]
        sfc_t = sfc_tv * jnp.ones((n_horiz, n_horiz), dtype=jnp.float_)
        p = conv3d(pressure_allsites[rfmip_site, :])
        pl = conv3d(pressure_level_allsites[rfmip_site, :])
        t  = conv3d(temperature_allsites[rfmip_expt_id, rfmip_site, :])
        h2 = conv3d(h2o_allsites[rfmip_expt_id, rfmip_site, :])
        o3 = conv3d(o3_allsites[rfmip_expt_id, rfmip_site, :])
        mol = _air_molecules_per_area(pl, h2)
        vmr = {'h2o': h2, 'o3': o3}

        lw = two_stream.solve_lw(p, t, mol, optics_lib, atmos_state, vmr, sfc_t, use_scan=use_scan)
        sw = two_stream.solve_sw(p, t, mol, optics_lib, atmos_state, vmr, use_scan=use_scan)

        lw_up, lw_dn, sw_up, sw_dn = _load_expected_data()
        exp_up_lw = conv3d(lw_up[rfmip_expt_id, rfmip_site, :])
        exp_dn_lw = conv3d(lw_dn[rfmip_expt_id, rfmip_site, :])
        exp_up_sw = conv3d(sw_up[rfmip_expt_id, rfmip_site, :])
        exp_dn_sw = conv3d(sw_dn[rfmip_expt_id, rfmip_site, :])

        # The comparison spans the interior *and* the top-of-atmosphere face.
        atol=0.2
        np.testing.assert_allclose(_interior_and_toa(lw['flux_down']), _interior_and_toa(exp_dn_lw), rtol=1e-2, atol=atol)
        np.testing.assert_allclose(_interior_and_toa(lw['flux_up']),   _interior_and_toa(exp_up_lw), rtol=7e-3, atol=atol)

        rtol = 7e-3 if use_compact_lookup else 1e-3
        np.testing.assert_allclose(_interior_and_toa(sw['flux_down']), _interior_and_toa(exp_dn_sw), rtol=rtol, atol=atol)
        np.testing.assert_allclose(_interior_and_toa(sw['flux_up']),   _interior_and_toa(exp_up_sw), rtol=rtol, atol=atol)


if __name__ == '__main__':
    unittest.main()
