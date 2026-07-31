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

"""Top-of-atmosphere flux regression tests (issue #19).

These pin the two properties that are exactly known analytically at the top
boundary, independent of any reference dataset:

  1. With no absorbers at all, the outgoing longwave radiation must equal the
     surface emission `sigma * emissivity * Ts^4` -- the atmosphere neither
     absorbs nor emits, so the surface flux passes straight through.
  2. With no prescribed incoming longwave, the downwelling longwave flux at the
     top boundary must be exactly zero.

Both are violated by any post-hoc extrapolation onto the top face, which is how
a spurious top-boundary flux previously went unnoticed: the reference
comparison in `clear_sky_test` excluded that level.

A US Standard Atmosphere column is also run end to end as a coarse sanity band
on clear-sky OLR.
"""

import functools
from pathlib import Path
from typing import TypeAlias

import unittest
import jax
import jax.numpy as jnp
import numpy as np

from rrtmgp import constants
from rrtmgp import kernel_ops
from rrtmgp import test_util
from rrtmgp.config import radiative_transfer
from rrtmgp.optics import atmospheric_state
from rrtmgp.optics import constants as optics_constants
from rrtmgp.optics import optics
from rrtmgp.rte import two_stream

Array: TypeAlias = jax.Array

_ROOT = Path()
_LW_LOOKUP = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g256.nc'
_SW_LOOKUP = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-sw-g224.nc'
_CLOUD_LW = 'rrtmgp/optics/rrtmgp_data/cloudysky_lw.nc'
_CLOUD_SW = 'rrtmgp/optics/rrtmgp_data/cloudysky_sw.nc'
_VMR_GLOBAL_MEANS = 'rrtmgp/optics/test_data/vmr_global_means.json'

_HALO = 1
_N_HORIZ = 2
_SFC_EMIS = 0.98
_TS = 288.15

# US Standard Atmosphere 1976 layer bases: altitude, lapse rate, temperature,
# pressure.
_USSA_H = np.array([0.0, 11e3, 20e3, 32e3, 47e3, 51e3, 71e3])
_USSA_L = np.array([-6.5e-3, 0.0, 1.0e-3, 2.8e-3, 0.0, -2.8e-3, -2.0e-3])
_USSA_T = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65])
_USSA_P = np.array(
    [101325.0, 22632.06, 5474.889, 868.0187, 110.9063, 66.93887, 3.956420]
)
_G0 = 9.80665
_R_DRY_AIR = 287.0528


def _us_standard_atmosphere(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Temperature [K] and pressure [Pa] of the USSA-76 at altitude `h` [m]."""
    h = np.atleast_1d(h).astype(float)
    temperature = np.zeros_like(h)
    pressure = np.zeros_like(h)
    for i, (h_b, lapse, t_b, p_b) in enumerate(
        zip(_USSA_H, _USSA_L, _USSA_T, _USSA_P)
    ):
        if i == len(_USSA_H) - 1:
            in_layer = h >= h_b
        else:
            in_layer = (h >= h_b) & (h < _USSA_H[i + 1])
        dh = h[in_layer] - h_b
        temperature[in_layer] = t_b + lapse * dh
        if lapse == 0.0:
            pressure[in_layer] = p_b * np.exp(-_G0 * dh / (_R_DRY_AIR * t_b))
        else:
            pressure[in_layer] = p_b * (
                t_b / (t_b + lapse * dh)
            ) ** (_G0 / (_R_DRY_AIR * lapse))
    return temperature, pressure


def _saturation_vapor_pressure(temperature: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure over water [Pa] (Bolton 1980)."""
    t_celsius = temperature - 273.15
    return 611.2 * np.exp(17.67 * t_celsius / (t_celsius + 243.5))


def _radiation_setup(
    sfc_emis: float,
) -> tuple[optics.RRTMOptics, atmospheric_state.AtmosphericState]:
    """Build the RRTMGP optics library and atmospheric state."""
    params = radiative_transfer.RadiativeTransfer(
        optics=radiative_transfer.OpticsParameters(
            optics=radiative_transfer.RRTMOptics(
                longwave_nc_filepath=str(_ROOT / _LW_LOOKUP),
                shortwave_nc_filepath=str(_ROOT / _SW_LOOKUP),
                cloud_longwave_nc_filepath=str(_ROOT / _CLOUD_LW),
                cloud_shortwave_nc_filepath=str(_ROOT / _CLOUD_SW),
            )
        ),
        atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
            sfc_emis=sfc_emis,
            sfc_alb=0.06,
            zenith=0.0,
            irrad=0.0,
            toa_flux_lw=0.0,
            vmr_global_mean_filepath=_ROOT / _VMR_GLOBAL_MEANS,
        ),
    )
    atmos_state = atmospheric_state.from_config(params.atmospheric_state_cfg)
    optics_lib = optics.optics_factory(params.optics, atmos_state.vmr)
    return optics_lib, atmos_state


def _build_ussa_column(n_layers: int = 60, p_top: float = 1.0):
    """Construct a US Standard Atmosphere column on the solver's halo layout."""
    p_level = np.exp(
        np.linspace(np.log(_USSA_P[0]), np.log(p_top), n_layers + 1)
    )
    z_fine = np.linspace(0.0, 84e3, 200001)
    _, p_fine = _us_standard_atmosphere(z_fine)
    # Pressure decreases monotonically with altitude, so -log(p) increases.
    z_level = np.interp(-np.log(p_level), -np.log(p_fine), z_fine)
    z_layer = 0.5 * (z_level[:-1] + z_level[1:])
    p_layer = np.exp(0.5 * (np.log(p_level[:-1]) + np.log(p_level[1:])))
    t_layer, _ = _us_standard_atmosphere(z_layer)
    t_level, _ = _us_standard_atmosphere(z_level)

    # 50% relative humidity in the troposphere, 3 ppmv above it.
    vmr_h2o = 0.5 * _saturation_vapor_pressure(t_layer) / p_layer
    vmr_h2o = np.maximum(np.where(z_layer < 11e3, vmr_h2o, 3e-6), 3e-6)

    # Ozone: ~0.03 ppmv at the surface rising to ~8 ppmv near 10 hPa.
    log_p = np.log10(p_layer)
    vmr_o3 = 0.03e-6 + 8e-6 * np.exp(
        -0.5 * ((log_p - np.log10(1000.0)) / 0.55) ** 2
    )

    pressure = np.pad(p_layer, (_HALO, _HALO), mode='edge')
    pressure_level = np.pad(p_level, (_HALO, _HALO - 1), mode='edge')
    temperature = np.zeros(n_layers + 2 * _HALO)
    temperature[_HALO:-_HALO] = t_layer
    temperature[0] = 2 * t_level[0] - t_layer[0]
    temperature[-1] = 2 * t_level[-1] - t_layer[-1]

    convert = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=_N_HORIZ
    )
    p_3d = convert(jnp.asarray(pressure))
    p_level_3d = convert(jnp.asarray(pressure_level))
    t_3d = convert(jnp.asarray(temperature))
    h2o_3d = convert(jnp.asarray(np.pad(vmr_h2o, (_HALO, _HALO), mode='edge')))
    o3_3d = convert(jnp.asarray(np.pad(vmr_o3, (_HALO, _HALO), mode='edge')))

    # Dry-air molecules per unit area in each layer.
    dp = kernel_ops.forward_difference(p_level_3d, dim=2)
    molecules = (
        -(dp / constants.G)
        * constants.AVOGADRO
        / (constants.DRY_AIR_MOL_MASS + constants.WATER_MOL_MASS * h2o_3d)
    )
    sfc_temperature = _TS * jnp.ones((_N_HORIZ, _N_HORIZ), dtype=p_3d.dtype)
    return p_3d, t_3d, molecules, h2o_3d, o3_3d, sfc_temperature


class ToaFluxTest(unittest.TestCase):

    def test_transparent_atmosphere_olr_equals_surface_emission(self):
        """With no absorbers, OLR must equal sigma * emissivity * Ts^4."""
        optics_lib, atmos_state = _radiation_setup(_SFC_EMIS)
        p, t, molecules, _, _, sfc_temperature = _build_ussa_column()

        # Zero out every gas the longwave lookup table knows about.
        vmr_fields = {
            gas: 1e-30 * jnp.ones_like(t)
            for gas in optics_lib.gas_optics_lw.idx_gases
        }

        fluxes = two_stream.solve_lw(
            p, t, molecules, optics_lib, atmos_state, vmr_fields,
            sfc_temperature, use_scan=True,
        )
        olr = np.asarray(fluxes['flux_up'])[:, :, -1]
        expected = (
            optics_constants.STEFAN_BOLTZMANN * _SFC_EMIS * _TS ** 4
        )
        # The lookup tables span 10-3250 cm^-1, which holds all but ~1e-5 of the
        # Planck emission at this temperature, so the agreement is tight.
        np.testing.assert_allclose(olr, expected, atol=1.0)

    def test_toa_downwelling_longwave_is_zero(self):
        """No prescribed incoming longwave means exactly zero flux down at TOA."""
        optics_lib, atmos_state = _radiation_setup(_SFC_EMIS)
        p, t, molecules, h2o, o3, sfc_temperature = _build_ussa_column()
        vmr_fields = {'h2o': h2o, 'o3': o3}

        fluxes = two_stream.solve_lw(
            p, t, molecules, optics_lib, atmos_state, vmr_fields,
            sfc_temperature, use_scan=True,
        )
        flux_down_toa = np.asarray(fluxes['flux_down'])[:, :, -1]
        np.testing.assert_allclose(flux_down_toa, 0.0, atol=1e-5)

    def test_us_standard_atmosphere_clear_sky_olr(self):
        """Clear-sky OLR for a USSA column sits in a physically sane band.

        This is a coarse regression guard, not a line-by-line benchmark: the
        humidity and ozone profiles here are idealised, so the band is wide.
        Published line-by-line values for this profile are ~260-265 W/m^2.
        """
        optics_lib, atmos_state = _radiation_setup(_SFC_EMIS)
        p, t, molecules, h2o, o3, sfc_temperature = _build_ussa_column()
        vmr_fields = {
            'h2o': h2o, 'o3': o3, 'co2': 400e-6 * jnp.ones_like(t),
        }

        fluxes = two_stream.solve_lw(
            p, t, molecules, optics_lib, atmos_state, vmr_fields,
            sfc_temperature, use_scan=True,
        )
        olr = float(np.asarray(fluxes['flux_up'])[0, 0, -1])
        self.assertGreater(olr, 240.0)
        self.assertLess(olr, 275.0)

        flux_up = np.asarray(fluxes['flux_up'])[0, 0, 1:]
        flux_down = np.asarray(fluxes['flux_down'])[0, 0, 1:]
        self.assertTrue(np.all(np.isfinite(flux_up)))
        self.assertTrue(np.all(np.isfinite(flux_down)))

        # Surface energy balance: the upwelling flux leaving the surface is its
        # own emission plus the reflected part of the downwelling flux. Note it
        # therefore exceeds `sigma * emissivity * Ts^4` whenever emissivity < 1.
        surface_emission = (
            optics_constants.STEFAN_BOLTZMANN * _SFC_EMIS * _TS ** 4
        )
        np.testing.assert_allclose(
            flux_up[0],
            surface_emission + (1.0 - _SFC_EMIS) * flux_down[0],
            atol=1.0,
        )

        # An absorbing atmosphere must reduce the outgoing flux relative to the
        # surface. The upwelling flux is deliberately not required to decrease
        # monotonically: the USSA stratosphere warms with height through the
        # ozone layer, so it legitimately rises slightly there.
        self.assertLess(olr, float(flux_up[0]))


if __name__ == '__main__':
    unittest.main()
