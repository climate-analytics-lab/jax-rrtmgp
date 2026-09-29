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
import jax
import jax.numpy as jnp
import numpy as np
from rrtmgp import constants
from rrtmgp import test_util
from rrtmgp.config import radiative_transfer
from rrtmgp.optics import atmospheric_state
from rrtmgp.optics import constants as optics_constants
from rrtmgp.optics import optics
from rrtmgp.optics import optics_base
from rrtmgp.rte import two_stream

Array: TypeAlias = jax.Array


def _remove_halos(f: Array) -> Array:
  """Remove the halos from the output."""
  return f[:, :, 1:-1]


def _setup_atmospheric_profiles(
    params: radiative_transfer.GrayAtmosphereOptics,
    pe: float,
    tt: float,
    nlayers: int,
) -> tuple[Array, Array]:
  """Create 1D profiles of temperature and pressure for the gray atmosphere."""
  halo_width = 1
  dp = (params.p0 - pe) / nlayers  # Δp for linear pressure distribution.
  p_lower = params.p0 + dp * (halo_width - 0.5)
  p_higher = pe - dp * (halo_width - 0.5)
  pressure = np.linspace(p_lower, p_higher, nlayers + 2 * halo_width)
  d0_lw, p0, alpha = params.d0_lw, params.p0, params.alpha
  temperature = tt * (1.0 + d0_lw * (pressure / p0) ** alpha) ** 0.25
  return temperature, pressure


def _update_temperature_from_heating_rate(
    temperature: Array,
    sfc_temperature: Array,
    flux_down: Array,
    flux_net: Array,
    heating_rate: Array,
    dt: float,
) -> tuple[Array, Array]:
  """Update cell center and face temperatures based on the heating rate."""
  halo_width = 1  # Assumed and hardcoded.
  # First-order Euler steps are used here.
  updated_temperature = temperature + dt * heating_rate

  # temperature_halo = 2 * sfc_temperature - temperature[:, :, 1]
  # updated_temperature = updated_temperature.at[:, :, 0].set(temperature_halo)
  # Set the lower halo layer to the surface temperature.
  # In the future, don't want to use halo layer value.
  updated_temperature = updated_temperature.at[:, :, halo_width - 1].set(
      sfc_temperature
  )

  # Set the upper halo layer using Neumann BC.
  updated_temperature = updated_temperature.at[:, :, -1].set(
      temperature[:, :, -2]
  )

  temperature_stefan_boltzmann = (
      (flux_down + flux_net / 2.0) / optics_constants.STEFAN_BOLTZMANN
  ) ** (0.25)
  return updated_temperature, temperature_stefan_boltzmann


class TestTwoStream(unittest.TestCase):

  @parameterized.expand([(True,)])
  def test_gray_atmosphere_longwave_equilibrium(self, use_scan: bool):
    """Check the longwave fluxes converge to an equilibrium state."""
    # SETUP
    p0 = 100_000  # Surface pressure [Pa].
    pe = 9000  # Top-of-atmosphere pressure [Pa].
    sfc_emis = 1.0  # Surface emissivity.
    toa_flux_lw = 0.0  # Incoming longwave flux.
    temperature_sfc = 320.0  # Surface temperature [K].
    skin_temperature = 200.0  # Skin temperature at the top of atmosphere [K].
    alpha = 3.5  # Lapse rate of radiative equilibrium [units?].

    n_layers = 64

    # Longwave optical depth of the entire gray atmosphere.
    d0_lw = (temperature_sfc / skin_temperature) ** 4 - 1.0

    radiation_params = radiative_transfer.RadiativeTransfer(
        optics=radiative_transfer.OpticsParameters(
            optics=radiative_transfer.GrayAtmosphereOptics(
                p0=p0,
                alpha=alpha,
                d0_lw=d0_lw,
                d0_sw=0.22,
            )
        ),
        atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
            sfc_emis=sfc_emis,
            sfc_alb=0.0,
            zenith=0.0,
            irrad=0.0,
            toa_flux_lw=toa_flux_lw,
        ),
    )
    temperature, pressure = _setup_atmospheric_profiles(
        radiation_params.optics.optics, pe, skin_temperature, n_layers
    )

    atmos_state = atmospheric_state.from_config(
        radiation_params.atmospheric_state_cfg
    )
    optics_lib = optics.optics_factory(radiation_params.optics, atmos_state.vmr)

    # Timestep in hours.
    dt_hrs = 6
    # Timestep in seconds.
    dt = dt_hrs * 3600

    # Convert from 1D to 3D arrays.
    n_horiz = 2
    convert_to_3d = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=n_horiz
    )
    temperature = convert_to_3d(temperature)
    pressure = convert_to_3d(pressure)
    sfc_temperature = temperature_sfc * jnp.ones(
        (n_horiz, n_horiz), dtype=pressure.dtype
    )

    # Define the body function to be executed during the loop.
    def body_fn(i, states):
      """Body function for updating temperature from fluxes."""
      del i
      temperature = states['temperature']
      molecules = jnp.zeros_like(pressure)
      fluxes = two_stream.solve_lw(
          pressure,
          temperature,
          molecules,
          optics_lib,
          atmos_state,
          sfc_temperature=sfc_temperature,
          use_scan=use_scan,
      )
      heating_rate = two_stream.compute_heating_rate(
          fluxes['flux_net'], pressure
      )
      updated_temperature, updated_temperature_stefan_boltzmann = (
          _update_temperature_from_heating_rate(
              temperature,
              sfc_temperature,
              fluxes['flux_down'],
              fluxes['flux_net'],
              heating_rate,
              dt,
          )
      )
      return {
          'temperature': updated_temperature,
          'temperature_stefan_boltzmann': updated_temperature_stefan_boltzmann,
      }

    # ACTION
    # Number of timesteps corresponding to 24 years.

    n_steps = 24 * 365 * 24 // dt_hrs #TODO: check numerics

    init_states = {
        'temperature': temperature,
        'temperature_stefan_boltzmann': jnp.zeros_like(temperature),
    }

    states = jax.lax.fori_loop(0, n_steps, body_fn, init_states)
    temperature = states['temperature']
    temperature_face, _ = optics_base.reconstruct_face_values(
        temperature, f_lower_bc=sfc_temperature
    )
    temperature_sb_reference = states['temperature_stefan_boltzmann']

    # VERIFICATION
    temperature_face = _remove_halos(temperature_face)
    temperature_sb_reference = _remove_halos(temperature_sb_reference)

    assert not np.any(np.isnan(temperature_face))
    np.testing.assert_allclose(
        temperature_face, temperature_sb_reference, atol=0.1, rtol=2e-3
    )

  @parameterized.expand([(True,)])
  def test_gray_atmosphere_shortwave(self, use_scan: bool):
    """Check the direct solar radiation reaching the surface."""
    # SETUP
    p0 = 100_000  # Surface pressure [Pa].
    pe = 9000  # Top of atmosphere pressure [Pa].
    skin_temperature = 200.0  # Skin temperature at the top of atmosphere [K].

    n_layers = 64
    halo_width = 1

    # Parameters for shortwave radiation
    # acsc(1.66)
    deg_to_rad = np.pi / 180
    zenith = deg_to_rad * 52.95  # Zenith angle, in radians.
    # Total shortwave optical depth of the entire gray atmosphere.
    d0_sw = 0.22
    sfc_alb = 0.1  # Surface albedo.
    irrad = 1407.679  # Solar radiation flux [W/m²].

    radiation_params = radiative_transfer.RadiativeTransfer(
        optics=radiative_transfer.OpticsParameters(
            optics=radiative_transfer.GrayAtmosphereOptics(
                p0=p0,
                d0_lw=5.35,
                d0_sw=d0_sw,
            )
        ),
        atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
            sfc_alb=sfc_alb,
            zenith=zenith,
            irrad=irrad,
        ),
    )

    temperature, pressure = _setup_atmospheric_profiles(
        radiation_params.optics.optics, pe, skin_temperature, n_layers
    )
    # Convert from 1D to 3D arrays.
    n_horiz = 2
    convert_to_3d = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=n_horiz
    )
    temperature = convert_to_3d(temperature)
    pressure = convert_to_3d(pressure)

    atmos_state = atmospheric_state.from_config(
        radiation_params.atmospheric_state_cfg
    )
    optics_lib = optics.optics_factory(radiation_params.optics, atmos_state.vmr)

    # ACTION
    molecules = jnp.zeros_like(pressure)
    fluxes = two_stream.solve_sw(
        pressure,
        temperature,
        molecules,
        optics_lib,
        atmos_state,
        use_scan=use_scan,
    )

    # Compute the optical depth for all layers.
    tau = optics_lib.compute_sw_optical_properties(
        pressure,
        temperature,  # Unused argument for Gray Atmosphere Optics.
        molecules,  # Unused argument for Gray Atmosphere Optics.
        0,  # Unused argument for Gray Atmosphere Optics.
    )['optical_depth']
    # Remove halos.
    tau_no_halos = _remove_halos(tau)

    # Numerical optical depth of the entire gray atmosphere.
    tau_tot = jnp.sum(tau_no_halos, axis=2) / np.cos(zenith)

    # VERIFICATION
    # Extract the downward flux on the lower wall.
    flux_down = fluxes['flux_down'][:, :, halo_width]
    expected_flux_down_direct_sfc = irrad * np.cos(zenith) * np.exp(-tau_tot)
    np.testing.assert_allclose(
        flux_down, expected_flux_down_direct_sfc, rtol=1e-5, atol=0
    )

  def test_gray_atmosphere_direct_and_diffuse_surface_albedo(self):
    """A non-scattering atmosphere delivers only direct light to the surface.

    The gray shortwave optics absorb and never scatter, so the whole surface
    downward flux is the attenuated direct beam, the surface reflects it with
    the direct albedo alone, and the diffuse albedo has nothing to act on.
    """
    zenith = np.deg2rad(52.95)
    irrad = 1407.679
    sfc_alb = 0.1
    halo_width = 1
    radiation_params = radiative_transfer.RadiativeTransfer(
        optics=radiative_transfer.OpticsParameters(
            optics=radiative_transfer.GrayAtmosphereOptics(
                p0=100_000, d0_lw=5.35, d0_sw=0.22
            )
        ),
        atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
            sfc_alb=sfc_alb, zenith=zenith, irrad=irrad
        ),
    )
    temperature, pressure = _setup_atmospheric_profiles(
        radiation_params.optics.optics, 9000, 200.0, 64
    )
    convert_to_3d = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=2
    )
    temperature = convert_to_3d(temperature)
    pressure = convert_to_3d(pressure)
    molecules = jnp.zeros_like(pressure)
    atmos_state = atmospheric_state.from_config(
        radiation_params.atmospheric_state_cfg
    )
    optics_lib = optics.optics_factory(radiation_params.optics, atmos_state.vmr)

    def solve(atmos_state=atmos_state, **kwargs):
      return two_stream.solve_sw(
          pressure, temperature, molecules, optics_lib, atmos_state, **kwargs
      )

    default = solve()
    matching = solve(sfc_alb_dir=sfc_alb, sfc_alb_dif=sfc_alb)
    for key in ('flux_up', 'flux_down', 'flux_net', 'flux_down_dir_sfc'):
      np.testing.assert_array_equal(matching[key], default[key], err_msg=key)

    tau = optics_lib.compute_sw_optical_properties(
        pressure, temperature, molecules, 0
    )['optical_depth']
    tau_tot = jnp.sum(_remove_halos(tau), axis=2) / np.cos(zenith)
    expected_direct = irrad * np.cos(zenith) * np.exp(-tau_tot)

    a_dir, a_dif = 0.3, 0.9
    fluxes = solve(sfc_alb_dir=a_dir, sfc_alb_dif=a_dif)
    direct = fluxes['flux_down_dir_sfc']
    np.testing.assert_allclose(direct, expected_direct, rtol=1e-5, atol=0)
    np.testing.assert_allclose(
        fluxes['flux_down'][:, :, halo_width], direct, rtol=1e-6, atol=0
    )
    np.testing.assert_allclose(
        fluxes['flux_up'][:, :, halo_width], a_dir * direct, rtol=1e-6, atol=0
    )
    # With nothing diffuse at the surface the diffuse albedo is inert.
    other_dif = solve(sfc_alb_dir=a_dir, sfc_alb_dif=0.0)
    for key in ('flux_up', 'flux_down', 'flux_down_dir_sfc'):
      np.testing.assert_allclose(
          other_dif[key], fluxes[key], rtol=1e-6, atol=1e-6, err_msg=key
      )

    # A transparent atmosphere lets the full beam reach the surface.
    transparent = dataclasses.replace(
        radiation_params,
        optics=radiative_transfer.OpticsParameters(
            optics=radiative_transfer.GrayAtmosphereOptics(
                p0=100_000, d0_lw=5.35, d0_sw=0.0
            )
        ),
    )
    optics_lib = optics.optics_factory(transparent.optics, atmos_state.vmr)
    fluxes = solve(sfc_alb_dir=a_dir, sfc_alb_dif=a_dif)
    np.testing.assert_allclose(
        fluxes['flux_down_dir_sfc'], irrad * np.cos(zenith), rtol=1e-6
    )
    np.testing.assert_allclose(
        fluxes['flux_down'][:, :, halo_width], fluxes['flux_down_dir_sfc'],
        rtol=1e-6,
    )

    # And no sun, no beam.
    night = dataclasses.replace(atmos_state, zenith=0.6 * np.pi)
    fluxes = solve(atmos_state=night, sfc_alb_dir=a_dir, sfc_alb_dif=a_dif)
    np.testing.assert_array_equal(fluxes['flux_down_dir_sfc'], 0.0)

  def test_compute_heating_rate(self):
    """Check the heating rate as derived from fluxes and the pressure field."""
    # SETUP
    # Pressure is linearly decreasing by 200 Pa per grid cell.
    pressure = np.linspace(1e5, 98600, 8)
    # Net flux is linearly increasing by 0.1 W/m² per grid cell.
    flux_net = np.linspace(0.1, 0.8, 8)

    n_horiz = 2
    convert_to_3d = functools.partial(
        test_util.convert_to_3d_array_and_tile, dim=2, num_repeats=n_horiz
    )
    flux_net = convert_to_3d(flux_net)
    pressure = convert_to_3d(pressure)

    # ACTION
    heating_rate = two_stream.compute_heating_rate(flux_net, pressure)

    # VERIFICATION
    heating_rate = _remove_halos(heating_rate)

    expected_heating_rate = (
        -constants.G * 0.1 / 200 / constants.CP_D * np.ones_like(heating_rate)
    )
    np.testing.assert_allclose(
        heating_rate, expected_heating_rate, rtol=1e-5, atol=0
    )


if __name__ == '__main__':
  unittest.main()
