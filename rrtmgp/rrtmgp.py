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

"""Implementation of a radiative transfer solver."""

import dataclasses
from collections.abc import Sequence
from typing import TypeAlias

import jax
import jax.numpy as jnp
from rrtmgp import constants
from rrtmgp import kernel_ops
from rrtmgp import stretched_grid_util
from rrtmgp import rrtmgp_common
from rrtmgp.config import radiative_transfer
from rrtmgp.optics import atmospheric_state
from rrtmgp.optics import lookup_volume_mixing_ratio
from rrtmgp.optics import optics
from rrtmgp.rte import longwave_noscat
from rrtmgp.rte import two_stream

Array: TypeAlias = jax.Array


def _humidity_to_volume_mixing_ratio(
    q_t: Array, q_c: Array
) -> Array:
  """For water vapor, convert humidity to a volume mixing ratio."""
  mol_ratio = constants.R_V / constants.R_D
  q_v = q_t - q_c
  mix_ratio = q_v / (1 - q_t)
  return mol_ratio * mix_ratio


def _air_molecules_per_area(p_xxc: Array, vmr_h2o_xxc: Array) -> Array:
  """Compute the number of molecules in a grid cell per area."""
  # centered_difference returns p[k+1] - p[k-1], which spans two layers; the
  # thickness of the layer centered on k is half of it.
  dp_xxc = 0.5 * kernel_ops.centered_difference(p_xxc, dim=2)
  mol_m_air_xxc = (
      constants.DRY_AIR_MOL_MASS + constants.WATER_MOL_MASS * vmr_h2o_xxc
  )
  return -(dp_xxc / constants.G) * constants.AVOGADRO / mol_m_air_xxc


def _compute_cloud_path(
    rho: Array, q_c: Array, dz: float, sg_map: dict[str, Array]
) -> Array:
  """Compute the cloud water/ice path in each atmospheric grid cell."""
  use_stretched_grid_z = stretched_grid_util.get_use_stretched_grid(sg_map)[2]
  if use_stretched_grid_z:
    h = sg_map[stretched_grid_util.hc_key(2)]
  else:
    h = dz
  return rho * q_c * h


def _horiz_mean(f: Array) -> Array:
  """Compute the horizontal mean of `f`.

  Accumulate means in float64 to avoid potential numerical issues when
  accumulating, but then convert back to float32.

  Args:
    f: The array to compute the horizontal mean of.

  Returns:
    The horizontal mean of `f`.
  """
  acc_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
  return jnp.mean(f, axis=(0, 1), dtype=acc_dtype).astype(jnp.float32)


def _horiz_mean_zc(f_zc: Array) -> Array:
  """Compute the horizontal mean of a column-minor `(nzh, nc)` array."""
  acc_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
  return jnp.mean(f_zc, axis=1, dtype=acc_dtype).astype(jnp.float32)


def _air_molecules_per_area_zc(p_zc: Array, vmr_h2o_zc: Array) -> Array:
  """Column-minor `(nzh, nc)` counterpart of `_air_molecules_per_area`."""
  dp_zc = 0.5 * (jnp.roll(p_zc, -1, axis=0) - jnp.roll(p_zc, 1, axis=0))
  mol_m_air_zc = (
      constants.DRY_AIR_MOL_MASS + constants.WATER_MOL_MASS * vmr_h2o_zc
  )
  return -(dp_zc / constants.G) * constants.AVOGADRO / mol_m_air_zc


class RRTMGP:
  """Rapid Radiative Transfer Model for General Circulation Models (RRTMGP)."""

  def __init__(
      self,
      radiative_transfer_cfg: radiative_transfer.RadiativeTransfer,
      dz: float,
      diagnostic_fields: Sequence[str] = tuple(),
  ):
    self._dz = dz  # Store dz (only used if not using stretched grid in z).
    self._diagnostic_fields = diagnostic_fields
    self._save_lw_sw_heating_rates = (
        radiative_transfer_cfg.save_lw_sw_heating_rates
    )
    self._do_clear_sky = radiative_transfer_cfg.do_clear_sky

    # Load and store the atmospheric gas concentrations.
    self.atmospheric_state = atmospheric_state.from_config(
        radiative_transfer_cfg.atmospheric_state_cfg
    )
    # Create the optics library.
    self.optics_lib = optics.optics_factory(
        radiative_transfer_cfg.optics, self.atmospheric_state.vmr
    )

  def compute_heating_rate(
      self,
      rho_xxc: Array,
      q_t: Array,
      q_liq: Array,
      q_ice: Array,
      q_c: Array,
      cloud_r_eff_liq: Array,
      cloud_r_eff_ice: Array,
      temperature: Array,
      sfc_temperature: Array,
      p_ref_xxc: Array,
      sg_map: dict[str, Array],
      use_scan: bool = False,
      zenith: float | Array | None = None,
      irrad: float | Array | None = None,
      cloud_path_liq_lw_per_gpt: Array | None = None,
      cloud_path_ice_lw_per_gpt: Array | None = None,
      cloud_path_liq_sw_per_gpt: Array | None = None,
      cloud_path_ice_sw_per_gpt: Array | None = None,
      vmr_fields: dict[str, Array] | None = None,
      aerosol_optics_lw: dict[str, Array] | None = None,
      aerosol_optics_sw: dict[str, Array] | None = None,
      # New optional parameters go at the END of the signature so existing
      # positional callers (e.g. of the McICA cloud-path arrays) keep their
      # bindings.
      sfc_alb: float | Array | None = None,
      sfc_emis: float | Array | None = None,
      gpt_chunk: int = two_stream.DEFAULT_GPT_CHUNK,
      cloud_tau_scale_liq: float | Array | None = None,
      cloud_tau_scale_ice: float | Array | None = None,
      sfc_alb_dir: float | Array | None = None,
      sfc_alb_dif: float | Array | None = None,
      clear_sky_fluxes: bool = False,
      lw_solver: str = two_stream.LW_SOLVER_TWO_STREAM,
      lw_secant: float = longwave_noscat.LW_NOSCAT_DEFAULT_SECANT,
      gas_optics_mode: str = two_stream.GAS_OPTICS_MODE_PER_GPT,
      aerosol_free_fluxes: bool = False,
  ) -> dict[str, Array]:
    """Compute the local heating rate due to radiative transfer.

    The optical properties of the layered atmosphere are computed using RRTMGP
    and the two-stream radiative transfer equation is solved for the net fluxes
    at the grid cell faces.  Based on the overall net radiative flux of the grid
    cell, a local heating rate is determined.

    For canonical McICA, the caller can supply per-g-point cloud water paths
    via `cloud_path_{liq,ice}_{lw,sw}_per_gpt`, each shaped
    `[n_gpt_{lw,sw}, nx, ny, nz]`. When provided, these replace the broadcast
    cloud paths derived from `q_liq` and `q_ice` inside the per-g-point loop,
    so each g-point sees its own stochastic sub-column. The clear-sky branch
    is unaffected.

    The surface boundary condition can be overridden per call via `sfc_alb`
    (broadband SW surface albedo) and `sfc_emis` (broadband LW surface
    emissivity). Like `zenith`/`irrad`, these replace the corresponding
    `AtmosphericStateCfg` scalars for this call only, and may be traced
    values — e.g. per-column albedos from a host model's surface scheme
    under vmap. Anything broadcastable against the `[nx, ny]` surface plane
    is accepted. When omitted, the configured scalars are used.

    Separate direct-beam and diffuse SW surface albedos can be given via
    `sfc_alb_dir` and `sfc_alb_dif`. Each is either a per-column field like
    `sfc_alb` (broadcastable against the `[nx, ny]` surface plane) or a
    per-band one with a trailing SW-band axis, `(..., n_bnd_sw)`, e.g.
    `[nx, ny, n_bnd_sw]` or `[n_bnd_sw]` (one spectrum for every column, e.g.
    a single column under a column vmap); each g-point then uses the albedo
    of its band. An array that
    broadcasts against the surface plane is read as per column, so the rank-3
    form is the unambiguous per-band one. Either may be omitted and falls back
    to `sfc_alb` (the argument or the configured value), which keeps its
    meaning of a single broadband albedo for both. They apply to the clear-sky
    diagnostic as well.

    Per-cell gas concentrations can be supplied via `vmr_fields`, a dict keyed
    by chemical formula (e.g. `'o3'`, `'co2'`, `'ch4'`, `'n2o'`) mapping to a
    3D `[nx, ny, nz]` volume mixing ratio array. These override both the
    sounding-based reconstruction from pressure (`o3` by default) and the
    global-mean fallbacks loaded from `vmr_global_means.json`. The `h2o` entry
    is always recomputed from `q_t` and `q_c` to keep the radiative water
    consistent with the simulation's humidity state; any caller-supplied
    `'h2o'` is therefore ignored. The same `vmr_fields` are used for the
    clear-sky diagnostic when enabled.

    Per-band aerosol optical properties can be supplied via
    `aerosol_optics_{lw,sw}`, each a dict with keys `'optical_depth'`, `'ssa'`,
    and `'asymmetry_factor'`, each shaped `[n_bnd_{lw,sw}, nx, ny, nz]`. The
    per-band slice is mixed in alongside the gas + cloud optical properties
    using the mass-weighted combine inside the per-g-point loop. Values are
    passed through unmodified (no delta-scaling). LW and SW are independent —
    pass one or both. Aerosols are also applied to the clear-sky diagnostic
    (CMIP convention: "clear-sky" = cloud-free, aerosols included).

    Sub-grid cloud inhomogeneity can be represented with
    `cloud_tau_scale_{liq,ice}`, multipliers of the liquid and ice cloud
    optical depths (e.g. ECHAM's `zinhoml` / `zinhomi`), each a scalar or an
    array broadcastable against the `[nx, ny, nz]` cloud path. Unlike scaling
    `q_liq` / `q_ice`, they leave the single-scattering albedo and asymmetry
    factor weighted by the physical per-phase optical depths, which is what
    ECHAM does and is not the same thing once the two factors differ. They
    apply to both the longwave and the shortwave cloud optics, and to the McICA
    per-g-point cloud paths when those are given; the clear-sky diagnostic has
    no clouds and is unaffected. When omitted, the optical depths are unscaled.

    `gpt_chunk` sets how many g-points the spectral loop solves per iteration.
    It changes no physics -- only the order in which independent g-point fluxes
    are summed -- and exists to cut GPU kernel launches on a solve that is
    launch bound rather than flop bound. See
    `two_stream._solve_over_gpoints`, which also records the two alternative
    fixes that were measured and failed.

    `gas_optics_mode` selects where the gas optics is evaluated:
    `'per_gpt'` (default) inside the spectral loop, `'all_gpts'` for every
    g-point ahead of it, with each table read a row gather over the g-points.
    It changes no physics. See `two_stream.solve_lw`.

    The clear-sky diagnostics (`do_clear_sky` in the config, or
    `clear_sky_fluxes=True` for this call) are the cloud-free atmosphere --
    gas plus aerosol, with the same surface albedos, surface emissivity and
    surface temperature. They are solved inside the all-sky g-point loop: the
    gas optics and Planck sources, which dominate the cost of a g-point, are
    evaluated once and shared, and only the cloud mix-in, cell properties and
    two-stream transport are repeated. That is much cheaper than a second
    solve, and returns the same fluxes as one (see
    `two_stream.solve_lw(with_clear_sky=True)`). Whether they are requested
    changes nothing in the all-sky outputs.

    `aerosol_free_fluxes=True` adds, for every problem solved (the all-sky
    one, and the clear-sky one when that is requested), its aerosol-free twin:
    the same atmosphere and cloud with no `aerosol_optics_{lw,sw}` mixed in,
    which is what the AeroCom `*noa` diagnostics (`rsutnoa`, `rlutnoa` and
    their clear-sky variants) are. They are solved inside the same g-point
    loop, sharing the gas optics, Planck sources and cloud mix-in, and equal
    a separate call without the aerosol arguments (see
    `two_stream.solve_lw(with_aerosol_free=True)`); the outputs carry an
    `aerosolfree` tag (`sw_flux_up_aerosolfree_full`,
    `toa_sw_flux_outgoing_aerosolfree_2d_xy`, and `aerosolfree_clearsky` for
    the twin of the clear sky). Without aerosol arguments the twins are the
    originals, at no extra cost.

    `lw_solver` selects the longwave transport: `'two_stream'` (the default) or
    `'noscat'`, RTE's absorption-only solver (one angle with secant
    `lw_secant`, default `1 / 0.6096748751`; cloud and aerosol scattering
    dropped, as in ECHAM6). It applies to the clear-sky diagnostic as well. The
    shortwave is always two-stream. See `two_stream.solve_lw`.

    Returns:
      A dictionary containing the following keys:
        rrtmgp_common.KEY_STORED_RADIATION: The heating rate [K/s].
      Optional keys depending on config:
        'rad_heat_sw_3d': The net shortwave radiative heating rate [K/s].
        'rad_heat_lw_3d': The net longwave radiative heating rate [K/s].
        'sw_flux_up_full': Full shortwave upward flux profile [W/m²]
        'sw_flux_down_full': Full shortwave downward flux profile [W/m²]
        'sw_flux_down_dir_sfc': Direct-beam shortwave flux incident on the
          surface, `[nx, ny]` [W/m²]. It is part of `sw_flux_down_full[:, :,
          0]`; the diffuse surface flux is the difference.
        'lw_flux_up_full': Full longwave upward flux profile [W/m²]
        'lw_flux_down_full': Full longwave downward flux profile [W/m²]
      With the clear-sky diagnostics requested, the same names with
      `clearsky` after the quantity (`'sw_flux_up_clearsky_full'`,
      `'sw_flux_down_dir_clearsky_sfc'`, `'rad_heat_sw_clearsky_3d'`, ...), and
      the `*_clearsky_2d_xy` / `*_clearsky_1d_z` diagnostics listed in
      `diagnostic_fields`.
    """
    atm_state = self.atmospheric_state
    if (
        zenith is not None
        or irrad is not None
        or sfc_alb is not None
        or sfc_emis is not None
    ):
      # Per-call overrides of the configured atmospheric-state scalars.
      # `sfc_alb` / `sfc_emis` may be traced per-column values (e.g. from a
      # host model's surface scheme under vmap); the solvers broadcast them
      # against the horizontal plane, so anything broadcastable against the
      # `[nx, ny]` surface works.
      atm_state = dataclasses.replace(
          atm_state,
          **({"zenith": zenith} if zenith is not None else {}),
          **({"irrad": irrad} if irrad is not None else {}),
          **({"sfc_alb": sfc_alb} if sfc_alb is not None else {}),
          **({"sfc_emis": sfc_emis} if sfc_emis is not None else {}),
      )

    # Temperature may have NaNs in the halos (this is intentional).  These NaNs
    # cause problems later on, so fill in the halo values with linear
    # extrapolations.  Note that NaNs in other fields do not cause issues
    # because halos are discarded later on, but there are places where the halos
    # of the temperature are used to determine interior values.
    def fill_halo(f: Array) -> Array:
      bottom_halo_val = (2 * f[:, :, 1] - f[:, :, 2])[..., jnp.newaxis]
      top_halo_val = (2 * f[:, :, -2] - f[:, :, -3])[..., jnp.newaxis]
      return jnp.concatenate(
          [bottom_halo_val, f[:, :, 1:-1], top_halo_val], axis=2
      )

    temperature = fill_halo(temperature)

    # Sometimes, the simulation can result in q_t or q_c with negative values.
    # Clip these to zero, otherwise there can be extremely deleterious effects
    # in the radiation solver.  E.g., negative relative abundance of gas species
    # and negative values in optical depth and Planck fraction that are
    # inherently nonnegative.
    q_t = jnp.clip(q_t, 0.0, None)
    q_liq = jnp.clip(q_liq, 0.0, None)
    q_ice = jnp.clip(q_ice, 0.0, None)
    q_c = jnp.clip(q_c, 0.0, None)

    # Reconstruct the volume mixing ratio (vmr) of relevant gas species from
    # the sounding-based lookup. Then apply any caller-supplied per-cell
    # overrides on top; this is how upstream models inject ozone climatologies,
    # CO2/CH4/N2O forcing, etc. Finally, h2o is always set from the simulation
    # humidity state so it stays consistent with q_t/q_c (any caller h2o entry
    # is intentionally clobbered).
    vmr_lib = atm_state.vmr
    merged_vmr_fields = (
        lookup_volume_mixing_ratio.reconstruct_vmr_fields_from_pressure(
            vmr_lib, p_ref_xxc
        )
    )
    if vmr_fields is not None:
      merged_vmr_fields.update(vmr_fields)
    merged_vmr_fields['h2o'] = _humidity_to_volume_mixing_ratio(q_t, q_c)
    vmr_fields = merged_vmr_fields

    # Compute molecules
    molecules_per_area = _air_molecules_per_area(p_ref_xxc, vmr_fields['h2o'])

    # Whether to also return the cloud-free fluxes, solved alongside the
    # all-sky ones (see `two_stream.solve_lw(with_clear_sky=...)`).
    with_clear_sky = self._do_clear_sky or clear_sky_fluxes
    with_aerosol_free = aerosol_free_fluxes

    # Compute water paths for liquid and ice cloud condensate.
    liq_water_path = _compute_cloud_path(rho_xxc, q_liq, self._dz, sg_map)
    ice_water_path = _compute_cloud_path(rho_xxc, q_ice, self._dz, sg_map)

    with jax.named_scope('lw'):
      lw_fluxes = two_stream.solve_lw(
          p_ref_xxc,
          temperature,
          molecules_per_area,
          self.optics_lib,
          atm_state,
          vmr_fields,
          sfc_temperature,
          cloud_r_eff_liq=cloud_r_eff_liq,
          cloud_path_liq=liq_water_path,
          cloud_r_eff_ice=cloud_r_eff_ice,
          cloud_path_ice=ice_water_path,
          use_scan=use_scan,
          gpt_chunk=gpt_chunk,
          cloud_path_liq_per_gpt=cloud_path_liq_lw_per_gpt,
          cloud_path_ice_per_gpt=cloud_path_ice_lw_per_gpt,
          aerosol_optics=aerosol_optics_lw,
          cloud_tau_scale_liq=cloud_tau_scale_liq,
          cloud_tau_scale_ice=cloud_tau_scale_ice,
          with_clear_sky=with_clear_sky,
          lw_solver=lw_solver,
          lw_secant=lw_secant,
          gas_optics_mode=gas_optics_mode,
          with_aerosol_free=with_aerosol_free,
      )
    with jax.named_scope('sw'):
      sw_fluxes = two_stream.solve_sw(
          p_ref_xxc,
          temperature,
          molecules_per_area,
          self.optics_lib,
          atm_state,
          vmr_fields,
          cloud_r_eff_liq=cloud_r_eff_liq,
          cloud_path_liq=liq_water_path,
          cloud_r_eff_ice=cloud_r_eff_ice,
          cloud_path_ice=ice_water_path,
          use_scan=use_scan,
          gpt_chunk=gpt_chunk,
          cloud_path_liq_per_gpt=cloud_path_liq_sw_per_gpt,
          cloud_path_ice_per_gpt=cloud_path_ice_sw_per_gpt,
          aerosol_optics=aerosol_optics_sw,
          cloud_tau_scale_liq=cloud_tau_scale_liq,
          cloud_tau_scale_ice=cloud_tau_scale_ice,
          sfc_alb_dir=sfc_alb_dir,
          sfc_alb_dif=sfc_alb_dif,
          with_clear_sky=with_clear_sky,
          gas_optics_mode=gas_optics_mode,
          with_aerosol_free=with_aerosol_free,
      )
    # Take the extra problems apart: the aerosol-free twins first (they may
    # still hold `clear_` entries), then the clear sky of each half. `extra`
    # maps an output tag to its (lw, sw) flux dictionaries.
    extra: dict[str, tuple[dict[str, Array], dict[str, Array]]] = {}
    if with_aerosol_free:
      lw_fluxes, lw_noaer = two_stream.split_aerosol_free_fluxes(lw_fluxes)
      sw_fluxes, sw_noaer = two_stream.split_aerosol_free_fluxes(sw_fluxes)
      if with_clear_sky:
        lw_noaer, lw_noaer_clear = two_stream.split_clear_sky_fluxes(lw_noaer)
        sw_noaer, sw_noaer_clear = two_stream.split_clear_sky_fluxes(sw_noaer)
        extra['aerosolfree_clearsky'] = (lw_noaer_clear, sw_noaer_clear)
      extra['aerosolfree'] = (lw_noaer, sw_noaer)
    if with_clear_sky:
      lw_fluxes, lw_fluxes_clearsky = two_stream.split_clear_sky_fluxes(
          lw_fluxes
      )
      sw_fluxes, sw_fluxes_clearsky = two_stream.split_clear_sky_fluxes(
          sw_fluxes
      )
      extra['clearsky'] = (lw_fluxes_clearsky, sw_fluxes_clearsky)

    # Compute the heating rate in K/s.
    q_v = q_t - q_c
    lw_heating_rate = two_stream.compute_heating_rate(
        lw_fluxes['flux_net'], p_ref_xxc, q_v
    )
    sw_heating_rate = two_stream.compute_heating_rate(
        sw_fluxes['flux_net'], p_ref_xxc, q_v
    )
    # Compute the total heating rate (temperature tendency due to radiation).
    heating_rate = lw_heating_rate + sw_heating_rate

    output = {rrtmgp_common.KEY_STORED_RADIATION: heating_rate}

    # add LW, SW heating rate or fluxes if desired.
    if self._save_lw_sw_heating_rates:
      output['rad_heat_sw_3d'] = sw_heating_rate
      output['rad_heat_lw_3d'] = lw_heating_rate

    # Compute diagnostics, if desired.
    lw_flux_down = lw_fluxes['flux_down']
    lw_flux_up = lw_fluxes['flux_up']
    sw_flux_down = sw_fluxes['flux_down']
    sw_flux_up = sw_fluxes['flux_up']
    hw = 1  # halo width.

    # Add full flux profiles (remove surface halos)
    output['sw_flux_up_full'] = sw_flux_up[:, :, hw:]  # Full profile (..., nlev+1)
    output['sw_flux_down_full'] = sw_flux_down[:, :, hw:]  # Full profile (..., nlev+1)
    output['lw_flux_up_full'] = lw_flux_up[:, :, hw:]  # Full profile (..., nlev+1)
    output['lw_flux_down_full'] = lw_flux_down[:, :, hw:]  # Full profile (..., nlev+1)
    # Direct beam at the surface, for the host's direct / diffuse partition.
    output['sw_flux_down_dir_sfc'] = sw_fluxes['flux_down_dir_sfc']

    # 2D diagnostics
    if (v := 'surf_lw_flux_down_2d_xy') in self._diagnostic_fields:
      output[v] = lw_flux_down[:, :, hw]
    if (v := 'surf_lw_flux_up_2d_xy') in self._diagnostic_fields:
      output[v] = lw_flux_up[:, :, hw]
    if (v := 'surf_sw_flux_down_2d_xy') in self._diagnostic_fields:
      output[v] = sw_flux_down[:, :, hw]
    if (v:= 'surf_sw_flux_up_2d_xy') in self._diagnostic_fields:
      output[v] = sw_flux_up[:, :, hw]
    if (v := 'surf_sw_flux_down_dir_2d_xy') in self._diagnostic_fields:
      output[v] = sw_fluxes['flux_down_dir_sfc']
    # Add clear sky surf

    if (v := 'toa_sw_flux_incoming_2d_xy') in self._diagnostic_fields:
      output[v] = sw_flux_down[:, :, -hw]
    if (v := 'toa_sw_flux_outgoing_2d_xy') in self._diagnostic_fields:
      output[v] = sw_flux_up[:, :, -hw]
    if (v := 'toa_lw_flux_outgoing_2d_xy') in self._diagnostic_fields:
      output[v] = lw_flux_up[:, :, -hw]
    # Add clear sky toa

    # 1D diagnostics
    if (v := 'rad_heat_lw_1d_z') in self._diagnostic_fields:
      output[v] = _horiz_mean(lw_heating_rate)
    if (v := 'rad_heat_sw_1d_z') in self._diagnostic_fields:
      output[v] = _horiz_mean(sw_heating_rate)

    # The extra problems (clear sky, aerosol-free, aerosol-free clear sky):
    # the same profiles and diagnostics as the all-sky ones, each key tagged
    # with the problem's name. The fluxes were solved above.
    for tag, (lw_extra, sw_extra) in extra.items():
      self._add_tagged_outputs(output, tag, lw_extra, sw_extra, p_ref_xxc, q_v)

    return output

  def _add_tagged_outputs(
      self,
      output: dict[str, Array],
      tag: str,
      lw_fluxes: dict[str, Array],
      sw_fluxes: dict[str, Array],
      p_ref_xxc: Array,
      q_v: Array,
  ) -> None:
    """Adds one extra problem's profiles and diagnostics to `output`.

    `tag` is `'clearsky'`, `'aerosolfree'` or `'aerosolfree_clearsky'`; each
    all-sky key gets the tag before its `_full` / `_sfc` / `_2d_xy` / `_1d_z`
    / `_3d` suffix (`sw_flux_up_clearsky_full`, ...). The 2D and 1D
    diagnostics are added when their tagged name is in `diagnostic_fields`;
    the heating rates when `save_lw_sw_heating_rates` is set.
    """
    hw = 1  # halo width.
    lw_heating_rate = two_stream.compute_heating_rate(
        lw_fluxes['flux_net'], p_ref_xxc, q_v
    )
    sw_heating_rate = two_stream.compute_heating_rate(
        sw_fluxes['flux_net'], p_ref_xxc, q_v
    )
    if self._save_lw_sw_heating_rates:
      output[f'rad_heat_sw_{tag}_3d'] = sw_heating_rate
      output[f'rad_heat_lw_{tag}_3d'] = lw_heating_rate

    lw_flux_down = lw_fluxes['flux_down']
    lw_flux_up = lw_fluxes['flux_up']
    sw_flux_down = sw_fluxes['flux_down']
    sw_flux_up = sw_fluxes['flux_up']

    # Full flux profiles (surface halo removed), shaped (..., nlev+1).
    output[f'sw_flux_up_{tag}_full'] = sw_flux_up[:, :, hw:]
    output[f'sw_flux_down_{tag}_full'] = sw_flux_down[:, :, hw:]
    output[f'lw_flux_up_{tag}_full'] = lw_flux_up[:, :, hw:]
    output[f'lw_flux_down_{tag}_full'] = lw_flux_down[:, :, hw:]
    output[f'sw_flux_down_dir_{tag}_sfc'] = sw_fluxes['flux_down_dir_sfc']

    df = self._diagnostic_fields
    # 2D diagnostics
    if (v := f'surf_lw_flux_down_{tag}_2d_xy') in df:
      output[v] = lw_flux_down[:, :, hw]
    if (v := f'surf_lw_flux_up_{tag}_2d_xy') in df:
      output[v] = lw_flux_up[:, :, hw]
    if (v := f'surf_sw_flux_down_{tag}_2d_xy') in df:
      output[v] = sw_flux_down[:, :, hw]
    if (v := f'surf_sw_flux_up_{tag}_2d_xy') in df:
      output[v] = sw_flux_up[:, :, hw]
    if (v := f'surf_sw_flux_down_dir_{tag}_2d_xy') in df:
      output[v] = sw_fluxes['flux_down_dir_sfc']
    if (v := f'toa_sw_flux_outgoing_{tag}_2d_xy') in df:
      output[v] = sw_flux_up[:, :, -hw]
    if (v := f'toa_lw_flux_outgoing_{tag}_2d_xy') in df:
      output[v] = lw_flux_up[:, :, -hw]

    # 1D diagnostics
    if (v := f'rad_heat_lw_{tag}_1d_z') in df:
      output[v] = _horiz_mean(lw_heating_rate)
    if (v := f'rad_heat_sw_{tag}_1d_z') in df:
      output[v] = _horiz_mean(sw_heating_rate)

  def compute_heating_rate_zc(
      self,
      q_t: Array,
      cloud_r_eff_liq: Array | None,
      cloud_r_eff_ice: Array | None,
      temperature: Array,
      sfc_temperature: Array | float,
      p_ref_zc: Array,
      *,
      rho_zc: Array | None = None,
      q_liq: Array | None = None,
      q_ice: Array | None = None,
      q_c: Array | None = None,
      dz_zc: float | Array | None = None,
      zenith: float | Array | None = None,
      irrad: float | Array | None = None,
      cloud_path_liq_lw_per_gpt: Array | None = None,
      cloud_path_ice_lw_per_gpt: Array | None = None,
      cloud_path_liq_sw_per_gpt: Array | None = None,
      cloud_path_ice_sw_per_gpt: Array | None = None,
      vmr_fields: dict[str, Array] | None = None,
      aerosol_optics_lw: dict[str, Array] | None = None,
      aerosol_optics_sw: dict[str, Array] | None = None,
      sfc_alb: float | Array | None = None,
      sfc_emis: float | Array | None = None,
      gpt_chunk: int = 64,
      cloud_tau_scale_liq: float | Array | None = None,
      cloud_tau_scale_ice: float | Array | None = None,
      sfc_alb_dir: float | Array | None = None,
      sfc_alb_dif: float | Array | None = None,
      clear_sky_fluxes: bool = False,
      lw_solver: str = two_stream.LW_SOLVER_TWO_STREAM,
      lw_secant: float = longwave_noscat.LW_NOSCAT_DEFAULT_SECANT,
      aerosol_free_fluxes: bool = False,
      per_gpt_axis: int = 1,
  ) -> dict[str, Array]:
    """Column-minor `(nzh, nc)` counterpart of `compute_heating_rate` for TPU.

    Accepts column-minor `(nzh, nc)` atmospheric profiles (`nc` trailing),
    interior `(nz, nc)` or haloed `(nzh, nc)` effective radii, interior
    `(nz, n_gpt, nc)` McICA cloud paths (or `(n_gpt, nz, nc)` with
    `per_gpt_axis=0`), interior `(n_bnd, nz, nc)` aerosol optics, `(nc,)`
    surface fields, and `(n_bnd_sw, nc)` per-band SW surface albedos, returning
    column-minor heating rates `(nzh, nc)`, full flux profiles `(nz + 1, nc)`,
    and `(nc,)` surface/TOA diagnostics without `(ncols, ..., nlev)` transposes.
    """
    assert isinstance(self.optics_lib, optics.RRTMOptics), (
        'compute_heating_rate_zc requires the RRTMOptics scheme.'
    )
    atm_state = self.atmospheric_state
    if (
        zenith is not None
        or irrad is not None
        or sfc_alb is not None
        or sfc_emis is not None
    ):
      atm_state = dataclasses.replace(
          atm_state,
          **({'zenith': zenith} if zenith is not None else {}),
          **({'irrad': irrad} if irrad is not None else {}),
          **({'sfc_alb': sfc_alb} if sfc_alb is not None else {}),
          **({'sfc_emis': sfc_emis} if sfc_emis is not None else {}),
      )

    bottom_halo_val = 2.0 * temperature[1:2, :] - temperature[2:3, :]
    top_halo_val = 2.0 * temperature[-2:-1, :] - temperature[-3:-2, :]
    temperature = jnp.concatenate(
        [bottom_halo_val, temperature[1:-1, :], top_halo_val], axis=0
    )

    q_t = jnp.clip(q_t, 0.0, None)
    q_c_val = jnp.zeros_like(q_t) if q_c is None else jnp.clip(q_c, 0.0, None)

    vmr_lib = atm_state.vmr
    merged_vmr_fields = (
        lookup_volume_mixing_ratio.reconstruct_vmr_fields_from_pressure(
            vmr_lib, p_ref_zc
        )
    )
    if vmr_fields is not None:
      merged_vmr_fields.update(vmr_fields)
    merged_vmr_fields['h2o'] = _humidity_to_volume_mixing_ratio(q_t, q_c_val)

    molecules_per_area = _air_molecules_per_area_zc(
        p_ref_zc, merged_vmr_fields['h2o']
    )

    with_clear_sky = self._do_clear_sky or clear_sky_fluxes
    with_aerosol_free = aerosol_free_fluxes

    h = self._dz if dz_zc is None else dz_zc
    liq_water_path = None
    if q_liq is not None and rho_zc is not None:
      liq_water_path = rho_zc * jnp.clip(q_liq, 0.0, None) * h
    ice_water_path = None
    if q_ice is not None and rho_zc is not None:
      ice_water_path = rho_zc * jnp.clip(q_ice, 0.0, None) * h

    with jax.named_scope('lw'):
      lw_fluxes = two_stream.solve_lw_zc(
          p_ref_zc,
          temperature,
          molecules_per_area,
          self.optics_lib,
          atm_state,
          merged_vmr_fields,
          sfc_temperature,
          cloud_r_eff_liq=cloud_r_eff_liq,
          cloud_path_liq=liq_water_path,
          cloud_r_eff_ice=cloud_r_eff_ice,
          cloud_path_ice=ice_water_path,
          cloud_path_liq_per_gpt=cloud_path_liq_lw_per_gpt,
          cloud_path_ice_per_gpt=cloud_path_ice_lw_per_gpt,
          aerosol_optics=aerosol_optics_lw,
          gpt_chunk=gpt_chunk,
          cloud_tau_scale_liq=cloud_tau_scale_liq,
          cloud_tau_scale_ice=cloud_tau_scale_ice,
          with_clear_sky=with_clear_sky,
          lw_solver=lw_solver,
          lw_secant=lw_secant,
          with_aerosol_free=with_aerosol_free,
          per_gpt_axis=per_gpt_axis,
      )
    with jax.named_scope('sw'):
      sw_fluxes = two_stream.solve_sw_zc(
          p_ref_zc,
          temperature,
          molecules_per_area,
          self.optics_lib,
          atm_state,
          merged_vmr_fields,
          cloud_r_eff_liq=cloud_r_eff_liq,
          cloud_path_liq=liq_water_path,
          cloud_r_eff_ice=cloud_r_eff_ice,
          cloud_path_ice=ice_water_path,
          cloud_path_liq_per_gpt=cloud_path_liq_sw_per_gpt,
          cloud_path_ice_per_gpt=cloud_path_ice_sw_per_gpt,
          aerosol_optics=aerosol_optics_sw,
          gpt_chunk=gpt_chunk,
          cloud_tau_scale_liq=cloud_tau_scale_liq,
          cloud_tau_scale_ice=cloud_tau_scale_ice,
          sfc_alb_dir=sfc_alb_dir,
          sfc_alb_dif=sfc_alb_dif,
          with_clear_sky=with_clear_sky,
          with_aerosol_free=with_aerosol_free,
          per_gpt_axis=per_gpt_axis,
      )

    extra: dict[str, tuple[dict[str, Array], dict[str, Array]]] = {}
    if with_aerosol_free:
      lw_fluxes, lw_noaer = two_stream.split_aerosol_free_fluxes(lw_fluxes)
      sw_fluxes, sw_noaer = two_stream.split_aerosol_free_fluxes(sw_fluxes)
      if with_clear_sky:
        lw_noaer, lw_noaer_clear = two_stream.split_clear_sky_fluxes(lw_noaer)
        sw_noaer, sw_noaer_clear = two_stream.split_clear_sky_fluxes(sw_noaer)
        extra['aerosolfree_clearsky'] = (lw_noaer_clear, sw_noaer_clear)
      extra['aerosolfree'] = (lw_noaer, sw_noaer)
    if with_clear_sky:
      lw_fluxes, lw_fluxes_clearsky = two_stream.split_clear_sky_fluxes(
          lw_fluxes
      )
      sw_fluxes, sw_fluxes_clearsky = two_stream.split_clear_sky_fluxes(
          sw_fluxes
      )
      extra['clearsky'] = (lw_fluxes_clearsky, sw_fluxes_clearsky)

    q_v = q_t if q_c is None else (q_t - q_c_val)
    lw_heating_rate = two_stream.compute_heating_rate_zc(
        lw_fluxes['flux_net'], p_ref_zc, q_v
    )
    sw_heating_rate = two_stream.compute_heating_rate_zc(
        sw_fluxes['flux_net'], p_ref_zc, q_v
    )
    heating_rate = lw_heating_rate + sw_heating_rate

    output = {rrtmgp_common.KEY_STORED_RADIATION: heating_rate}
    if self._save_lw_sw_heating_rates:
      output['rad_heat_sw_3d'] = sw_heating_rate
      output['rad_heat_lw_3d'] = lw_heating_rate

    lw_flux_down = lw_fluxes['flux_down']
    lw_flux_up = lw_fluxes['flux_up']
    sw_flux_down = sw_fluxes['flux_down']
    sw_flux_up = sw_fluxes['flux_up']
    hw = 1

    output['sw_flux_up_full'] = sw_flux_up[hw:, :]
    output['sw_flux_down_full'] = sw_flux_down[hw:, :]
    output['lw_flux_up_full'] = lw_flux_up[hw:, :]
    output['lw_flux_down_full'] = lw_flux_down[hw:, :]
    output['sw_flux_down_dir_sfc'] = sw_fluxes['flux_down_dir_sfc']

    df = self._diagnostic_fields
    if (v := 'surf_lw_flux_down_2d_xy') in df:
      output[v] = lw_flux_down[hw, :]
    if (v := 'surf_lw_flux_up_2d_xy') in df:
      output[v] = lw_flux_up[hw, :]
    if (v := 'surf_sw_flux_down_2d_xy') in df:
      output[v] = sw_flux_down[hw, :]
    if (v := 'surf_sw_flux_up_2d_xy') in df:
      output[v] = sw_flux_up[hw, :]
    if (v := 'surf_sw_flux_down_dir_2d_xy') in df:
      output[v] = sw_fluxes['flux_down_dir_sfc']

    if (v := 'toa_sw_flux_incoming_2d_xy') in df:
      output[v] = sw_flux_down[-hw, :]
    if (v := 'toa_sw_flux_outgoing_2d_xy') in df:
      output[v] = sw_flux_up[-hw, :]
    if (v := 'toa_lw_flux_outgoing_2d_xy') in df:
      output[v] = lw_flux_up[-hw, :]

    if (v := 'rad_heat_lw_1d_z') in df:
      output[v] = _horiz_mean_zc(lw_heating_rate)
    if (v := 'rad_heat_sw_1d_z') in df:
      output[v] = _horiz_mean_zc(sw_heating_rate)

    for tag, (lw_extra, sw_extra) in extra.items():
      self._add_tagged_outputs_zc(output, tag, lw_extra, sw_extra, p_ref_zc, q_v)

    return output

  def _add_tagged_outputs_zc(
      self,
      output: dict[str, Array],
      tag: str,
      lw_fluxes: dict[str, Array],
      sw_fluxes: dict[str, Array],
      p_ref_zc: Array,
      q_v: Array,
  ) -> None:
    """Column-minor `(nzh, nc)` counterpart of `_add_tagged_outputs`."""
    hw = 1
    lw_heating_rate = two_stream.compute_heating_rate_zc(
        lw_fluxes['flux_net'], p_ref_zc, q_v
    )
    sw_heating_rate = two_stream.compute_heating_rate_zc(
        sw_fluxes['flux_net'], p_ref_zc, q_v
    )
    if self._save_lw_sw_heating_rates:
      output[f'rad_heat_sw_{tag}_3d'] = sw_heating_rate
      output[f'rad_heat_lw_{tag}_3d'] = lw_heating_rate

    lw_flux_down = lw_fluxes['flux_down']
    lw_flux_up = lw_fluxes['flux_up']
    sw_flux_down = sw_fluxes['flux_down']
    sw_flux_up = sw_fluxes['flux_up']

    output[f'sw_flux_up_{tag}_full'] = sw_flux_up[hw:, :]
    output[f'sw_flux_down_{tag}_full'] = sw_flux_down[hw:, :]
    output[f'lw_flux_up_{tag}_full'] = lw_flux_up[hw:, :]
    output[f'lw_flux_down_{tag}_full'] = lw_flux_down[hw:, :]
    output[f'sw_flux_down_dir_{tag}_sfc'] = sw_fluxes['flux_down_dir_sfc']

    df = self._diagnostic_fields
    if (v := f'surf_lw_flux_down_{tag}_2d_xy') in df:
      output[v] = lw_flux_down[hw, :]
    if (v := f'surf_lw_flux_up_{tag}_2d_xy') in df:
      output[v] = lw_flux_up[hw, :]
    if (v := f'surf_sw_flux_down_{tag}_2d_xy') in df:
      output[v] = sw_flux_down[hw, :]
    if (v := f'surf_sw_flux_up_{tag}_2d_xy') in df:
      output[v] = sw_flux_up[hw, :]
    if (v := f'surf_sw_flux_down_dir_{tag}_2d_xy') in df:
      output[v] = sw_fluxes['flux_down_dir_sfc']
    if (v := f'toa_sw_flux_outgoing_{tag}_2d_xy') in df:
      output[v] = sw_flux_up[-hw, :]
    if (v := f'toa_lw_flux_outgoing_{tag}_2d_xy') in df:
      output[v] = lw_flux_up[-hw, :]

    if (v := f'rad_heat_lw_{tag}_1d_z') in df:
      output[v] = _horiz_mean_zc(lw_heating_rate)
    if (v := f'rad_heat_sw_{tag}_1d_z') in df:
      output[v] = _horiz_mean_zc(sw_heating_rate)
