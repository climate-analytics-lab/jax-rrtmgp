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


def _air_molecules_per_area(
    p_xxc: Array, vmr_h2o_xxc: Array, axis: int = 2
) -> Array:
  """Compute the number of molecules in a grid cell per area."""
  # centered_difference returns p[k+1] - p[k-1], which spans two layers; the
  # thickness of the layer centered on k is half of it.
  dp_xxc = 0.5 * kernel_ops.centered_difference(p_xxc, dim=axis)
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


def _horiz_mean(f: Array, axis: int = 2) -> Array:
  """Compute the horizontal mean of `f`, whose vertical axis is `axis`.

  Accumulate means in float64 (when x64 is enabled) to avoid potential
  numerical issues when accumulating, but then convert back to float32.

  Args:
    f: The array to compute the horizontal mean of.
    axis: Its vertical axis; the mean is over every other axis.

  Returns:
    The horizontal mean of `f`.
  """
  acc_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
  horizontal = tuple(a for a in range(f.ndim) if a != axis % f.ndim)
  return jnp.mean(f, axis=horizontal, dtype=acc_dtype).astype(jnp.float32)


def _level(f: Array, k: int, axis: int) -> Array:
  """Level `k` of `f` along its vertical axis `axis` (a static slice)."""
  return f[(slice(None),) * axis + (k,)]


def _from_level(f: Array, k: int, axis: int) -> Array:
  """Levels `k:` of `f` along its vertical axis `axis`."""
  return f[(slice(None),) * axis + (slice(k, None),)]


def _fill_halo(f: Array, axis: int) -> Array:
  """`f` with each halo level linearly extrapolated from the two inside it."""
  n = f.shape[axis]
  take = lambda k: jax.lax.slice_in_dim(f, k, k + 1, axis=axis)
  return jnp.concatenate(
      [
          2 * take(1) - take(2),
          jax.lax.slice_in_dim(f, 1, n - 1, axis=axis),
          2 * take(n - 2) - take(n - 3),
      ],
      axis=axis,
  )


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
    atm_state = self._atmospheric_state(zenith, irrad, sfc_alb, sfc_emis)

    # Temperature may have NaNs in the halos (this is intentional).  These NaNs
    # cause problems later on, so fill in the halo values with linear
    # extrapolations.  Note that NaNs in other fields do not cause issues
    # because halos are discarded later on, but there are places where the halos
    # of the temperature are used to determine interior values.
    temperature = _fill_halo(temperature, axis=2)

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
    vmr_fields = self._vmr_fields(atm_state, p_ref_xxc, vmr_fields, q_t, q_c)

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
    return self._outputs(
        lw_fluxes, sw_fluxes, p_ref_xxc, q_t - q_c, with_clear_sky,
        with_aerosol_free, axis=2,
    )

  def _atmospheric_state(
      self,
      zenith: float | Array | None,
      irrad: float | Array | None,
      sfc_alb: float | Array | None,
      sfc_emis: float | Array | None,
  ) -> atmospheric_state.AtmosphericState:
    """The configured atmospheric state with this call's overrides."""
    overrides = {
        name: value
        for name, value in (
            ('zenith', zenith),
            ('irrad', irrad),
            ('sfc_alb', sfc_alb),
            ('sfc_emis', sfc_emis),
        )
        if value is not None
    }
    # Per-call overrides of the configured atmospheric-state scalars.
    # `sfc_alb` / `sfc_emis` may be traced per-column values (e.g. from a host
    # model's surface scheme under vmap); the solvers broadcast them against
    # the horizontal plane.
    if not overrides:
      return self.atmospheric_state
    return dataclasses.replace(self.atmospheric_state, **overrides)

  def _vmr_fields(
      self,
      atm_state: atmospheric_state.AtmosphericState,
      p: Array,
      vmr_fields: dict[str, Array] | None,
      q_t: Array,
      q_c: Array,
  ) -> dict[str, Array]:
    """The volume mixing ratios of the solve.

    Reconstruct the volume mixing ratio (vmr) of relevant gas species from
    the sounding-based lookup. Then apply any caller-supplied per-cell
    overrides on top; this is how upstream models inject ozone climatologies,
    CO2/CH4/N2O forcing, etc. Finally, h2o is always set from the simulation
    humidity state so it stays consistent with q_t/q_c (any caller h2o entry
    is intentionally clobbered).
    """
    merged = lookup_volume_mixing_ratio.reconstruct_vmr_fields_from_pressure(
        atm_state.vmr, p
    )
    if vmr_fields is not None:
      merged.update(vmr_fields)
    merged['h2o'] = _humidity_to_volume_mixing_ratio(q_t, q_c)
    return merged

  def _outputs(
      self,
      lw_fluxes: dict[str, Array],
      sw_fluxes: dict[str, Array],
      p: Array,
      q_v: Array,
      with_clear_sky: bool,
      with_aerosol_free: bool,
      axis: int,
  ) -> dict[str, Array]:
    """The heating rates, flux profiles and diagnostics of a solve.

    Args:
      lw_fluxes: The longwave fluxes, with any extra problems under their
        prefixes.
      sw_fluxes: Same as above, for the shortwave.
      p: The pressure [Pa].
      q_v: The water vapor specific humidity [kg/kg].
      with_clear_sky: Whether the fluxes include the clear-sky problems.
      with_aerosol_free: Whether they include the aerosol-free twins.
      axis: The vertical axis of the fields (2 for `(nx, ny, nz)`, 0 for
        column-minor `(nz, nc)`).

    Returns:
      The output dictionary of `compute_heating_rate`.
    """
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

    output = {}
    # The all-sky problem has no tag; its heating rate is the stored one.
    self._add_tagged_outputs(output, '', lw_fluxes, sw_fluxes, p, q_v, axis)
    # The extra problems (clear sky, aerosol-free, aerosol-free clear sky):
    # the same profiles and diagnostics as the all-sky ones, each key tagged
    # with the problem's name. The fluxes were solved above.
    for tag, (lw_extra, sw_extra) in extra.items():
      self._add_tagged_outputs(output, tag, lw_extra, sw_extra, p, q_v, axis)
    return output

  def _add_tagged_outputs(
      self,
      output: dict[str, Array],
      tag: str,
      lw_fluxes: dict[str, Array],
      sw_fluxes: dict[str, Array],
      p: Array,
      q_v: Array,
      axis: int = 2,
  ) -> None:
    """Adds one problem's heating rates, profiles and diagnostics to `output`.

    `tag` is `''` for the all-sky problem, whose total heating rate is the
    stored radiation, or `'clearsky'`, `'aerosolfree'` or
    `'aerosolfree_clearsky'`; each other key gets the tag before its `_full` /
    `_sfc` / `_2d_xy` / `_1d_z` / `_3d` suffix (`sw_flux_up_clearsky_full`,
    ...). The 2D and 1D diagnostics are added when their tagged name is in
    `diagnostic_fields`; the LW / SW heating rates when
    `save_lw_sw_heating_rates` is set. `axis` is the vertical axis of the
    fields: the "2D" diagnostics are a level of them, `(nx, ny)` or `(nc,)`.
    """
    hw = 1  # halo width.
    t = f'_{tag}' if tag else ''
    lw_heating_rate = two_stream.compute_heating_rate(
        lw_fluxes['flux_net'], p, q_v, axis=axis
    )
    sw_heating_rate = two_stream.compute_heating_rate(
        sw_fluxes['flux_net'], p, q_v, axis=axis
    )
    if not tag:
      # Compute the total heating rate (temperature tendency due to radiation).
      output[rrtmgp_common.KEY_STORED_RADIATION] = (
          lw_heating_rate + sw_heating_rate
      )
    if self._save_lw_sw_heating_rates:
      output[f'rad_heat_sw{t}_3d'] = sw_heating_rate
      output[f'rad_heat_lw{t}_3d'] = lw_heating_rate

    lw_flux_down = lw_fluxes['flux_down']
    lw_flux_up = lw_fluxes['flux_up']
    sw_flux_down = sw_fluxes['flux_down']
    sw_flux_up = sw_fluxes['flux_up']

    # Full flux profiles (surface halo removed), nlev + 1 faces.
    output[f'sw_flux_up{t}_full'] = _from_level(sw_flux_up, hw, axis)
    output[f'sw_flux_down{t}_full'] = _from_level(sw_flux_down, hw, axis)
    output[f'lw_flux_up{t}_full'] = _from_level(lw_flux_up, hw, axis)
    output[f'lw_flux_down{t}_full'] = _from_level(lw_flux_down, hw, axis)
    # Direct beam at the surface, for the host's direct / diffuse partition.
    output[f'sw_flux_down_dir{t}_sfc'] = sw_fluxes['flux_down_dir_sfc']

    df = self._diagnostic_fields
    # 2D diagnostics: the surface and the top of the atmosphere.
    for name, flux, k in (
        ('surf_lw_flux_down', lw_flux_down, hw),
        ('surf_lw_flux_up', lw_flux_up, hw),
        ('surf_sw_flux_down', sw_flux_down, hw),
        ('surf_sw_flux_up', sw_flux_up, hw),
        ('toa_sw_flux_incoming', sw_flux_down, -hw),
        ('toa_sw_flux_outgoing', sw_flux_up, -hw),
        ('toa_lw_flux_outgoing', lw_flux_up, -hw),
    ):
      if (v := f'{name}{t}_2d_xy') in df:
        output[v] = _level(flux, k, axis)
    if (v := f'surf_sw_flux_down_dir{t}_2d_xy') in df:
      output[v] = sw_fluxes['flux_down_dir_sfc']

    # 1D diagnostics
    if (v := f'rad_heat_lw{t}_1d_z') in df:
      output[v] = _horiz_mean(lw_heating_rate, axis)
    if (v := f'rad_heat_sw{t}_1d_z') in df:
      output[v] = _horiz_mean(sw_heating_rate, axis)

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
      cloud_path_liq: Array | None = None,
      cloud_path_ice: Array | None = None,
      cloud_path_liq_lw_per_gpt: Array | None = None,
      cloud_path_ice_lw_per_gpt: Array | None = None,
      cloud_path_liq_sw_per_gpt: Array | None = None,
      cloud_path_ice_sw_per_gpt: Array | None = None,
      cloud_mask_lw_per_gpt: Array | None = None,
      cloud_mask_sw_per_gpt: Array | None = None,
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
    """`compute_heating_rate` for a column-minor `(nz + 2, nc)` atmosphere.

    The same physics and outputs, for profiles laid out with the levels
    leading and the columns minor, which is how a host model that keeps its
    columns contiguous holds them: nothing is transposed to a column-leading
    layout, and the solve is the column-minor one on every backend
    (`two_stream.solve_lw_zc`, which also documents the layout and the
    measured costs). The gas optics are always hoisted out of the spectral loop
    and the recurrences always scanned; `gpt_chunk` g-points are solved per
    iteration (capped to fit vector memory with the TPU kernels, see
    `two_stream.TPU_CHUNK_VMEM_BYTES`).

    Inputs, with `nz` interior levels, `nc` columns and a halo level at each
    end of a `(nz + 2, nc)` profile:

      * `q_t`, `temperature`, `p_ref_zc`, `q_c` (and `rho_zc`, `q_liq`,
        `q_ice`, `dz_zc` when the cloud paths are derived from them): `(nz +
        2, nc)`. The temperature halos are re-filled by linear extrapolation,
        as in `compute_heating_rate`.
      * `cloud_r_eff_{liq,ice}` [m], `cloud_path_{liq,ice}` [kg/m^2]:
        `(nz, nc)` or `(nz + 2, nc)`. Given cloud paths take precedence over
        `rho_zc * q_{liq,ice} * dz_zc`.
      * `cloud_path_{liq,ice}_{lw,sw}_per_gpt`: McICA paths, `(nz[+2], n_gpt,
        nc)`, or `(n_gpt, nz[+2], nc)` with `per_gpt_axis=0`.
      * `cloud_mask_{lw,sw}_per_gpt`: boolean McICA masks of the bulk paths,
        in the same layout, as an alternative to per-g-point paths (one byte
        per entry rather than four floats; see `two_stream.solve_lw_zc`).
      * `aerosol_optics_{lw,sw}`: `(n_bnd, nz[+2], nc)` per key.
      * `vmr_fields`: `(nz + 2, nc)` per gas.
      * `zenith`, `irrad`, `sfc_alb`, `sfc_emis`, `sfc_temperature`: scalars
        or `(nc,)`; `sfc_alb_{dir,dif}`: those, or per band `(n_bnd_sw, nc)`.
      * `cloud_tau_scale_{liq,ice}`: a scalar, `(nc,)` or `(nz[+2], nc)`.

    Returns:
      The outputs of `compute_heating_rate`, column-minor: heating rates
      `(nz + 2, nc)`, flux profiles `(nz + 1, nc)` from the surface up, and
      surface / top-of-atmosphere diagnostics `(nc,)`.
    """
    assert isinstance(self.optics_lib, optics.RRTMOptics), (
        'compute_heating_rate_zc requires the RRTMOptics scheme.'
    )
    axis = 0
    atm_state = self._atmospheric_state(zenith, irrad, sfc_alb, sfc_emis)
    temperature = _fill_halo(temperature, axis)
    # See `compute_heating_rate` for the clipping.
    q_t = jnp.clip(q_t, 0.0, None)
    q_c = jnp.zeros_like(q_t) if q_c is None else jnp.clip(q_c, 0.0, None)
    vmr_fields = self._vmr_fields(atm_state, p_ref_zc, vmr_fields, q_t, q_c)
    molecules_per_area = _air_molecules_per_area(
        p_ref_zc, vmr_fields['h2o'], axis
    )
    with_clear_sky = self._do_clear_sky or clear_sky_fluxes
    with_aerosol_free = aerosol_free_fluxes

    h = self._dz if dz_zc is None else dz_zc
    if cloud_path_liq is None and q_liq is not None and rho_zc is not None:
      cloud_path_liq = rho_zc * jnp.clip(q_liq, 0.0, None) * h
    if cloud_path_ice is None and q_ice is not None and rho_zc is not None:
      cloud_path_ice = rho_zc * jnp.clip(q_ice, 0.0, None) * h

    common = dict(
        cloud_r_eff_liq=cloud_r_eff_liq,
        cloud_path_liq=cloud_path_liq,
        cloud_r_eff_ice=cloud_r_eff_ice,
        cloud_path_ice=cloud_path_ice,
        gpt_chunk=gpt_chunk,
        cloud_tau_scale_liq=cloud_tau_scale_liq,
        cloud_tau_scale_ice=cloud_tau_scale_ice,
        with_clear_sky=with_clear_sky,
        with_aerosol_free=with_aerosol_free,
        per_gpt_axis=per_gpt_axis,
    )
    with jax.named_scope('lw'):
      lw_fluxes = two_stream.solve_lw_zc(
          p_ref_zc,
          temperature,
          molecules_per_area,
          self.optics_lib,
          atm_state,
          vmr_fields,
          sfc_temperature,
          cloud_path_liq_per_gpt=cloud_path_liq_lw_per_gpt,
          cloud_path_ice_per_gpt=cloud_path_ice_lw_per_gpt,
          aerosol_optics=aerosol_optics_lw,
          lw_solver=lw_solver,
          lw_secant=lw_secant,
          cloud_mask_per_gpt=cloud_mask_lw_per_gpt,
          **common,
      )
    with jax.named_scope('sw'):
      sw_fluxes = two_stream.solve_sw_zc(
          p_ref_zc,
          temperature,
          molecules_per_area,
          self.optics_lib,
          atm_state,
          vmr_fields,
          cloud_path_liq_per_gpt=cloud_path_liq_sw_per_gpt,
          cloud_path_ice_per_gpt=cloud_path_ice_sw_per_gpt,
          aerosol_optics=aerosol_optics_sw,
          sfc_alb_dir=sfc_alb_dir,
          sfc_alb_dif=sfc_alb_dif,
          cloud_mask_per_gpt=cloud_mask_sw_per_gpt,
          **common,
      )
    return self._outputs(
        lw_fluxes, sw_fluxes, p_ref_zc, q_t - q_c, with_clear_sky,
        with_aerosol_free, axis,
    )
