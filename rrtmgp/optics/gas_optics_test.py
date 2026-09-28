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

"""Tests whether the atmospheric conditions are loaded properly from a proto."""

from typing import TypeAlias

import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from rrtmgp.config import radiative_transfer
from rrtmgp.optics import gas_optics
from rrtmgp.optics import lookup_gas_optics_longwave
from rrtmgp.optics import lookup_gas_optics_shortwave
from rrtmgp.optics import lookup_volume_mixing_ratio
from rrtmgp.optics import optics_utils

Array: TypeAlias = jax.Array
IndexAndWeight: TypeAlias = optics_utils.IndexAndWeight
Interpolant: TypeAlias = optics_utils.Interpolant

_LW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-lw-g256.nc'
_SW_LOOKUP_TABLE_FILENAME = 'rrtmgp/optics/rrtmgp_data/rrtmgp-gas-sw-g224.nc'
_GLOBAL_MEANS_FILENAME = 'rrtmgp/optics/test_data/vmr_global_means.json'

root = Path()
_LW_LOOKUP_TABLE_FILEPATH = root / _LW_LOOKUP_TABLE_FILENAME
_SW_LOOKUP_TABLE_FILEPATH = root / _SW_LOOKUP_TABLE_FILENAME
_GLOBAL_MEANS_FILEPATH = root / _GLOBAL_MEANS_FILENAME


def assert_interpolant_allclose(i1: Interpolant, i2: Interpolant):
  rtol = 1e-5
  atol = 1e-6
  np.testing.assert_allclose(i1.interp_low.idx, i2.interp_low.idx, rtol, atol)
  np.testing.assert_allclose(
      i1.interp_low.weight, i2.interp_low.weight, rtol, atol
  )
  np.testing.assert_allclose(i1.interp_high.idx, i2.interp_high.idx, rtol, atol)
  np.testing.assert_allclose(
      i1.interp_high.weight, i2.interp_high.weight, rtol, atol
  )


class GasOpticsTest(unittest.TestCase):

  def setUp(self):
    super(GasOpticsTest, self).setUp()
    self.gas_optics_lw = lookup_gas_optics_longwave.from_nc_file(
        _LW_LOOKUP_TABLE_FILEPATH
    )
    self.gas_optics_sw = lookup_gas_optics_shortwave.from_nc_file(
        _SW_LOOKUP_TABLE_FILEPATH
    )
    atmospheric_state_cfg = radiative_transfer.AtmosphericStateCfg(
        vmr_global_mean_filepath=_GLOBAL_MEANS_FILEPATH,
    )
    self.vmr_lib = lookup_volume_mixing_ratio.from_config(atmospheric_state_cfg)

  def test_get_vmr(self):
    """Tests that correct variable and global mean vmr's are returned."""
    major_species_idx_h20 = jnp.ones((1, 4), dtype=jnp.int_)
    major_species_idx_co2 = 2 * jnp.ones((1, 4), dtype=jnp.int_)
    major_species_idx_o3 = 3 * jnp.ones((1, 4), dtype=jnp.int_)
    major_species_idx = jnp.concatenate(
        [major_species_idx_h20, major_species_idx_co2, major_species_idx_o3],
        axis=0,
    )
    precomputed_vmr_h2o = jnp.array(
        [[3.7729078e-06, 1.61512e-05, 0.00273486, 0.018282978]],
        dtype=jnp.float_,
    )
    precomputed_vmr_o3 = jnp.array(
        [[1.9249276e-06, 4.4498346e-08, 4.7968513e-08, 3.5275427e-08]]
    )
    vmr_fields = {
        self.gas_optics_lw.idx_h2o: precomputed_vmr_h2o,
        self.gas_optics_lw.idx_o3: precomputed_vmr_o3,
    }
    # A global mean is used for CO2, so vmr does not change with pressure level.
    expected_vmr_co2 = 3.9754697e-4 * jnp.ones((1, 4), dtype=jnp.float_)
    expected_vmr = jnp.concatenate(
        [precomputed_vmr_h2o, expected_vmr_co2, precomputed_vmr_o3], axis=0
    )

    with self.subTest('VMRWithinRange'):
      vmr = gas_optics.get_vmr(
          self.gas_optics_lw, self.vmr_lib, major_species_idx, vmr_fields
      )
      np.testing.assert_allclose(vmr, expected_vmr, rtol=1e-5, atol=0)

  def test_compute_relative_abundance_interpolant(self):
    """Tests that the correct relative abundance interpolants are computed."""
    troposphere_idx = jnp.ones((3, 4), dtype=jnp.int_)
    temperature_idx = 10 * jnp.ones((3, 4), dtype=jnp.int_)
    ibnd = 4
    relative_abundance_interp = (
        gas_optics._compute_relative_abundance_interpolant(
            self.gas_optics_lw,
            self.vmr_lib,
            troposphere_idx,
            temperature_idx,
            ibnd,
            True,
        )
    )
    idx_low = jnp.zeros((3, 4), dtype=jnp.int_)
    idx_high = jnp.ones((3, 4), dtype=jnp.int_)
    weight_low = 5.2951004292976034e-06 * jnp.ones((3, 4), dtype=jnp.float_)
    weight_high = 3.5275427000000043e-07 * jnp.ones((3, 4), dtype=jnp.float_)
    idx_and_weight_low = IndexAndWeight(idx_low, weight_low)
    idx_and_weight_high = IndexAndWeight(idx_high, weight_high)
    expected_interp = Interpolant(idx_and_weight_low, idx_and_weight_high)
    assert_interpolant_allclose(relative_abundance_interp, expected_interp)

  def test_compute_major_optical_depth(self):
    """Checks the optical depth computation for different values of t and p."""
    temperature = jnp.array(
        [[160.0, 200.0, 300.0], [280.0, 290.0, 355.0]], dtype=jnp.float_
    )
    pressure = jnp.array(
        [
            [1.09663316e5, 90000.0, 80000.0],
            [7.35095189e04, 30000.0, 1.00518357],
        ],
        dtype=jnp.float_,
    )
    molecules = jnp.array([[1e24]], dtype=jnp.float_)
    major_optical_depth = gas_optics.compute_major_optical_depth(
        self.gas_optics_lw, self.vmr_lib, molecules, temperature, pressure, 70
    )
    expected_major_optical_depth = jnp.array(
        [
            [1.071459e-5, 2.313674e-5, 1.053062e-4],
            [7.901242e-5, 4.897409e-5, 1.766592e-7],
        ],
        dtype=jnp.float_,
    )
    np.testing.assert_allclose(
        expected_major_optical_depth, major_optical_depth, rtol=1e-5, atol=0
    )

  def test_compute_minor_optical_depth(self):
    """Checks the minor optical depth computation for a particular g-point."""
    # Temperature corresponding to the 10th reference point.
    temperature = jnp.array([[310.0]], dtype=jnp.float_)
    p = jnp.array([[73509.51892419]], dtype=jnp.float_)
    moles = jnp.array([[1e24]], dtype=jnp.float_)

    with jax.disable_jit():
      # Disable jit commpile to ensure the if/else branch is refreshed at each call
      with self.subTest('PrecomputedVmrH2O'):
        # Precomputed VMR for H2O.
        vmr_fields = {
            self.gas_optics_lw.idx_h2o: jnp.array([[1.2e-3]], dtype=jnp.float_),
        }
        minor_optical_depth = gas_optics.compute_minor_optical_depth(
            self.gas_optics_lw,
            self.vmr_lib,
            moles,
            temperature,
            p,
            100,
            vmr_fields,
        )
        np.testing.assert_allclose(
            minor_optical_depth, [[3.755038e-7]], rtol=1e-5, atol=1e-12
        )

      with self.subTest('AbsentH2OVmr'):
        vmr_fields = {
            self.gas_optics_lw.idx_o3: jnp.array([[1.2e-3]], dtype=jnp.float_),
        }
        minor_optical_depth = gas_optics.compute_minor_optical_depth(
            self.gas_optics_lw,
            self.vmr_lib,
            moles,
            temperature,
            p,
            100,
            vmr_fields,
        )
        np.testing.assert_allclose(
            [[1.768588e-7]], minor_optical_depth, rtol=1e-5, atol=1e-12
        )

  def test_minor_optical_depth_across_the_troposphere_split(self):
    """Cells either side of `p_ref_tropo` draw on their own absorber table.

    RRTMGP splits the minor absorbers into a lower- and an upper-atmosphere
    table and a cell uses exactly one of them. The two are walked as one merged
    interval list with a per-cell mask, so a field spanning the split is the
    case that pins the mask: getting it wrong silently gives a cell the wrong
    half's absorbers, which no single-regime test would see.
    """
    lookup = self.gas_optics_lw
    p_tropo = float(lookup.p_ref_tropo)
    # One cell below the reference pressure, one above.
    p = jnp.array([[73509.51892419, 5000.0]], dtype=jnp.float_)
    self.assertGreater(float(p[0, 0]), p_tropo)
    self.assertLess(float(p[0, 1]), p_tropo)
    temperature = jnp.array([[310.0, 260.0]], dtype=jnp.float_)
    moles = jnp.array([[1e24, 1e24]], dtype=jnp.float_)
    vmr_fields = {
        lookup.idx_h2o: jnp.array([[1.2e-3, 4.0e-6]], dtype=jnp.float_),
    }

    for igpt, expected in ((100, [[3.7550384e-07, 2.6859974e-09]]),
                           (20, [[3.4000088e-05, 3.0309238e-08]])):
      with self.subTest(f'gpt{igpt}'):
        minor_optical_depth = gas_optics.compute_minor_optical_depth(
            lookup, self.vmr_lib, moles, temperature, p, igpt, vmr_fields
        )
        np.testing.assert_allclose(
            minor_optical_depth, expected, rtol=1e-5, atol=1e-12
        )

  def test_minor_optical_depth_gradients(self):
    """Reverse-mode gradients through the absorber batch must be right.

    This code path is a fixed-size batch rather than a data-dependent loop
    precisely so that it can be differentiated -- a `while_loop` with a
    data-dependent trip count cannot be -- so a gradient that is wrong or
    non-finite defeats the reason for the structure.
    """
    lookup = self.gas_optics_lw
    p = jnp.array([[73509.51892419, 5000.0]], dtype=jnp.float_)
    temperature = jnp.array([[310.0, 260.0]], dtype=jnp.float_)
    moles = jnp.array([[1e24, 1e24]], dtype=jnp.float_)
    vmr_h2o = jnp.array([[1.2e-3, 4.0e-6]], dtype=jnp.float_)

    def total_optical_depth(h2o):
      return jnp.sum(gas_optics.compute_minor_optical_depth(
          lookup, self.vmr_lib, moles, temperature, p, 100,
          {lookup.idx_h2o: h2o},
      ))

    grad = jax.grad(total_optical_depth)(vmr_h2o)
    self.assertTrue(bool(jnp.isfinite(grad).all()))

    # Central differences on the same quantity. The step is large relative to
    # float32 epsilon because the optical depth itself is ~1e-7.
    eps = 1e-5
    for i in range(vmr_h2o.shape[1]):
      with self.subTest(f'cell{i}'):
        bump = jnp.zeros_like(vmr_h2o).at[0, i].set(eps)
        expected = (total_optical_depth(vmr_h2o + bump)
                    - total_optical_depth(vmr_h2o - bump)) / (2.0 * eps)
        np.testing.assert_allclose(
            grad[0, i], expected, rtol=1e-2, atol=1e-6
        )

  def test_optical_depth_gradients_finite_at_zero_vmr(self):
    """A cell with both dominant species at exactly zero must still linearise.

    The relative-abundance interpolant divides by the combined volume mixing
    ratio of the two dominant species and falls back to 0.5 where that is zero.
    `jnp.where` evaluates both arms, so an unguarded denominator makes the
    discarded arm NaN and reverse-mode differentiation carries it back into the
    gradient with respect to every gas concentration -- fine in value, fatal to
    `jax.grad`.
    """
    lookup = self.gas_optics_lw
    shape = (1, 2)
    p = jnp.array([[73509.51892419, 5000.0]], dtype=jnp.float_)
    temperature = jnp.array([[310.0, 260.0]], dtype=jnp.float_)
    moles = jnp.array([[1e24, 1e24]], dtype=jnp.float_)
    zero_vmr = jnp.zeros(shape, dtype=jnp.float_)

    for name, fn in (
        ('minor', gas_optics.compute_minor_optical_depth),
        ('major', gas_optics.compute_major_optical_depth),
    ):
      with self.subTest(name):
        def total(h2o, fn=fn):
          return jnp.sum(fn(
              lookup, self.vmr_lib, moles, temperature, p, 100,
              {lookup.idx_h2o: h2o, lookup.idx_o3: zero_vmr},
          ))

        grad = jax.grad(total)(zero_vmr)
        self.assertTrue(bool(jnp.isfinite(grad).all()),
                        msg=f'{name} gradient: {grad}')

  def test_compute_rayleigh_optical_depth(self):
    """Checks the Rayleigh scattering contribution for a particular g-point."""
    # Temperature corresponding to the 10th reference point.
    temperature = jnp.array([[310.0]], dtype=jnp.float_)
    p = jnp.array([[1e5]], dtype=jnp.float_)
    moles = jnp.array([[1e24]], dtype=jnp.float_)
    # Precomputed VMR for H2O.
    vmr_fields = {1: jnp.array([[1.2e-5]], dtype=jnp.float_)}

    with self.subTest('PrecomputedVmrH2O'):
      expected_minor_optical_depth = jnp.array(
          [[9.647629e-9]], dtype=jnp.float_
      )
      minor_optical_depth = gas_optics.compute_rayleigh_optical_depth(
          self.gas_optics_sw,
          self.vmr_lib,
          moles,
          temperature,
          p,
          100,
          vmr_fields,
      )
      np.testing.assert_allclose(
          minor_optical_depth,
          expected_minor_optical_depth,
          rtol=1e-5,
          atol=1e-14,
      )

    with self.subTest('AbsentVMRFields'):
      expected_minor_optical_depth = jnp.array(
          [[9.642172e-9]], dtype=jnp.float_
      )
      minor_optical_depth = gas_optics.compute_rayleigh_optical_depth(
          self.gas_optics_sw,
          self.vmr_lib,
          moles,
          temperature,
          p,
          100,
      )
      np.testing.assert_allclose(
          minor_optical_depth,
          expected_minor_optical_depth,
          rtol=1e-5,
          atol=1e-14,
      )

  def test_compute_planck_fraction(self):
    """Tests the Planck fraction computation."""
    # Temperature corresponding to the 10th reference point.
    temperature = jnp.array([[310.0]], dtype=jnp.float_)
    # Pressure corresponding to pressure index 4.
    pressure = jnp.array([[4.92749041e+04]], dtype=jnp.float_)
    vmr_fields = {
        self.gas_optics_lw.idx_h2o: jnp.array([[1.2e-3]], dtype=jnp.float_),
        self.gas_optics_lw.idx_o3: jnp.array([[3.124e-6]], dtype=jnp.float_),
    }

    planck_fraction = gas_optics.compute_planck_fraction(
        self.gas_optics_lw,
        self.vmr_lib,
        pressure,
        temperature,
        100,
        vmr_fields,
    )
    np.testing.assert_allclose(planck_fraction, [[0.116895]], rtol=1e-5, atol=0)

  def test_compute_planck_sources(self):
    """Checks the Planck source computation for different temperature fields."""
    # Temperature corresponding to the 10th reference point.
    temperature_center = jnp.array([[310.0]], dtype=jnp.float_)
    # Temperature corresponding to the 135th reference Planck temperature.
    temperature_top = jnp.array([[295.0]], dtype=jnp.float_)
    planck_fraction = jnp.array([[0.116895]], dtype=jnp.float_)

    def planck_src_fn(temp: Array) -> Array:
      return gas_optics.compute_planck_sources(
          self.gas_optics_lw,
          planck_fraction,
          temp,
          100,
      )
    np.testing.assert_allclose(
        planck_src_fn(temperature_center), [[1.287805]], rtol=1e-5, atol=0
    )
    np.testing.assert_allclose(
        planck_src_fn(temperature_top), [[1.008423]], rtol=1e-5, atol=0
    )

  def test_major_optical_depth_extrapolates_below_the_temperature_table(self):
    """Below 160 K the absorption coefficient continues the end interval.

    Every axis but temperature is held fixed, and 145, 160 and 175 K all use
    the first temperature interval (the eta axis's temperature dependence goes
    through the same two corners), so the optical depth is exactly linear in
    temperature across them. The mirrored lookup of issue #39 returned the
    175 K value at 145 K instead.
    """
    temperature = jnp.array([[145.0, 160.0, 175.0]], dtype=jnp.float_)
    pressure = jnp.full((1, 3), 8000.0, dtype=jnp.float_)
    molecules = jnp.array([[1e24]], dtype=jnp.float_)
    for igpt in (10, 70, 200):
      tau = np.asarray(
          gas_optics.compute_major_optical_depth(
              self.gas_optics_lw, self.vmr_lib, molecules, temperature,
              pressure, igpt,
          ),
          np.float64,
      )[0]
      np.testing.assert_allclose(
          tau[0], max(2 * tau[1] - tau[2], 0.0), rtol=1e-4,
          atol=1e-6 * tau.max(), err_msg=str(igpt),
      )

  def test_optical_depths_non_negative_and_finite_far_outside_the_tables(self):
    """Extrapolation is floored: no negative absorption, no NaN, finite grads.

    The temperatures and pressures span well beyond RRTMGP's tables
    (160-355 K, 1.005 Pa - 1096 hPa) on both sides.
    """
    temperature = jnp.array(
        [[40.0, 100.0, 150.0, 400.0, 500.0]], dtype=jnp.float_
    )
    pressure = jnp.array(
        [[0.01, 0.5, 5e3, 1.2e5, 2e5]], dtype=jnp.float_
    )
    molecules = jnp.array([[1e24]], dtype=jnp.float_)
    lw, sw = self.gas_optics_lw, self.gas_optics_sw

    def taus(t):
      return {
          'major': gas_optics.compute_major_optical_depth(
              lw, self.vmr_lib, molecules, t, pressure, 70),
          'minor': gas_optics.compute_minor_optical_depth(
              lw, self.vmr_lib, molecules, t, pressure, 70),
          'rayleigh': gas_optics.compute_rayleigh_optical_depth(
              sw, self.vmr_lib, molecules, t, pressure, 50),
          'planck_fraction': gas_optics.compute_planck_fraction(
              lw, self.vmr_lib, pressure, t, 70),
      }

    for name, tau in taus(temperature).items():
      self.assertTrue(np.all(np.isfinite(tau)), name)
      self.assertTrue(np.all(np.asarray(tau) >= 0), name)
    grads = jax.grad(
        lambda t: sum(jnp.sum(v) for v in taus(t).values())
    )(temperature)
    self.assertTrue(np.all(np.isfinite(grads)))

  def test_planck_source_outside_the_temperature_table(self):
    """The band Planck source continues its end interval, floored at zero.

    This is RRTMG's `planckFunction` (ECHAM6 `mo_lrtm_driver.f90`): index
    limited to the table, signed fraction. Above the table the source keeps
    rising along the last interval; below it the first interval's line is
    followed until it reaches zero. The source is monotone in temperature
    throughout, so a colder layer never emits more (issue #39).
    """
    lw = self.gas_optics_lw
    igpt = 100
    ibnd = int(lw.g_point_to_bnd[igpt])
    table = np.asarray(lw.totplnk[ibnd], np.float64)
    t_min, t_max = float(lw.t_planck[0]), float(lw.t_planck[-1])
    dt = float(lw.t_planck[1] - lw.t_planck[0])

    def src(t):
      return gas_optics.compute_planck_sources(
          lw, jnp.ones_like(t), t, igpt
      )

    temps = jnp.arange(60.0, 450.0, 0.5, dtype=jnp.float_)
    values = np.asarray(src(temps), np.float64)
    self.assertTrue(np.all(np.isfinite(values)))
    self.assertTrue(np.all(values >= 0))
    self.assertTrue(np.all(np.diff(values) >= 0))

    below = np.array([t_min - 0.5, t_min - 3.0])
    expected_below = np.maximum(
        table[0] + (below - t_min) / dt * (table[1] - table[0]), 0.0
    )
    above = np.array([t_max + 0.5, t_max + 20.0])
    expected_above = table[-1] + (above - t_max) / dt * (table[-1] - table[-2])
    np.testing.assert_allclose(
        src(jnp.asarray(below, jnp.float_)), expected_below, rtol=1e-5
    )
    np.testing.assert_allclose(
        src(jnp.asarray(above, jnp.float_)), expected_above, rtol=1e-5
    )

    # The derivative is the end interval's slope just outside the table and
    # zero once the source has been floored.
    grad = jax.vmap(jax.grad(lambda t: src(t[None])[0]))
    slopes = np.asarray(
        grad(jnp.array([t_min - 0.5, t_max + 5.0, 20.0], dtype=jnp.float_))
    )
    np.testing.assert_allclose(
        slopes[:2],
        [(table[1] - table[0]) / dt, (table[-1] - table[-2]) / dt],
        rtol=1e-4,
    )
    self.assertEqual(slopes[2], 0.0)

if __name__ == '__main__':
  unittest.main()
