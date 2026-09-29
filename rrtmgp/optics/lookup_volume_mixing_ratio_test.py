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

import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from rrtmgp.config import radiative_transfer
from rrtmgp.optics import lookup_volume_mixing_ratio

_GLOBAL_MEANS_FILENAME = 'rrtmgp/optics/test_data/vmr_global_means.json'
_SOUNDING_CSV_FILENAME = 'rrtmgp/optics/test_data/vmr_interpolated.csv'

root = Path()
_GLOBAL_MEANS_FILEPATH = root / _GLOBAL_MEANS_FILENAME
_SOUNDING_CSV_FILEPATH = root / _SOUNDING_CSV_FILENAME

# First 10 vmr values from the sounding csv file.
_EXPECTED_VMR_CH4_FROM_SOUNDING = [
    1.52882521e-07, 1.52882521e-07, 1.52882521e-07, 1.52882521e-07,
    1.52882521e-07, 1.54751092e-07, 1.58478497e-07, 1.62205903e-07,
    1.65933308e-07, 1.69660714e-07
]


class LookupVolumeMixingRatioTest(unittest.TestCase):

  def test_volume_mixing_ratio_lookup_loads_data(self):
    atmospheric_state_cfg = radiative_transfer.AtmosphericStateCfg(
        vmr_global_mean_filepath=_GLOBAL_MEANS_FILEPATH,
        vmr_sounding_filepath=_SOUNDING_CSV_FILEPATH,
    )
    lookup_vmr = lookup_volume_mixing_ratio.from_config(atmospheric_state_cfg)
    np.testing.assert_allclose(
        lookup_vmr.profiles['ch4'][:10],
        np.array(_EXPECTED_VMR_CH4_FROM_SOUNDING),
        rtol=1e-5,
        atol=0,
    )
    self.assertEqual(lookup_vmr.global_means['co2'], 3.9754697e-4)
    self.assertAlmostEqual(
        lookup_vmr.global_means['n2o'], 3.2698801e-7, delta=1e-12
    )
    self.assertEqual(lookup_vmr.global_means['co'], 1.2e-7)

  def test_sounding_held_at_its_end_values_outside_its_range(self):
    """Beyond the sounding the profile is clamped, not mirrored (#39).

    The sounding spans 10 Pa to 1032.5 hPa. A model top at 1 Pa used to get
    the mixing ratio of 100 Pa (the sounding reflected about its top); it now
    gets the value at 10 Pa, the `np.interp` convention.
    """
    atmospheric_state_cfg = radiative_transfer.AtmosphericStateCfg(
        vmr_global_mean_filepath=_GLOBAL_MEANS_FILEPATH,
        vmr_sounding_filepath=_SOUNDING_CSV_FILEPATH,
    )
    lookup_vmr = lookup_volume_mixing_ratio.from_config(atmospheric_state_cfg)
    p_ref = np.asarray(lookup_vmr.profiles['p_ref'], np.float64)
    pressure = jnp.array([1.0, 5.0, 10.0, 523.0, 5.0e4, 103250.0, 1.2e5])
    fields = lookup_volume_mixing_ratio.reconstruct_vmr_fields_from_pressure(
        lookup_vmr, pressure
    )
    self.assertTrue(fields)
    for gas, field in fields.items():
      if gas == 'o3' and lookup_volume_mixing_ratio._USE_RCEMIP_OZONE_PROFILE.value:
        continue  # Analytic profile, not the sounding.
      profile = np.asarray(lookup_vmr.profiles[gas], np.float64)
      field = np.asarray(field, np.float64)
      # Outside (and at) the ends: exactly the end values.
      np.testing.assert_allclose(field[:3], profile[0], rtol=1e-6, err_msg=gas)
      np.testing.assert_allclose(field[-2:], profile[-1], rtol=1e-6, err_msg=gas)
      # Inside: linear in log-pressure. The helper assumes the sounding's
      # levels are evenly spaced in log-pressure, which this file's are only to
      # about 1e-4, hence the looser tolerance.
      expected = np.interp(np.log(pressure), np.log(p_ref), profile)
      np.testing.assert_allclose(field, expected, rtol=1e-3, err_msg=gas)

      # Zero derivative beyond either end, finite everywhere.
      grad = jax.vmap(
          jax.grad(
              lambda p, gas=gas:
              lookup_volume_mixing_ratio.reconstruct_vmr_fields_from_pressure(
                  lookup_vmr, p[None]
              )[gas][0]
          )
      )(pressure)
      self.assertTrue(np.all(np.isfinite(grad)), gas)
      np.testing.assert_array_equal(np.asarray(grad)[[0, 1, -1]], 0.0)


if __name__ == '__main__':
  unittest.main()
