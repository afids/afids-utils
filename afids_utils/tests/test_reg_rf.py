from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from afids_utils.features import reg_rf
from afids_utils.tests.strategies import image_data_arrays


class TestIntegralVolume:
    @given(nii_img=image_data_arrays())
    def test_integral_float_volume(self, nii_img: NDArray[np.float_]):
        # Compute expected result
        expected_iv_img = np.zeros(
            (nii_img.shape[0] + 1, nii_img.shape[1] + 1, nii_img.shape[2] + 1)
        )
        expected_iv_img[1:, 1:, 1:] = nii_img.cumsum(0).cumsum(1).cumsum(2)

        # Compute integral volume and check output
        iv_img = reg_rf.integral_volume(nii_img)
        assert iv_img.all() == expected_iv_img.all()

        # Check output datatype is correct
        assert iv_img.dtype == np.float_

    @given(nii_img=image_data_arrays(float=False))
    def test_integral_int_volume(self, nii_img: NDArray[np.int_]):
        # Compute expected result
        expected_iv_img = np.zeros(
            (nii_img.shape[0] + 1, nii_img.shape[1] + 1, nii_img.shape[2] + 1)
        )
        expected_iv_img[1:, 1:, 1:] = nii_img.cumsum(0).cumsum(1).cumsum(2)

        # Compute integral volume and check output
        iv_img = reg_rf.integral_volume(nii_img)
        assert iv_img.all() == expected_iv_img.all()

        # Check output datatype is correct
        assert iv_img.dtype == np.int_


class TestSampleCoordRegion:
    @given(
        coord=arrays(
            shape=(3),
            dtype=np.int_,
            elements=st.integers(min_value=0, max_value=16),
        ),
        sampling_rate=st.integers(min_value=2, max_value=10),
        multiplier=st.integers(min_value=1, max_value=4),
    )
    def test_sample_coord_region(
        self,
        coord: NDArray[np.int_],
        sampling_rate: int,
        multiplier: int,
    ):
        sampled_region = reg_rf.sample_coord_region(
            coord=coord, sampling_rate=sampling_rate, multiplier=multiplier
        )

        # Check correct return type
        assert isinstance(sampled_region, pl.DataFrame)

        # Check outputs correctly generated
        assert np.all(
            (
                np.unique(sampled_region["x"].to_numpy())
                == range(
                    coord[0] - sampling_rate * multiplier,
                    coord[0] + sampling_rate * multiplier + 1,
                    multiplier,
                )
            )
        )
        assert np.all(
            (
                np.unique(sampled_region["y"].to_numpy())
                == range(
                    coord[1] - sampling_rate * multiplier,
                    coord[1] + sampling_rate * multiplier + 1,
                    multiplier,
                )
            )
        )
        assert np.all(
            (
                np.unique(sampled_region["z"].to_numpy())
                == range(
                    coord[2] - sampling_rate * multiplier,
                    coord[2] + sampling_rate * multiplier + 1,
                    multiplier,
                )
            )
        )
