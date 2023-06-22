from __future__ import annotations

import numpy as np
from hypothesis import given
from numpy.typing import NDArray

from afids_utils.features import regRF
from afids_utils.tests.strategies import nii_img


class TestIntegralVolume:
    @given(nii_img=nii_img())
    def test_integral_volume(self, nii_img: NDArray[np.float_ | np.int_]):
        # Compute expected result
        expected_iv_img = np.zeros(
            (nii_img.shape[0] + 1, nii_img.shape[1] + 1, nii_img.shape[2] + 1)
        )
        expected_iv_img[1:, 1:, 1:] = nii_img.cumsum(0).cumsum(1).cumsum(2)

        # Compute integral volume and check output
        iv_img = regRF.integral_volume(nii_img)
        assert iv_img.all() == expected_iv_img.all()

        # Check output datatype is correct
        if nii_img.dtype == np.float_:
            assert iv_img.dtype == np.float_
        else:
            assert iv_img.dtype == np.int_
