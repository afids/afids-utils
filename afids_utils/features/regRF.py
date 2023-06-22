"""Core methods / functions for regRF"""
from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import NDArray


@overload
def integral_volume(nii_img: NDArray[np.int_]) -> NDArray[np.int_]:
    ...


@overload
def integral_volume(nii_img: NDArray[np.float_]) -> NDArray[np.float_]:
    ...


def integral_volume(nii_img: NDArray[np.int_ | np.float_]):
    """Compute zero-padded (resampled) volume of image"""
    iv_img = nii_img.cumsum(0).cumsum(1).cumsum(2)
    iv_zeropad = np.zeros(
        (iv_img.shape[0] + 1, iv_img.shape[1] + 1, iv_img.shape[2] + 1)
    )
    iv_zeropad[1:, 1:, 1:] = iv_img

    return iv_zeropad
