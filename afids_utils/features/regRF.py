"""Core methods / functions for regRF"""
from __future__ import annotations

import itertools as it
from typing import overload

import numpy as np
import polars as pl
from numpy.typing import NDArray


@overload
def integral_volume(nii_img: NDArray[np.int_]) -> NDArray[np.int_]:
    ...  # pragma: no cover


@overload
def integral_volume(nii_img: NDArray[np.float_]) -> NDArray[np.float_]:
    ...  # pragma: no cover


def integral_volume(nii_img: NDArray[np.int_ | np.float_]):
    """Compute zero-padded (resampled) volume of image"""
    iv_img = nii_img.cumsum(0).cumsum(1).cumsum(2)
    iv_zeropad = np.zeros(
        (iv_img.shape[0] + 1, iv_img.shape[1] + 1, iv_img.shape[2] + 1),
        dtype=iv_img.dtype,
    )
    iv_zeropad[1:, 1:, 1:] = iv_img

    return iv_zeropad


def sample_coord_region(
    coord: NDArray[np.int_], sampling_rate: int, multiplier: int = 1
) -> pl.DataFrame:
    return pl.DataFrame(
        list(
            it.product(
                *[
                    range(
                        coord[0] - sampling_rate * multiplier,
                        coord[0] + sampling_rate * multiplier + 1,
                        multiplier,
                    ),
                    range(
                        coord[1] - sampling_rate * multiplier,
                        coord[1] + sampling_rate * multiplier + 1,
                        multiplier,
                    ),
                    range(
                        coord[2] - sampling_rate * multiplier,
                        coord[2] + sampling_rate * multiplier + 1,
                        multiplier,
                    ),
                ]
            )
        ),
        orient="row",
        schema=["x", "y", "z"],
    )
