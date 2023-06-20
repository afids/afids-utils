from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from afids_utils.tests.strategies import world_coord, affine_xfm
from afids_utils.transforms import afid_world2voxel


class TestAfidWorld2Voxel:
    @given(world_coord=world_coord(), nii_affine=affine_xfm())
    @settings(
        deadline=400,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_transform_output(
        self, world_coord: NDArray[np.single], nii_affine: NDArray[np.single]
    ):
        voxel_coord = afid_world2voxel(world_coord, nii_affine)

        assert isinstance(voxel_coord, np.ndarray)
        assert voxel_coord.dtype == np.int_

    @given(
        world_coord=world_coord(),
        nii_affine=affine_xfm(),
        resample_size=st.floats(min_value=0.0, max_value=4.0),
    )
    @settings(
        deadline=400,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_resample_transform(
        self,
        world_coord: NDArray[np.single],
        nii_affine: NDArray[np.single],
        resample_size: float,
    ):
        voxel_coord = afid_world2voxel(
            world_coord,
            nii_affine,
            resample_size=resample_size,
        )

        assert isinstance(voxel_coord, np.ndarray)
        assert voxel_coord.dtype == np.int_

    @given(
        world_coord=world_coord(),
        nii_affine=affine_xfm(),
        padding=st.integers(min_value=1, max_value=10),
    )
    @settings(
        deadline=400,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_padding_transform(
        self,
        world_coord: NDArray[np.single],
        nii_affine: NDArray[np.single],
        padding: int,
    ):
        voxel_coord = afid_world2voxel(
            world_coord,
            nii_affine,
            padding=padding,
        )
        assert isinstance(voxel_coord, np.ndarray)
        assert voxel_coord.dtype == np.int_
