from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from numpy.typing import NDArray

from afids_utils.tests.strategies import affine_xfm, world_coord
from afids_utils.transforms import afid_world2voxel


class TestAfidWorld2Voxel:
    @given(world_coord=world_coord(), nii_affine=affine_xfm())
    @settings(
        deadline=400,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_transform_output(
        self, world_coord: NDArray[np.float_], nii_affine: NDArray[np.float_]
    ):
        voxel_coord = afid_world2voxel(world_coord, nii_affine)

        assert isinstance(voxel_coord, np.ndarray)
        assert voxel_coord.dtype == np.int_
        assert voxel_coord.shape == (3,)
