from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from numpy.typing import NDArray

from afids_utils.afids import AfidPosition, AfidVoxel
from afids_utils.tests.strategies import (
    affine_xfms,
    voxel_coords,
    world_coords,
)
from afids_utils.transforms import voxel_to_world, world_to_voxel


class TestAfidWorld2Voxel:
    @given(world_coord=world_coords(), nii_affine=affine_xfms())
    @settings(
        deadline=400,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_world_to_voxel_xfm(
        self, world_coord: AfidPosition, nii_affine: NDArray[np.float_]
    ):
        voxel_coord = world_to_voxel(world_coord, nii_affine)

        assert isinstance(voxel_coord, AfidVoxel)
        # Have to assert specific int dtype
        assert isinstance(voxel_coord.i, np.int64)
        assert isinstance(voxel_coord.j, np.int64)
        assert isinstance(voxel_coord.k, np.int64)

    @given(voxel_coord=voxel_coords(), nii_affine=affine_xfms())
    @settings(
        deadline=400,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_world_type(
        self, voxel_coord: AfidVoxel, nii_affine: NDArray[np.float_]
    ):
        with pytest.raises(TypeError, match="Not an AfidPosition.*"):
            world_to_voxel(voxel_coord, nii_affine)


class TestAfidVoxel2World:
    @given(voxel_coord=voxel_coords(), nii_affine=affine_xfms())
    @settings(
        deadline=400,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_voxel_to_world_xfm(
        self, voxel_coord: AfidVoxel, nii_affine: NDArray[np.float_]
    ):
        world_coord = voxel_to_world(voxel_coord, nii_affine)

        assert isinstance(world_coord, AfidPosition)
        # Have to assert specific float dtype
        assert isinstance(world_coord.x, np.float64)
        assert isinstance(world_coord.y, np.float64)
        assert isinstance(world_coord.z, np.float64)

    @given(world_coord=world_coords(), nii_affine=affine_xfms())
    @settings(
        deadline=400,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_voxel_type(
        self, world_coord: AfidPosition, nii_affine: NDArray[np.float_]
    ):
        with pytest.raises(TypeError, match="Not an AfidVoxel.*"):
            voxel_to_world(world_coord, nii_affine)
