from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from numpy.typing import NDArray

import afids_utils.tests.strategies as af_st
import afids_utils.transforms as af_xfm
from afids_utils.afids import AfidPosition, AfidSet, AfidVoxel


class TestAfidWorld2Voxel:
    @given(
        afid_position=af_st.afid_positions(), nii_affine=af_st.affine_xfms()
    )
    @settings(
        deadline=400,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_world_to_voxel_xfm(
        self, afid_position: AfidPosition, nii_affine: NDArray[np.float_]
    ):
        afid_voxel = af_xfm.world_to_voxel(afid_position, nii_affine)

        assert isinstance(afid_voxel, AfidVoxel)
        # Have to assert specific int dtype
        assert isinstance(afid_voxel.i, np.int64)
        assert isinstance(afid_voxel.j, np.int64)
        assert isinstance(afid_voxel.k, np.int64)

    @given(afid_voxel=af_st.afid_voxels(), nii_affine=af_st.affine_xfms())
    @settings(
        deadline=400,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_world_type(
        self, afid_voxel: AfidVoxel, nii_affine: NDArray[np.float_]
    ):
        with pytest.raises(TypeError, match="Not an AfidPosition.*"):
            af_xfm.world_to_voxel(afid_voxel, nii_affine)


class TestAfidVoxel2World:
    @given(afid_voxel=af_st.afid_voxels(), nii_affine=af_st.affine_xfms())
    @settings(
        deadline=400,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_voxel_to_world_xfm(
        self, afid_voxel: AfidVoxel, nii_affine: NDArray[np.float_]
    ):
        afid_position = af_xfm.voxel_to_world(afid_voxel, nii_affine)

        assert isinstance(afid_position, AfidPosition)
        # Have to assert specific float dtype
        assert isinstance(afid_position.x, np.float64)
        assert isinstance(afid_position.y, np.float64)
        assert isinstance(afid_position.z, np.float64)

    @given(
        afid_position=af_st.afid_positions(), nii_affine=af_st.affine_xfms()
    )
    @settings(
        deadline=400,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_voxel_type(
        self, afid_position: AfidPosition, nii_affine: NDArray[np.float_]
    ):
        with pytest.raises(TypeError, match="Not an AfidVoxel.*"):
            af_xfm.voxel_to_world(afid_position, nii_affine)


class TestAfidRoundTripConvert:
    @given(
        afid_position=af_st.afid_positions(), nii_affine=af_st.affine_xfms()
    )
    @settings(
        deadline=400,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_round_trip_world(
        self, afid_position: AfidPosition, nii_affine: NDArray[np.float_]
    ):
        afid_voxel = af_xfm.world_to_voxel(afid_position, nii_affine)
        afid_position_approx = af_xfm.voxel_to_world(afid_voxel, nii_affine)
        print(afid_position, afid_position_approx, nii_affine)

        # Check to see if round-trip approximates to within 10mm
        # Very loose approx, due to lack of imposed constraints
        assert afid_position_approx.x == pytest.approx(afid_position.x, abs=10)
        assert afid_position_approx.y == pytest.approx(afid_position.y, abs=10)
        assert afid_position_approx.z == pytest.approx(afid_position.z, abs=10)

    @given(afid_voxel=af_st.afid_voxels(), nii_affine=af_st.affine_xfms())
    @settings(
        deadline=400,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_round_trip_voxel(
        self, afid_voxel: AfidVoxel, nii_affine: NDArray[np.float_]
    ):
        afid_world = af_xfm.voxel_to_world(afid_voxel, nii_affine)
        afid_voxel_approx = af_xfm.world_to_voxel(afid_world, nii_affine)

        # Check to see if round-trip approximates to within 2 voxels
        assert afid_voxel_approx.i == pytest.approx(afid_voxel.i, abs=2)
        assert afid_voxel_approx.j == pytest.approx(afid_voxel.j, abs=2)
        assert afid_voxel_approx.k == pytest.approx(afid_voxel.k, abs=2)


class TestCoordSystemXfm:
    @given(afid_set=af_st.afid_sets())
    @settings(
        deadline=400,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_new_coord_system(self, afid_set: AfidSet):
        with pytest.raises(
            ValueError, match=r"Unrecognized coordinate system.*"
        ):
            af_xfm.coord_system_xfm(afid_set, new_coord_system="invalid")

    @given(afid_set=af_st.afid_sets(randomize_header=False))
    @settings(
        deadline=400,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_same_coord_system(self, afid_set: AfidSet):
        with pytest.raises(ValueError, match=r"Already saved in.*"):
            af_xfm.coord_system_xfm(afid_set, new_coord_system="LPS")

    @given(afid_set=af_st.afid_sets(randomize_header=False))
    @settings(
        deadline=400,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_valid_new_coord_system(self, afid_set):
        new_afid_set = af_xfm.coord_system_xfm(afid_set)

        assert isinstance(new_afid_set, AfidSet)
        assert new_afid_set.coord_system == "RAS"

        for idx in range(len(new_afid_set.afids)):
            assert (
                new_afid_set.afids[idx].x,
                new_afid_set.afids[idx].y,
                new_afid_set.afids[idx].z,
            ) == (
                -afid_set.afids[idx].x,
                -afid_set.afids[idx].y,
                afid_set.afids[idx].z,
            )
