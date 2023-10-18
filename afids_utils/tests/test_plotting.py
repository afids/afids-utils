from __future__ import annotations

from pathlib import Path

from nibabel.nifti1 import Nifti1Image
from nibabel.loadsave import load
from nilearn.plotting.html_stat_map import StatMapView

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import afids_utils.plotting as af_plot
import afids_utils.tests.helpers as af_helpers
import afids_utils.tests.strategies as af_st
from afids_utils.afids import AfidPosition, AfidVoxel


@pytest.fixture
def template_t1w() -> Nifti1Image:
    return load(
        Path(__file__).parent
        / "data"
        / "tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz"
    )


class TestCreateAfidNii:
    @given(afid_voxels=af_st.afid_voxels())
    @af_helpers.allow_function_scoped
    def test_create_nii(
        self, afid_voxels: AfidVoxel, template_t1w: Nifti1Image
    ):
        # Constrain coordinates to within template dimensions
        assume(
            0 <= afid_voxels.i <= template_t1w.shape[0] - 1
            and 0 <= afid_voxels.j <= template_t1w.shape[1] - 1
            and 0 <= afid_voxels.k <= template_t1w.shape[2] - 1
        )

        nii = af_plot._create_afid_nii(
            afid_voxels=[afid_voxels], nii_img=template_t1w
        )
        assert isinstance(nii, Nifti1Image)


class TestPlotOrtho:
    @given(afid_voxels=af_st.afid_voxels())
    @af_helpers.allow_function_scoped_deadline(time=2000)
    def test_plot_ortho_afid_voxel(
        self, afid_voxels: AfidVoxel, template_t1w: Nifti1Image
    ):
        # Constrain coordinates to within template dimensions
        assume(
            0 <= afid_voxels.i <= template_t1w.shape[0] - 1
            and 0 <= afid_voxels.j <= template_t1w.shape[1] - 1
            and 0 <= afid_voxels.k <= template_t1w.shape[2] - 1
        )

        view = af_plot.plot_ortho(
            afids=afid_voxels,
            nii_img=template_t1w,
        )
        assert isinstance(view, StatMapView)

    @given(afid_voxels=af_st.afid_voxels())
    @af_helpers.allow_function_scoped_deadline(time=2000)
    def test_plot_ortho_afid_voxels(
        self, afid_voxels: AfidVoxel, template_t1w: Nifti1Image
    ):
        # Constrain coordinates to within template dimensions
        assume(
            0 <= afid_voxels.i <= template_t1w.shape[0] - 1
            and 0 <= afid_voxels.j <= template_t1w.shape[1] - 1
            and 0 <= afid_voxels.k <= template_t1w.shape[2] - 1
        )

        afid_voxels: list[AfidVoxel] = [afid_voxels for _ in range(2)]
        view = af_plot.plot_ortho(
            afids=afid_voxels,
            nii_img=template_t1w,
        )
        assert isinstance(view, StatMapView)

    @given(afid_positions=af_st.afid_positions())
    @af_helpers.allow_function_scoped_deadline(time=2000)
    def test_plot_ortho_afid_position(
        self, afid_positions: AfidPosition, template_t1w: Nifti1Image
    ):
        # Constrain coordinates to within template dimensions
        assume(
            0 <= afid_positions.x <= template_t1w.shape[0] / 2.0
            and 0 <= afid_positions.y <= template_t1w.shape[1] / 2.0
            and 0 <= afid_positions.z <= template_t1w.shape[2] / 2.0
        )

        view = af_plot.plot_ortho(
            afids=afid_positions,
            nii_img=template_t1w,
        )
        assert isinstance(view, StatMapView)

    @given(afid_positions=af_st.afid_positions())
    @af_helpers.allow_function_scoped_deadline(time=2000)
    def test_plot_ortho_afid_positions(
        self, afid_positions: AfidPosition, template_t1w: Nifti1Image
    ):
        # Constrain coordinates to within template dimensions
        assume(
            0 <= afid_positions.x <= template_t1w.shape[0] / 2.0
            and 0 <= afid_positions.y <= template_t1w.shape[1] / 2.0
            and 0 <= afid_positions.z <= template_t1w.shape[2] / 2.0
        )

        afid_positions: list[AfidPosition] = [afid_positions for _ in range(2)]
        view = af_plot.plot_ortho(
            afids=afid_positions,
            nii_img=template_t1w,
        )
        assert isinstance(view, StatMapView)
