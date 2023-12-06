from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest
from hypothesis import assume, given
from nibabel.loadsave import load  # pyright: ignore
from nibabel.nifti1 import Nifti1Image
from nilearn.plotting.displays._projectors import LYRZProjector
from nilearn.plotting.html_stat_map import StatMapView
from plotly.graph_objs._figure import Figure as goFigure

import afids_utils.plotting as af_plot
import afids_utils.tests.helpers as af_helpers
import afids_utils.tests.strategies as af_st
from afids_utils.afids import AfidPosition, AfidVoxel


@pytest.fixture
def human_mappings() -> list[dict[str, str]]:
    with resources.open_text(
        "afids_utils.resources", "afids_descs.json"
    ) as json_fpath:
        mappings = json.load(json_fpath)

    return mappings["human"]


@pytest.fixture
def template_t1w() -> Nifti1Image:
    return load(  # pyright: ignore
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
            afid_voxels=[afid_voxels], afid_nii=template_t1w
        )
        assert isinstance(nii, Nifti1Image)


class TestPlotOrtho:
    @given(afid_voxels=af_st.afid_voxels())
    @af_helpers.allow_function_scoped_deadline(time=None)
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
            afid_nii=template_t1w,
        )
        assert isinstance(view, StatMapView)
        del view

    @given(afid_voxels=af_st.afid_voxels())
    @af_helpers.allow_function_scoped_deadline(time=None)
    def test_plot_ortho_afid_voxels(
        self,
        afid_voxels: AfidVoxel,  # pyright: ignore
        template_t1w: Nifti1Image,
    ):
        # Constrain coordinates to within template dimensions
        assume(
            0 <= afid_voxels.i <= template_t1w.shape[0] - 1
            and 0 <= afid_voxels.j <= template_t1w.shape[1] - 1
            and 0 <= afid_voxels.k <= template_t1w.shape[2] - 1
        )

        afid_voxels: list[AfidVoxel | AfidPosition] = [
            afid_voxels for _ in range(2)
        ]
        view = af_plot.plot_ortho(
            afids=afid_voxels,
            afid_nii=template_t1w,
        )
        assert isinstance(view, StatMapView)
        del view

    @given(afid_positions=af_st.afid_positions())
    @af_helpers.allow_function_scoped_deadline(time=None)
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
            afid_nii=template_t1w,
        )
        assert isinstance(view, StatMapView)
        del view

    @given(afid_positions=af_st.afid_positions())
    @af_helpers.allow_function_scoped_deadline(time=None)
    def test_plot_ortho_afid_positions(
        self,
        afid_positions: AfidPosition,  # pyright: ignore
        template_t1w: Nifti1Image,
    ):
        # Constrain coordinates to within template dimensions
        assume(
            0 <= afid_positions.x <= template_t1w.shape[0] / 2.0
            and 0 <= afid_positions.y <= template_t1w.shape[1] / 2.0
            and 0 <= afid_positions.z <= template_t1w.shape[2] / 2.0
        )

        afid_positions: list[AfidVoxel | AfidPosition] = [
            afid_positions for _ in range(2)
        ]
        view = af_plot.plot_ortho(
            afids=afid_positions,
            afid_nii=template_t1w,
        )
        assert isinstance(view, StatMapView)
        del view


class TestPlotConnectome:
    @given(afid_distances=af_st.afid_distances())
    @af_helpers.deadline(time=None)
    def test_create_connectome_plot(self, afid_distances: list[float]):
        view = af_plot._create_connectome_plot(afid_distances=afid_distances)
        assert view is not None
        assert isinstance(view, LYRZProjector)
        view.close()  # pyright: ignore


class TestPlotHistogram:
    @given(afid_distances=af_st.afid_distances())
    def test_do_binning(self, afid_distances: list[float]):
        bin_strs = af_plot._do_binning(in_data=afid_distances)
        assert all(isinstance(bin_str, str) for bin_str in bin_strs)

    @given(afid_distances=af_st.afid_distances())
    def test_create_histogram_plot_no_labels(
        self, afid_distances: list[float]
    ):
        view = af_plot._create_histogram_plot(afid_distances=afid_distances)
        assert view is not None
        assert isinstance(view, goFigure)
        del view

    @given(afid_distances=af_st.afid_distances())
    @af_helpers.allow_function_scoped
    def test_create_histogram_plot_labels(
        self, afid_distances: list[float], human_mappings: list[dict[str, str]]
    ):
        afid_labels: list[str] = [
            human_mappings[idx]["desc"] for idx in range(len(afid_distances))
        ]
        view = af_plot._create_histogram_plot(
            afid_distances=afid_distances, afid_labels=afid_labels
        )
        assert view is not None
        assert isinstance(view, goFigure)
        del view


class TestPlotScatter:
    @given(afid_distances=af_st.afid_distances())
    def test_create_scatter_plot_no_labels(self, afid_distances: list[float]):
        view = af_plot._create_scatter_plot(afid_distances=afid_distances)
        assert view is not None
        assert isinstance(view, goFigure)
        del view

    @given(afid_distances=af_st.afid_distances())
    @af_helpers.allow_function_scoped
    def test_create_scatter_plot_labels(
        self, afid_distances: list[float], human_mappings: list[dict[str, str]]
    ):
        afid_labels: list[str] = [
            human_mappings[idx]["desc"] for idx in range(len(afid_distances))
        ]
        view = af_plot._create_scatter_plot(
            afid_distances=afid_distances, afid_labels=afid_labels
        )
        assert view is not None
        assert isinstance(view, goFigure)
        del view


class TestPlotDistanceSummary:
    @given(
        afid_distances=af_st.afid_distances(),
        plot_type=af_st.short_ascii_text(),
    )
    def test_invalid_plot_type(
        self, afid_distances: list[float], plot_type: str
    ):
        with pytest.raises(ValueError, match="Invalid plot type provided*"):
            af_plot.plot_distance_summary(
                afid_distances=afid_distances, plot_type=plot_type
            )

    @given(afid_distances=af_st.afid_distances())
    @af_helpers.deadline(time=None)
    @pytest.mark.parametrize(
        "plot_type, view_type",
        [
            ("connectome", LYRZProjector),
            ("scatter", goFigure),
            ("histogram", goFigure),
        ],
    )
    def test_plot_summary(
        self,
        afid_distances: list[float],
        plot_type: str,
        view_type: type,
    ):
        view = af_plot.plot_distance_summary(
            afid_distances=afid_distances,
            afid_labels=None,
            plot_type=plot_type,
        )
        assert isinstance(view, view_type)

        # Remove figure to avoid having too many open (memory considerations)
        if plot_type == "connectome":
            view.close()  # pyright: ignore
        else:
            del view


class TestPlot3D:
    @given(afid_positions=af_st.position_lists())
    def test_plot_scatter3d(self, afid_positions: list[AfidPosition]):
        view = af_plot.plot_3d(afids=afid_positions)
        assert isinstance(view, goFigure)

        del view

    @given(afid_positions=af_st.position_lists())
    def test_plot_scatter3d_dict(self, afid_positions: list[AfidPosition]):
        # Static dict for testing
        scatter_dict = {
            "size": 8,
            "color": "rgba(125, 125, 125, 0.5)",
            "line": {"width": 3.0, "color": "rgba(0,0,0,1.)"},
        }
        view = af_plot.plot_3d(
            afids=afid_positions, afids_scatter_dict=scatter_dict
        )
        assert isinstance(view, goFigure)

        del view

    @given(afid_positions=af_st.position_lists())
    def test_plot_scatter3d_title(self, afid_positions: list[AfidPosition]):
        # Static title for testing
        title = "Test Title"
        view = af_plot.plot_3d(afids=afid_positions, title=title)
        assert isinstance(view, goFigure)

        del view
