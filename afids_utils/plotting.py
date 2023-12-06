"""Methods for plotting anatomical fiducials"""
from __future__ import annotations

from importlib import resources

import nibabel as nib
import nilearn.plotting as niplot
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from nilearn.plotting.displays._projectors import LYRZProjector
from numpy.typing import NDArray
from plotly.graph_objs._figure import Figure as goFigure

from afids_utils.afids import AfidPosition, AfidSet, AfidVoxel
from afids_utils.transforms import world_to_voxel

# Matplotlib colormap object with 32-discrete colors
COLORS: list[str] = [
    "#FF0000",  # Red
    "#00FF00",  # Green
    "#0000FF",  # Blue
    "#00FFFF",  # Cyan
    "#FF00FF",  # Magenta
    "#FFFF00",  # Yellow
    "#FFA500",  # Orange
    "#800080",  # Purple
    "#008080",  # Teal
    "#00FF00",  # Lime
    "#A52A2A",  # Brown
    "#FFC0CB",  # Pink
    "#E6E6FA",  # Lavender
    "#40E0D0",  # Turquoise
    "#4B0082",  # Indigo
    "#FA8072",  # Salmon
    "#808000",  # Olive
    "#708090",  # Slate
    "#FFD700",  # Gold
    "#DDA0DD",  # Plum
    "#87CEEB",  # Sky Blue
    "#228B22",  # Forest Green
    "#FF6347",  # Tomato
    "#008B8B",  # Dark Cyan
    "#BA55D3",  # Medium Orchid
    "#FF8C00",  # Dark Orange
    "#F08080",  # Light Coral
    "#98FB98",  # Pale Green
    "#87CEFA",  # Light Sky Blue
    "#556B2F",  # Dark Olive
    "#FFB6C1",  # Light Pink
    "#9932CC",  # Dark Orchid
]
CMAP = LinearSegmentedColormap.from_list(  # pyright: ignore
    name="afids_cmap", colors=COLORS, N=len(COLORS)
)

SCATTER_DICT = {
    "size": 4,
    "color": "rgba(0,0,0,0.9)",
    "line": {"width": 1.5, "color": "rgba(50,50,50,1.0)"},
}


def _create_afid_nii(
    afid_voxels: list[AfidVoxel], afid_nii: nib.nifti1.Nifti1Image
) -> nib.nifti1.Nifti1Image:
    """Internal function to create a nifti image based on afid coordinates

    Parameters
    ----------
    afid_voxels
        List of voxel indices (AfidVoxels) to visualize

    afid_nii
        3D nifti input image AFIDs were placed on

    Returns
    -------
    nib.nifti1.Nifti1Image
        3D nifti image object associated with afid positions
    """
    # Initialize empty image with zeros
    afid_img = np.zeros(afid_nii.shape, dtype=int)

    # Update image with label values in associatd indices
    for afid in afid_voxels:
        afid_img[afid.i, afid.j, afid.k] = afid.label

    affine: NDArray[np.float_] = afid_nii.affine  # pyright: ignore
    header: nib.nifti1.Nifti1Header = afid_nii.header

    return nib.nifti1.Nifti1Image(afid_img, affine=affine, header=header)


def _create_connectome_plot(
    afid_distances: list[float],
) -> LYRZProjector:
    """Internal function to generate a connectome plot of distances
    for a complete ``AfidSet`` collection.

    Parameters
    ----------
    afid_distances
        List of average distances either along a spatial component
        or Euclidean distance

    Returns
    -------
    LYRZProjector
        Afids overlaid on a glass connectome
    """
    # Get AFID coordinates in MNI
    with resources.open_text(
        "afids_utils.resources", "template.fcsv"
    ) as fcsv_fname:
        template_afid_set = AfidSet.load(fcsv_fname.name)
    template_coords: list[list[float]] = [
        [afid.x, afid.y, afid.z] for afid in template_afid_set.afids
    ]

    # Plot connectome
    view: LYRZProjector = niplot.plot_markers(  # pyright: ignore
        node_values=afid_distances,
        node_coords=template_coords,
        node_size=20,
        node_cmap="magma",
        node_vmin=min(0, *afid_distances),  # Set to 0 or negative value
        node_vmax=max(afid_distances),
        alpha=0.8,
        display_mode="lyrz",
    )

    return view  # pyright: ignore


def _do_binning(in_data: list[float], n_bins: int = 6) -> list[str]:
    """Internal function to manually bin list of numbers

    Parameters
    ----------
    in_data
        Values to bin

    nbins
        The number of bins to use (default: 6)

    Returns
    -------
    list[str]
        List of string describing bin limits
    """
    output: list[str] = []

    full_range = int(max(in_data)) + 1
    interval = full_range / n_bins

    for val in in_data:
        for bin_idx in range(n_bins):
            val -= interval
            if val < 0:
                out = interval * bin_idx
                break

        out_formatted = (
            f"{round(out, 2)} - {round(out + interval, 2)}"  # pyright: ignore
        )
        output.append(out_formatted)

    return output


def _create_histogram_plot(
    afid_distances: list[float],
    afid_labels: list[str] | None = None,
) -> goFigure:
    """Internal function to create a histogram figure of distances
    with plotly and optional labels.

    Parameters
    ----------
    afid_distances
        List of average distances either along a spatial component
        or Euclidean distance

    afid_labels
        List of strings denoting associated labels with distances. If none
        provided, will use integer representation in order distances are
        provided

    Returns
    -------
    go.Figure
        Figure object created with Plotly, demonstrating the histogram
        of distances
    """
    # If labels not provided, use index
    if not afid_labels:
        afid_labels = [f"{idx+1}" for idx in range(len(afid_distances))]

    # Sort distances by magnitude
    afid_distances, afid_labels = zip(  # pyright: ignore
        *sorted(zip(afid_distances, afid_labels), key=lambda x: x[0])
    )

    view: goFigure = go.Figure(  # pyright: ignore
        data=go.Bar(  # pyright: ignore
            x=_do_binning(afid_distances),
            y=[1 for _ in afid_labels],  # pyright: ignore
            text=[
                f"Label: {afid_labels[idx]} <br>"  # pyright:ignore
                f"Distance: {round(dist, 3)} mm"
                for idx, dist in enumerate(afid_distances)
            ],
            marker_color=afid_distances,
            marker_colorscale="magma",
            showlegend=False,
        )
    )
    view.update_layout(  # pyright: ignore
        autosize=True,
        coloraxis={"colorscale": "magma"},
    )

    return view  # pyright: ignore


def _create_scatter_plot(
    afid_distances: list[float],
    afid_labels: list[str] | None = None,
) -> goFigure:
    """Internal function to create a scatter figure of distances with plotly
    and optional labels.

    Parameters
    ----------
    afid_distances
        List of average distances either along a spatial component
        or Euclidean distance

    afid_labels
        List of strings denoting associated labels with distances. If none
        provided, will use integer representation in order distances are
        provided

    Returns
    -------
    go.Figure
        Figure object created with Plotly, demonstrating the distances as a
        scatter plot
    """
    # If labels not provided, use index
    if not afid_labels:
        afid_labels = [f"{idx+1}" for idx in range(len(afid_distances))]

    view: goFigure = go.Figure(  # pyright: ignore
        data=go.Scatter(  # pyright: ignore
            x=afid_labels,
            y=afid_distances,  # pyright: ignore
            text=[
                f"Label: {afid_labels[idx]} <br>"  # pyright:ignore
                f"Distance: {round(dist, 3)} mm"
                for idx, dist in enumerate(afid_distances)
            ],
            mode="markers",
            marker=dict(size=[14] * len(afid_distances), color=COLORS),
        )
    )
    view.update_xaxes(  # pyright: ignore
        title_text="AFID Label",
        tickangle=-45,
    )
    view.update_yaxes(title_text="Distance [mm]")  # pyright: ignore

    return view  # pyright: ignore


def plot_ortho(
    afids: AfidVoxel | AfidPosition | list[AfidVoxel | AfidPosition],
    afid_nii: nib.nifti1.Nifti1Image,
    opacity: float = 1,
) -> niplot.html_stat_map.StatMapView:
    """Generate interactive, html ortho view of the slices. Uses
    ``nilearn.plotting`` to generate the figures.

    The generated view can either be opened interactively or saved as a figure
    per ``nilearn.plotting`` functionality:
        * Interactive view - ``view.open_in_browser()``
        * Save - ``view.save_as_html(file_name.html)``

    Parameters
    ----------
    afids
        List of AFIDs to visualize, with assumption that they are in the
        same space as the provided nifti image. If AFIDs are provided as
        AfidPositions, transformation to voxel coordinates will be performed
        using the affine from the provided nifti image.

    afid_nii
        Input nifti image object to overlay afids on

    opacity:
        Opacity value [0 - transparent, 1 - opaque] of overlaid AFIDs

    Returns
    -------
    niplot.html_stat_map.StatMapView
        View object with fiducials overlaid on provided background nifti image
    """

    # If single position provided, set to list
    if isinstance(afids, (AfidVoxel, AfidPosition)):
        afids = [afids]  # pyright: ignore

    # If list[AfidPosition], convert to list[AfidVoxel]
    nii_affine: NDArray[np.float_] = afid_nii.affine  # pyright: ignore
    afid_voxels: list[AfidVoxel] = [
        world_to_voxel(afid_world=afid, nii_affine=nii_affine)
        if isinstance(afid, AfidPosition)
        else afid
        for afid in afids  # pyright: ignore
    ]

    # Create temporary overlay image
    afid_img = _create_afid_nii(afid_voxels=afid_voxels, afid_nii=afid_nii)

    # Create view
    view = niplot.view_img(  # pyright: ignore
        stat_map_img=afid_img,
        bg_img=afid_nii,  # pyright: ignore
        cmap=CMAP,
        symmetric_cmap=False,
        opacity=opacity,  # pyright: ignore
    )

    return view  # pyright: ignore


def plot_distance_summary(
    afid_distances: list[float],
    afid_labels: list[str] | None = None,
    plot_type: str = "connectome",
) -> LYRZProjector | goFigure:
    """Generate a summary plot of average distances for a
    complete ``AfidSet`` collection.

    Parameters
    ----------
    afid_distances
        List of average distances either along a spatial component or Euclidean
        distance

    afid_labels
        List of strings denoting associated labels with distances. Note, these
        are only used with plot type "histogram" and "scatter".

    plot_type
        Type of plot to generate - one of ["connectome", "scatter",
        "histogram"].

    Returns
    -------
    LYRZProjector | goFigure
        View object as either a connectome, scatter or histogram dependent
        on plot_type

    """
    # Make plot_type case-insensitive
    plot_type = plot_type.lower()

    # Generate connectome plot
    if plot_type == "connectome":
        view = _create_connectome_plot(afid_distances=afid_distances)
    elif plot_type == "scatter":
        view = _create_scatter_plot(
            afid_distances=afid_distances, afid_labels=afid_labels
        )
    elif plot_type == "histogram":
        view = _create_histogram_plot(
            afid_distances=afid_distances, afid_labels=afid_labels
        )
    # Throw error if invalid plot type
    else:
        raise ValueError(
            "Invalid plot type provided - choose one of 'connectome', "
            "'scatter', or 'histogram."
        )

    return view  # pyright: ignore


def plot_3d(
    afids: list[AfidPosition],
    afids_scatter_dict: dict[
        str, int | str | dict[str, float | str]
    ] = SCATTER_DICT,
    title: str = "",
) -> goFigure:
    """Generate 3D plot of AFIDs. Optionally visualize distance against
    template and/or overlay with surface mesh.

    Parameters
    ----------
    afids
        Collection of AfidPositions to visualize.

    afids_scatter_dict
        Dictionary containing parameters for modifying visualization of afid
        scatter points

    title
        Main title of figure
    """
    go_afids = go.Scatter3d(  # pyright: ignore
        x=[afid.x for afid in afids],
        y=[afid.y for afid in afids],
        z=[afid.z for afid in afids],
        showlegend=True,
        mode="markers",
        marker=afids_scatter_dict,
        hovertemplate=("%{text}<br>x: %{x:.4f}<br>y: %{y:.4f}<br>z: %{z:.4f}"),
        text=[f"<b>{afid.desc} ({afid.label})</b>" for afid in afids],
        name="Subject AFIDs",
    )

    view = go.Figure()  # pyright: ignore
    view.add_trace(go_afids)  # pyright: ignore

    view.update_layout(  # pyright: ignore
        title_text=title,
        autosize=True,
        barmode="stack",
        coloraxis={"colorscale": "Bluered"},
        legend_orientation="h",
    )

    return view  # pyright: ignore
