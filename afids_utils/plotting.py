"""Methods for plotting anatomical fiducials"""
from __future__ import annotations

import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from nilearn.plotting import view_img  # pyright: ignore
from nilearn.plotting.html_stat_map import StatMapView
from numpy.typing import NDArray

from afids_utils.afids import AfidPosition, AfidVoxel
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


def _create_afid_nii(
    afid_voxels: list[AfidVoxel], nii_img: nib.nifti1.Nifti1Image
) -> nib.nifti1.Nifti1Image:
    """Internal function to create a nifti image based on afid coordinates

    Parameters
    ----------
    afid_voxels
        List of voxel indices (AfidVoxels) to visualize

    nii_img
        3D nifti input image

    Returns
    -------
    nib.nifti1.Nifti1Image
        3D nifti image object associated with afid positions
    """
    # Initialize empty image with zeros
    afid_img = np.zeros(nii_img.shape, dtype=int)

    # Update image with label values in associatd indices
    for afid in afid_voxels:
        afid_img[afid.i, afid.j, afid.k] = afid.label

    affine: NDArray[np.float_] = nii_img.affine  # pyright: ignore
    header: nib.nifti1.Nifti1Header = nii_img.header

    return nib.nifti1.Nifti1Image(afid_img, affine=affine, header=header)


def plot_ortho(
    afids: AfidVoxel | AfidPosition | list[AfidVoxel | AfidPosition],
    nii_img: nib.nifti1.Nifti1Image,
    opacity: float = 1,
) -> StatMapView:
    """Generate interactive, html ortho view of the slices. Uses ``nilearn``
    to generate the figures.

    The generated view can either be opened interactively or saved as a figure
    per ``nilearn`` functionality:
        * Interactive view - ``view.open_in_browser()``
        * Save - ``view.save_as_html(file_name.html)``

    Parameters
    ----------
    afids
        List of AFIDs to visualize, with assumption that they are in the
        same space as the provided nifti image. If AFIDs are provided as
        AfidPositions, transformation to voxel coordinates will be performed
        using the affine from the provided nifti image.

    nii_img
        Input nifti image object to overlay afids on

    opacity:
        Opacity value [0 - transparent, 1 - opaque] of overlaid AFIDs
        (default: 0.8)

    Returns
    -------
    StatMapView
        View object with fiducials overlaid on provided background nifti image
    """

    # If single position provided, set to list
    if isinstance(afids, (AfidVoxel, AfidPosition)):
        afids = [afids]  # pyright: ignore

    # If list[AfidPosition], convert to list[AfidVoxel]
    nii_affine: NDArray[np.float_] = nii_img.affine  # pyright: ignore
    afid_voxels: list[AfidVoxel] = [
        world_to_voxel(afid_world=afid, nii_affine=nii_affine)
        if isinstance(afid, AfidPosition)
        else afid
        for afid in afids  # pyright: ignore
    ]

    # Create temporary overlay image
    afid_img = _create_afid_nii(afid_voxels=afid_voxels, nii_img=nii_img)

    # Create view
    view = view_img(  # pyright: ignore
        stat_map_img=afid_img,
        bg_img=nii_img,  # pyright: ignore
        cmap=CMAP,
        symmetric_cmap=False,
        opacity=opacity,  # pyright: ignore
    )

    return view  # pyright: ignore
