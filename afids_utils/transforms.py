"""Methods for transforming between different coordinate systems"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from afids_utils.afids import AfidPosition, AfidSet, AfidVoxel


def world_to_voxel(
    afid_world: AfidPosition,
    nii_affine: NDArray[np.float_],
) -> AfidVoxel:
    """Transform fiducials from world coordinates to voxel coordinates

    Parameters
    ----------
    afid_world
        AfidPosition containing floating-point spatial coordinates (x, y, z)
        to transform

    nii_affine
        NumPy array containing affine transformation associated with
        NifTI image

    Returns
    -------
    AfidVoxel
        Object containing transformed integer voxel coordinates (i, j, k)
    """
    if not isinstance(afid_world, AfidPosition):
        raise TypeError("Not an AfidPosition object")

    # Put into numpy array for easier computation
    world_pos = np.asarray([afid_world.x, afid_world.y, afid_world.z])

    # Translation, rotation, and round to nearest voxel
    voxel_pos = np.linalg.inv(nii_affine[:3, :3]).dot(
        world_pos - nii_affine[:3, 3]
    )
    voxel_pos = np.rint(voxel_pos).astype(int)

    return AfidVoxel(
        label=afid_world.label,
        i=voxel_pos[0],
        j=voxel_pos[1],
        k=voxel_pos[2],
        desc=afid_world.desc,
    )


def voxel_to_world(
    afid_voxel: AfidVoxel,
    nii_affine: NDArray[np.float_],
) -> AfidPosition:
    """Transform fiducials from world coordinates to voxel coordinates

    Parameters
    ----------
    afid_voxel
        AfidVoxel containing integer voxel coordinates (i, j, k)

    nii_affine
        NumPy array containing affine transformation associated with
        NifTI image

    Returns
    -------
    AfidPosition
        Object containing approximate floating-point spatial coordinates
        (x, y, z)
    """
    if not isinstance(afid_voxel, AfidVoxel):
        raise TypeError("Not an AfidVoxel object")

    # Put into numpy array for easier computation
    voxel_pos = np.asarray([afid_voxel.i, afid_voxel.j, afid_voxel.k])

    # Perform rotation, translation
    world_pos = nii_affine[:3, :3].dot(voxel_pos) + nii_affine[:3, 3]

    return AfidPosition(
        label=afid_voxel.label,
        x=world_pos[0],
        y=world_pos[1],
        z=world_pos[2],
        desc=afid_voxel.desc,
    )


def coord_system_xfm(
    afid_set: AfidSet, new_coord_system: str = "RAS"
) -> AfidSet:
    """Convert AFID set between LPS and RAS coordinates

    Parameters
    ----------
    afid_set
        Object containing valid AfidSet

    new_coord_sys
        Convert AFID set to defined coordinate system (default: 'RAS')

    Returns
    -------
    AfidSet
        Object containing AFIDs stored in defined coordinate system

    Raises
    ------
    ValueError
        If invalid coordinate system, or if already in defined
        cordinate system
    """
    if new_coord_system not in ["RAS", "LPS"]:
        raise ValueError(
            "Unrecognized coordinate system - please select RAS or LPS"
        )

    if afid_set.coord_system == new_coord_system:
        raise ValueError(f"Already saved in {new_coord_system}")

    # Create copy and update coordinate system
    new_afid_set = deepcopy(afid_set)
    new_afid_set.coord_system = new_coord_system

    # Update afid positions
    for idx in range(len(new_afid_set.afids)):
        new_afid_set.afids[idx].x = -new_afid_set.afids[idx].x
        new_afid_set.afids[idx].y = -new_afid_set.afids[idx].y

    return new_afid_set
