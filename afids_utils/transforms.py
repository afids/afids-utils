"""Methods for transforming between different coordinate systems"""
from __future__ import annotations

import attrs
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
    if not isinstance(afid_world, AfidPosition):  # type: ignore
        raise TypeError("Not an AfidPosition object")

    # Put into numpy array for easier computation
    world_pos = np.asarray([afid_world.x, afid_world.y, afid_world.z])

    # Translation, rotation, and round to nearest voxel
    voxel_pos = np.linalg.inv(nii_affine[:3, :3]).dot(  # type: ignore
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
    if not isinstance(afid_voxel, AfidVoxel):  # type: ignore
        raise TypeError("Not an AfidVoxel object")

    # Put into numpy array for easier computation
    voxel_pos = np.asarray([afid_voxel.i, afid_voxel.j, afid_voxel.k])

    # Perform rotation, translation
    world_pos = (
        nii_affine[:3, :3].dot(voxel_pos) + nii_affine[:3, 3]  # type: ignore
    )

    return AfidPosition(
        label=afid_voxel.label,
        x=world_pos[0],
        y=world_pos[1],
        z=world_pos[2],
        desc=afid_voxel.desc,
    )


def xfm_coord_system(
    afid_set: AfidSet, new_coord_system: str = "LPS"
) -> AfidSet:
    """Convert AFID set between RAS and LPS coordinates

    Parameters
    ----------
    afid_set
        Object containing valid AfidSet

    new_coord_system
        Convert AFID set to defined coordinate system (default: 'LPS')

    Returns
    -------
    AfidSet
        Object containing AFIDs stored in defined coordinate system

    Raises
    ------
    ValueError
        If invalid coordinate system
    """
    if new_coord_system not in ["RAS", "LPS"]:
        raise ValueError(
            "Unrecognized coordinate system - please select RAS or LPS"
        )

    if afid_set.coord_system == new_coord_system:
        return afid_set

    # Create copy and update AFIDs for new coordinate system
    new_afids = [
        attrs.evolve(afid, x=-afid.x, y=-afid.y) for afid in afid_set.afids
    ]
    return attrs.evolve(
        afid_set, coord_system=new_coord_system, afids=new_afids
    )
