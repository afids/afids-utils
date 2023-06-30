"""Methods for transforming between different coordinate systems"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def afid_world2voxel(
    afid_world: NDArray[np.float_],
    nii_affine: NDArray[np.float_],
) -> NDArray[np.int_]:
    """
    Transform fiducials from world coordinates to voxel coordinates

    Parameters
    ----------
    afid_world : numpy.ndarray[shape=(3,), dtype=numpy.float_]
        NumPy array containing floating-point spatial coordinates (x, y, z) to
        transform

    nii_affine : numpy.ndarray[shape=(4, 4), dtype=numpy.float_]
        NumPy array containing affine transformation associated with
        NifTI image

    Returns
    -------
    numpy.ndarray[shape=(3,), dtype=numpy.float_]
        NumPy array containing indices corresponding to voxel location along
        spatial dimensions
    """

    # Translation
    afid_voxel = afid_world.T - nii_affine[:3, 3:4]
    # Rotation
    afid_voxel = np.dot(afid_voxel, np.linalg.inv(nii_affine[:3, :3]))

    # Return coordinates rounded to nearest voxel
    return np.rint(np.diag(afid_voxel)).astype(int)
