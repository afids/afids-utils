"""Methods for transforming between different coordinate systems"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def afid_world2voxel(
    afid_world: NDArray[np.float_],
    nii_affine: NDArray[np.float_],
) -> NDArray[np.int_]:
    """Transform fiducials in world coordinates to voxel coordinates"""

    # Translation
    afid_voxel = afid_world.T - nii_affine[:3, 3:4]
    # Rotation
    afid_voxel = np.dot(afid_voxel, np.linalg.inv(nii_affine[:3, :3]))

    # Return coordinates rounded to nearest voxel
    return np.rint(np.diag(afid_voxel)).astype(int)
