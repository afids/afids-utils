"""Methods for transforming between different coordinate systems"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def afid_world2voxel(
    afid_world: NDArray[np.float_],
    nii_affine: NDArray[np.float_],
    resample_size: float = 1,
    padding: Optional[int] = None,
) -> NDArray[np.int_]:
    """Transform fiducials in world coordinates to voxel coordinates

    Optionally, resample to match resampled image
    """

    # Translation
    afid_voxel = afid_world.T - nii_affine[:3, 3:4]
    # Rotation
    afid_voxel = np.dot(afid_voxel, np.linalg.inv(nii_affine[:3, :3]))

    # Round to nearest voxel
    afid_voxel = np.rint(np.diag(afid_voxel) * resample_size)

    if padding:
        afid_voxel = np.pad(afid_voxel, pad_width=padding, mode="constant")

    return afid_voxel.astype(int)
