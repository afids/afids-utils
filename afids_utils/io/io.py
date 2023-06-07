"""Methods for loading and saving nifti files"""
from __future__ import annotations

import csv
from os import PathLike

import nibabel as nib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from afids_utils.exceptions import InvalidFiducialNumberError

AFIDS_FIELDNAMES = [
    "id",
    "x",
    "y",
    "z",
    "ow",
    "ox",
    "oy",
    "oz",
    "vis",
    "sel",
    "lock",
    "label",
    "desc",
    "associatedNodeID",
]


def load_nii(
    nii_path: PathLike[str] | str,
) -> tuple[NDArray[np.single], NDArray[np.single]]:
    """Load and normalize nifti data and scanner-to-world transform"""
    # Load unnormalized nifti volume and transform
    nii = nib.loadsave.load(nii_path)
    nii_affine = nii.affine.astype(np.single)
    nii_data = nii.get_fdata().astype(np.single)
    # Normalize nifti volume
    nii_data = (nii_data - nii_data.min()) / (nii_data.max() - nii_data.min())

    return nii_affine, nii_data


def get_afid(
    fcsv_path: PathLike[str] | str, fid_num: int
) -> NDArray[np.single]:
    """Extract specific fiducial's spatial coordinates"""
    if fid_num < 1 or fid_num > 32:
        raise InvalidFiducialNumberError(fid_num)
    fcsv_df = pd.read_csv(fcsv_path, sep=",", header=2)

    return fcsv_df.loc[fid_num - 1, ["x", "y", "z"]].to_numpy(dtype="single")


def afids_to_fcsv(
    afid_coords: NDArray[np.single],
    fcsv_template: PathLike[str] | str,
    fcsv_output: PathLike[str] | str,
) -> None:
    """AFIDS to Slicer-compatible .fcsv file"""
    # Read in fcsv template
    with open(
        fcsv_template, "r", encoding="utf-8", newline=""
    ) as template_fcsv_file:
        header = [template_fcsv_file.readline() for _ in range(3)]
        reader = csv.DictReader(
            template_fcsv_file, fieldnames=AFIDS_FIELDNAMES
        )
        fcsv = list(reader)

    # Loop over fiducials and update with fiducial spatial coordinates
    for idx, row in enumerate(fcsv):
        row["x"] = afid_coords[idx][0]
        row["y"] = afid_coords[idx][1]
        row["z"] = afid_coords[idx][2]

    # Write output fcsv
    with open(fcsv_output, "w", encoding="utf-8", newline="") as out_fcsv_file:
        for line in header:
            out_fcsv_file.write(line)
        writer = csv.DictWriter(out_fcsv_file, fieldnames=AFIDS_FIELDNAMES)
        for row in fcsv:
            writer.writerow(row)
