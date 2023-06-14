"""Methods for loading and saving nifti files"""
from __future__ import annotations

import csv
from importlib import resources
from os import PathLike

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from afids_utils.exceptions import InvalidFiducialNumberError

FCSV_FIELDNAMES = [
    "# columns = id",
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


def get_afid(
    fcsv_path: PathLike[str] | str, fid_num: int
) -> NDArray[np.single]:
    """Extract specific fiducial's spatial coordinates"""
    if fid_num < 1 or fid_num > 32:
        raise InvalidFiducialNumberError(fid_num)
    fcsv_df = pd.read_csv(
        fcsv_path, sep=",", header=2, usecols=FCSV_FIELDNAMES
    )

    return fcsv_df.loc[fid_num - 1, ["x", "y", "z"]].to_numpy(dtype="single")


def afids_to_fcsv(
    afid_coords: NDArray[np.single],
    fcsv_output: PathLike[str] | str,
) -> None:
    """AFIDS to Slicer-compatible .fcsv file"""
    # Read in fcsv template
    with resources.open_text(
        "afids_utils.resources", "template.fcsv"
    ) as template_fcsv_file:
        header = [template_fcsv_file.readline() for _ in range(3)]
        reader = csv.DictReader(template_fcsv_file, fieldnames=FCSV_FIELDNAMES)
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
        writer = csv.DictWriter(out_fcsv_file, fieldnames=FCSV_FIELDNAMES)
        for row in fcsv:
            writer.writerow(row)
