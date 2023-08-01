"""Methods for loading and saving files associated with AFIDs"""
from __future__ import annotations

import csv
from importlib import resources
from os import PathLike

import numpy as np
import polars as pl
from numpy.typing import NDArray

from afids_utils.exceptions import InvalidFiducialNumberError

FCSV_FIELDNAMES = {
    "# columns = id": pl.Utf8,
    "x": pl.Float32,
    "y": pl.Float32,
    "z": pl.Float32,
    "ow": pl.UInt8,
    "ox": pl.UInt8,
    "oy": pl.UInt8,
    "oz": pl.UInt8,
    "vis": pl.UInt8,
    "sel": pl.UInt8,
    "lock": pl.UInt8,
    "label": pl.UInt8,
    "desc": pl.Utf8,
    "associatedNodeID": pl.Utf8,
}


def get_afid(
    fcsv_path: PathLike[str] | str, fid_num: int
) -> NDArray[np.single]:
    """
    Extract specific fiducial's spatial coordinates

    Parameters
    ----------
    fcsv_path : os.PathLike[str] | str
        Path to .fcsv file to extract AFID coordinates from

    fid_num : int
        Unique fiducial number to extract from .fcsv file

    Returns
    -------
    numpy.ndarray[shape=(3,), dtype=numpy.single]
        NumPy array containing spatial coordinates (x, y, z) of single AFID
        coordinate
    """
    if fid_num < 1 or fid_num > 32:
        raise InvalidFiducialNumberError(fid_num)
    fcsv_df = pl.scan_csv(
        fcsv_path, separator=",", skip_rows=2, dtypes=FCSV_FIELDNAMES
    )

    return (
        fcsv_df.filter(pl.col("label") == fid_num)
        .select("x", "y", "z")
        .collect()
        .to_numpy()[0]
    )


def afids_to_fcsv(
    afid_coords: NDArray[np.single],
    fcsv_output: PathLike[str] | str,
) -> None:
    """
    Save AFIDS to Slicer-compatible .fcsv file

    Parameters
    ----------
    afid_coords : numpy.ndarray[shape=(N, 3), dtype=numpy.single]
        Floating-point NumPy array containing spatial coordinates (x, y, z) of
        `N` AFIDs

    fcsv_output : os.PathLike[str] | str
        Path of file (including filename and extension) to save AFIDs to

    """
    # Read in fcsv template
    with resources.open_text(
        "afids_utils.resources", "template.fcsv"
    ) as template_fcsv_file:
        header = [template_fcsv_file.readline() for _ in range(3)]
        reader = csv.DictReader(
            template_fcsv_file, fieldnames=list(FCSV_FIELDNAMES.keys())
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
        writer = csv.DictWriter(
            out_fcsv_file, fieldnames=list(FCSV_FIELDNAMES.keys())
        )
        for row in fcsv:
            writer.writerow(row)
