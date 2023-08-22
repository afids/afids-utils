"""Methods for handling .fcsv files associated with AFIDs"""
from __future__ import annotations

import csv
import json
import re
from importlib import resources
from itertools import islice
from os import PathLike
from typing import Dict

import polars as pl
from numpy.typing import NDArray

from afids_utils.afids import AfidSet
from afids_utils.exceptions import InvalidFileError

HEADER_ROWS: int = 2
FCSV_FIELDNAMES = (
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
)
FCSV_COLS: Dict[str] = {
    "x": pl.Float32,
    "y": pl.Float32,
    "z": pl.Float32,
    "label": pl.Utf8,
    "desc": pl.Utf8,
}


def _get_metadata(fcsv_path: PathLike[str] | str) -> tuple[str, str]:
    """
    Internal function to extract metadata from header of fcsv files

    Parameters
    ----------
    fcsv_path
        Path to .fcsv file containing AFIDs coordinates

    Returns
    -------
    parsed_version
        Slicer version associated with fiducial file

    parsed_coord
        Coordinate system of fiducials

    Raises
    ------
    InvalidFileError
        If header is missing or invalid from .fcsv file
    """
    try:
        with open(fcsv_path, "r") as fcsv:
            header = list(islice(fcsv, HEADER_ROWS))

        parsed_version = re.findall(r"\d+\.\d+", header[0])[0]
        parsed_coord = re.split(r"\s", header[1])[-2]
    except IndexError:
        raise InvalidFileError("Missing or invalid header in .fcsv file")

    # Set to human-understandable coordinate system
    if parsed_coord == "0":
        parsed_coord = "LPS"
    elif parsed_coord == "1":
        parsed_coord = "RAS"

    if parsed_coord not in ["LPS", "RAS"]:
        raise InvalidFileError("Invalid coordinate system in header")

    return parsed_version, parsed_coord


def _get_afids(fcsv_path: PathLike[str] | str) -> pl.DataFrame:
    """
    Internal function for converting .fcsv file to a pl.DataFrame

    Parameters
    ----------
    fcsv_path
        Path to .fcsv file containing AFID coordinates

    Returns
    -------
    pl.DataFrame
        Dataframe containing afids ids, descriptions, and coordinates
    """
    # Read in fiducials to dataframe, shortening id header
    afids_df = pl.read_csv(
        fcsv_path,
        skip_rows=HEADER_ROWS,
        columns=list(FCSV_COLS.keys()),
        new_columns=["x_mm", "y_mm", "z_mm"],
        dtypes=FCSV_COLS,
    )

    return afids_df


def load_fcsv(
    fcsv_path: PathLike[str] | str,
) -> AfidSet:
    """
    Read in fcsv to an AfidSet

    Parameters
    ----------
    fcsv_path
        Path to .fcsv file containing AFIDs coordinates

    Returns
    -------
    AfidSet
        Set of anatomical fiducials containing spatial coordinates and metadata
    """
    # Grab metadata
    slicer_version, coord_system = _get_metadata(fcsv_path)

    # Grab afids
    afids_set = AfidSet(
        slicer_version=slicer_version,
        coord_system=coord_system,
        afids_df=_get_afids(fcsv_path),
    )

    return afids_set


def save_fcsv(
    afid_coords: NDArray[np.single],
    out_fcsv: PathLike[str] | str,
) -> None:
    """
    Save fiducials to output fcsv file

    Parameters
    ----------
    afid_coords
        Floating-point NumPy array containing spatial coordinates (x, y, z)

    out_fcsv
        Path of fcsv file to save AFIDs to

    """
    # Read in fcsv template
    with resources.open_text(
        "afids_utils.resources", "template.fcsv"
    ) as template_fcsv_file:
        header = [template_fcsv_file.readline() for _ in range(HEADER_ROWS+1)]
        reader = csv.DictReader(template_fcsv_file, fieldnames=FCSV_FIELDNAMES)
        fcsv = list(reader)

    # Check to make sure shape of AFIDs array matches expected template
    if afid_coords.shape[0] != len(fcsv):
        raise TypeError(
            f"Expected {len(fcsv)} AFIDs, but received {afid_coords.shape[0]}"
        )
    if afid_coords.shape[1] != 3:
        raise TypeError(
            "Expected 3 spatial dimensions (x, y, z),"
            f"but received {afid_coords.shape[1]}"
        )

    # Loop over fiducials and update with fiducial spatial coordinates
    for idx, row in enumerate(fcsv):
        row["x"] = afid_coords[idx][0]
        row["y"] = afid_coords[idx][1]
        row["z"] = afid_coords[idx][2]

    # Write output fcsv
    with open(out_fcsv, "w", encoding="utf-8", newline="") as out_fcsv_file:
        for line in header:
            out_fcsv_file.write(line)
        writer = csv.DictWriter(out_fcsv_file, fieldnames=FCSV_FIELDNAMES)

        for row in fcsv:
            writer.writerow(row)
            