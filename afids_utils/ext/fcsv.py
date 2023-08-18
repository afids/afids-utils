"""Methods for handling .fcsv files associated with AFIDs"""
from __future__ import annotations

import json
import re
from importlib import resources
from itertools import islice
from os import PathLike
from typing import Dict

import polars as pl

from afids_utils.afids import AfidSet
from afids_utils.exceptions import InvalidFileError

HEADER_ROWS: int = 2
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
    fcsv_path: PathLike[str] | str, species: str = "human"
) -> AfidSet:
    """
    Read in fcsv to an AfidSet

    Parameters
    ----------
    fcsv_path
        Path to .fcsv file containing AFIDs coordinates

    species
        The species associated with the .fcsv file (default: human)

    Returns
    -------
    AfidSet
        Set of anatomical fiducials containing spatial coordinates and metadata
    """
    # Load expected mappings
    with resources.open_text(
        "afids_utils.resources", "afids_descs.json"
    ) as json_fpath:
        json.load(json_fpath)

    # Grab metadata
    slicer_version, coord_system = _get_metadata(fcsv_path)

    # Grab afids
    afids_set = AfidSet(
        slicer_version=slicer_version,
        coord_system=coord_system,
        afids_df=_get_afids(fcsv_path),
    )

    return afids_set
