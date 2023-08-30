"""Methods for handling .fcsv files associated with AFIDs"""
from __future__ import annotations

import csv
import re
from importlib import resources
from os import PathLike

from afids_utils.afids import AfidPosition, AfidSet
from afids_utils.exceptions import InvalidFileError

HEADER_ROWS: int = 2
FCSV_FIELDNAMES: tuple[str] = (
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


def _get_metadata(in_fcsv: list[str]) -> tuple[str, str]:
    """
    Internal function to extract metadata from header of fcsv files

    Parameters
    ----------
    in_fcsv
        Data from provided fcsv file to parse metadata from

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
        header = in_fcsv[: HEADER_ROWS + 1]

        # Parse version and coordinate system
        parsed_version = re.findall(r"\d+\.\d+", header[0])[0]
        parsed_coord = re.split(r"\s", header[1])[-2]

    except IndexError:
        raise InvalidFileError("Missing or invalid header in .fcsv file")

    # Transform coordinate system so human-understandable
    if parsed_coord == "0":
        parsed_coord = "LPS"
    elif parsed_coord == "1":
        parsed_coord = "RAS"

    if parsed_coord not in ["LPS", "RAS"]:
        raise InvalidFileError("Invalid coordinate system in header")

    return parsed_version, parsed_coord


def _get_afids(in_fcsv: list[str]) -> list[AfidPosition]:
    """
    Internal function for converting .fcsv file to a pl.DataFrame

    Parameters
    ----------
    in_fcsv
        Data from provided fcsv file to parse metadata from

    Returns
    -------
    afid_positions
        List containing spatial position of afids
    """
    # Read in AFIDs from fcsv (set to start from 1 to skip header fields)
    afids = in_fcsv[HEADER_ROWS + 1 :]

    # Add to list of AfidPosition
    afids_positions = []
    for afid in afids:
        afid = afid.split(",")
        afids_positions.append(
            AfidPosition(
                label=int(afid[-3]),
                x=float(afid[1]),
                y=float(afid[2]),
                z=float(afid[3]),
                desc=afid[-2],
            )
        )

    return afids_positions


def load_fcsv(
    fcsv_path: PathLike[str] | str,
) -> tuple[str, str, list[AfidPosition]]:
    """
    Read in fcsv to an AfidSet

    Parameters
    ----------
    fcsv_path
        Path to .fcsv file containing AFIDs coordinates

    Returns
    -------
    slicer_version
        Slicer version associated with fiducial file

    coord_system
        Coordinate system of fiducials

    afids_positions
        List containing spatial position of afids
    """
    with open(fcsv_path) as in_fcsv_fpath:
        in_fcsv = in_fcsv_fpath.readlines()

    # Grab metadata
    slicer_version, coord_system = _get_metadata(in_fcsv)
    # Grab afids
    afids_positions = _get_afids(in_fcsv)

    return slicer_version, coord_system, afids_positions


def save_fcsv(
    afid_set: AfidSet,
    out_fcsv: PathLike[str] | str,
) -> None:
    """
    Save fiducials to output fcsv file

    Parameters
    ----------
    afid_set
        A complete AfidSet containing metadata and positions of AFIDs

    out_fcsv
        Path of fcsv file to save AFIDs to

    Raises
    ------
    TypeError
        If number of fiducials to write does not match expected number
    """
    # Read in fcsv template
    with resources.open_text(
        "afids_utils.resources", "template.fcsv"
    ) as template_fcsv_file:
        header = [
            template_fcsv_file.readline() for _ in range(HEADER_ROWS + 1)
        ]
        reader = csv.DictReader(template_fcsv_file, fieldnames=FCSV_FIELDNAMES)
        fcsv = list(reader)

    # Update header coordinate system
    header[1] = f"# CoordinateSystem = {afid_set.coord_system}\n"

    # Loop over fiducials and update with fiducial spatial coordinates
    for idx, row in enumerate(fcsv):
        row["x"] = afid_set.afids[idx].x
        row["y"] = afid_set.afids[idx].y
        row["z"] = afid_set.afids[idx].z

    # Write output fcsv
    with open(out_fcsv, "w", encoding="utf-8", newline="") as out_fcsv_file:
        for line in header:
            out_fcsv_file.write(line)
        writer = csv.DictWriter(out_fcsv_file, fieldnames=FCSV_FIELDNAMES)

        for row in fcsv:
            writer.writerow(row)
