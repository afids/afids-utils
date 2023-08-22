"""General methods for loading and saving files associated with AFIDs"""
from __future__ import annotations

import json
from importlib import resources
from os import PathLike
from pathlib import Path

import polars as pl

from afids_utils.afids import AfidSet
from afids_utils.exceptions import InvalidFiducialError, InvalidFileError
from afids_utils.ext.fcsv import load_fcsv, save_fcsv


def load(
    afids_fpath: PathLike[str] | str,
) -> AfidSet:
    """
    Load an AFIDs file

    Parameters
    ----------
    afids_fpath
        Path to .fcsv or .json file containing AFIDs information

    Returns
    -------
    AfidSet
        Set of anatomical fiducials containing spatial coordinates and metadata

    Raises
    ------
    IOError
        If extension to fiducial file is not .fcsv or .json or does not exist

    InvalidFileError
        If fiducial file has none or more than expected number of fiducials

    InvalidFiducialError
        If description in fiducial file does not match expected
    """
    # Check if file exists
    afids_fpath = Path(afids_fpath)
    if not afids_fpath.exists():
        raise FileNotFoundError("Provided AFID file does not exist")

    afids_fpath_ext = afids_fpath.suffix

    # Loading fcsv
    if afids_fpath_ext == ".fcsv":
        afids_set = load_fcsv(afids_fpath)
    # Loading json
    # if afids_fpath_ext = ".json":
    #   load_json(afids_path)
    else:
        raise ValueError("Unsupported file extension")

    # Perform validation of loaded file
    # Load expected mappings
    with resources.open_text(
        "afids_utils.resources", "afids_descs.json"
    ) as json_fpath:
        mappings = json.load(json_fpath)
    # Check expected number of fiducials exist
    if len(afids_set["afids"]) != len(mappings["human"]):
        raise InvalidFileError("Unexpected number of fiducials")

    # Validate descriptions, before dropping
    for label in range(1, len(afids_set["afids"] + 1)):
        desc = (
            afids_set["afids"]
            .filter(pl.col("label") == str(label))
            .select("desc")
            .item()
        )
        if desc not in mappings["human"][label - 1]:
            raise InvalidFiducialError(
                f"Description for label {label} does not match expected"
            )

    # Drop description column
    afids_set["afids"] = afids_set["afids"].drop("desc")

    return afids_set


# TODO: Handle the metadata - specifically setting the coordinate system
def save(
    afids_set: AfidSet,
    out_fpath: PathLike[str] | str,
) -> None:
    """Save AFIDs to Slicer-compatible file

    Parameters
    ----------
    afids_set
        An AFID dataset containing metadata and coordinates

    fcsv_output : os.PathLike[str] | str
        Path of file (including filename and extension) to save AFIDs to

    Raises
    ------
    IOError
        If file extension is not supported
    """

    out_fpath_ext = Path(out_fpath).suffix

    afids_coords = afids_set["afids"].select("x_mm", "y_mm", "z_mm").to_numpy()

    # Saving fcsv
    if out_fpath_ext == ".fcsv":
        save_fcsv(afids_coords, out_fpath)
    # Saving json
    # if out_fpath_ext = ".json":
    #   save_json(afids_coords, out_fpath)
    else:
        raise ValueError("Unsupported file extension")
