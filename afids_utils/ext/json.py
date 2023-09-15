"""Methods for handling .json files associated with AFIDS"""
from __future__ import annotations

import json
from importlib import resources
from os import PathLike

from typing_extensions import TypedDict

from afids_utils.afids import AfidPosition, AfidSet
from afids_utils.exceptions import InvalidFileError


class ControlPoint(TypedDict):
    id: str
    label: str
    description: str
    associatedNodeID: str
    position: list[float]
    orientation: list[float]
    selected: bool
    locked: bool
    visibility: bool
    positionStatus: str


def _get_metadata(coord_system: str) -> tuple[str, str]:
    """Internal function to extract metadata from json files

    Note: Slicer version is not currently included in the json file

    Parameters:
    -----------
    coord_system
        Coordinate system parsed from json_file

    Returns
    -------
    parsed_version
        Slicer version associated with fiducial file

    parsed_coord
        Coordinate system of fiducials

    Raises
    ------
    InvalidFileError
        If header is invalid from .json file
    """

    # Update if future json versions include Slicer version
    parsed_version = "Unknown"
    parsed_coord = coord_system

    # Transform coordinate system so human-understandable
    if parsed_coord == "0":
        parsed_coord = "RAS"
    elif parsed_coord == "1":
        parsed_coord = "LPS"

    if parsed_coord not in ["RAS", "LPS"]:
        raise InvalidFileError("Invalid coordinate system")

    return parsed_version, parsed_coord


def _get_afids(control_points: ControlPoint) -> list[AfidPosition]:
    """Internal function to parse fiducial information from json file

    Parameters
    ----------
    ctrl_points
        List of dicts containing fiducial information from parsed json file

    Returns
    -------
    afid_positions
        List containing spatial position of afids
    """
    afids_positions = [
        AfidPosition(
            label=int(afid["label"]),
            x=float(afid["position"][0]),
            y=float(afid["position"][1]),
            z=float(afid["position"][2]),
            desc=afid["description"],
        )
        for afid in control_points
    ]

    return afids_positions


def load_json(
    json_path: PathLike[str] | str,
) -> tuple[str, str, list[AfidPosition]]:
    """Read in json and extract relevant information for an AfidSet

    Parameters
    ----------
    json_path
        Path to .json file containing AFIDs coordinates

    Returns
    -------
    slicer_version
        Slicer version associated with fiducial file

    coord_system
        Coordinate system of fiducials

    afids_positions
        List containing spatial position of afids
    """
    with open(json_path) as json_file:
        afids_json = json.load(json_file)

    # Grab metadata
    slicer_version, coord_system = _get_metadata(
        afids_json["markups"][0]["coordinateSystem"]
    )
    # Grab afids
    afids_positions = _get_afids(afids_json["markups"][0]["controlPoints"])

    return slicer_version, coord_system, afids_positions


def save_json(
    afid_set: AfidSet,
    out_json: PathLike[str] | str,
) -> None:
    """Save fiducials to output json file

    Parameters
    ----------
    afid_set
        A complete AfidSet containing metadata and positions of AFIDs

    out_json
        Path of json file to save AFIDs to
    """
    # Read in json template
    with resources.open_text(
        "afids_utils.resources", "template.json"
    ) as template_json_file:
        template_content = json.load(template_json_file)

        # Update header
        template_content["markups"][0][
            "coordinateSystem"
        ] = afid_set.coord_system

    # Loop and update with fiducial coordinates
    for idx in range(len(template_content["markups"][0]["controlPoints"])):
        template_content["markups"][0]["controlPoints"][idx]["position"] = [
            afid_set.afids[idx].x,
            afid_set.afids[idx].y,
            afid_set.afids[idx].z,
        ]

    # Write output json
    with open(out_json, "w") as out_json_file:
        json.dump(template_content, out_json_file, indent=4)
