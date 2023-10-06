"""Anatomical fiducial classes"""
from __future__ import annotations

import json
import warnings
from collections.abc import Iterable
from importlib import resources
from os import PathLike
from pathlib import Path

import attrs

from afids_utils.exceptions import InvalidFiducialError, InvalidFileError

with resources.open_text(
    "afids_utils.resources", "afids_descs.json"
) as json_fpath:
    HUMAN_PROTOCOL_MAP = json.load(json_fpath)["human"]


def _validate_desc(
    self: AfidPosition,
    attribute: attrs.Attribute[str],
    value: str,
):
    if value not in [
        HUMAN_PROTOCOL_MAP[self.label - 1]["desc"],
        HUMAN_PROTOCOL_MAP[self.label - 1]["acronym"],
    ]:
        raise InvalidFiducialError(
            f"Description {value} does not correspond to label {self.label}."
        )


@attrs.define(kw_only=True)
class AfidPosition:
    """Base class for a single AFID position

    Parameters
    ----------
    label
        Unique label for AFID

    x
        Spatial position along x-axis (in mm)

    y
        Spatial position along y-axis (in mm)

    z
        Spatial position along z-axis (in mm)

    desc
        Description for AFID (e.g. AC, PC)
    """

    label: int = attrs.field(validator=attrs.validators.in_(range(1, 33)))
    x: float = attrs.field()
    y: float = attrs.field()
    z: float = attrs.field()
    desc: str = attrs.field(validator=_validate_desc)


def sort_afids(afids: Iterable[AfidPosition]) -> list[AfidPosition]:
    return list(sorted(afids, key=lambda afid: afid.label))


def _validate_afids(
    instance: AfidSet,
    attribute: attrs.Attribute[list[AfidPosition]],
    value: list[AfidPosition],
):
    if len(value) != (expected_length := len(HUMAN_PROTOCOL_MAP)):
        raise ValueError(
            f"Incorrect number of AFIDs. Expected {expected_length}, "
            f"found: {len(value)}"
        )
    incorrect_afids = [
        (afid, expected_label)
        for expected_label, afid in enumerate(value, start=1)
        if afid.label != expected_label
    ]
    if incorrect_afids:
        msg = "\n".join(
            [
                f"Found {len(incorrect_afids)} afids with incorrect labels",
                *[
                    f"AFID: {afid}, expected label: {expected_label}"
                    for afid, expected_label in incorrect_afids
                ],
            ]
        )
        raise ValueError(msg)


@attrs.define
class AfidVoxel:
    """Class for Afid voxel position

    Parameters
    ----------
    label
        Unique label for AFID

    i
        Spatial position along i-axis

    j
        Spatial position along j-axis

    k
        Spatial position along k-axis

    desc
        Description for AFID (e.g. AC, PC)
    """

    label: int = attrs.field()
    i: int = attrs.field()
    j: int = attrs.field()
    k: int = attrs.field()
    desc: str = attrs.field()


@attrs.define(kw_only=True)
class AfidSet:
    """Base class for a set of AFIDs

    Parameters
    ----------
    slicer_version
        Version of Slicer associated with AfidSet

    coord_system
        Coordinate system AFIDs are placed in (e.g. RAS)

    afids
        List of AFID labels and their coordinates
    """

    slicer_version: str = attrs.field()
    coord_system: str = attrs.field()

    afids: list[AfidPosition] = attrs.field(
        converter=sort_afids,
        validator=_validate_afids,
    )

    @classmethod
    def load(cls, afids_fpath: PathLike[str] | str) -> AfidSet:
        """
        Load an AFIDs file

        Parameters
        ----------
        afids_fpath
            Path to .fcsv or .json file containing AFIDs information

        Returns
        -------
        AfidSet
            Set of anatomical fiducials containing coordinates and metadata

        Raises
        ------
        IOError
            If extension to fiducial file is not supported

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
            from afids_utils.ext.fcsv import load_fcsv

            slicer_version, coord_system, afids_positions = load_fcsv(
                afids_fpath
            )
        # Loading json
        elif afids_fpath_ext == ".json":
            from afids_utils.ext.json import load_json

            slicer_version, coord_system, afids_positions = load_json(
                afids_fpath
            )
        else:
            raise ValueError("Unsupported file extension")

        # Perform validation of loaded file
        # Load expected mappings
        with resources.open_text(
            "afids_utils.resources", "afids_descs.json"
        ) as json_fpath:
            mappings = json.load(json_fpath)
        # Check expected number of fiducials exist
        if len(afids_positions) != len(mappings["human"]):
            raise InvalidFileError("Unexpected number of fiducials")

        return cls(
            slicer_version=slicer_version,
            coord_system=coord_system,
            afids=afids_positions,
        )

    def save(self, out_fpath: PathLike[str] | str) -> None:
        """Save AFIDs to Slicer-compatible file

        Parameters
        ----------
        out_fpath
            Path of file (including filename and extension) to save AFIDs to

        Raises
        ------
        ValueError
            If file extension is not supported
        """

        out_fpath_ext = Path(out_fpath).suffix

        # Update coordinate system for template
        if self.coord_system not in ["RAS", "LPS", "0", "1"]:
            raise ValueError("AfidSet contains an invalid coordinate system")
        elif self.coord_system == "RAS":
            self.coord_system = "0"
        elif self.coord_system == "LPS":
            self.coord_system = "1"

        # Saving fcsv
        if out_fpath_ext == ".fcsv":
            from afids_utils.ext.fcsv import save_fcsv

            save_fcsv(self, out_fpath)
        # Saving json
        elif out_fpath_ext == ".json":
            from afids_utils.ext.json import save_json

            save_json(self, out_fpath)
        else:
            raise ValueError("Unsupported file extension")

    def get_afid(self, label: int) -> AfidPosition:
        """
        Extract a specific AFID's spatial coordinates

        Parameters
        ----------
        label
            Unique AFID label to extract from

        Returns
        -------
        AfidPosition
            Spatial position of Afid (as class AfidPosition)

        Raises
        ------
        InvalidFiducialError
            If AFID label given out of valid range
        """

        # Fiducial selection out of bounds
        if label < 1 or label > len(self.afids):
            raise InvalidFiducialError(f"AFID label {label} is not valid")

        return self.afids[label - 1]


@attrs.define()
class AfidDistance:
    """Class to store distances between two ``AfidPosition`` objects

    Parameters
    ----------
    afid_position1
        An AfidPosition object containing floating-point spatial coordinates
        (x, y, z)

    afid_position2
        Other AfidPosition object containing floating-point spatial
        coordinates (x, y, z) to compute distance against
    """

    afid_position1: AfidPosition = attrs.field()
    afid_position2: AfidPosition = attrs.field()

    def __attrs_post_init__(self):
        # Always throw warning if label/desc don't match between AFIDs
        if (self.afid_position1.label, self.afid_position1.desc) != (
            self.afid_position2.label,
            self.afid_position2.desc,
        ):
            warnings.simplefilter("always", category=UserWarning)
            warnings.warn(
                "Computing distances between non-corresponding AFIDs"
            )

    @property
    def x(self):
        """Floating-point distance between AFIDs along x-axis"""
        return self.afid_position1.x - self.afid_position2.x

    @property
    def y(self):
        """Floating-point distance between AFIDs along y-axis"""
        return self.afid_position1.y - self.afid_position2.y

    @property
    def z(self):
        """Floating-point distance between AFIDs along z-axis"""
        return self.afid_position1.z - self.afid_position2.z

    @property
    def distance(self):
        """Floating-point distance between a pair of AFIDs"""
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def get(self, component: str):
        """Return value of specified component"""
        valid_components = ["x", "y", "z", "distance"]

        if component in valid_components:
            return getattr(self, component)
        else:
            raise ValueError(f"Invalid component '{component}'")


@attrs.define()
class AfidDistanceSet:
    """Class to store distances between a pair of valid ``AfidSet`` objects

    Parameters
    ----------
    afid_set1
        One set of anatomical fiducials containing coordinates and metadata

    afid_set2
        Another set of anatomical fiducials containing coordinates and metadata
    """

    afid_set1: AfidSet = attrs.field()
    afid_set2: AfidSet = attrs.field()

    @property
    def afids(self):
        """List of distances of corresponding AFIDs between the two ``AfidSet``
        objects

        Raises
        ------
        ValueError
            If coordinate systems are mismatched between ``AfidSet`` objects
        """
        # Check if the coordinate systems match
        if self.afid_set1.coord_system != self.afid_set2.coord_system:
            raise ValueError("Mismatched coordinate systems")

        # Compute distances between AfidSets
        distances = [
            AfidDistance(afid_set1_position, afid_set2_position)
            for afid_set1_position, afid_set2_position in zip(
                self.afid_set1.afids, self.afid_set2.afids
            )
        ]

        return distances
