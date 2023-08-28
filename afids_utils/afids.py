"""Anatomical fiducial classes"""
from __future__ import annotations

import json
from importlib import resources
from os import PathLike
from pathlib import Path

import attrs

from afids_utils.exceptions import InvalidFiducialError, InvalidFileError


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

    label: int = attrs.field()
    x: float = attrs.field()
    y: float = attrs.field()
    z: float = attrs.field()
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
    afids: list[AfidPosition] = attrs.field()

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
        if len(afids_positions) != len(mappings["human"]):
            raise InvalidFileError("Unexpected number of fiducials")

        # Validate descriptions, before dropping
        for label in range(len(afids_positions)):
            if afids_positions[label].desc not in mappings["human"][label]:
                raise InvalidFiducialError(
                    f"Description for label {label+1} does not match expected"
                )

        return cls(
            slicer_version=slicer_version,
            coord_system=coord_system,
            afids=afids_positions,
        )

    # TODO: Handle the metadata - specifically setting the coordinate system
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

        # Saving fcsv
        if out_fpath_ext == ".fcsv":
            from afids_utils.ext.fcsv import save_fcsv

            save_fcsv(self.afids, out_fpath)
        # Saving json
        # if out_fpath_ext = ".json":
        #   save_json(afids_coords, out_fpath)
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
        afid_position
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
