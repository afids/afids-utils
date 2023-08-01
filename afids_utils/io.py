"""Methods for loading and saving files associated with AFIDs"""
from __future__ import annotations

import csv
import json
from importlib import resources
from os import PathLike
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from afids_utils.exceptions import InvalidFiducialNumberError

EXPECTED_DESCS = [
    ["AC"],
    ["PC"],
    ["infracollicular sulcus", "ICS"],
    ["PMJ"],
    ["superior interpeduncular fossa", "SIPF"],
    ["R superior LMS", "RSLMS"],
    ["L superior LMS", "LSLMS"],
    ["R inferior LMS", "RILMS"],
    ["L inferior LMS", "LILMS"],
    ["Culmen", "CUL"],
    ["Intermammillary sulcus", "IMS"],
    ["R MB", "RMB"],
    ["L MB", "LMB"],
    ["pineal gland", "PG"],
    ["R LV at AC", "RLVAC"],
    ["L LV at AC", "LLVAC"],
    ["R LV at PC", "RLVPC"],
    ["L LV at PC", "LLVPC"],
    ["Genu of CC", "GENU"],
    ["Splenium of CC", "SPLE"],
    ["R AL temporal horn", "RALTH"],
    ["L AL temporal horn", "LALTH"],
    ["R superior AM temporal horn", "RSAMTH"],
    ["L superior AM temporal horn", "LSAMTH"],
    ["R inferior AM temporal horn", "RIAMTH"],
    ["L inferior AM temporal horn", "LIAMTH"],
    ["R indusium griseum origin", "RIGO"],
    ["L indusium griseum origin", "LIGO"],
    ["R ventral occipital horn", "RVOH"],
    ["L ventral occipital horn", "LVOH"],
    ["R olfactory sulcal fundus", "ROSF"],
    ["L olfactory sulcal fundus", "LOSF"],
]

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
    afids_fpath: PathLike[str] | str, fid_num: int
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
        raise InvalidFiducialNumberError(
            f"Invalid fiducial number: {fid_num}."
        )

    afids_fpath = Path(afids_fpath)
    afids_fpath_ext = afids_fpath.suffix

    # Handling of different file extensions
    if afids_fpath_ext == ".fcsv":
        afids_df = pl.scan_csv(
            afids_fpath, separator=",", skip_rows=2, dtypes=FCSV_FIELDNAMES
        )

        # Check fiducial number matches expected description
        if (
            afids_df.filter(pl.col("label") == fid_num)
            .select("desc")
            .collect()
            .item()
            not in EXPECTED_DESCS[fid_num - 1]
        ):
            raise ValueError(
                f"Fiducial {fid_num} does not match expected description"
            )

        return (
            afids_df.filter(pl.col("label") == fid_num)
            .select("x", "y", "z")
            .collect()
            .to_numpy()[0]
        )

    elif afids_fpath_ext == ".json":
        # Polars currently cannot handle Slicer-esque JSONs directly
        with open(afids_fpath, "r", encoding="utf-8") as json_file:
            afids_json = json.load(json_file)
        afids_data = afids_json["markups"][0]["controlPoints"][fid_num - 1]

        # Check fiducial number matches expected description
        if afids_data["description"] not in EXPECTED_DESCS[fid_num - 1]:
            raise ValueError(
                f"Fiducial {fid_num} does not match expected description"
            )

        return np.array(afids_data["position"], dtype=np.single)

    else:
        # Unknown extension
        raise IOError("Invalid file extension.")


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

    # Check to make sure shape of AFIDs array matches expected template
    if afid_coords.shape[0] != len(fcsv):
        raise TypeError(
            f"Expected {len(fcsv)} AFIDs, but received {afid_coords.shape[0]}."
        )
    if afid_coords.shape[1] != 3:
        raise TypeError(
            "Expected 3 spatial dimensions (x, y, z),"
            f"but received {afid_coords.shape[1]}."
        )

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
