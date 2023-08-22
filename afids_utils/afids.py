"""Anatomical fiducial classes"""
from __future__ import annotations

import attr
import numpy as np
import polars as pl
from numpy.typing import NDArray

from afids_utils.exceptions import InvalidFiducialError


@attr.define
class AfidSet(dict):
    """Base class for a set of fiducials"""

    slicer_version: str = attr.field()
    coord_system: str = attr.field()
    afids_df: pl.DataFrame = attr.field()

    def __attrs_post_init__(self):
        self["metadata"] = {
            "slicer_version": self.slicer_version,
            "coord_system": self.coord_system,
        }
        self["afids"] = self.afids_df

    def get_afid(self, label: int) -> NDArray[np.single]:
        """
        Extract a specific AFID's spatial coordinates

        Parameters
        ----------
        label
            Unique AFID label to extract from

        Returns
        -------
        numpy.ndarray[shape=(3,), dtype=numpy.single]
            NumPy array containing spatial coordinates (x, y, z) of single AFID
            coordinate

        Raises
        ------
        InvalidFiducialError
            If none or more than expected number of fiducials exist
        """

        # Filter based off of integer type
        if isinstance(label, int):
            # Fiducial selection out of bounds
            if label < 1 or label > len(self["afids"]):
                raise InvalidFiducialError(f"AFID label {label} is not valid")

            return (
                self["afids"]
                .filter(pl.col("label") == str(label))
                .select("x_mm", "y_mm", "z_mm")
                .to_numpy()[0]
            )
