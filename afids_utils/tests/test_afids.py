from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from afids_utils.io import load
from afids_utils.exceptions import InvalidFiducialError


@pytest.fixture
def valid_fcsv_file() -> PathLike[str]:
    return (
        Path(__file__).parent / "data" / "tpl-MNI152NLin2009cAsym_afids.fcsv"
    )

class TestAfids:
    def test_init(self, valid_fcsv_file: PathLike[str]):
        # Load valid file to check internal types
        afids_set = load(valid_fcsv_file)

        # Check to make sure internal types are correct
        assert isinstance(afids_set["metadata"], dict)
        assert isinstance(afids_set["metadata"]["slicer_version"], str)
        assert isinstance(afids_set["metadata"]["coord_system"], str)
        assert isinstance(afids_set["afids"], pl.DataFrame)


    @given(label=st.integers(min_value=1, max_value=32))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_valid_get_afid(self, valid_fcsv_file: PathLike[str], label: int):
        afids_set = load(valid_fcsv_file)
        afid_pos = afids_set.get_afid(label)

        # Check array type
        assert isinstance(afid_pos, np.ndarray)
        # Check array values
        assert afid_pos.dtype == np.single

    @given(label=st.integers(min_value=-100, max_value=100))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )    
    def test_invalid_get_afid(self, valid_fcsv_file: PathLike[str], label: int):
        afids_set = load(valid_fcsv_file)
        assume(not 1 <= label <= len(afids_set['afids']))
        
        with pytest.raises(InvalidFiducialError, match=".*not valid"):
            afids_set.get_afid(label)