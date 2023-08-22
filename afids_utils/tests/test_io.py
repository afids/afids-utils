from __future__ import annotations

import json
import tempfile
from os import PathLike
from importlib import resources
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from afids_utils.afids import AfidSet
from afids_utils.io import load, save
from afids_utils.exceptions import InvalidFiducialError, InvalidFileError
from afids_utils.tests.strategies import afid_coords

whitelist_strs = ['Lu', 'Ll', 'Lt']

@pytest.fixture
def valid_fcsv_file() -> PathLike[str]:
    return (
        Path(__file__).parent / "data" / "tpl-MNI152NLin2009cAsym_afids.fcsv"
    )


@pytest.fixture
def human_mappings() -> List[List[str] | str]:
    with resources.open_text(
        "afids_utils.resources", "afids_descs.json"
    ) as json_fpath:
        mappings = json.load(json_fpath)

    return mappings['human']


class TestLoad:
    def test_valid_file(self, valid_fcsv_file: PathLike[str]):
        afids_set = load(valid_fcsv_file)

        assert isinstance(afids_set, AfidSet)
        

    def test_invalid_fpath(self):
        with pytest.raises(IOError, match=".*does not exist"):
            load('invalid/fpath.fcsv')


    @given(
        ext=st.text(
            min_size=2, 
            max_size=5, 
            alphabet=st.characters(whitelist_categories=whitelist_strs)
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_ext(self, valid_fcsv_file: PathLike[str], ext: str):
        assume(not ext == "fcsv" or not ext == "json")

        with tempfile.NamedTemporaryFile(
            mode="w", prefix="sub-test_desc-", suffix=f"_afids.{ext}",
        ) as invalid_file_ext:
            with pytest.raises(IOError, match="Unsupported .* extension"):
                load(invalid_file_ext.name)


    def test_invalid_label_range(self, valid_fcsv_file: PathLike[str]):
        # Create additional line of fiducials
        with open(valid_fcsv_file) as valid_fcsv:
            fcsv_data = valid_fcsv.readlines()
            fcsv_data.append(fcsv_data[-1])

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="sub-test_desc-",
            suffix="_afids.fcsv",
        ) as out_fcsv_file:
            out_fcsv_file.writelines(fcsv_data)
            out_fcsv_file.flush()

            # Test that InvalidFileError raised containing correct message
            with pytest.raises(InvalidFileError, match="Unexpected number.*"):
                load(out_fcsv_file.name)


    @given(
        label=st.integers(min_value=0, max_value=31),
        desc=st.text(
            min_size=2,
            max_size=5,
            alphabet=st.characters(whitelist_categories=whitelist_strs),
        ),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_desc(
        self, 
        valid_fcsv_file: PathLike[str], 
        human_mappings: List[List[str] | str], 
        label: int, 
        desc: str
    ) -> None:
        assume(desc not in human_mappings[label])

        # Replace valid description with a mismatch
        with open(valid_fcsv_file) as valid_fcsv:
            fcsv_data = valid_fcsv.readlines()
            fcsv_data[label+3] = fcsv_data[label+3].replace(
                human_mappings[label][0], desc
            )
            print(fcsv_data)

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="sub-test_desc-",
            suffix="_afids.fcsv",
        ) as out_fcsv_file:
            out_fcsv_file.writelines(fcsv_data)
            out_fcsv_file.flush()

            # Test for description match error raised
            with pytest.raises(
                InvalidFiducialError, match="Description for label .*"
            ):
                load(out_fcsv_file.name)


class TestSave:
    @given(afids_coords=afid_coords())
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_save_fcsv(self, afids_coords: NDArray[np.single]):
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="sub-test_desc-",
            suffix="_afids.fcsv"
        ) as out_fcsv_file:
            save(afids_coords, out_fcsv_file.name)

            assert Path(out_fcsv_file.name).exists()

    
    @given(
        afids_coords=afid_coords(),
        ext=st.text(
            min_size=2, 
            max_size=5, 
            alphabet=st.characters(whitelist_categories=whitelist_strs)
        )
    )
    def test_save_invalid_ext(
        self, afids_coords: NDArray[np.single], ext: str
    ):
        assume(not ext == "fcsv" or not ext == "json")
        
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="sub-test_desc-",
            suffix=f"_afids.{ext}"
        ) as out_file:

            with pytest.raises(IOError, match="Unsupported file extension"):
                save(afids_coords, out_file.name)