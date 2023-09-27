from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import afids_utils.tests.strategies as af_st
from afids_utils.afids import AfidPosition, AfidSet
from afids_utils.exceptions import InvalidFileError
from afids_utils.ext import fcsv as af_fcsv
from afids_utils.ext import json as af_json
from afids_utils.tests.helpers import allow_function_scoped


@pytest.fixture
def valid_fcsv_file() -> Path:
    return (
        Path(__file__).parent / "data" / "tpl-MNI152NLin2009cAsym_afids.fcsv"
    )


@pytest.fixture
def valid_json_file() -> Path:
    return (
        Path(__file__).parent / "data" / "tpl-MNI152NLin2009cAsym_afids.json"
    )


class TestLoadFcsv:
    @given(coord_num=st.integers(min_value=0, max_value=1))
    @allow_function_scoped
    def test_get_valid_metadata(self, valid_fcsv_file: Path, coord_num: int):
        # Randomize coordinate system
        with open(valid_fcsv_file) as valid_fcsv:
            valid_fcsv_data = valid_fcsv.readlines()
            valid_fcsv_data[1] = valid_fcsv_data[1].replace(
                "# CoordinateSystem = 0",
                f"# CoordinateSystem = {str(coord_num)}",
            )

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="sub-test_desc-",
            suffix="_afids.fcsv",
        ) as temp_valid_fcsv_file:
            temp_valid_fcsv_file.writelines(valid_fcsv_data)
            temp_valid_fcsv_file.flush()

            with open(temp_valid_fcsv_file.name) as temp_in_fcsv:
                parsed_ver, parsed_coord = af_fcsv._get_metadata(
                    temp_in_fcsv.readlines()
                )

        # Check version pattern matches expected
        ver_regex = re.compile(r"\d+\.\d+")
        assert ver_regex.match(parsed_ver)

        # Check to make sure coordinate system is correct
        if coord_num == 0:
            assert parsed_coord == "RAS"
        else:
            assert parsed_coord == "LPS"

    @given(coord_num=st.integers(min_value=2))
    @allow_function_scoped
    def test_invalid_num_coord(self, valid_fcsv_file: Path, coord_num: int):
        with open(valid_fcsv_file) as valid_fcsv:
            fcsv_data = valid_fcsv.readlines()
            fcsv_data[1] = fcsv_data[1].replace(
                "# CoordinateSystem = 0",
                f"# CoordinateSystem = {str(coord_num)}",
            )

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="sub-test_desc-",
            suffix="_afids.fcsv",
        ) as temp_invalid_fcsv_file:
            temp_invalid_fcsv_file.writelines(fcsv_data)
            temp_invalid_fcsv_file.flush()

            with open(temp_invalid_fcsv_file.name) as temp_in_fcsv:
                with pytest.raises(
                    InvalidFileError, match="Invalid coordinate.*"
                ):
                    af_fcsv._get_metadata(temp_in_fcsv.readlines())

    @given(coord_str=af_st.short_ascii_text())
    @allow_function_scoped
    def test_invalid_str_coord(self, valid_fcsv_file: Path, coord_str: str):
        assume(coord_str not in ["RAS", "LPS"])

        with open(valid_fcsv_file) as valid_fcsv:
            fcsv_data = valid_fcsv.readlines()
            fcsv_data[1] = fcsv_data[1].replace(
                "# CoordinateSystem = 0",
                f"# CoordinateSystem = {coord_str}",
            )

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="sub-test_desc-",
            suffix="_afids.fcsv",
        ) as temp_invalid_fcsv_file:
            temp_invalid_fcsv_file.writelines(fcsv_data)
            temp_invalid_fcsv_file.flush()

            with open(temp_invalid_fcsv_file.name) as temp_in_fcsv:
                with pytest.raises(
                    InvalidFileError, match="Invalid coordinate.*"
                ):
                    af_fcsv._get_metadata(temp_in_fcsv.readlines())

    def test_fcsv_invalid_header(self, valid_fcsv_file: Path):
        with open(valid_fcsv_file) as valid_fcsv:
            valid_fcsv_data = valid_fcsv.readlines()
            invalid_fcsv_data = valid_fcsv_data[0]

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="sub-test_desc-",
            suffix="_afids.fcsv",
        ) as temp_invalid_fcsv_file:
            temp_invalid_fcsv_file.writelines(invalid_fcsv_data)
            temp_invalid_fcsv_file.flush()

            with open(temp_invalid_fcsv_file.name) as temp_in_fcsv:
                with pytest.raises(
                    InvalidFileError, match="Missing or invalid.*"
                ):
                    af_fcsv._get_metadata(temp_in_fcsv.readlines())

    @given(label=af_st.valid_labels())
    @allow_function_scoped
    def test_valid_get_afids(self, valid_fcsv_file: Path, label: int):
        with open(valid_fcsv_file) as valid_fcsv:
            afids_positions = af_fcsv._get_afids(valid_fcsv.readlines())

        assert isinstance(afids_positions, list)
        assert isinstance(afids_positions[label - 1], AfidPosition)


class TestSaveFcsv:
    @given(afid_set=af_st.afid_sets())
    def test_save_fcsv_invalid_template(
        self,
        afid_set: AfidSet,
    ):
        with pytest.raises(FileNotFoundError):
            af_fcsv.save_fcsv(afid_set, "/invalid/template/path.fcsv")

    @given(afid_set=af_st.afid_sets(randomize_header=False))
    def test_save_fcsv_valid_template(self, afid_set: AfidSet):
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="sub-test_desc-", suffix="_afids.fcsv"
        ) as out_fcsv_file:
            # Create and check output file
            af_fcsv.save_fcsv(afid_set, out_fcsv_file.name)

            # Check if file loads correctly and contents are the same
            test_load = AfidSet.load(out_fcsv_file.name)
            assert test_load == afid_set
            assert isinstance(test_load, AfidSet)


class TestLoadJson:
    @given(coord=st.sampled_from(["RAS", "LPS", "0", "1"]))
    @allow_function_scoped
    def test_json_get_valid_metadata(self, valid_json_file: Path, coord: str):
        # Randomize coordinate system
        with open(valid_json_file) as valid_json:
            afids_json = json.load(valid_json)
            afids_json["markups"][0]["coordinateSystem"] = coord

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="sub-test_desc-",
            suffix="_afids.json",
        ) as temp_valid_json_file:
            json.dump(afids_json, temp_valid_json_file, indent=4)
            temp_valid_json_file.flush()

            with open(temp_valid_json_file.name) as temp_in_json:
                temp_afids_json = json.load(temp_in_json)
                parsed_ver, parsed_coord = af_json._get_metadata(
                    temp_afids_json["markups"][0]["coordinateSystem"]
                )

        # Check version is not given / unknown
        assert parsed_ver == "Unknown"

        # Check to make sure coordinate system is correct
        if coord in ["0" or "RAS"]:
            assert parsed_coord == "RAS"
        elif coord in ["1" or "LPS"]:
            assert parsed_coord == "LPS"

    @given(coord_num=st.integers(min_value=2))
    @allow_function_scoped
    def test_json_invalid_num_coord(
        self, valid_json_file: Path, coord_num: int
    ):
        with open(valid_json_file) as valid_json:
            afids_json = json.load(valid_json)
            afids_json["markups"][0]["coordinateSystem"] = str(coord_num)

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="sub-test_desc-",
            suffix="_afids.json",
        ) as temp_valid_json_file:
            json.dump(afids_json, temp_valid_json_file, indent=4)
            temp_valid_json_file.flush()

            with open(temp_valid_json_file.name) as temp_in_json:
                with pytest.raises(
                    InvalidFileError, match=r"Invalid coordinate.*"
                ):
                    temp_afids_json = json.load(temp_in_json)
                    af_json._get_metadata(temp_afids_json["markups"][0])

    @given(
        coord_str=st.text(
            min_size=3,
            alphabet=st.characters(
                min_codepoint=ord("A"), max_codepoint=ord("z")
            ),
        )
    )
    @allow_function_scoped
    def test_json_invalid_str_coord(
        self, valid_json_file: Path, coord_str: str
    ):
        assume(coord_str not in ["RAS", "LPS"])

        with open(valid_json_file) as valid_json:
            afids_json = json.load(valid_json)
            afids_json["markups"][0]["coordinateSystem"] = coord_str

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="sub-test_desc-",
            suffix="_afids.json",
        ) as temp_valid_json_file:
            json.dump(afids_json, temp_valid_json_file, indent=4)
            temp_valid_json_file.flush()

            with open(temp_valid_json_file.name) as temp_in_json:
                with pytest.raises(
                    InvalidFileError, match=r"Invalid coordinate.*"
                ):
                    temp_afids_json = json.load(temp_in_json)
                    af_json._get_metadata(temp_afids_json["markups"][0])

    @given(label=st.integers(min_value=0, max_value=31))
    @allow_function_scoped
    def test_json_valid_get_afids(self, valid_json_file: Path, label: int):
        with open(valid_json_file) as valid_json:
            afids_json = json.load(valid_json)
            afids_positions = af_json._get_afids(
                afids_json["markups"][0]["controlPoints"]
            )

        assert isinstance(afids_positions, list)
        assert isinstance(afids_positions[label], AfidPosition)


class TestSaveJson:
    @given(afid_set=af_st.afid_sets())
    def test_save_json_invalid_template(
        self,
        afid_set: AfidSet,
    ):
        with pytest.raises(FileNotFoundError):
            af_json.save_json(afid_set, "/invalid/template/path.json")

    @given(afid_set=af_st.afid_sets(randomize_header=False))
    def test_save_json_valid_template(self, afid_set: AfidSet):
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="sub-test_desc-", suffix="_afids.json"
        ) as out_json_file:
            # Create and check output file
            af_json.save_json(afid_set, out_json_file.name)

            # Check if file loads correctly and contents are the same
            # (except for the version)
            test_load = AfidSet.load(out_json_file.name)
            assert test_load.coord_system == afid_set.coord_system
            assert test_load.afids == afid_set.afids
            assert isinstance(test_load, AfidSet)
