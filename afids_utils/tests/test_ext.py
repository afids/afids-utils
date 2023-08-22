from __future__ import annotations

import csv
import re
import tempfile
from os import PathLike
from pathlib import Path

import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from afids_utils.afids import AfidSet
from afids_utils.ext.fcsv import (
    FCSV_FIELDNAMES, _get_metadata, load_fcsv, save_fcsv
)
from afids_utils.exceptions import InvalidFileError
from afids_utils.tests.strategies import afid_coords


@pytest.fixture
def valid_fcsv_file() -> PathLike[str]:
    return (
        Path(__file__).parent / "data" / "tpl-MNI152NLin2009cAsym_afids.fcsv"
    )


class TestLoadFcsv:
    def test_load_valid_fcsv(self, valid_fcsv_file: PathLike[str]):
        afids_set = load_fcsv(valid_fcsv_file)

        assert isinstance(afids_set, AfidSet)

    @given(coord_num=st.integers(min_value=0, max_value=1))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_get_valid_metadata(
        self, valid_fcsv_file: PathLike[str], coord_num: int
    ):
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

            parsed_ver, parsed_coord = _get_metadata(temp_valid_fcsv_file.name)

        # Check version pattern matches expected
        ver_regex = re.compile(r"\d+\.\d+")
        assert ver_regex.match(parsed_ver)

        # Check to make sure coordinate system is correct
        if coord_num == 0:
            assert parsed_coord == "LPS"
        else:
            assert parsed_coord == "RAS"

    @given(coord_num=st.integers(min_value=2))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_num_coord(
        self, valid_fcsv_file: PathLike[str], coord_num: int
    ):
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

            with pytest.raises(
                InvalidFileError, match="Invalid coordinate.*"
            ):
                _get_metadata(temp_invalid_fcsv_file.name)


    @given(coord_str=st.text(min_size=3, alphabet=st.characters(whitelist_categories=['Lu', 'Ll', 'Lt'])))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_str_coord(
        self, valid_fcsv_file: PathLike[str], coord_str: int
    ):
        assume(coord_str not in ["LPS", "RAS"])

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

            with pytest.raises(
                InvalidFileError, match="Invalid coordinate.*"
            ):
                _get_metadata(temp_invalid_fcsv_file.name)


    def test_invalid_header(
        self, valid_fcsv_file: PathLike[str]
    ):
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

            with pytest.raises(InvalidFileError, match="Missing or invalid.*"):
                _get_metadata(temp_invalid_fcsv_file.name)


class TestSaveFcsv:
    @given(afids_coords=afid_coords())
    def test_save_fcsv_invalid_template(
        self, afids_coords: NDArray[np.single]
    ):
        with pytest.raises(FileNotFoundError):
            save_fcsv(afids_coords, "/invalid/template/path.fcsv")

    @given(afids_coords=afid_coords())
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_save_fcsv_valid_template(
        self, afids_coords: NDArray[np.single], valid_fcsv_file: PathLike[str],
    ):
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="sub-test_desc-", suffix="_afids.fcsv"
        ) as out_fcsv_file:
            # Create and check output file
            save_fcsv(afids_coords, out_fcsv_file.name)

            # Load files
            with open(
                valid_fcsv_file, "r", encoding="utf-8", newline=""
            ) as template_fcsv_file, open(
                out_fcsv_file.name, "r", encoding="utf-8", newline=""
            ) as output_fcsv_file:
                template_header = [
                    template_fcsv_file.readline() for _ in range(3)
                ]
                output_header = [output_fcsv_file.readline() for _ in range(3)]
                reader = csv.DictReader(
                    output_fcsv_file, fieldnames=FCSV_FIELDNAMES
                )
                output_fcsv = list(reader)

            # Check header
            assert output_header == template_header
            # Check contents
            for idx, row in enumerate(output_fcsv):
                assert (row["x"], row["y"], row["z"]) == (
                    str(afids_coords[idx][0]),
                    str(afids_coords[idx][1]),
                    str(afids_coords[idx][2]),
                )


    @given(afids_coords=afid_coords(bad_range=True))
    def test_invalid_num_afids(self, afids_coords: NDArray[np.single]) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="sub-test_desc-", suffix="_afids.fcsv"
        ) as out_fcsv_file:
            with pytest.raises(TypeError) as err:
                save_fcsv(afids_coords, out_fcsv_file)

            assert "AFIDs, but received" in str(err.value)


    @given(afids_coords=afid_coords(bad_dims=True))
    def test_invalid_afids_dims(
        self, afids_coords: NDArray[np.single]
    ) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="sub-test_desc-", suffix="_afids.fcsv"
        ) as out_fcsv_file:
            with pytest.raises(TypeError) as err:
                save_fcsv(afids_coords, out_fcsv_file)

            assert "Expected 3 spatial dimensions" in str(err.value)