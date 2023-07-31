from __future__ import annotations

import csv
import tempfile
from os import PathLike
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from afids_utils.exceptions import InvalidFiducialNumberError
from afids_utils.io import FCSV_FIELDNAMES, afids_to_fcsv, get_afid
from afids_utils.tests.strategies import afid_coords


@pytest.fixture
def valid_fcsv_file() -> PathLike[str]:
    return (
        Path(__file__).parent / "data" / "tpl-MNI152NLin2009cAsym_afids.fcsv"
    )


@pytest.fixture
def valid_json_file() -> PathLike[str]:
    return (
        Path(__file__).parent / "data" / "tpl-MNI152NLin2009cAsym_afids.json"
    )


class TestGetAfid:
    @given(afid_num=st.integers(min_value=1, max_value=32))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_valid_fcsv_get_afid(
        self, valid_fcsv_file: PathLike[str], afid_num: int
    ):
        afid = get_afid(valid_fcsv_file, afid_num)

        # Check array type
        assert isinstance(afid, np.ndarray)
        # Check array values
        assert afid.dtype == np.single

    @given(afid_num=st.integers(min_value=-1000, max_value=1000))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_fcsv_get_afid(
        self,
        valid_fcsv_file: PathLike[str],
        afid_num: int,
    ):
        assume(afid_num < 1 or afid_num > 32)

        with pytest.raises(
            InvalidFiducialNumberError, match=".*is not valid."
        ):
            get_afid(valid_fcsv_file, afid_num)

    @given(afid_num=st.integers(min_value=1, max_value=32))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_valid_json_get_afid(
        self, valid_json_file: PathLike[str], afid_num: int
    ):
        afid = get_afid(valid_json_file, afid_num)

        # Check array type
        assert isinstance(afid, np.ndarray)
        # Check array values
        assert afid.dtype == np.single

    @given(afid_num=st.integers(min_value=-1000, max_value=1000))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_json_get_afid(
        self,
        valid_json_file: PathLike[str],
        afid_num: int,
    ):
        assume(afid_num < 1 or afid_num > 32)

        with pytest.raises(
            InvalidFiducialNumberError, match=".*is not valid."
        ):
            get_afid(valid_json_file, afid_num)

    @given(
        afid_num=st.integers(min_value=1, max_value=32),
        ext=st.text(
            min_size=2,
            max_size=5,
            alphabet=st.characters(blacklist_categories=("P", "Nd")),
        ),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_extension(
        self, valid_fcsv_file: PathLike[str], afid_num: int, ext: str
    ):
        assume(not ext == "fcsv" or not ext == "json")
        invalid_file_ext = valid_fcsv_file.with_suffix(f".{ext}")

        with pytest.raises(IOError, match="Unexpected file extension"):
            get_afid(invalid_file_ext, afid_num)


class TestAfidsToFcsv:
    @given(afids_coords=afid_coords())
    def test_invalid_template(self, afids_coords: NDArray[np.single]) -> None:
        with pytest.raises(FileNotFoundError):
            afids_to_fcsv(
                afids_coords,
                "/invalid/fcsv/path",
            )

    @given(afids_coords=afid_coords(bad_range=True))
    def test_invalid_num_afids(self, afids_coords: NDArray[np.single]) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="sub-test_desc-", suffix="_afids.fcsv"
        ) as out_fcsv_file:
            with pytest.raises(TypeError) as err:
                afids_to_fcsv(afids_coords, out_fcsv_file)

            assert "AFIDs, but received" in str(err.value)

    @given(afids_coords=afid_coords(bad_dims=True))
    def test_invalid_afids_dims(
        self, afids_coords: NDArray[np.single]
    ) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="sub-test_desc-", suffix="_afids.fcsv"
        ) as out_fcsv_file:
            with pytest.raises(TypeError) as err:
                afids_to_fcsv(afids_coords, out_fcsv_file)

            assert "Expected 3 spatial dimensions" in str(err.value)

    @given(afids_coords=afid_coords())
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_write_fcsv(
        self, afids_coords: NDArray[np.single], valid_fcsv_file: PathLike[str]
    ) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="sub-test_desc-", suffix="_afids.fcsv"
        ) as out_fcsv_file:
            # Create and check output file
            afids_to_fcsv(afids_coords, out_fcsv_file.name)

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
