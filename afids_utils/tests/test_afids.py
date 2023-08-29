from __future__ import annotations

import json
import re
import tempfile
from importlib import resources
from os import PathLike
from pathlib import Path

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from afids_utils.afids import AfidPosition, AfidSet
from afids_utils.exceptions import InvalidFiducialError, InvalidFileError
from afids_utils.tests.strategies import afid_set


@pytest.fixture
def valid_fcsv_file() -> PathLike[str]:
    return (
        Path(__file__).parent / "data" / "tpl-MNI152NLin2009cAsym_afids.fcsv"
    )


@pytest.fixture
def human_mappings() -> list[list[str] | str]:
    with resources.open_text(
        "afids_utils.resources", "afids_descs.json"
    ) as json_fpath:
        mappings = json.load(json_fpath)

    return mappings["human"]


class TestAfidsIO:
    @given(label=st.integers(min_value=0, max_value=31))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_valid_load(self, valid_fcsv_file: PathLike[str], label: int):
        # Load valid file to check internal types
        afids_set = AfidSet.load(valid_fcsv_file)

        # Check correct type created after loading
        assert isinstance(afids_set, AfidSet)

        # Check to make sure internal types are correct
        assert isinstance(afids_set.slicer_version, str)
        assert isinstance(afids_set.coord_system, str)
        assert isinstance(afids_set.afids, list)
        assert isinstance(afids_set.afids[label], AfidPosition)

    def test_invalid_fpath(self):
        with pytest.raises(FileNotFoundError, match=".*does not exist"):
            AfidSet.load("invalid/fpath.fcsv")

    @given(
        ext=st.text(
            min_size=2,
            max_size=5,
            alphabet=st.characters(
                min_codepoint=ord("A"), max_codepoint=ord("z")
            ),
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_ext(self, valid_fcsv_file: PathLike[str], ext: str):
        assume(not ext == "fcsv" or not ext == "json")

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="sub-test_desc-",
            suffix=f"_afids.{ext}",
        ) as invalid_file_ext:
            with pytest.raises(ValueError, match="Unsupported .* extension"):
                AfidSet.load(invalid_file_ext.name)

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
                AfidSet.load(out_fcsv_file.name)

    @given(
        label=st.integers(min_value=0, max_value=31),
        desc=st.text(
            min_size=2,
            max_size=5,
            alphabet=st.characters(
                min_codepoint=ord("A"), max_codepoint=ord("z")
            ),
        ),
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_desc(
        self,
        valid_fcsv_file: PathLike[str],
        human_mappings: list[list[str] | str],
        label: int,
        desc: str,
    ) -> None:
        assume(desc not in human_mappings[label])

        # Replace valid description with a mismatch
        with open(valid_fcsv_file) as valid_fcsv:
            fcsv_data = valid_fcsv.readlines()
            fcsv_data[label + 3] = fcsv_data[label + 3].replace(
                human_mappings[label][0], desc
            )

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
                InvalidFiducialError, match="Description for label.*"
            ):
                AfidSet.load(out_fcsv_file.name)

    def test_valid_save(self, valid_fcsv_file: PathLike[str]):
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="sub-test_desc-", suffix="_afids.fcsv"
        ) as out_fcsv_file:
            afids_set = AfidSet.load(valid_fcsv_file)
            afids_set.save(out_fcsv_file.name)

            assert Path(out_fcsv_file.name).exists()

    @given(
        ext=st.text(
            min_size=2,
            max_size=5,
            alphabet=st.characters(
                min_codepoint=ord("A"), max_codepoint=ord("z")
            ),
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_ext_save(self, valid_fcsv_file: PathLike[str], ext: str):
        assume(not ext == "fcsv" or not ext == "json")

        with tempfile.NamedTemporaryFile(
            mode="w", prefix="sub-test_desc-", suffix=f"_afids.{ext}"
        ) as out_file:
            afids_set = AfidSet.load(valid_fcsv_file)
            with pytest.raises(ValueError, match="Unsupported file extension"):
                afids_set.save(out_file.name)

    @given(afids_set=afid_set())
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_save_invalid_coord_system(self, afids_set: AfidSet):
        afids_set.coord_system = "invalid"

        with tempfile.NamedTemporaryFile(
            mode="w", prefix="sub-test_desc-", suffix="_afids.fcsv"
        ) as out_file:
            with pytest.raises(
                ValueError, match=".*invalid coordinate system"
            ):
                afids_set.save(out_file.name)

    @given(afids_set=afid_set(), coord_sys=st.sampled_from(["LPS", "RAS"]))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_update_coord_system(self, afids_set: AfidSet, coord_sys: str):
        afids_set.coord_system = coord_sys

        with tempfile.NamedTemporaryFile(
            mode="w", prefix="sub-test_desc-", suffix="_afids.fcsv"
        ) as out_file:
            afids_set.save(out_file.name)

            with open(
                out_file.name, "r", encoding="utf-8", newline=""
            ) as in_file:
                in_data = in_file.readlines()
                saved_header = in_data[:3]
                parsed_coord = re.split(r"\s", saved_header[1])[-2]

                if coord_sys == "LPS":
                    assert parsed_coord == "0"
                else:
                    assert parsed_coord == "1"


class TestAfidsCore:
    @given(label=st.integers(min_value=1, max_value=32))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_valid_get_afid(self, valid_fcsv_file: PathLike[str], label: int):
        afids_set = AfidSet.load(valid_fcsv_file)
        afid_pos = afids_set.get_afid(label)

        # Check array type
        assert isinstance(afid_pos, AfidPosition)

    @given(label=st.integers(min_value=-100, max_value=100))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_invalid_get_afid(
        self, valid_fcsv_file: PathLike[str], label: int
    ):
        afids_set = AfidSet.load(valid_fcsv_file)
        assume(not 1 <= label <= len(afids_set.afids))

        with pytest.raises(InvalidFiducialError, match=".*not valid"):
            afids_set.get_afid(label)
