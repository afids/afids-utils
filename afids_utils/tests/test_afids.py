from __future__ import annotations

import json
import re
import tempfile
from importlib import resources
from pathlib import Path

import pytest
from hypothesis import assume, example, given
from hypothesis import strategies as st
from more_itertools import pairwise

import afids_utils.tests.strategies as af_st
from afids_utils.afids import (
    AfidDistance,
    AfidDistanceSet,
    AfidPosition,
    AfidSet,
)
from afids_utils.exceptions import InvalidFiducialError, InvalidFileError
from afids_utils.tests.helpers import allow_function_scoped, slow_generation


@pytest.fixture
def valid_file() -> Path:
    return (
        Path(__file__).parent / "data" / "tpl-MNI152NLin2009cAsym_afids.fcsv"
    )


@pytest.fixture
def human_mappings() -> list[dict[str, str]]:
    with resources.open_text(
        "afids_utils.resources", "afids_descs.json"
    ) as json_fpath:
        mappings = json.load(json_fpath)

    return mappings["human"]


class TestAfidPosition:
    @given(pos=af_st.afid_positions())
    def test_valid_position(self, pos: AfidPosition):
        """Just checks that a hypothesis-generated AfidsPosition inits."""

    @given(
        label=st.integers().filter(lambda label: label not in range(1, 33)),
        x=af_st.valid_position_coords(),
        y=af_st.valid_position_coords(),
        z=af_st.valid_position_coords(),
        desc=st.text(),
    )
    def test_invalid_label(
        self, label: int, x: int, y: int, z: int, desc: str
    ):
        with pytest.raises(ValueError, match=r".*must be in.*"):
            AfidPosition(label=label, x=x, y=y, z=z, desc=desc)

    @given(
        label_with_desc=af_st.labels_with_mismatched_descs(),
        x=af_st.valid_position_coords(),
        y=af_st.valid_position_coords(),
        z=af_st.valid_position_coords(),
    )
    def test_mismatched_desc(
        self, label_with_desc: tuple[int, str], x: int, y: int, z: int
    ):
        label, desc = label_with_desc
        with pytest.raises(
            InvalidFiducialError, match=r".*does not correspond.*"
        ):
            AfidPosition(label=label, x=x, y=y, z=z, desc=desc)


class TestAfidSet:
    @given(
        slicer_version=st.from_regex(r"\d\.\d+"),
        coord_system=st.sampled_from(["RAS", "LPS", "0", "1"]),
        positions=af_st.position_lists(),
    )
    @slow_generation
    def test_valid_afid_set(
        self,
        slicer_version: str,
        coord_system: str,
        positions: list[AfidPosition],
    ):
        afid_set = AfidSet(
            slicer_version=slicer_version,
            coord_system=coord_system,
            afids=positions,
        )
        # Check that afids are sorted
        for first, second in pairwise(afid_set.afids):
            assert first.label <= second.label

    @given(
        slicer_version=st.from_regex(r"\d\.\d+"),
        coord_system=st.sampled_from(["RAS", "LPS", "0", "1"]),
        positions=af_st.position_lists(complete=False),
    )
    def test_incomplete_afid_set(
        self,
        slicer_version: str,
        coord_system: str,
        positions: list[AfidPosition],
    ):
        with pytest.raises(ValueError, match=r"Incorrect number.*"):
            AfidSet(
                slicer_version=slicer_version,
                coord_system=coord_system,
                afids=positions,
            )

    @given(
        slicer_version=st.from_regex(r"\d\.\d+"),
        coord_system=st.sampled_from(["RAS", "LPS", "0", "1"]),
        positions=af_st.position_lists(unique=False),
    )
    @slow_generation
    def test_repeated_afid_set(
        self,
        slicer_version: str,
        coord_system: str,
        positions: list[AfidPosition],
    ):
        with pytest.raises(ValueError, match=r"Incorrect number.*"):
            AfidSet(
                slicer_version=slicer_version,
                coord_system=coord_system,
                afids=positions,
            )

    @given(
        slicer_version=st.from_regex(r"\d\.\d+"),
        coord_system=st.sampled_from(["RAS", "LPS", "0", "1"]),
        positions=af_st.position_lists(unique=False, complete=False),
    )
    @example(  # ensure case of 32 AFIDs with repeats gets hit
        slicer_version="1.0",
        coord_system="RAS",
        positions=[
            AfidPosition(desc="AC", label=1, x=1, y=1, z=1) for _ in range(32)
        ],
    )
    def test_repeated_incomplete_afid_set(
        self,
        slicer_version: str,
        coord_system: str,
        positions: list[AfidPosition],
    ):
        with pytest.raises(
            ValueError,
            match=r"(?:Incorrect number.*)|(?:.*incorrect labels.*)",
        ):
            AfidSet(
                slicer_version=slicer_version,
                coord_system=coord_system,
                afids=positions,
            )


class TestAfidsIO:
    @given(label=af_st.valid_labels(), ext=st.sampled_from(["fcsv", "json"]))
    @allow_function_scoped
    def test_valid_load(self, valid_file: Path, label: int, ext: str):
        # Load valid file to check internal types
        afids_set = AfidSet.load(valid_file.with_suffix(f".{ext}"))

        # Check correct type created after loading
        assert isinstance(afids_set, AfidSet)

        # Check to make sure internal types are correct
        assert isinstance(afids_set.slicer_version, str)
        assert isinstance(afids_set.coord_system, str)
        assert isinstance(afids_set.afids, list)
        assert isinstance(afids_set.afids[label - 1], AfidPosition)

    def test_invalid_fpath(self):
        with pytest.raises(FileNotFoundError, match=".*does not exist"):
            AfidSet.load("invalid/fpath.fcsv")

    @given(ext=af_st.short_ascii_text())
    def test_invalid_ext(self, ext: str):
        assume(ext not in {"fcsv", "json"})

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="sub-test_desc-",
            suffix=f"_afids.{ext}",
        ) as invalid_file_ext:
            with pytest.raises(ValueError, match="Unsupported .* extension"):
                AfidSet.load(invalid_file_ext.name)

    def test_invalid_label_range(self, valid_file: Path):
        # Create additional line of fiducials
        with open(valid_file) as valid_fcsv:
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

    @given(label=af_st.valid_labels(), desc=af_st.short_ascii_text())
    @allow_function_scoped
    def test_invalid_desc(
        self,
        valid_file: Path,
        human_mappings: list[dict[str, str]],
        label: int,
        desc: str,
    ) -> None:
        assume(
            desc
            not in [
                human_mappings[label - 1]["desc"],
                human_mappings[label - 1]["acronym"],
            ]
        )

        # Replace valid description with a mismatch
        with open(valid_file) as valid_fcsv:
            fcsv_data = valid_fcsv.readlines()
            fcsv_data[label + 2] = fcsv_data[label + 2].replace(
                human_mappings[label - 1]["desc"], desc
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
                InvalidFiducialError, match=r".*does not correspond to label.*"
            ):
                AfidSet.load(out_fcsv_file.name)

    @given(ext=st.sampled_from(["fcsv", "json"]))
    @allow_function_scoped
    def test_valid_save(self, valid_file: Path, ext: str):
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="sub-test_desc-", suffix=f"_afids.{ext}"
        ) as out_fcsv_file:
            afids_set = AfidSet.load(valid_file.with_suffix(f".{ext}"))
            afids_set.save(out_fcsv_file.name)

            assert Path(out_fcsv_file.name).exists()

    @given(
        ext=st.sampled_from(["fcsv", "json"]),
        invalid_ext=af_st.short_ascii_text(),
    )
    @allow_function_scoped
    def test_invalid_ext_save(
        self, valid_file: Path, ext: str, invalid_ext: str
    ):
        assume(invalid_ext not in {"fcsv", "json"})

        with tempfile.NamedTemporaryFile(
            mode="w", prefix="sub-test_desc-", suffix=f"_afids.{invalid_ext}"
        ) as out_file:
            afid_set = AfidSet.load(valid_file.with_suffix(f".{ext}"))
            with pytest.raises(ValueError, match="Unsupported file extension"):
                afid_set.save(out_file.name)

    @given(afid_set=af_st.afid_sets(), ext=st.sampled_from(["fcsv", "json"]))
    def test_save_invalid_coord_system(self, afid_set: AfidSet, ext: str):
        afid_set.coord_system = "invalid"

        with tempfile.NamedTemporaryFile(
            mode="w", prefix="sub-test_desc-", suffix=f"_afids.{ext}"
        ) as out_file:
            with pytest.raises(
                ValueError, match=".*invalid coordinate system"
            ):
                afid_set.save(out_file.name)

    @given(
        afid_set=af_st.afid_sets(), coord_sys=st.sampled_from(["RAS", "LPS"])
    )
    def test_update_coord_system(self, afid_set: AfidSet, coord_sys: str):
        afid_set.coord_system = coord_sys

        with tempfile.NamedTemporaryFile(
            mode="w", prefix="sub-test_desc-", suffix="_afids.fcsv"
        ) as out_file:
            afid_set.save(out_file.name)

            with open(
                out_file.name, "r", encoding="utf-8", newline=""
            ) as in_file:
                in_data = in_file.readlines()
                saved_header = in_data[:3]
                parsed_coord = re.split(r"\s", saved_header[1])[-2]

                if coord_sys == "RAS":
                    assert parsed_coord == "0"
                else:
                    assert parsed_coord == "1"


class TestAfidsCore:
    @given(label=af_st.valid_labels())
    @allow_function_scoped
    def test_valid_get_afid(self, valid_file: Path, label: int):
        afid_set = AfidSet.load(valid_file)
        afid_pos = afid_set.get_afid(label)

        # Check array type
        assert isinstance(afid_pos, AfidPosition)

    @given(label=st.integers(min_value=-100, max_value=100))
    @allow_function_scoped
    def test_invalid_get_afid(self, valid_file: Path, label: int):
        afid_set = AfidSet.load(valid_file)
        assume(not 1 <= label <= len(afid_set.afids))

        with pytest.raises(InvalidFiducialError, match=".*not valid"):
            afid_set.get_afid(label)


class TestAfidsDistance:
    @given(
        afid1=af_st.afid_positions(label=1),
        afid2=af_st.afid_positions(label=1),
    )
    def test_same_labels(self, afid1: AfidPosition, afid2: AfidPosition):
        afid_distance = AfidDistance(
            afid_position1=afid1, afid_position2=afid2
        )

        # Check output and properties
        assert isinstance(afid_distance, AfidDistance)
        assert isinstance(afid_distance.x, float)
        assert isinstance(afid_distance.z, float)
        assert isinstance(afid_distance.y, float)
        assert (
            isinstance(afid_distance.distance, float)
            and afid_distance.distance >= 0
        )

    @given(
        afid1=af_st.afid_positions(),
        afid2=af_st.afid_positions(),
    )
    def test_diff_labels(self, afid1: AfidPosition, afid2: AfidPosition):
        assume(afid1.label != afid2.label)

        with pytest.warns(UserWarning, match=r".*non-corresponding AFIDs"):
            AfidDistance(afid_position1=afid1, afid_position2=afid2)

    @given(
        afid1=af_st.afid_positions(label=1),
        afid2=af_st.afid_positions(label=1),
        component=st.sampled_from(["x", "y", "z", "distance"]),
    )
    def test_get_valid_component(
        self, afid1: AfidPosition, afid2: AfidPosition, component: str
    ):
        afid_distance = AfidDistance(
            afid_position1=afid1, afid_position2=afid2
        )
        # Check to make sure get works and a value is returned
        assert float("-inf") < afid_distance.get(component) < float("inf")

    @given(
        afid1=af_st.afid_positions(label=1),
        afid2=af_st.afid_positions(label=1),
    )
    def test_get_invalid_component(
        self,
        afid1: AfidPosition,
        afid2: AfidPosition,
    ):
        afid_distance = AfidDistance(
            afid_position1=afid1, afid_position2=afid2
        )

        with pytest.raises(ValueError, match="Invalid component"):
            afid_distance.get("invalid_component")


class TestAfidsDistanceSet:
    @given(
        afid_set1=af_st.afid_sets(randomize_header=False),
        afid_set2=af_st.afid_sets(randomize_header=False),
    )
    def test_valid_afid_set(self, afid_set1: AfidSet, afid_set2: AfidSet):
        afid_distance_set = AfidDistanceSet(
            afid_set1=afid_set1, afid_set2=afid_set2
        )

        # Check afids property is correct (list[AfidDistances])
        assert isinstance(afid_distance_set.afids, list) and all(
            list(
                map(
                    lambda x: isinstance(x, AfidDistance),
                    afid_distance_set.afids,
                )
            )
        )

    @given(
        afid_set1=af_st.afid_sets(randomize_header=False),
        afid_set2=af_st.afid_sets(randomize_header=False),
    )
    @slow_generation
    def test_mismatched_coords(self, afid_set1: AfidSet, afid_set2: AfidSet):
        # Manually mismatch the coord system due to slow data generation
        afid_set2.coord_system = "LPS"

        with pytest.raises(ValueError, match=r"Mismatched coord.*"):
            AfidDistanceSet(afid_set1, afid_set2).afids
