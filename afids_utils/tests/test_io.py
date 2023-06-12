from __future__ import annotations

import csv
import tempfile
from os import PathLike, remove
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from afids_utils.exceptions import InvalidFiducialNumberError
from afids_utils.io import FCSV_FIELDNAMES, afids_to_fcsv, get_afid


@pytest.fixture
def valid_fcsv_file() -> PathLike[str]:
    return (
        Path(__file__).parent / "data" / "tpl-MNI152NLin2009cAsym_afids.fcsv"
    )


class TestGetAfid:
    @given(afid_num=st.integers(min_value=1, max_value=32))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_valid_num_get_afid(
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
    def test_invalid_num_get_afid(
        self,
        valid_fcsv_file: PathLike[str],
        afid_num: int,
    ):
        assume(afid_num < 1 or afid_num > 32)

        with pytest.raises(
            InvalidFiducialNumberError, match=".*is not valid."
        ):
            get_afid(valid_fcsv_file, afid_num)


@st.composite
def afid_coords(
    draw: st.DrawFn,
    min_value: float = -50.0,
    max_value: float = 50.0,
    width: int = 16,
) -> NDArray[np.single]:
    coords = draw(
        arrays(
            shape=(32, 3),
            dtype=np.single,
            elements=st.floats(
                min_value=min_value, max_value=max_value, width=width
            ),
        )
    )

    return coords


class TestAfidsToFcsv:
    @given(afids_coords=afid_coords())
    def test_invalid_template(self, afids_coords: NDArray[np.single]) -> None:
        with pytest.raises(FileNotFoundError):
            afids_to_fcsv(
                afids_coords, "/invalid/fcsv/path", "/random/output/path"
            )

    @given(afids_coords=afid_coords())
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_write_fcsv(
        self, afids_coords: NDArray[np.single], valid_fcsv_file: PathLike[str]
    ) -> None:
        out_fcsv_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, prefix="sub-test_afids.fcsv"
        )
        out_fcsv_path = Path(out_fcsv_file.name)

        afids_to_fcsv(afids_coords, out_fcsv_path)

        # Check file was created
        assert out_fcsv_path.exists()

        # Load files
        with open(
            valid_fcsv_file, "r", encoding="utf-8", newline=""
        ) as template_fcsv_file:
            template_header = [template_fcsv_file.readline() for _ in range(3)]

        with open(
            out_fcsv_path, "r", encoding="utf-8", newline=""
        ) as output_fcsv_file:
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

        # Delete temporary file
        remove(out_fcsv_path)
