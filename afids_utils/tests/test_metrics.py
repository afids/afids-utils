from __future__ import annotations

import pytest
from hypothesis import given

import afids_utils.metrics as af_metrics
import afids_utils.tests.strategies as af_st
from afids_utils.afids import AfidPosition, AfidSet


class TestMeanAfidSet:
    @given(
        afid_set1=af_st.afid_sets(randomize_header=False),
        afid_set2=af_st.afid_sets(randomize_header=False),
    )
    def test_mismatched_coords(self, afid_set1: AfidSet, afid_set2: AfidSet):
        # Manually set the coord system to mistmatch
        afid_set2.coord_system = "LPS"

        with pytest.raises(ValueError, match=r"Mismatched coordinate.*"):
            af_metrics.compute_mean_afid_sets([afid_set1, afid_set2])

    @given(
        afid_set1=af_st.afid_sets(randomize_header=False),
        afid_set2=af_st.afid_sets(randomize_header=False),
    )
    def test_valid_afid_sets(self, afid_set1: AfidSet, afid_set2: AfidSet):
        mean_afid_set = af_metrics.compute_mean_afid_sets(
            [afid_set1, afid_set2]
        )

        # Check internals
        assert isinstance(mean_afid_set.afids, list) and all(
            list(
                map(lambda x: isinstance(x, AfidPosition), mean_afid_set.afids)
            )
        )
