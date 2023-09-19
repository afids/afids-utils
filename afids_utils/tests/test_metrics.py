from __future__ import annotations

from copy import deepcopy

import pytest
from hypothesis import given

import afids_utils.metrics as af_metrics
import afids_utils.tests.strategies as af_st
from afids_utils.afids import AfidDistanceSet, AfidPosition, AfidSet


class TestMeanAfidSet:
    @given(
        afid_set1=af_st.afid_sets(randomize_header=False),
    )
    def test_mismatched_coords(self, afid_set1: AfidSet):
        # Manually set the coord system to mistmatch
        afid_set2 = deepcopy(afid_set1)
        afid_set2.coord_system = "LPS"

        with pytest.raises(ValueError, match=r"Mismatched coordinate.*"):
            af_metrics.mean_afid_sets([afid_set1, afid_set2])

    @given(
        afid_set1=af_st.afid_sets(randomize_header=False),
    )
    def test_valid_afid_sets(self, afid_set1: AfidSet):
        mean_afid_set = af_metrics.mean_afid_sets([afid_set1, afid_set1])

        # Check internals
        assert isinstance(mean_afid_set.afids, list) and all(
            list(
                map(
                    lambda afid: isinstance(afid, AfidPosition),
                    mean_afid_set.afids,
                )
            )
        )

    def test_invalid_input_type(self):
        with pytest.raises(ValueError, match=r".*collection of AfidSet.*"):
            af_metrics.mean_afid_sets(["random string"])


class TestMeanDistances:
    @given(
        afid_set1=af_st.afid_sets(randomize_header=False),
        afid_set2=af_st.afid_sets(randomize_header=False),
    )
    def test_valid_afid_sets(self, afid_set1: AfidSet, afid_set2: AfidSet):
        afid_distance_sets = [
            AfidDistanceSet(afid_set1, afid_set2),
            AfidDistanceSet(afid_set2, afid_set2),
        ]
        mean_distances = af_metrics.mean_distances(afid_distance_sets)

        # Check list objects
        assert isinstance(mean_distances, list) and all(
            list(map(lambda dist: isinstance(dist, float), mean_distances))
        )
        assert all(list(map(lambda dist: dist >= 0, mean_distances)))

    def test_invalid_input_type(self):
        with pytest.raises(
            ValueError, match=r".*collection of AfidDistanceSet.*"
        ):
            af_metrics.mean_distances(["random string"])

    @given(
        afid_set1=af_st.afid_sets(randomize_header=False),
        afid_set2=af_st.afid_sets(min_value=1.0, randomize_header=False),
    )
    def test_no_common_afid_sets_short(
        self, afid_set1: AfidSet, afid_set2: AfidSet
    ):
        afid_distance_sets = [
            AfidDistanceSet(afid_set1, afid_set1),
            AfidDistanceSet(afid_set2, afid_set2),
        ]

        with pytest.raises(ValueError, match=r"No single common AfidSet.*"):
            af_metrics.mean_distances(afid_distance_sets)

    @given(
        afid_set1=af_st.afid_sets(randomize_header=False),
        afid_set2=af_st.afid_sets(min_value=1.0, randomize_header=False),
    )
    def test_no_common_afid_sets_long(
        self,
        afid_set1: AfidSet,
        afid_set2: AfidSet,
    ):
        afid_distance_sets = [
            AfidDistanceSet(afid_set1, afid_set1),
            AfidDistanceSet(afid_set1, afid_set2),
            AfidDistanceSet(afid_set2, afid_set2),
        ]

        with pytest.raises(ValueError, match=r"No single common AfidSet.*"):
            af_metrics.mean_distances(afid_distance_sets)
