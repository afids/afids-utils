from __future__ import annotations

from copy import deepcopy

import pytest
from hypothesis import given
from hypothesis import strategies as st

import afids_utils.metrics as af_metrics
import afids_utils.tests.strategies as af_st
from afids_utils.afids import AfidPosition, AfidSet


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


class TestMeanDistances:
    @given(
        afid_set=af_st.afid_sets(randomize_header=False),
        template_set=af_st.afid_sets(randomize_header=False),
        component=st.sampled_from(["x", "y", "z", "distance"]),
    )
    def test_valid_afid_sets(
        self,
        afid_set: AfidSet,
        template_set: AfidSet,
        component: str,
    ):
        mean_component = af_metrics.mean_distances(
            afid_sets=[afid_set, afid_set],
            template_afid_set=template_set,
            component=component,
        )

        # Check list objects
        assert isinstance(mean_component, list) and all(
            list(map(lambda dist: isinstance(dist, float), mean_component))
        )
        assert all(
            list(
                map(
                    lambda dist: float("-inf") < dist < float("inf"),
                    mean_component,
                )
            )
        )
