from __future__ import annotations

from hypothesis import given

import afids_utils.metrics as af_metrics
import afids_utils.tests.strategies as af_st
from afids_utils.afids import AfidDistance, AfidPosition


class TestMetrics:
    @given(pos1=af_st.afid_positions(), pos2=af_st.afid_positions())
    def test_compute_afle(self, pos1: AfidPosition, pos2: AfidPosition):
        res = af_metrics.compute_distance(pos1, pos2)
        assert isinstance(res, AfidDistance)
        # Check internals
        assert isinstance(res.x, float)
        assert isinstance(res.y, float)
        assert isinstance(res.z, float)
        assert isinstance(res.euc, float)
