from __future__ import annotations

from hypothesis import given

import afids_utils.metrics as af_metrics
import afids_utils.tests.strategies as af_st
from afids_utils.afids import AfidPosition


class TestMetrics:
    @given(
        pos1=af_st.afid_positions(label=1), pos2=af_st.afid_positions(label=1)
    )
    def test_compute_afle(self, pos1: AfidPosition, pos2: AfidPosition):
        res = af_metrics.compute_AFLE(pos1, pos2)
        assert isinstance(res, tuple) and list(map(type, res)) == [
            float,
            float,
            float,
            float,
        ]
