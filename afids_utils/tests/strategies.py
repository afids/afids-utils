from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray


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


def world_coord(
    min_value: float = -50.0, max_value: float = 50.0, width: int = 16
) -> NDArray[np.single]:
    return arrays(
        shape=(3, 1),
        dtype=np.single,
        elements=st.floats(
            min_value=min_value, max_value=max_value, width=width
        ),
    )
