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
    return draw(
        arrays(
            shape=(32, 3),
            dtype=np.single,
            elements=st.floats(
                min_value=min_value, max_value=max_value, width=width
            ),
        )
    )


@st.composite
def world_coord(
    draw: st.DrawFn,
    min_value: float = -50.0,
    max_value: float = 50.0,
    width: int = 16,
) -> NDArray[np.single]:
    return draw(
        arrays(
            shape=(3, 1),
            dtype=np.single,
            elements=st.floats(
                min_value=min_value, max_value=max_value, width=width
            ),
        )
    )


@st.composite
def affine_xfm(
    draw: st.DrawFn,
) -> NDArray[np.single]:
    affine = np.eye(4)

    scale = affine.copy()
    for i in range(3):
        scale[i, i] = draw(st.floats(min_value=0.1, max_value=10))

    rotation_vals = draw(
        arrays(
            shape=(3),
            dtype=np.float_,
            elements=st.floats(min_value=-np.pi, max_value=np.pi),
        )
    )
    rotation_x = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rotation_vals[0]), -np.sin(rotation_vals[0]), 0],
            [0, np.sin(rotation_vals[0]), np.cos(rotation_vals[0]), 0],
            [0, 0, 0, 1],
        ]
    )
    rotation_y = np.array(
        [
            [np.cos(rotation_vals[1]), 0, np.sin(rotation_vals[1]), 0],
            [0, 1, 0, 0],
            [-np.sin(rotation_vals[1]), 0, np.cos(rotation_vals[1]), 0],
            [0, 0, 0, 1],
        ]
    )
    rotation_z = np.array(
        [
            [np.cos(rotation_vals[2]), -np.sin(rotation_vals[2]), 0, 0],
            [np.sin(rotation_vals[2]), np.cos(rotation_vals[2]), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    rotation = rotation_x @ rotation_y @ rotation_z

    translation = affine.copy()
    translation[:3, 3] = draw(
        arrays(
            shape=(3),
            dtype=np.float_,
            elements=st.floats(min_value=-50.0, max_value=50.0),
        )
    )

    # Create mapping for permutations
    matrices = {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
    }

    # Compose affine matrix
    order = draw(st.permutations(["scale", "rotation", "translation"]))
    for xfm in order:
        affine = matrices[xfm] @ affine

    return affine
