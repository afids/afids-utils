from __future__ import annotations

import json
from importlib import resources
from itertools import chain
from typing import Literal

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays  # type: ignore
from numpy.typing import NDArray

from afids_utils.afids import AfidPosition, AfidSet, AfidVoxel

with resources.open_text(
    "afids_utils.resources", "afids_descs.json"
) as json_fpath:
    HUMAN_PROTOCOL_MAP = json.load(json_fpath)["human"]


def short_ascii_text():
    return st.text(
        min_size=2,
        max_size=5,
        alphabet=st.characters(min_codepoint=ord("A"), max_codepoint=ord("z")),
    )


def valid_labels():
    return st.integers(min_value=1, max_value=32)


# Constrain coordinates to be within 180mm or 200 voxels in any direction
def valid_position_coords():
    return st.floats(
        allow_nan=False, min_value=-90, max_value=90, allow_infinity=False
    )


def valid_voxel_coords():
    return st.integers(min_value=0, max_value=200)


@st.composite
def labels_with_mismatched_descs(draw: st.DrawFn):
    label = draw(valid_labels())
    desc = draw(
        st.text().filter(
            lambda desc: desc
            not in [
                HUMAN_PROTOCOL_MAP[label - 1]["desc"],
                HUMAN_PROTOCOL_MAP[label - 1]["acronym"],
            ]
        )
    )
    return label, desc


@st.composite
def afid_positions(draw: st.DrawFn, label: int | None = None) -> AfidPosition:
    if not label:
        label = draw(valid_labels())
    x, y, z = (draw(valid_position_coords()) for _ in range(3))
    desc = HUMAN_PROTOCOL_MAP[label - 1]["desc"]
    return AfidPosition(label=label, x=x, y=y, z=z, desc=desc)


@st.composite
def afid_voxels(draw: st.DrawFn, label: int | None = None) -> AfidVoxel:
    if not label:
        label = draw(valid_labels())
    i, j, k = (draw(valid_voxel_coords()) for _ in range(3))
    desc = HUMAN_PROTOCOL_MAP[label - 1]["desc"]
    return AfidVoxel(label=label, i=i, j=j, k=k, desc=desc)


@st.composite
def position_lists(
    draw: st.DrawFn, unique: bool = True, complete: bool = True
) -> list[AfidPosition]:
    if unique and complete:
        labels = range(1, 33)
    else:
        values = (
            range(1, 33)
            if complete
            else draw(
                st.lists(valid_labels(), min_size=1, max_size=31, unique=True)
            )
        )
        multiples = [
            1 if unique else draw(st.integers(min_value=1, max_value=3))
            for _ in values
        ]
        assume(unique or (not all(multiple == 1 for multiple in multiples)))
        labels = chain(
            *[[value] * multiple for value, multiple in zip(values, multiples)]
        )

    return [
        draw(afid_positions(label=label))
        for label in draw(st.permutations(list(labels)))
    ]


@st.composite
def afid_sets(
    draw: st.DrawFn,
    min_value: float = -50.0,
    max_value: float = 50.0,
    width: Literal[16, 32, 64] = 16,
    randomize_header: bool = True,
) -> AfidSet:
    slicer_version = draw(st.from_regex(r"\d+\.\d+"))
    coord_system = draw(st.sampled_from(["RAS", "LPS"]))

    # Set (in)valid number of Afid coordinates in a list
    afid_pos: list[AfidPosition] = []
    num_afids = 32
    # Load expected mappings
    with resources.open_text(
        "afids_utils.resources", "afids_descs.json"
    ) as json_fpath:
        mappings = json.load(json_fpath)
    for afid in range(num_afids):
        afid_pos.append(
            AfidPosition(
                label=afid + 1,
                x=draw(
                    st.floats(
                        min_value=min_value, max_value=max_value, width=width
                    )
                ),
                y=draw(
                    st.floats(
                        min_value=min_value, max_value=max_value, width=width
                    )
                ),
                z=draw(
                    st.floats(
                        min_value=min_value, max_value=max_value, width=width
                    )
                ),
                desc=mappings["human"][afid]["desc"]
                if afid < 32
                else "Unknown",
            )
        )

    # Create AfidSet
    st_afid_set = AfidSet(
        slicer_version=slicer_version if randomize_header else "4.6",
        coord_system=coord_system if randomize_header else "RAS",
        afids=afid_pos,
    )

    return st_afid_set


@st.composite
def affine_xfms(
    draw: st.DrawFn,
) -> NDArray[np.single]:
    affine = np.eye(4)

    scale = affine.copy()
    for i in range(3):
        scale[i, i] = draw(st.floats(min_value=0.1, max_value=10))

    rotation_vals = draw(  # type: ignore
        arrays(  # type: ignore
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
        arrays(  # type: ignore
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
