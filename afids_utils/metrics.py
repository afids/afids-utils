"""Methods for computing various metrics pertaining to AFIDs"""
from __future__ import annotations

import statistics as stats

from afids_utils.afids import AfidDistanceSet, AfidPosition, AfidSet


def mean_afid_sets(afid_sets: list[AfidSet]) -> AfidSet:
    """Calculate the average spatial coordinates for corresponding AFIDs
    within a list of ``AfidSet`` objects.

    Parameters
    ----------
    afid_sets
        List of ``AfidSet`` to compute mean from

    Returns
    -------
    AfidSet
        Object containing mean spatial components for each AFID

    Raises
    ------
    ValueError
        If input list does not consist of all ``AfidSet`` objects or if there
        are different coordinate systems in provided list of ``AfidSet``
        objects
    """
    # Check if coordinate systems are all the same
    if not all(
        afid_set.coord_system == afid_sets[0].coord_system
        for afid_set in afid_sets
    ):
        raise ValueError(
            "Mismatched coordinate system in provided list of AfidSet"
        )

    mean_afid_set = AfidSet(
        slicer_version="Unknown",
        coord_system=afid_sets[0].coord_system,
        afids=[
            AfidPosition(
                label=afid.label,
                x=stats.mean(
                    [afid_set.afids[idx].x for afid_set in afid_sets]
                ),
                y=stats.mean(
                    [afid_set.afids[idx].y for afid_set in afid_sets]
                ),
                z=stats.mean(
                    [afid_set.afids[idx].z for afid_set in afid_sets]
                ),
                desc=afid.desc,
            )
            for idx, afid in enumerate(afid_sets[0].afids)
        ],
    )

    return mean_afid_set


def mean_distances(
    afid_sets: list[AfidSet],
    template_afid_set: AfidSet,
    component: str = "distance",
) -> list[float]:
    """Calculate the average distance for a given spatial component betweeen
    a collection of ``AfidSet`` objects and a common / template ``AfidSet``.

    Parameters
    ----------
    afid_sets
        List of ``AfidSet`` objects to compute distances with

    template_afid_set
        Template / common ``AfidSet`` to compute distances against

    component
        Spatial component to compute - if "distance" will compute Euclidean
        distance (default: "distance")

    Returns
    -------
    list[float]
        List of average distances along each spatial component and Euclidean
        distance

    """
    # Compute distances
    afid_distance_sets = [
        AfidDistanceSet(afid_set1=afid_set, afid_set2=template_afid_set)
        for afid_set in afid_sets
    ]

    # Compute mean distance for each AFID
    mean_component = [
        stats.mean(
            [
                afid_distance_set.afids[idx].get(component)
                for afid_distance_set in afid_distance_sets
            ]
        )
        for idx in range(len(afid_distance_sets[0].afids))
    ]

    return mean_component
