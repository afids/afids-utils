"""Methods for computing various metrics pertaining to AFIDs"""
from __future__ import annotations

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
    # Check to make sure input datatype is correct
    if not all(list(map(lambda x: isinstance(x, AfidSet), afid_sets))):
        raise ValueError("Input is not a collection of AfidSet objects")

    # Check if coordinate systems are all the same
    if not all(
        afid_set.coord_system == afid_sets[0].coord_system
        for afid_set in afid_sets
    ):
        raise ValueError(
            "Mismatched coordinate system in provided list of AfidSet"
        )

    num_sets = len(afid_sets)
    mean_afid_set = AfidSet(
        slicer_version="Unknown",
        coord_system=afid_sets[0].coord_system,
        afids=[
            AfidPosition(
                label=afid_sets[0].afids[idx].label,
                x=sum(afid_set.afids[idx].x for afid_set in afid_sets)
                / num_sets,
                y=sum(afid_set.afids[idx].y for afid_set in afid_sets)
                / num_sets,
                z=sum(afid_set.afids[idx].z for afid_set in afid_sets)
                / num_sets,
                desc=afid_sets[0].afids[idx].desc,
            )
            for idx in range(len(afid_sets[0].afids))
        ],
    )

    return mean_afid_set


def mean_distances(
    afid_distance_sets: list[AfidDistanceSet],
) -> list[float]:
    """Calculate the average distance from a collection of
    ``AfidDistanceSet`` objects. Ensure that one of the ``AfidSet`` objects
    used to compute each ``AfidDistanceSet`` is consistent across all sets in
    the list.

    Parameters
    ----------
    afid_distance_sets
        List of ``AfidDistanceSet`` objects to compute mean distance from

    Returns
    -------
    list[float]
        Dictionary object describing average spatial component and Euclidean
        distances

    Raises
    ------
    ValueError
        If no single common ``AfidSet`` used to compute ``AfidDistance`` or
        list does not consist of all ``AfidDistanceSet`` objects
    """
    # Check to make sure all input types are correct
    if not all(
        list(map(lambda x: isinstance(x, AfidDistanceSet), afid_distance_sets))
    ):
        raise ValueError(
            "Input is not a collection of AfidDistanceSet objects"
        )

    # Check for common AfidSet
    if afid_distance_sets[0].afid_set1 in [
        afid_distance_sets[1].afid_set1,
        afid_distance_sets[1].afid_set2,
    ]:
        common_set = afid_distance_sets[0].afid_set1
    elif afid_distance_sets[0].afid_set2 in [
        afid_distance_sets[1].afid_set1,
        afid_distance_sets[1].afid_set2,
    ]:
        common_set = afid_distance_sets[0].afid_set2
    else:
        raise ValueError(
            "No single common AfidSet found within AfidDistanceSet objects"
        )

    for idx in range(2, len(afid_distance_sets)):
        if common_set not in [
            afid_distance_sets[idx].afid_set1,
            afid_distance_sets[idx].afid_set2,
        ]:
            raise ValueError(
                "No single common AfidSet found within AfidDistanceSet objects"
            )

    # Compute mean distance for each AFID
    num_pairs = len(afid_distance_sets)
    mean_distance = [
        sum(
            afid_distance_set.afids[idx].distance
            for afid_distance_set in afid_distance_sets
        )
        / num_pairs
        for idx in range(len(afid_distance_sets[0].afids))
    ]

    return mean_distance
