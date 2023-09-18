"""Methods for computing various metrics pertaining to AFIDs"""
from __future__ import annotations

from afids_utils.afids import AfidPosition, AfidSet


def compute_mean_afid_sets(afid_sets: list[AfidSet]) -> AfidSet:
    """Compute the mean spatial coordinates of corresponding AFIDs across a
    list of ``AfidSet`` objects.

    Parameters
    ----------
    afid_sets
        List of AfidSets to compute mean from

    Returns
    -------
    AfidSet
        Object containing mean spatial components for each AFID

    Raises
    ------
    ValueError
        If there are different coordinate systems in provided list of
        ``AfidSet`` objects
    """
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
