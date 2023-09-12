"""Methods for computing various metrics pertaining to AFIDs"""
from __future__ import annotations

from afids_utils.afids import AfidDistance, AfidPosition


def compute_distance(
    afid_position: AfidPosition, template_position: AfidPosition
) -> AfidDistance:
    """Compute distance between two AfidPositions, returning Euclidean
    distance, as well as distances along each spatial dimension

    Parameters
    ----------
    afid_position
        Input AfidPosition containing floating-point spatial coordinates
        (x, y, z)

    template_position
        Template AfidPosition to compute distance against

    Returns
    -------
    AfidDistance
        Object containing distances along each spatial dimension
        (x, y, z) and Euclidean distance respectively
    """
    # Compute distances
    x_dist, y_dist, z_dist = afid_position - template_position
    euc = (x_dist**2 + y_dist**2 + z_dist**2) ** 0.5

    return AfidDistance(
        label=afid_position.label,
        desc=afid_position.desc,
        x=x_dist,
        y=y_dist,
        z=z_dist,
        euc=euc,
    )
