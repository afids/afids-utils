"""Methods for computing various metrics pertaining to AFIDs"""
from __future__ import annotations

from afids_utils.afids import AfidPosition


def compute_AFLE(
    afid_position: AfidPosition, template_position: AfidPosition
) -> tuple[float, float, float, float]:
    """Compute distance errors along each spatial dimension and
    AFID localizaton error (AFLE)

    Parameters
    ----------
    afid_position
        Input AfidPosition containing floating-point spatial coordinates
        (x, y, z)

    template_position
        Template AfidPosition to compute AFLE against

    Returns
    -------
    tuple[float, float, float, float]
        Distance errors along each spatial dimension (x, y, z) and
        AFLE respectively
    """
    # Compute distances and AFLE
    x_dist, y_dist, z_dist = afid_position - template_position
    afle = (x_dist**2 + y_dist**2 + z_dist**2) ** 0.5

    return x_dist, y_dist, z_dist, afle
