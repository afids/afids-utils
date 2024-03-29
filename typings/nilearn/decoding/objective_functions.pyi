"""
This type stub file was generated by pyright.
"""

"""Common functions and base classes."""
def spectral_norm_squared(X): # -> Any:
    """Compute square of the operator 2-norm (spectral norm) of X.

    This corresponds to the Lipschitz constant of the gradient of the
    squared-loss function:

        w -> .5 * ||y - Xw||^2

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
      Design matrix.

    Returns
    -------
    lipschitz_constant : float
      The square of the spectral norm of X.

    """
    ...

_squared_loss_grad = ...
