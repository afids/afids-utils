"""
This type stub file was generated by pyright.
"""

from nibabel.onetime import auto_attr
from nilearn.glm.model import LikelihoodModelResults

"""Implement some standard regression models: OLS and WLS \
models, as well as an AR(p) regression model.

Models are specified with a design matrix and are fit using their
'fit' method.

Subclasses that have more complicated covariance matrices
should write over the 'whiten' method as the fit method
prewhitens the response by calling 'whiten'.

General reference for regression models:

'Introduction to Linear Regression Analysis', Douglas C. Montgomery,
    Elizabeth A. Peck, G. Geoffrey Vining. Wiley, 2006.

"""
__docformat__ = ...
class OLSModel:
    """A simple ordinary least squares model.

    Parameters
    ----------
    design : array-like
        This is your design matrix.  Data are assumed to be column ordered
        with observations in rows.

    Methods
    -------
    model.__init___(design)
    model.logL(b=self.beta, Y)

    Attributes
    ----------
    design : ndarray
        This is the design, or X, matrix.

    whitened_design : ndarray
        This is the whitened design matrix.
        `design` == `whitened_design` by default for the OLSModel,
        though models that inherit from the OLSModel will whiten the design.

    calc_beta : ndarray
        This is the Moore-Penrose pseudoinverse of the whitened design matrix.

    normalized_cov_beta : ndarray
        ``np.dot(calc_beta, calc_beta.T)``

    df_residuals : scalar
        Degrees of freedom of the residuals.  Number of observations less the
        rank of the design.

    df_model : scalar
        Degrees of freedome of the model.  The rank of the design.

    """
    def __init__(self, design) -> None:
        """Construct instance.

        Parameters
        ----------
        design : array-like
            This is your design matrix.
            Data are assumed to be column ordered with
            observations in rows.

        """
        ...
    
    def initialize(self, design): # -> None:
        """Construct instance."""
        ...
    
    def logL(self, beta, Y, nuisance=...):
        r"""Return the value of the loglikelihood function at beta.

        Given the whitened design matrix, the loglikelihood is evaluated
        at the parameter vector, beta, for the dependent variable, Y
        and the nuisance parameter, sigma :footcite:`Greene2003`.

        Parameters
        ----------
        beta : ndarray
            The parameter estimates.  Must be of length df_model.

        Y : ndarray
            The dependent variable

        nuisance : dict, optional
            A dict with key 'sigma', which is an optional estimate of sigma.
            If None, defaults to its maximum likelihood estimate
            (with beta fixed) as
            ``sum((Y - X*beta)**2) / n``, where n=Y.shape[0], X=self.design.

        Returns
        -------
        loglf : float
            The value of the loglikelihood function.

        Notes
        -----
        The log-Likelihood Function is defined as

        .. math::

            \ell(\beta,\sigma,Y)=
            -\frac{n}{2}\log(2\pi\sigma^2) - \|Y-X\beta\|^2/(2\sigma^2)

        The parameter :math:`\sigma` above is what is sometimes referred to
        as a nuisance parameter. That is, the likelihood is considered as a
        function of :math:`\beta`, but to evaluate it, a value of
        :math:`\sigma` is needed.

        If :math:`\sigma` is not provided,
        then its maximum likelihood estimate:

        .. math::

            \hat{\sigma}(\beta) = \frac{\text{SSE}(\beta)}{n}

        is plugged in. This likelihood is now a function of only :math:`\beta`
        and is technically referred to as a profile-likelihood.

        References
        ----------
        .. footbibliography::

        """
        ...
    
    def whiten(self, X):
        """Whiten design matrix.

        Parameters
        ----------
        X : array
            design matrix

        Returns
        -------
        whitened_X : array
            This matrix is the matrix whose pseudoinverse is ultimately
            used in estimating the coefficients. For OLSModel, it is
            does nothing. For WLSmodel, ARmodel, it pre-applies
            a square root of the covariance matrix to X.

        """
        ...
    
    def fit(self, Y): # -> RegressionResults:
        """Fit model to data `Y`.

        Full fit of the model including estimate of covariance matrix,
        (whitened) residuals and scale.

        Parameters
        ----------
        Y : array-like
            The dependent variable for the Least Squares problem.

        Returns
        -------
        fit : RegressionResults

        """
        ...
    


class ARModel(OLSModel):
    """A regression model with an AR(p) covariance structure.

    In terms of a LikelihoodModel, the parameters
    are beta, the usual regression parameters,
    and sigma, a scalar nuisance parameter that
    shows up as multiplier in front of the AR(p) covariance.

    """
    def __init__(self, design, rho) -> None:
        """Initialize AR model instance.

        Parameters
        ----------
        design : ndarray
            2D array with design matrix.

        rho : int or array-like
            If int, gives order of model, and initializes rho to zeros.  If
            ndarray, gives initial estimate of rho. Be careful as ``ARModel(X,
            1) != ARModel(X, 1.0)``.

        """
        ...
    
    def whiten(self, X): # -> NDArray[float64]:
        """Whiten a series of columns according to AR(p) covariance structure.

        Parameters
        ----------
        X : array-like of shape (n_features)
            Array to whiten.

        Returns
        -------
        whitened_X : ndarray
            X whitened with order self.order AR.

        """
        ...
    


class RegressionResults(LikelihoodModelResults):
    """Summarize the fit of a linear regression model.

    It handles the output of contrasts, estimates of covariance, etc.

    """
    def __init__(self, theta, Y, model, whitened_Y, whitened_residuals, cov=..., dispersion=..., nuisance=...) -> None:
        """See LikelihoodModelResults constructor.

        The only difference is that the whitened Y and residual values
        are stored for a regression model.

        """
        ...
    
    @auto_attr
    def residuals(self): # -> Any:
        """Residuals from the fit."""
        ...
    
    @auto_attr
    def normalized_residuals(self): # -> Any:
        """Residuals, normalized to have unit length.

        See :footcite:`Montgomery2006` and :footcite:`Davidson2004`.

        Notes
        -----
        Is this supposed to return "stanardized residuals,"
        residuals standardized
        to have mean zero and approximately unit variance?

        d_i = e_i / sqrt(MS_E)

        Where MS_E = SSE / (n - k)

        References
        ----------
        .. footbibliography::

        """
        ...
    
    @auto_attr
    def predicted(self): # -> Any:
        """Return linear predictor values from a design matrix."""
        ...
    
    @auto_attr
    def SSE(self):
        """Error sum of squares.

        If not from an OLS model this is "pseudo"-SSE.
        """
        ...
    
    @auto_attr
    def r_square(self): # -> Any:
        """Proportion of explained variance.

        If not from an OLS model this is "pseudo"-R2.
        """
        ...
    
    @auto_attr
    def MSE(self):
        """Return Mean square (error)."""
        ...
    


class SimpleRegressionResults(LikelihoodModelResults):
    """Contain only information of the model fit necessary \
    for contrast computation.

    Its intended to save memory when details of the model are unnecessary.

    """
    def __init__(self, results) -> None:
        """See LikelihoodModelResults constructor.

        The only difference is that the whitened Y and residual values
        are stored for a regression model.
        """
        ...
    
    def logL(self, Y):
        """Return the maximized log-likelihood."""
        ...
    
    def residuals(self, Y):
        """Residuals from the fit."""
        ...
    
    def normalized_residuals(self, Y):
        """Residuals, normalized to have unit length.

        See :footcite:`Montgomery2006` and :footcite:`Davidson2004`.

        Notes
        -----
        Is this supposed to return "stanardized residuals,"
        residuals standardized
        to have mean zero and approximately unit variance?

        d_i = e_i / sqrt(MS_E)

        Where MS_E = SSE / (n - k)

        References
        ----------
        .. footbibliography::

        """
        ...
    
    @auto_attr
    def predicted(self): # -> Any:
        """Return linear predictor values from a design matrix."""
        ...
    


