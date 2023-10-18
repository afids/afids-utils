"""
This type stub file was generated by pyright.
"""

from sklearn.base import BaseEstimator
from .._utils import fill_doc

"""The searchlight is a widely used approach for the study \
of the fine-grained patterns of information in fMRI analysis, \
in which multivariate statistical relationships are iteratively tested \
in the neighborhood of each location of a domain."""
ESTIMATOR_CATALOG = ...
@fill_doc
def search_light(X, y, estimator, A, groups=..., scoring=..., cv=..., n_jobs=..., verbose=...):
    """Compute a search_light.

    Parameters
    ----------
    X : array-like of shape at least 2D
        data to fit.

    y : array-like
        target variable to predict.

    estimator : estimator object implementing 'fit'
        object to use to fit the data

    A : scipy sparse matrix.
        adjacency matrix. Defines for each feature the neigbhoring features
        following a given structure of the data.

    groups : array-like, optional, (default None)
        group label for each sample for cross validation.

        .. note::
            This will have no effect for scikit learn < 0.18

    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        for possible values.
        If callable, it takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.
    %(n_jobs_all)s
    %(verbose0)s

    Returns
    -------
    scores : array-like of shape (number of rows in A)
        search_light scores
    """
    ...

@fill_doc
class GroupIterator:
    """Group iterator.

    Provides group of features for search_light loop
    that may be used with Parallel.

    Parameters
    ----------
    n_features : int
        Total number of features
    %(n_jobs)s

    """
    def __init__(self, n_features, n_jobs=...) -> None:
        ...
    
    def __iter__(self): # -> Generator[NDArray[signedinteger[Any]], Unknown, None]:
        ...
    


@fill_doc
class SearchLight(BaseEstimator):
    """Implement search_light analysis using an arbitrary type of classifier.

    Parameters
    ----------
    mask_img : Niimg-like object
        See :ref:`extracting_data`.
        Boolean image giving location of voxels containing usable signals.

    process_mask_img : Niimg-like object, optional
        See :ref:`extracting_data`.
        Boolean image giving voxels on which searchlight should be
        computed.

    radius : float, optional
        radius of the searchlight ball, in millimeters. Defaults to 2.

    estimator : 'svr', 'svc', or an estimator object implementing 'fit'
        The object to use to fit the data
    %(n_jobs)s
    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.
    %(verbose0)s

    Notes
    -----
    The searchlight [Kriegeskorte 06] is a widely used approach for the
    study of the fine-grained patterns of information in fMRI analysis.
    Its principle is relatively simple: a small group of neighboring
    features is extracted from the data, and the prediction function is
    instantiated on these features only. The resulting prediction
    accuracy is thus associated with all the features within the group,
    or only with the feature on the center. This yields a map of local
    fine-grained information, that can be used for assessing hypothesis
    on the local spatial layout of the neural code under investigation.

    Nikolaus Kriegeskorte, Rainer Goebel & Peter Bandettini.
    Information-based functional brain mapping.
    Proceedings of the National Academy of Sciences
    of the United States of America,
    vol. 103, no. 10, pages 3863-3868, March 2006
    """
    def __init__(self, mask_img, process_mask_img=..., radius=..., estimator=..., n_jobs=..., scoring=..., cv=..., verbose=...) -> None:
        ...
    
    def fit(self, imgs, y, groups=...): # -> Self@SearchLight:
        """Fit the searchlight.

        Parameters
        ----------
        imgs : Niimg-like object
            See :ref:`extracting_data`.
            4D image.

        y : 1D array-like
            Target variable to predict. Must have exactly as many elements as
            3D images in img.

        groups : array-like, optional
            group label for each sample for cross validation. Must have
            exactly as many elements as 3D images in img. default None
            NOTE: will have no effect for scikit learn < 0.18

        """
        ...
    

