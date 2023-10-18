"""
This type stub file was generated by pyright.
"""

from nilearn._utils import fill_doc
from ._base import _BaseDecomposition

"""Dictionary learning estimator.

Perform a map learning algorithm by learning
a temporal dense dictionary along with sparse spatial loadings, that
constitutes output maps
"""
sparse_encode_args = ...
@fill_doc
class DictLearning(_BaseDecomposition):
    """Perform a map learning algorithm based on spatial component sparsity, \
    over a :term:`CanICA` initialization.

    This yields more stable maps than :term:`CanICA`.

    See :footcite:`Mensch2016`.

     .. versionadded:: 0.2

    Parameters
    ----------
    mask : Niimg-like object or MultiNiftiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    n_components : int, optional
        Number of components to extract. Default=20.

    batch_size : int, optional
        The number of samples to take in each batch. Default=20.

    n_epochs : float, optional
        Number of epochs the algorithm should run on the data. Default=1.

    alpha : float, optional
        Sparsity controlling parameter. Default=10.

    dict_init : Niimg-like object, optional
        Initial estimation of dictionary maps. Would be computed from CanICA if
        not provided.

    reduction_ratio : 'auto' or float between 0. and 1., optional
        - Between 0. or 1. : controls data reduction in the temporal domain.
          1. means no reduction, < 1. calls for an SVD based reduction.
        - if set to 'auto', estimator will set the number of components per
          reduced session to be n_components. Default='auto'.

    method : {'cd', 'lars'}, optional
        Coding method used by sklearn backend. Below are the possible values.
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.
        Default='cd'.

    random_state : int or RandomState, optional
        Pseudo number generator state used for random sampling.
    %(smoothing_fwhm)s
        Default=4mm.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension. Default=True.

    detrend : boolean, optional
        If detrend is True, the time-series will be detrended before
        components extraction. Default=True.

    target_affine : 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape : 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    low_pass : None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    high_pass : None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    t_r : float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    %(mask_strategy)s

        .. note::
             Depending on this value, the mask will be computed from
             :func:`nilearn.masking.compute_background_mask`,
             :func:`nilearn.masking.compute_epi_mask`, or
             :func:`nilearn.masking.compute_brain_mask`.

        Default='epi'.

    mask_args : dict, optional
        If mask is None, these are additional parameters passed to
        masking.compute_background_mask or masking.compute_epi_mask
        to fine-tune mask computation. Please see the related documentation
        for details.

    memory : instance of joblib.Memory or string, optional
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level : integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching. Default=0.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on. Default=1.

    verbose : integer, optional
        Indicate the level of verbosity. By default, nothing is printed.
        Default=0.

    Attributes
    ----------
    `components_` : 2D numpy array (n_components x n-voxels)
        Masked dictionary components extracted from the input images.

        .. note::

            Use attribute `components_img_` rather than manually unmasking
            `components_` with `masker_` attribute.

    `components_img_` : 4D Nifti image
        4D image giving the extracted components. Each 3D image is a component.

        .. versionadded:: 0.4.1

    `masker_` : instance of MultiNiftiMasker
        Masker used to filter and mask data as first step. If an instance of
        MultiNiftiMasker is given in `mask` parameter,
        this is a copy of it. Otherwise, a masker is created using the value
        of `mask` and other NiftiMasker related parameters as initialization.

    `mask_img_` : Niimg-like object
        See :ref:`extracting_data`.
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    References
    ----------
    .. footbibliography::

    """
    def __init__(self, n_components=..., n_epochs=..., alpha=..., reduction_ratio=..., dict_init=..., random_state=..., batch_size=..., method=..., mask=..., smoothing_fwhm=..., standardize=..., detrend=..., low_pass=..., high_pass=..., t_r=..., target_affine=..., target_shape=..., mask_strategy=..., mask_args=..., n_jobs=..., verbose=..., memory=..., memory_level=...) -> None:
        ...
    

