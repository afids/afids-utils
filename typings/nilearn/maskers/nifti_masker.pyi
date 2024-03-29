"""
This type stub file was generated by pyright.
"""

from nilearn import _utils
from nilearn.maskers.base_masker import BaseMasker

"""Transformer used to apply basic transformations on MRI data."""
class _ExtractionFunctor:
    func_name = ...
    def __init__(self, mask_img_) -> None:
        ...
    
    def __call__(self, imgs): # -> tuple[ndarray[Any, dtype[Unknown]] | Unknown, Unknown]:
        ...
    


@_utils.fill_doc
class NiftiMasker(BaseMasker, _utils.CacheMixin):
    """Applying a mask to extract time-series from Niimg-like objects.

    NiftiMasker is useful when preprocessing (detrending, standardization,
    resampling, etc.) of in-mask :term:`voxels<voxel>` is necessary.
    Use case: working with time series of resting-state or task maps.

    Parameters
    ----------
    mask_img : Niimg-like object, optional
        See :ref:`extracting_data`.
        Mask for the data. If not given, a mask is computed in the fit step.
        Optional parameters (mask_args and mask_strategy) can be set to
        fine tune the mask extraction.
        If the mask and the images have different resolutions, the images
        are resampled to the mask resolution.
        If target_shape and/or target_affine are provided, the mask is
        resampled first. After this, the images are resampled to the
        resampled mask.

    runs : :obj:`numpy.ndarray`, optional
        Add a run level to the preprocessing. Each run will be
        detrended independently. Must be a 1D array of n_samples elements.
    %(smoothing_fwhm)s
    %(standardize_maskers)s
    %(standardize_confounds)s
    high_variance_confounds : :obj:`bool`, optional
        If True, high variance confounds are computed on provided image with
        :func:`nilearn.image.high_variance_confounds` and default parameters
        and regressed out. Default=False.
    %(detrend)s
    %(low_pass)s
    %(high_pass)s
    %(t_r)s
    target_affine : 3x3 or 4x4 :obj:`numpy.ndarray`, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape : 3-:obj:`tuple` of :obj:`int`, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.
    %(mask_strategy)s

            .. note::
                Depending on this value, the mask will be computed from
                :func:`nilearn.masking.compute_background_mask`,
                :func:`nilearn.masking.compute_epi_mask`, or
                :func:`nilearn.masking.compute_brain_mask`.

        Default is 'background'.

    mask_args : :obj:`dict`, optional
        If mask is None, these are additional parameters passed to
        masking.compute_background_mask or masking.compute_epi_mask
        to fine-tune mask computation. Please see the related documentation
        for details.

    dtype : {dtype, "auto"}, optional
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.
    %(memory)s
    %(memory_level1)s
    %(verbose0)s
    reports : :obj:`bool`, optional
        If set to True, data is saved in order to produce a report.
        Default=True.

    %(masker_kwargs)s

    Attributes
    ----------
    mask_img_ : :obj:`nibabel.nifti1.Nifti1Image`
        The mask of the data, or the computed one.

    affine_ : 4x4 :obj:`numpy.ndarray`
        Affine of the transformed image.

    n_elements_ : :obj:`int`
        The number of voxels in the mask.

        .. versionadded:: 0.9.2

    See Also
    --------
    nilearn.masking.compute_background_mask
    nilearn.masking.compute_epi_mask
    nilearn.image.resample_img
    nilearn.image.high_variance_confounds
    nilearn.masking.apply_mask
    nilearn.signal.clean

    """
    def __init__(self, mask_img=..., runs=..., smoothing_fwhm=..., standardize=..., standardize_confounds=..., detrend=..., high_variance_confounds=..., low_pass=..., high_pass=..., t_r=..., target_affine=..., target_shape=..., mask_strategy=..., mask_args=..., dtype=..., memory_level=..., memory=..., verbose=..., reports=..., **kwargs) -> None:
        ...
    
    def generate_report(self): # -> HTMLReport:
        """Generate a report of the masker."""
        ...
    
    def fit(self, imgs=..., y=...): # -> Self@NiftiMasker:
        """Compute the mask corresponding to the data.

        Parameters
        ----------
        imgs : :obj:`list` of Niimg-like objects
            See :ref:`extracting_data`.
            Data on which the mask must be calculated. If this is a list,
            the affine is considered the same for all.

        y : None
            This parameter is unused. It is solely included for scikit-learn
            compatibility.

        """
        ...
    
    def transform_single_imgs(self, imgs, confounds=..., sample_mask=..., copy=...): # -> Any | None:
        """Apply mask, spatial and temporal preprocessing.

        Parameters
        ----------
        imgs : 3D/4D Niimg-like object
            See :ref:`extracting_data`.
            Images to process.
            If a 3D niimg is provided, a singleton dimension will be added to
            the output to represent the single scan in the niimg.

        confounds : CSV file or array-like or :obj:`pandas.DataFrame`, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details: :func:`nilearn.signal.clean`.
            shape: (number of scans, number of confounds)

        sample_mask : Any type compatible with numpy-array indexing, optional
            shape: (number of scans - number of volumes removed, )
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to signal.clean.

        copy : :obj:`bool`, optional
            Indicates whether a copy is returned or not. Default=True.

        Returns
        -------
        region_signals : 2D :obj:`numpy.ndarray`
            Signal for each voxel inside the mask.
            shape: (number of scans, number of voxels)

        Warns
        -----
        DeprecationWarning
            If a 3D niimg input is provided, the current behavior
            (adding a singleton dimension to produce a 2D array) is deprecated.
            Starting in version 0.12, a 1D array will be returned for 3D
            inputs.

        """
        ...
    


