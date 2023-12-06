"""
This type stub file was generated by pyright.
"""

from nilearn._utils import CacheMixin, fill_doc
from nilearn.maskers.base_masker import BaseMasker

"""Transformer for computing seeds signals.

Mask nifti images by spherical volumes for seed-region analyses
"""
class _ExtractionFunctor:
    func_name = ...
    def __init__(self, seeds_, radius, mask_img, allow_overlap, dtype) -> None:
        ...
    
    def __call__(self, imgs): # -> tuple[NDArray[Any], None]:
        ...
    


@fill_doc
class NiftiSpheresMasker(BaseMasker, CacheMixin):
    """Class for masking of Niimg-like objects using seeds.

    NiftiSpheresMasker is useful when data from given seeds should be
    extracted. Use case: Summarize brain signals from seeds that were
    obtained from prior knowledge.

    Parameters
    ----------
    seeds : :obj:`list` of triplet of coordinates in native space
        Seed definitions. List of coordinates of the seeds in the same space
        as the images (typically MNI or TAL).

    radius : :obj:`float`, optional
        Indicates, in millimeters, the radius for the sphere around the seed.
        Default is None (signal is extracted on a single voxel).

    mask_img : Niimg-like object, optional
        See :ref:`extracting_data`.
        Mask to apply to regions before extracting signals.

    allow_overlap : :obj:`bool`, optional
        If False, an error is raised if the maps overlaps (ie at least two
        maps have a non-zero value for the same voxel). Default=False.
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
    dtype : {dtype, "auto"}, optional
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.
    %(memory)s
    %(memory_level1)s
    %(verbose0)s
    %(masker_kwargs)s

    Attributes
    ----------
    n_elements_ : :obj:`int`
        The number of seeds in the masker.

        .. versionadded:: 0.9.2

    seeds_ : :obj:`list` of :obj:`list`
        The coordinates of the seeds in the masker.

    See Also
    --------
    nilearn.maskers.NiftiMasker

    """
    def __init__(self, seeds, radius=..., mask_img=..., allow_overlap=..., smoothing_fwhm=..., standardize=..., standardize_confounds=..., high_variance_confounds=..., detrend=..., low_pass=..., high_pass=..., t_r=..., dtype=..., memory=..., memory_level=..., verbose=..., **kwargs) -> None:
        ...
    
    def fit(self, X=..., y=...): # -> Self@NiftiSpheresMasker:
        """Prepare signal extraction from regions.

        All parameters are unused; they are for scikit-learn compatibility.

        """
        ...
    
    def fit_transform(self, imgs, confounds=..., sample_mask=...):
        """Prepare and perform signal extraction.

        Parameters
        ----------
        imgs : 3D/4D Niimg-like object
            See :ref:`extracting_data`.
            Images to process.
            If a 3D niimg is provided, a singleton dimension will be added to
            the output to represent the single scan in the niimg.

        confounds : CSV file or array-like or :obj:`pandas.DataFrame`, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        sample_mask : Any type compatible with numpy-array indexing, optional
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to signal.clean.
            shape: (number of scans - number of volumes removed, )

                .. versionadded:: 0.8.0

        Returns
        -------
        region_signals : 2D :obj:`numpy.ndarray`
            Signal for each sphere.
            shape: (number of scans, number of spheres)

        """
        ...
    
    def transform_single_imgs(self, imgs, confounds=..., sample_mask=...): # -> Any:
        """Extract signals from a single 4D niimg.

        Parameters
        ----------
        imgs : 3D/4D Niimg-like object
            See :ref:`extracting_data`.
            Images to process.
            If a 3D niimg is provided, a singleton dimension will be added to
            the output to represent the single scan in the niimg.

        confounds : CSV file or array-like or :obj:`pandas.DataFrame`, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        sample_mask : Any type compatible with numpy-array indexing, optional
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to signal.clean.
            shape: (number of scans - number of volumes removed, )

                .. versionadded:: 0.8.0

        Returns
        -------
        region_signals : 2D :obj:`numpy.ndarray`
            Signal for each sphere.
            shape: (number of scans, number of spheres)

        Warns
        -----
        DeprecationWarning
            If a 3D niimg input is provided, the current behavior
            (adding a singleton dimension to produce a 2D array) is deprecated.
            Starting in version 0.12, a 1D array will be returned for 3D
            inputs.

        """
        ...
    
    def inverse_transform(self, region_signals): # -> list[Unknown] | Nifti1Image | FileBasedImage:
        """Compute voxel signals from spheres signals.

        Any mask given at initialization is taken into account. Throws an error
        if mask_img==None

        Parameters
        ----------
        region_signals : 1D/2D :obj:`numpy.ndarray`
            Signal for each region.
            If a 1D array is provided, then the shape should be
            (number of elements,), and a 3D img will be returned.
            If a 2D array is provided, then the shape should be
            (number of scans, number of elements), and a 4D img will be
            returned.

        Returns
        -------
        voxel_signals : :obj:`nibabel.nifti1.Nifti1Image`
            Signal for each sphere.
            shape: (mask_img, number of scans).

        """
        ...
    


