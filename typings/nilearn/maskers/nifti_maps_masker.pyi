"""
This type stub file was generated by pyright.
"""

from nilearn import _utils
from nilearn.maskers.base_masker import BaseMasker

"""Transformer for computing ROI signals."""
class _ExtractionFunctor:
    func_name = ...
    def __init__(self, _resampled_maps_img_, _resampled_mask_img_, keep_masked_maps) -> None:
        ...
    
    def __call__(self, imgs): # -> tuple[Unknown, list[Any]]:
        ...
    


@_utils.fill_doc
class NiftiMapsMasker(BaseMasker, _utils.CacheMixin):
    """Class for masking of Niimg-like objects.

    NiftiMapsMasker is useful when data from overlapping volumes should be
    extracted (contrarily to :class:`nilearn.maskers.NiftiLabelsMasker`).
    Use case: Summarize brain signals from large-scale networks obtained by
    prior PCA or :term:`ICA`.

    Note that, Inf or NaN present in the given input images are automatically
    put to zero rather than considered as missing data.

    Parameters
    ----------
    maps_img : 4D niimg-like object
        See :ref:`extracting_data`.
        Set of continuous maps. One representative time course per map is
        extracted using least square regression.

    mask_img : 3D niimg-like object, optional
        See :ref:`extracting_data`.
        Mask to apply to regions before extracting signals.

    allow_overlap : :obj:`bool`, optional
        If False, an error is raised if the maps overlaps (ie at least two
        maps have a non-zero value for the same voxel). Default=True.
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

    resampling_target : {"data", "mask", "maps", None}, optional.
        Gives which image gives the final shape/size. For example, if
        `resampling_target` is "mask" then maps_img and images provided to
        fit() are resampled to the shape and affine of mask_img. "None" means
        no resampling: if shapes and affines do not match, a ValueError is
        raised. Default="data".
    %(memory)s
    %(memory_level)s
    %(verbose0)s
    %(keep_masked_maps)s

    reports : :obj:`bool`, optional
        If set to True, data is saved in order to produce a report.
        Default=True.

    %(masker_kwargs)s

    Attributes
    ----------
    maps_img_ : :obj:`nibabel.nifti1.Nifti1Image`
        The maps mask of the data.

    n_elements_ : :obj:`int`
        The number of overlapping maps in the mask.
        This is equivalent to the number of volumes in the mask image.

        .. versionadded:: 0.9.2

    Notes
    -----
    If resampling_target is set to "maps", every 3D image processed by
    transform() will be resampled to the shape of maps_img. It may lead to a
    very large memory consumption if the voxel number in maps_img is large.

    See Also
    --------
    nilearn.maskers.NiftiMasker
    nilearn.maskers.NiftiLabelsMasker

    """
    def __init__(self, maps_img, mask_img=..., allow_overlap=..., smoothing_fwhm=..., standardize=..., standardize_confounds=..., high_variance_confounds=..., detrend=..., low_pass=..., high_pass=..., t_r=..., dtype=..., resampling_target=..., keep_masked_maps=..., memory=..., memory_level=..., verbose=..., reports=..., **kwargs) -> None:
        ...
    
    def generate_report(self, displayed_maps=...): # -> HTMLReport:
        """Generate an HTML report for the current ``NiftiMapsMasker`` object.

        .. note::
            This functionality requires to have ``Matplotlib`` installed.

        Parameters
        ----------
        displayed_maps : :obj:`int`, or :obj:`list`,\
        or :class:`~numpy.ndarray`, or "all", optional
            Indicates which maps will be displayed in the HTML report.

                - If "all": All maps will be displayed in the report.

                .. code-block:: python

                    masker.generate_report("all")

                .. warning:
                    If there are too many maps, this might be time and
                    memory consuming, and will result in very heavy
                    reports.

                - If a :obj:`list` or :class:`~numpy.ndarray`: This indicates
                  the indices of the maps to be displayed in the report. For
                  example, the following code will generate a report with maps
                  6, 3, and 12, displayed in this specific order:

                .. code-block:: python

                    masker.generate_report([6, 3, 12])

                - If an :obj:`int`: This will only display the first n maps,
                  n being the value of the parameter. By default, the report
                  will only contain the first 10 maps. Example to display the
                  first 16 maps:

                .. code-block:: python

                    masker.generate_report(16)

            Default=10.

        Returns
        -------
        report : `nilearn.reporting.html_report.HTMLReport`
            HTML report for the masker.
        """
        ...
    
    def fit(self, imgs=..., y=...): # -> Self@NiftiMapsMasker:
        """Prepare signal extraction from regions.

        Parameters
        ----------
        imgs : :obj:`list` of Niimg-like objects
            See :ref:`extracting_data`.
            Image data passed to the reporter.

        y : None
            This parameter is unused. It is solely included for scikit-learn
            compatibility.
        """
        ...
    
    def fit_transform(self, imgs, confounds=..., sample_mask=...):
        """Prepare and perform signal extraction."""
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

        confounds : CSV file or array-like, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        sample_mask : Any type compatible with numpy-array indexing, optional
            shape: (number of scans - number of volumes removed, )
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to signal.clean.

                .. versionadded:: 0.8.0

        Returns
        -------
        region_signals : 2D numpy.ndarray
            Signal for each map.
            shape: (number of scans, number of maps)

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
        """Compute voxel signals from region signals.

        Any mask given at initialization is taken into account.

        Parameters
        ----------
        region_signals : 1D/2D numpy.ndarray
            Signal for each region.
            If a 1D array is provided, then the shape should be
            (number of elements,), and a 3D img will be returned.
            If a 2D array is provided, then the shape should be
            (number of scans, number of elements), and a 4D img will be
            returned.

        Returns
        -------
        voxel_signals : nibabel.Nifti1Image
            Signal for each voxel. shape: that of maps.

        """
        ...
    

