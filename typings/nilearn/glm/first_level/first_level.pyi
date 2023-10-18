"""
This type stub file was generated by pyright.
"""

from nilearn._utils import fill_doc
from nilearn.glm._base import BaseGLM

"""Contains the GLM and contrast classes that are meant to be the main \
objects of fMRI data analyses.

Author: Bertrand Thirion, Martin Perez-Guevara, 2016

"""
def mean_scaling(Y, axis=...): # -> tuple[Unknown, Any]:
    """Scaling of the data to have percent of baseline change \
    along the specified axis.

    Parameters
    ----------
    Y : array of shape (n_time_points, n_voxels)
       The input data.

    axis : int, optional
        Axis along which the scaling mean should be calculated. Default=0.

    Returns
    -------
    Y : array of shape (n_time_points, n_voxels),
       The data after mean-scaling, de-meaning and multiplication by 100.

    mean : array of shape (n_voxels,)
        The data mean.

    """
    ...

def run_glm(Y, X, noise_model=..., bins=..., n_jobs=..., verbose=..., random_state=...): # -> tuple[NDArray[Any] | NDArray[float64], dict[float, RegressionResults]]:
    """GLM fit for an fMRI data matrix.

    Parameters
    ----------
    Y : array of shape (n_time_points, n_voxels)
        The fMRI data.

    X : array of shape (n_time_points, n_regressors)
        The design matrix.

    noise_model : {'ar(N)', 'ols'}, optional
        The temporal variance model.
        To specify the order of an autoregressive model place the
        order after the characters `ar`, for example to specify a third order
        model use `ar3`.
        Default='ar1'.

    bins : int, optional
        Maximum number of discrete bins for the AR coef histogram.
        If an autoregressive model with order greater than one is specified
        then adaptive quantification is performed and the coefficients
        will be clustered via K-means with `bins` number of clusters.
        Default=100.

    n_jobs : int, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'. Default=1.

    verbose : int, optional
        The verbosity level. Default=0.

    random_state : int or numpy.random.RandomState, optional
        Random state seed to sklearn.cluster.KMeans for autoregressive models
        of order at least 2 ('ar(N)' with n >= 2). Default=None.

        .. versionadded:: 0.9.1

    Returns
    -------
    labels : array of shape (n_voxels,),
        A map of values on voxels used to identify the corresponding model.

    results : dict,
        Keys correspond to the different labels values
        values are RegressionResults instances corresponding to the voxels.

    """
    ...

@fill_doc
class FirstLevelModel(BaseGLM):
    """Implement the General Linear Model for single session fMRI data.

    Parameters
    ----------
    t_r : float
        This parameter indicates repetition times of the experimental runs.
        In seconds. It is necessary to correctly consider times in the design
        matrix. This parameter is also passed to :func:`nilearn.signal.clean`.
        Please see the related documentation for details.

    slice_time_ref : float, optional
        This parameter indicates the time of the reference slice used in the
        slice timing preprocessing step of the experimental runs. It is
        expressed as a fraction of the t_r (time repetition), so it can have
        values between 0. and 1. Default=0.
    %(hrf_model)s
        Default='glover'.
    drift_model : string, optional
        This parameter specifies the desired drift model for the design
        matrices. It can be 'polynomial', 'cosine' or None.
        Default='cosine'.

    high_pass : float, optional
        This parameter specifies the cut frequency of the high-pass filter in
        Hz for the design matrices. Used only if drift_model is 'cosine'.
        Default=0.01.

    drift_order : int, optional
        This parameter specifies the order of the drift model (in case it is
        polynomial) for the design matrices. Default=1.

    fir_delays : array of shape(n_onsets) or list, optional
        In case of FIR design, yields the array of delays used in the FIR
        model, in scans. Default=[0].

    min_onset : float, optional
        This parameter specifies the minimal onset relative to the design
        (in seconds). Events that start before (slice_time_ref * t_r +
        min_onset) are not considered. Default=-24.

    mask_img : Niimg-like, NiftiMasker object or False, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a NiftiMasker with default
        parameters. If False is given then the data will not be masked.

    target_affine : 3x3 or 4x4 matrix, optional
        This parameter is passed to nilearn.image.resample_img.
        Please see the related documentation for details.

    target_shape : 3-tuple of integers, optional
        This parameter is passed to nilearn.image.resample_img.
        Please see the related documentation for details.
    %(smoothing_fwhm)s
    memory : string or pathlib.Path, optional
        Path to the directory used to cache the masking process and the glm
        fit. By default, no caching is done.
        Creates instance of joblib.Memory.

    memory_level : integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension. Default=False.

    signal_scaling : False, int or (int, int), optional
        If not False, fMRI signals are
        scaled to the mean value of scaling_axis given,
        which can be 0, 1 or (0, 1).
        0 refers to mean scaling each voxel with respect to time,
        1 refers to mean scaling each time point with respect to all voxels &
        (0, 1) refers to scaling with respect to voxels and time,
        which is known as grand mean scaling.
        Incompatible with standardize (standardize=False is enforced when
        signal_scaling is not False).
        Default=0.

    noise_model : {'ar1', 'ols'}, optional
        The temporal variance model. Default='ar1'.

    verbose : integer, optional
        Indicate the level of verbosity. By default, nothing is printed.
        If 0 prints nothing. If 1 prints progress by computation of
        each run. If 2 prints timing details of masker and GLM. If 3
        prints masker computation details. Default=0.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.
        Default=1.

    minimize_memory : boolean, optional
        Gets rid of some variables on the model fit results that are not
        necessary for contrast computation and would only be useful for
        further inspection of model details. This has an important impact
        on memory consumption. Default=True.

    subject_label : string, optional
        This id will be used to identify a `FirstLevelModel` when passed to
        a `SecondLevelModel` object.

    random_state : int or numpy.random.RandomState, optional
        Random state seed to sklearn.cluster.KMeans for autoregressive models
        of order at least 2 ('ar(N)' with n >= 2). Default=None.

        .. versionadded:: 0.9.1

    Attributes
    ----------
    labels_ : array of shape (n_voxels,),
        a map of values on voxels used to identify the corresponding model

    results_ : dict,
        with keys corresponding to the different labels values.
        Values are SimpleRegressionResults corresponding to the voxels,
        if minimize_memory is True,
        RegressionResults if minimize_memory is False

    """
    def __init__(self, t_r=..., slice_time_ref=..., hrf_model=..., drift_model=..., high_pass=..., drift_order=..., fir_delays=..., min_onset=..., mask_img=..., target_affine=..., target_shape=..., smoothing_fwhm=..., memory=..., memory_level=..., standardize=..., signal_scaling=..., noise_model=..., verbose=..., n_jobs=..., minimize_memory=..., subject_label=..., random_state=...) -> None:
        ...
    
    @property
    def scaling_axis(self): # -> int:
        """Return scaling of axis."""
        ...
    
    def fit(self, run_imgs, events=..., confounds=..., sample_masks=..., design_matrices=..., bins=...):
        """Fit the GLM.

        For each run:
        1. create design matrix X
        2. do a masker job: fMRI_data -> Y
        3. fit regression to (Y, X)

        Parameters
        ----------
        run_imgs : Niimg-like object or list of Niimg-like objects,
            Data on which the GLM will be fitted. If this is a list,
            the affine is considered the same for all.

        events : pandas Dataframe or string or list of pandas DataFrames \
                 or strings, optional
            fMRI events used to build design matrices. One events object
            expected per run_img. Ignored in case designs is not None.
            If string, then a path to a csv file is expected.

        confounds : pandas Dataframe, numpy array or string or
            list of pandas DataFrames, numpy arrays or strings, optional
            Each column in a DataFrame corresponds to a confound variable
            to be included in the regression model of the respective run_img.
            The number of rows must match the number of volumes in the
            respective run_img. Ignored in case designs is not None.
            If string, then a path to a csv file is expected.

        sample_masks : array_like, or list of array_like, optional
            shape of array: (number of scans - number of volumes removed, )
            Indices of retained volumes. Masks the niimgs along time/fourth
            dimension to perform scrubbing (remove volumes with high motion)
            and/or remove non-steady-state volumes.
            Default=None.

            .. versionadded:: 0.9.2

        design_matrices : pandas DataFrame or \
                          list of pandas DataFrames, optional
            Design matrices that will be used to fit the GLM. If given it
            takes precedence over events and confounds.

        bins : int, optional
            Maximum number of discrete bins for the AR coef histogram.
            If an autoregressive model with order greater than one is specified
            then adaptive quantification is performed and the coefficients
            will be clustered via K-means with `bins` number of clusters.
            Default=100.

        """
        ...
    
    def compute_contrast(self, contrast_def, stat_type=..., output_type=...): # -> Any | dict[Unknown, Unknown] | None:
        """Generate different outputs corresponding to \
        the contrasts provided e.g. z_map, t_map, effects and variance.

        In multi-session case, outputs the fixed effects map.

        Parameters
        ----------
        contrast_def : str or array of shape (n_col) or list of (string or
                       array of shape (n_col))

            where ``n_col`` is the number of columns of the design matrix,
            (one array per run). If only one array is provided when there
            are several runs, it will be assumed that the same contrast is
            desired for all runs. One can use the name of the conditions as
            they appear in the design matrix of the fitted model combined with
            operators +- and combined with numbers with operators +-`*`/. In
            this case, the string defining the contrasts must be a valid
            expression for compatibility with :meth:`pandas.DataFrame.eval`.

        stat_type : {'t', 'F'}, optional
            Type of the contrast.

        output_type : str, optional
            Type of the output map. Can be 'z_score', 'stat', 'p_value',
            :term:`'effect_size'<Parameter Estimate>`, 'effect_variance' or
            'all'.
            Default='z_score'.

        Returns
        -------
        output : Nifti1Image or dict
            The desired output image(s). If ``output_type == 'all'``, then
            the output is a dictionary of images, keyed by the type of image.

        """
        ...
    


def first_level_from_bids(dataset_path, task_label, space_label=..., sub_labels=..., img_filters=..., t_r=..., slice_time_ref=..., hrf_model=..., drift_model=..., high_pass=..., drift_order=..., fir_delays=..., min_onset=..., mask_img=..., target_affine=..., target_shape=..., smoothing_fwhm=..., memory=..., memory_level=..., standardize=..., signal_scaling=..., noise_model=..., verbose=..., n_jobs=..., minimize_memory=..., derivatives_folder=...): # -> tuple[list[Unknown], list[Unknown], list[Unknown], list[Unknown]]:
    """Create FirstLevelModel objects and fit arguments from a BIDS dataset.

    If t_r is `None` this function will attempt to load it from a bold.json.
    If `slice_time_ref` is  `None` this function will attempt
    to infer it from a bold.json.
    Otherwise t_r and slice_time_ref are taken as given,
    but a warning may be raised if they are not consistent with the bold.json.

    Parameters
    ----------
    dataset_path : :obj:`str` or :obj:`pathlib.Path`
        Directory of the highest level folder of the BIDS dataset.
        Should contain subject folders and a derivatives folder.

    task_label : :obj:`str`
        Task_label as specified in the file names like _task-<task_label>_.

    space_label : :obj:`str`, optional
        Specifies the space label of the preprocessed bold.nii images.
        As they are specified in the file names like _space-<space_label>_.

    sub_labels : :obj:`list` of :obj:`str`, optional
        Specifies the subset of subject labels to model.
        If 'None', will model all subjects in the dataset.
        .. versionadded:: 0.10.1

    img_filters : :obj:`list` of :obj:`tuple` (str, str), optional
        Filters are of the form (field, label). Only one filter per field
        allowed.
        A file that does not match a filter will be discarded.
        Possible filters are 'acq', 'ce', 'dir', 'rec', 'run', 'echo', 'res',
        'den', and 'desc'.
        Filter examples would be ('desc', 'preproc'), ('dir', 'pa')
        and ('run', '10').

    slice_time_ref : :obj:`float` between 0.0 and 1.0, optional
        This parameter indicates the time of the reference slice used in the
        slice timing preprocessing step of the experimental runs. It is
        expressed as a fraction of the t_r (time repetition), so it can have
        values between 0. and 1. Default=0.0

        .. deprecated:: 0.10.1

            The default=0 for ``slice_time_ref`` will be deprecated.
            The default value will change to 'None' in 0.12.

    derivatives_folder : :obj:`str`, Defaults="derivatives".
        derivatives and app folder path containing preprocessed files.
        Like "derivatives/FMRIPREP".

    All other parameters correspond to a `FirstLevelModel` object, which
    contains their documentation.
    The subject label of the model will be determined directly
    from the BIDS dataset.

    Returns
    -------
    models : list of `FirstLevelModel` objects
        Each FirstLevelModel object corresponds to a subject.
        All runs from different sessions are considered together
        for the same subject to run a fixed effects analysis on them.

    models_run_imgs : list of list of Niimg-like objects,
        Items for the FirstLevelModel fit function of their respective model.

    models_events : list of list of pandas DataFrames,
        Items for the FirstLevelModel fit function of their respective model.

    models_confounds : list of list of pandas DataFrames or None,
        Items for the FirstLevelModel fit function of their respective model.

    """
    ...

