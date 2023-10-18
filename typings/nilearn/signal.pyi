"""
This type stub file was generated by pyright.
"""

from nilearn._utils import fill_doc

"""
Preprocessing functions for time series.

All functions in this module should take X matrices with samples x
features
"""
availiable_filters = ...
@fill_doc
def butterworth(signals, sampling_rate, low_pass=..., high_pass=..., order=..., padtype=..., padlen=..., copy=...): # -> NDArray[Any]:
    """Apply a low-pass, high-pass or band-pass \
    `Butterworth filter <https://en.wikipedia.org/wiki/Butterworth_filter>`_.

    Apply a filter to remove signal below the `low` frequency and above the
    `high` frequency.

    Parameters
    ----------
    signals : :class:`numpy.ndarray` (1D sequence or n_samples x n_sources)
        Signals to be filtered. A signal is assumed to be a column
        of `signals`.

    sampling_rate : :obj:`float`
        Number of samples per second (sample frequency, in Hertz).
    %(low_pass)s
    %(high_pass)s
    order : :obj:`int`, optional
        Order of the `Butterworth filter
        <https://en.wikipedia.org/wiki/Butterworth_filter>`_.
        When filtering signals, the filter has a decay to avoid ringing.
        Increasing the order sharpens this decay. Be aware that very high
        orders can lead to numerical instability.
        Default=5.

    padtype : {"odd", "even", "constant", None}, optional
        Type of padding to use for the Butterworth filter.
        For more information about this, see :func:`scipy.signal.filtfilt`.

    padlen : :obj:`int` or None, optional
        The size of the padding to add to the beginning and end of ``signals``.
        If None, the default value from :func:`scipy.signal.filtfilt` will be
        used.

    copy : :obj:`bool`, optional
        If False, `signals` is modified inplace, and memory consumption is
        lower than for ``copy=True``, though computation time is higher.

    Returns
    -------
    filtered_signals : :class:`numpy.ndarray`
        Signals filtered according to the given parameters.
    """
    ...

@fill_doc
def high_variance_confounds(series, n_confounds=..., percentile=..., detrend=...):
    """Return confounds time series extracted from series \
    with highest variance.

    Parameters
    ----------
    series : :class:`numpy.ndarray`
        Timeseries. A timeseries is a column in the "series" array.
        shape (sample number, feature number)

    n_confounds : :obj:`int`, optional
        Number of confounds to return. Default=5.

    percentile : :obj:`float`, optional
        Highest-variance series percentile to keep before computing the
        singular value decomposition, 0. <= `percentile` <= 100.
        ``series.shape[0] * percentile / 100`` must be greater
        than ``n_confounds``. Default=2.0.
    %(detrend)s
        Default=True.

    Returns
    -------
    v : :class:`numpy.ndarray`
        Highest variance confounds. Shape: (samples, n_confounds)

    Notes
    -----
    This method is related to what has been published in the literature
    as 'CompCor' :footcite:`Behzadi2007`.

    The implemented algorithm does the following:

    - compute sum of squares for each time series (no mean removal)
    - keep a given percentile of series with highest variances (percentile)
    - compute an svd of the extracted series
    - return a given number (n_confounds) of series from the svd with
      highest singular values.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    nilearn.image.high_variance_confounds
    """
    ...

@fill_doc
def clean(signals, runs=..., detrend=..., standardize=..., sample_mask=..., confounds=..., standardize_confounds=..., filter=..., low_pass=..., high_pass=..., t_r=..., ensure_finite=..., **kwargs): # -> NDArray[Unknown] | ndarray[Unknown, Unknown] | NDArray[float32 | float64] | Any:
    """Improve :term:`SNR` on masked :term:`fMRI` signals.

    This function can do several things on the input signals. With the default
    options, the procedures are performed in the following order:

    - detrend
    - low- and high-pass butterworth filter
    - remove confounds
    - standardize

    Low-pass filtering improves specificity.

    High-pass filtering should be kept small, to keep some sensitivity.

    Butterworth filtering is only meaningful on evenly-sampled signals.

    When performing scrubbing (censoring high-motion volumes) with butterworth
    filtering, the signal is processed in the following order, based on the
    second recommendation in :footcite:`Lindquist2018`:

    - interpolate high motion volumes with cubic spline interpolation
    - detrend
    - low- and high-pass butterworth filter
    - censor high motion volumes
    - remove confounds
    - standardize

    According to :footcite:`Lindquist2018`, removal of confounds will be done
    orthogonally to temporal filters (low- and/or high-pass filters), if both
    are specified. The censored volumes should be removed in both signals and
    confounds before the nuisance regression.

    When performing scrubbing with cosine drift term filtering, the signal is
    processed in the following order, based on the first recommendation in
    :footcite:`Lindquist2018`:

    - generate cosine drift term
    - censor high motion volumes in both signal and confounds
    - detrend
    - remove confounds
    - standardize

    Parameters
    ----------
    signals : :class:`numpy.ndarray`
        Timeseries. Must have shape (instant number, features number).
        This array is not modified.

    runs : :class:`numpy.ndarray`, optional
        Add a run level to the cleaning process. Each run will be
        cleaned independently. Must be a 1D array of n_samples elements.
        Default is None.

    confounds : :class:`numpy.ndarray`, :obj:`str`, :class:`pathlib.Path`,\
    :class:`pandas.DataFrame` or :obj:`list` of confounds timeseries.
        Shape must be (instant number, confound number), or just
        (instant number,).
        The number of time instants in ``signals`` and ``confounds`` must be
        identical (i.e. ``signals.shape[0] == confounds.shape[0]``).
        If a string is provided, it is assumed to be the name of a csv file
        containing signals as columns, with an optional one-line header.
        If a list is provided, all confounds are removed from the input
        signal, as if all were in the same array.
        Default is None.

    sample_mask : None, Any type compatible with numpy-array indexing, \
        or :obj:`list` of
        shape: (number of scans - number of volumes removed, ) for explicit \
            index, or (number of scans, ) for binary mask
        Masks the niimgs along time/fourth dimension to perform scrubbing
        (remove volumes with high motion) and/or non-steady-state volumes.
        When passing binary mask with boolean values, ``True`` refers to
        volumes kept, and ``False`` for volumes removed.
        This masking step is applied before signal cleaning. When supplying run
        information, sample_mask must be a list containing sets of indexes for
        each run.

            .. versionadded:: 0.8.0

        Default is None.
    %(t_r)s
        Default=2.5.
    filter : {'butterworth', 'cosine', False}, optional
        Filtering methods:

            - 'butterworth': perform butterworth filtering.
            - 'cosine': generate discrete cosine transformation drift terms.
            - False: Do not perform filtering.

        Default='butterworth'.
    %(low_pass)s

        .. note::
            `low_pass` is not implemented for filter='cosine'.

    %(high_pass)s
    %(detrend)s
    standardize : {'zscore_sample', 'zscore', 'psc', True, False}, optional
        Strategy to standardize the signal:

            - 'zscore_sample': The signal is z-scored. Timeseries are shifted
              to zero mean and scaled to unit variance. Uses sample std.
            - 'zscore': The signal is z-scored. Timeseries are shifted
              to zero mean and scaled to unit variance. Uses population std
              by calling default :obj:`numpy.std` with N - ``ddof=0``.
            - 'psc':  Timeseries are shifted to zero mean value and scaled
              to percent signal change (as compared to original mean signal).
            - True: The signal is z-scored (same as option `zscore`).
              Timeseries are shifted to zero mean and scaled to unit variance.
            - False: Do not standardize the data.

        Default="zscore".
    %(standardize_confounds)s

    ensure_finite : :obj:`bool`, default=False
        If `True`, the non-finite values (NANs and infs) found in the data
        will be replaced by zeros.

    kwargs : dict
        Keyword arguments to be passed to functions called within ``clean``.
        Kwargs prefixed with ``'butterworth__'`` will be passed to
        :func:`~nilearn.signal.butterworth`.


    Returns
    -------
    cleaned_signals : :class:`numpy.ndarray`
        Input signals, cleaned. Same shape as `signals` unless `sample_mask`
        is applied.

    Notes
    -----
    Confounds removal is based on a projection on the orthogonal
    of the signal space. See :footcite:`Friston1994`.

    Orthogonalization between temporal filters and confound removal is based on
    suggestions in :footcite:`Lindquist2018`.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    nilearn.image.clean_img
    """
    ...

