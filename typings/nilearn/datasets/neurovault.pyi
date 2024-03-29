"""
This type stub file was generated by pyright.
"""

import numpy as np

"""Download statistical maps available \
on Neurovault (http://neurovault.org)."""
_NEUROVAULT_BASE_URL = ...
_NEUROVAULT_COLLECTIONS_URL = ...
_NEUROVAULT_IMAGES_URL = ...
_NEUROSYNTH_FETCH_WORDS_URL = ...
_COL_FILTERS_AVAILABLE_ON_SERVER = ...
_IM_FILTERS_AVAILABLE_ON_SERVER = ...
_DEFAULT_BATCH_SIZE = ...
_DEFAULT_MAX_IMAGES = ...
STD_AFFINE = np.array([[3, 0, 0, -90], [0, 3, 0, -126], [0, 0, 3, -72], [0, 0, 0, 1]])
_MAX_CONSECUTIVE_FAILS = ...
_MAX_FAILS_IN_COLLECTION = ...
_DEBUG = ...
_INFO = ...
_WARNING = ...
_ERROR = ...
class _SpecialValue:
    """Base class for special values used to filter terms.

    Derived classes should override ``__eq__`` in order to create
    objects that can be used for comparisons to particular sets of
    values in filters.

    """
    def __eq__(self, other) -> bool:
        ...
    
    def __req__(self, other):
        ...
    
    def __ne__(self, other) -> bool:
        ...
    
    def __rne__(self, other): # -> bool:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class IsNull(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class will always be equal to, and only to,
    any null value of any type (by null we mean for which bool
    returns False).

    See Also
    --------
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import IsNull
    >>> null = IsNull()
    >>> null == 0
    True
    >>> null == ''
    True
    >>> null == None
    True
    >>> null == 'a'
    False

    """
    def __eq__(self, other) -> bool:
        ...
    


class NotNull(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class will always be equal to, and only to,
    any non-zero value of any type (by non-zero we mean for which bool
    returns True).

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import NotNull
    >>> not_null = NotNull()
    >>> not_null == 0
    False
    >>> not_null == ''
    False
    >>> not_null == None
    False
    >>> not_null == 'a'
    True

    """
    def __eq__(self, other) -> bool:
        ...
    


class NotEqual(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class is constructed with `NotEqual(obj)`. It
    will always be equal to, and only to, any value for which
    ``obj == value`` is ``False``.

    Parameters
    ----------
    negated : object
        The object from which a candidate should be different in order
        to pass through the filter.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import NotEqual
    >>> not_0 = NotEqual(0)
    >>> not_0 == 0
    False
    >>> not_0 == '0'
    True

    """
    def __init__(self, negated) -> None:
        ...
    
    def __eq__(self, other) -> bool:
        ...
    


class _OrderComp(_SpecialValue):
    """Base class for special values based on order comparisons."""
    def __init__(self, bound) -> None:
        ...
    
    def __eq__(self, other) -> bool:
        ...
    


class GreaterOrEqual(_OrderComp):
    """Special value used to filter terms.

    An instance of this class is constructed with `GreaterOrEqual(obj)`. It
    will always be equal to, and only to, any value for which
    ``obj <= value`` is ``True``.

    Parameters
    ----------
    bound : object
        The object to which a candidate should be superior or equal in
        order to pass through the filter.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import GreaterOrEqual
    >>> nonnegative = GreaterOrEqual(0.)
    >>> nonnegative == -.1
    False
    >>> nonnegative == 0
    True
    >>> nonnegative == .1
    True

    """
    ...


class GreaterThan(_OrderComp):
    """Special value used to filter terms.

    An instance of this class is constructed with `GreaterThan(obj)`. It
    will always be equal to, and only to, any value for which
    ``obj < value`` is ``True``.

    Parameters
    ----------
    bound : object
        The object to which a candidate should be strictly superior in
        order to pass through the filter.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import GreaterThan
    >>> positive = GreaterThan(0.)
    >>> positive == 0.
    False
    >>> positive == 1.
    True
    >>> positive == -1.
    False

    """
    ...


class LessOrEqual(_OrderComp):
    """Special value used to filter terms.

    An instance of this class is constructed with `LessOrEqual(obj)`. It
    will always be equal to, and only to, any value for which
    ``value <= obj`` is ``True``.

    Parameters
    ----------
    bound : object
        The object to which a candidate should be inferior or equal in
        order to pass through the filter.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import LessOrEqual
    >>> nonpositive = LessOrEqual(0.)
    >>> nonpositive == -1.
    True
    >>> nonpositive == 0.
    True
    >>> nonpositive == 1.
    False

    """
    ...


class LessThan(_OrderComp):
    """Special value used to filter terms.

    An instance of this class is constructed with `LessThan(obj)`. It
    will always be equal to, and only to, any value for which
    ``value < obj`` is ``True``.

    Parameters
    ----------
    bound : object
        The object to which a candidate should be strictly inferior in
        order to pass through the filter.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import LessThan
    >>> negative = LessThan(0.)
    >>> negative == -1.
    True
    >>> negative == 0.
    False
    >>> negative == 1.
    False

    """
    ...


class IsIn(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class is constructed with
    `IsIn(*accepted)`. It will always be equal to, and only to, any
    value for which ``value in accepted`` is ``True``.

    Parameters
    ----------
    accepted : container
        A value will pass through the filter if it is present in
        `accepted`.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import IsIn
    >>> vowels = IsIn('a', 'e', 'i', 'o', 'u', 'y')
    >>> 'a' == vowels
    True
    >>> vowels == 'b'
    False

    """
    def __init__(self, *accepted) -> None:
        ...
    
    def __eq__(self, other) -> bool:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class NotIn(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class is constructed with
    `NotIn(*rejected)`. It will always be equal to, and only to, any
    value for which ``value in rejected`` is ``False``.

    Parameters
    ----------
    rejected : container
        A value will pass through the filter if it is absent from
        `rejected`.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import NotIn
    >>> consonants = NotIn('a', 'e', 'i', 'o', 'u', 'y')
    >>> 'b' == consonants
    True
    >>> consonants == 'a'
    False

    """
    def __init__(self, *rejected) -> None:
        ...
    
    def __eq__(self, other) -> bool:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class Contains(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class is constructed with
    `Contains(*must_be_contained)`. It will always be equal to, and
    only to, any value for which ``item in value`` is ``True`` for
    every item in ``must_be_contained``.

    Parameters
    ----------
    must_be_contained : container
        A value will pass through the filter if it contains all the
        items in must_be_contained.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import Contains
    >>> contains = Contains('house', 'face')
    >>> 'face vs house' == contains
    True
    >>> 'smiling face vs frowning face' == contains
    False

    """
    def __init__(self, *must_be_contained) -> None:
        ...
    
    def __eq__(self, other) -> bool:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class NotContains(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class is constructed with
    `NotContains(*must_not_be_contained)`. It will always be equal
    to, and only to, any value for which ``item in value`` is
    ``False`` for every item in ``must_not_be_contained``.

    Parameters
    ----------
    must_not_be_contained : container
        A value will pass through the filter if it does not contain
        any of the items in must_not_be_contained.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import NotContains
    >>> no_garbage = NotContains('bad', 'test')
    >>> no_garbage == 'test image'
    False
    >>> no_garbage == 'good image'
    True

    """
    def __init__(self, *must_not_be_contained) -> None:
        ...
    
    def __eq__(self, other) -> bool:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class Pattern(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class is constructed with

    `Pattern(pattern[, flags])`. It will always be equal to, and only
    to, any value for which ``re.match(pattern, value, flags)`` is
    ``True``.

    Parameters
    ----------
    pattern : str
        The pattern to try to match to candidates.

    flags : int, optional (default=0)
        Value for ``re.match`` `flags` parameter,
        e.g. ``re.IGNORECASE``. The default (0), is the default value
        used by ``re.match``.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains.

    Documentation for standard library ``re`` module.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import Pattern
    >>> poker = Pattern(r'[0-9akqj]{5}$')
    >>> 'ak05q' == poker
    True
    >>> 'ak05e' == poker
    False

    """
    def __init__(self, pattern, flags=...) -> None:
        ...
    
    def __eq__(self, other) -> bool:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class ResultFilter:
    """Easily create callable (local) filters for ``fetch_neurovault``.

    Constructed from a mapping of key-value pairs (optional) and a
    callable filter (also optional), instances of this class are meant
    to be used as ``image_filter`` or ``collection_filter`` parameters
    for ``fetch_neurovault``.

    Such filters can be combined using their methods ``AND``, ``OR``,
    ``XOR``, and ``NOT``, with the usual semantics.

    Key-value pairs can be added by treating a ``ResultFilter`` as a
    dictionary: after evaluating ``res_filter[key] = value``, only
    metadata such that ``metadata[key] == value`` can pass through the
    filter.

    Parameters
    ----------
    query_terms : dict, optional
        A ``metadata`` dictionary will be blocked by the filter if it
        does not respect ``metadata[key] == value`` for all
        ``key``, ``value`` pairs in `query_terms`. If ``None``, the
        empty dictionary is used.

    callable_filter : callable, optional
        A ``metadata`` dictionary will be blocked by the filter if
        `callable_filter` does not return ``True`` for ``metadata``.
        Default=empty_filter

    As an alternative to the `query_terms` dictionary parameter,
    key, value pairs can be passed as keyword arguments.

    Attributes
    ----------
    query_terms_ : dict
        In order to pass through the filter, metadata must verify
        ``metadata[key] == value`` for each ``key``, ``value`` pair in
        `query_terms_`.

    callable_filters_ : list of callables
        In addition to ``(key, value)`` pairs, we can use this
        attribute to specify more elaborate requirements. Called with
        a dict representing metadata for an image or collection, each
        element of this list returns ``True`` if the metadata should
        pass through the filter and ``False`` otherwise.

    A dict of metadata will only pass through the filter if it
    satisfies all the `query_terms` AND all the elements of
    `callable_filters_`.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import ResultFilter
    >>> filt = ResultFilter(a=0).AND(ResultFilter(b=1).OR(ResultFilter(b=2)))
    >>> filt({'a': 0, 'b': 1})
    True
    >>> filt({'a': 0, 'b': 0})
    False

    """
    def __init__(self, query_terms=..., callable_filter=..., **kwargs) -> None:
        ...
    
    def __call__(self, candidate): # -> bool:
        """Return True if candidate satisfies the requirements.

        Parameters
        ----------
        candidate : dict
            A dictionary representing metadata for a file or a
            collection, to be filtered.

        Returns
        -------
        bool
            ``True`` if `candidate` passes through the filter and ``False``
            otherwise.

        """
        ...
    
    def OR(self, other_filter): # -> ResultFilter:
        """Implement the OR operator between two filters."""
        ...
    
    def AND(self, other_filter): # -> ResultFilter:
        """Implement the AND operator between two filters."""
        ...
    
    def XOR(self, other_filter): # -> ResultFilter:
        """Implement the XOR operator between two filters."""
        ...
    
    def NOT(self): # -> ResultFilter:
        """Implement the NOT operator between two filters."""
        ...
    
    def __getitem__(self, item):
        """Get item from query_terms_."""
        ...
    
    def __setitem__(self, item, value): # -> None:
        """Set item in query_terms_."""
        ...
    
    def __delitem__(self, item): # -> None:
        """Remove item from query_terms_."""
        ...
    
    def add_filter(self, callable_filter): # -> None:
        """Add a function to the callable_filters_.

        After a call add_filter(additional_filt), in addition to all
        the previous requirements, a candidate must also verify
        additional_filt(candidate) in order to pass through the
        filter.

        """
        ...
    
    def __str__(self) -> str:
        ...
    


class _TemporaryDirectory:
    """Context manager that provides a temporary directory.

    A temporary directory is created on __enter__
    and removed on __exit__ .

    Attributes
    ----------
    temp_dir_ : str or None
        location of temporary directory or None if not created.

    """
    def __init__(self) -> None:
        ...
    
    def __enter__(self): # -> str:
        ...
    
    def __exit__(self, *args): # -> None:
        ...
    


def neurosynth_words_vectorized(word_files, verbose=..., **kwargs): # -> tuple[None, None] | tuple[Unknown | NDArray[float64], NDArray[Unknown]]:
    """Load Neurosynth data from disk into an (n images, voc size) matrix.

    Neurosynth data is saved on disk as ``{word: weight}``
    dictionaries for each image, this function reads it and returns a
    vocabulary list and a term weight matrix.

    Parameters
    ----------
    word_files : Container
        The paths to the files from which to read word weights (each
        is supposed to contain the Neurosynth response for a
        particular image).

    verbose : int, optional
        An integer in [0, 1, 2, 3] to control the verbosity level.
        Default=3.

    Keyword arguments are passed on to
    ``sklearn.feature_extraction.DictVectorizer``.

    Returns
    -------
    frequencies : numpy.ndarray
        An (n images, vocabulary size) array. Each row corresponds to
        an image, and each column corresponds to a word. The words are
        in the same order as in returned value `vocabulary`, so that
        `frequencies[i, j]` corresponds to the weight of
        `vocabulary[j]` for image ``i``.  This matrix is computed by
        an ``sklearn.feature_extraction.DictVectorizer`` instance.

    vocabulary : list of str
        A list of all the words encountered in the word files.

    See Also
    --------
    sklearn.feature_extraction.DictVectorizer

    """
    ...

def basic_collection_terms(): # -> dict[str, NotNull]:
    """Return a term filter that excludes empty collections."""
    ...

def basic_image_terms(): # -> dict[str, bool | NotIn | NotEqual]:
    """Filter that selects unthresholded F, T and Z maps in mni space.

    More precisely, an image is excluded if one of the following is
    true:

        - It is not in MNI space.
        - It is thresholded.
        - Its map type is one of "ROI/mask", "anatomical", or "parcellation".
        - Its image type is "atlas"

    """
    ...

def fetch_neurovault(max_images=..., collection_terms=..., collection_filter=..., image_terms=..., image_filter=..., mode=..., data_dir=..., fetch_neurosynth_words=..., resample=..., vectorize_words=..., verbose=..., **kwarg_image_filters): # -> Bunch:
    """Download data from neurovault.org that match certain criteria.

    Any downloaded data is saved on the local disk and subsequent
    calls to this function will first look for the data locally before
    querying the server for more if necessary.

    We explore the metadata for Neurovault collections and images,
    keeping those that match a certain set of criteria, until we have
    skimmed through the whole database or until an (optional) maximum
    number of images to fetch has been reached.

    For more information, see :footcite:`Gorgolewski2015`,
    and :footcite:`Yarkoni2011`.

    Parameters
    ----------
    max_images : int, optional
        Maximum number of images to fetch. Default=100.

    collection_terms : dict, optional
        Key, value pairs used to filter collection
        metadata. Collections for which
        ``collection_metadata['key'] == value`` is not ``True`` for
        every key, value pair will be discarded.
        See documentation for ``basic_collection_terms`` for a
        description of the default selection criteria.
        Default=basic_collection_terms().

    collection_filter : Callable, optional
        Collections for which `collection_filter(collection_metadata)`
        is ``False`` will be discarded.
        Default=empty_filter.

    image_terms : dict, optional
        Key, value pairs used to filter image metadata. Images for
        which ``image_metadata['key'] == value`` is not ``True`` for
        if image_filter != _empty_filter and image_terms =
        every key, value pair will be discarded.
        See documentation for ``basic_image_terms`` for a
        description of the default selection criteria.
        Default=basic_image_terms().

    image_filter : Callable, optional
        Images for which `image_filter(image_metadata)` is ``False``
        will be discarded. Default=empty_filter.

    mode : {'download_new', 'overwrite', 'offline'}
        When to fetch an image from the server rather than the local
        disk.

        - 'download_new' (the default) means download only files that
          are not already on disk (regardless of modify date).
        - 'overwrite' means ignore files on disk and overwrite them.
        - 'offline' means load only data from disk; don't query server.

    data_dir : str, optional
        The directory we want to use for nilearn data. A subdirectory
        named "neurovault" will contain neurovault data.

    fetch_neurosynth_words : bool, optional
        whether to collect words from Neurosynth. Default=False.

    vectorize_words : bool, optional
        If neurosynth words are downloaded, create a matrix of word
        counts and add it to the result. Also add to the result a
        vocabulary list. See ``sklearn.CountVectorizer`` for more info.
        Default=True.

    resample : bool, optional (default=False)
        Resamples downloaded images to a 3x3x3 grid before saving them,
        to save disk space.

    interpolation : str, optional
        Can be 'continuous', 'linear', or 'nearest'. Indicates the resample
        method. Default='continuous'.
        Argument passed to nilearn.image.resample_img.

    verbose : int, optional
        An integer in [0, 1, 2, 3] to control the verbosity level.
        Default=3.

    kwarg_image_filters
        Keyword arguments are understood to be filter terms for
        images, so for example ``map_type='Z map'`` means only
        download Z-maps; ``collection_id=35`` means download images
        from collection 35 only.

    Returns
    -------
    Bunch
        A dict-like object which exposes its items as attributes. It contains:

            - 'images', the paths to downloaded files.
            - 'images_meta', the metadata for the images in a list of
              dictionaries.
            - 'collections_meta', the metadata for the
              collections.
            - 'description', a short description of the Neurovault dataset.

        If `fetch_neurosynth_words` and `vectorize_words` were set, it
        also contains:

            - 'vocabulary', a list of words
            - 'word_frequencies', the weight of the words returned by
              neurosynth.org for each image, such that the weight of word
              `vocabulary[j]` for the image found in `images[i]` is
              `word_frequencies[i, j]`

    See Also
    --------
    nilearn.datasets.fetch_neurovault_ids
        Fetch collections and images from Neurovault by explicitly specifying
        their ids.

    Notes
    -----
    Images and collections from disk are fetched before remote data.

    Some helpers are provided in the ``neurovault`` module to express
    filtering criteria more concisely:

        ``ResultFilter``, ``IsNull``, ``NotNull``, ``NotEqual``,
        ``GreaterOrEqual``, ``GreaterThan``, ``LessOrEqual``,
        ``LessThan``, ``IsIn``, ``NotIn``, ``Contains``,
        ``NotContains``, ``Pattern``.

    If you pass a single value to match against the collection id
    (whether as the 'id' field of the collection metadata or as the
    'collection_id' field of the image metadata), the server is
    directly queried for that collection, so
    ``fetch_neurovault(collection_id=40)`` is as efficient as
    ``fetch_neurovault(collection_ids=[40])`` (but in the former
    version the other filters will still be applied). This is not true
    for the image ids. If you pass a single value to match against any
    of the fields listed in ``_COL_FILTERS_AVAILABLE_ON_SERVER``,
    i.e., 'DOI', 'name', and 'owner', these filters can be
    applied by the server, limiting the amount of metadata we have to
    download: filtering on those fields makes the fetching faster
    because the filtering takes place on the server side.

    In `download_new` mode, if a file exists on disk, it is not
    downloaded again, even if the version on the server is newer. Use
    `overwrite` mode to force a new download (you can filter on the
    field ``modify_date`` to re-download the files that are newer on
    the server - see Examples section).

    Tries to yield `max_images` images; stops early if we have fetched
    all the images matching the filters or if too many images fail to
    be downloaded in a row.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    To download **all** the collections and images from Neurovault::

        fetch_neurovault(max_images=None, collection_terms={}, image_terms={})

    To further limit the default selection to collections which
    specify a DOI (which reference a published paper, as they may be
    more likely to contain good images)::

        fetch_neurovault(
            max_images=None,
            collection_terms=dict(basic_collection_terms(), DOI=NotNull()))

    To update all the images (matching the default filters)::

        fetch_neurovault(
            max_images=None, mode='overwrite',
            modify_date=GreaterThan(newest))

    """
    ...

def fetch_neurovault_ids(collection_ids=..., image_ids=..., mode=..., data_dir=..., fetch_neurosynth_words=..., resample=..., vectorize_words=..., verbose=...): # -> Bunch:
    """Download specific images and collections from neurovault.org.

    Any downloaded data is saved on the local disk and subsequent
    calls to this function will first look for the data locally before
    querying the server for more if necessary.

    This is the fast way to get the data from the server if we already
    know which images or collections we want.

    For more information, see :footcite:`Gorgolewski2015`,
    and :footcite:`Yarkoni2011`.

    Parameters
    ----------
    collection_ids : Container, optional
        The ids of whole collections to be downloaded.
        Default=().

    image_ids : Container, optional
        The ids of particular images to be downloaded. The metadata for the
        corresponding collections is also downloaded.
        Default=().

    mode : {'download_new', 'overwrite', 'offline'}, optional
        When to fetch an image from the server rather than the local
        disk. Default='download_new'.

        - 'download_new' (the default) means download only files that
          are not already on disk (regardless of modify date).
        - 'overwrite' means ignore files on disk and overwrite them.
        - 'offline' means load only data from disk; don't query server.

    data_dir : str, optional
        The directory we want to use for nilearn data. A subdirectory
        named "neurovault" will contain neurovault data.

    fetch_neurosynth_words : bool, optional
        Whether to collect words from Neurosynth. Default=False.

    resample : bool, optional (default=False)
        Resamples downloaded images to a 3x3x3 grid before saving them,
        to save disk space.

    vectorize_words : bool, optional
        If neurosynth words are downloaded, create a matrix of word
        counts and add it to the result. Also add to the result a
        vocabulary list. See ``sklearn.CountVectorizer`` for more info.
        Default=True.

    verbose : int, optional
        An integer in [0, 1, 2, 3] to control the verbosity level.
        Default=3.

    Returns
    -------
    Bunch
        A dict-like object which exposes its items as attributes. It contains:

            - 'images', the paths to downloaded files.
            - 'images_meta', the metadata for the images in a list of
              dictionaries.
            - 'collections_meta', the metadata for the
              collections.
            - 'description', a short description of the Neurovault dataset.

        If `fetch_neurosynth_words` and `vectorize_words` were set, it
        also contains:

            - 'vocabulary', a list of words
            - 'word_frequencies', the weight of the words returned by
              neurosynth.org for each image, such that the weight of word
              `vocabulary[j]` for the image found in `images[i]` is
              `word_frequencies[i, j]`

    See Also
    --------
    nilearn.datasets.fetch_neurovault
        Fetch data from Neurovault, but use filters on metadata to select
        images and collections rather than giving explicit lists of ids.

    Notes
    -----
    Images and collections from disk are fetched before remote data.

    In `download_new` mode, if a file exists on disk, it is not
    downloaded again, even if the version on the server is newer. Use
    `overwrite` mode to force a new download.

    Stops early if too many images fail to be downloaded in a row.

    References
    ----------
    .. footbibliography::

    """
    ...

def fetch_neurovault_motor_task(data_dir=..., verbose=...): # -> Bunch:
    """Fetch left vs right button press group contrast map from NeuroVault.

    Parameters
    ----------
    data_dir : string, optional
        Path of the data directory. Used to force data storage in a specified
        location.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    data : Bunch
        A dict-like object which exposes its items as attributes. It contains:
            - 'images', the paths to downloaded files.
            - 'images_meta', the metadata for the images in a list of
              dictionaries.
            - 'collections_meta', the metadata for the
              collections.
            - 'description', a short description of the Neurovault dataset.

    Notes
    -----
    The 'left vs right button press' contrast is used:
    https://neurovault.org/images/10426/

    See Also
    --------
    nilearn.datasets.fetch_neurovault_ids
    nilearn.datasets.fetch_neurovault
    nilearn.datasets.fetch_neurovault_auditory_computation_task

    """
    ...

def fetch_neurovault_auditory_computation_task(data_dir=..., verbose=...): # -> Bunch:
    """Fetch a contrast map from NeuroVault showing \
    the effect of mental subtraction upon auditory instructions.

    Parameters
    ----------
    data_dir : string, optional
        Path of the data directory. Used to force data storage in a specified
        location.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    data : Bunch
        A dict-like object which exposes its items as attributes. It contains:
            - 'images', the paths to downloaded files.
            - 'images_meta', the metadata for the images in a list of
              dictionaries.
            - 'collections_meta', the metadata for the
              collections.
            - 'description', a short description of the Neurovault dataset.

    Notes
    -----
    The 'auditory_calculation_vs_baseline' contrast is used:
    https://neurovault.org/images/32980/

    See Also
    --------
    nilearn.datasets.fetch_neurovault_ids
    nilearn.datasets.fetch_neurovault
    nilearn.datasets.fetch_neurovault_motor_task

    """
    ...

