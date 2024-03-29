"""
This type stub file was generated by pyright.
"""

"""Helper functions for the manipulation of fmriprep output confounds."""
img_file_patterns = ...
img_file_error = ...
class MissingConfound(Exception):
    """
    Exception raised when failing to find params in the confounds.

    Parameters
    ----------
    params : list of missing params
        Default values are empty lists.
    keywords: list of missing keywords
        Default values are empty lists.
    """
    def __init__(self, params=..., keywords=...) -> None:
        """Set missing parameters and keywords."""
        ...
    


