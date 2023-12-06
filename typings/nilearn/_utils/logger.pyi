"""
This type stub file was generated by pyright.
"""

"""Logging facility for nilearn."""
def log(msg, verbose=..., object_classes=..., stack_level=..., msg_level=...): # -> None:
    """Display a message to the user, depending on the verbosity level.

    This function allows to display some information that references an object
    that is significant to the user, instead of a internal function. The goal
    is to make user's code as simple to debug as possible.

    Parameters
    ----------
    msg : str
        Message to display.

    verbose : int, optional
        Current verbosity level. Message is displayed if this value is greater
        or equal to msg_level. Default=1.

    object_classes : tuple of type, optional
        Classes that should appear to emit the message.
        Default=(BaseEstimator, ).

    stack_level : int, optional
        If no object in the call stack matches object_classes, go back that
        amount in the call stack and display class/function name thereof.
        Default=1.

    msg_level : int, optional
        Verbosity level at and above which message should be displayed to the
        user. Most of the time this parameter can be left unchanged.
        Default=1.

    Notes
    -----
    This function does tricky things to ensure that the proper object is
    referenced in the message. If it is called e.g. inside a function that is
    called by a method of an object inheriting from any class in
    object_classes, then the name of the object (and the method) will be
    displayed to the user. If several matching objects exist in the call
    stack, the highest one is used (first call chronologically), because this
    is the one which is most likely to have been written in the user's script.

    """
    ...

