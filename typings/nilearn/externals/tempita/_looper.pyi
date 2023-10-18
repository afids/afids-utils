"""
This type stub file was generated by pyright.
"""

import sys

"""
Helper for looping over sequences, particular in templates.

Often in a loop in a template it's handy to know what's next up,
previously up, if this is the first or last item in the sequence, etc.
These can be awkward to manage in a normal Python loop, but using the
looper you can get a better sense of the context.  Use like::

    >>> for loop, item in looper(['a', 'b', 'c']):
    ...     print(loop.number, item)
    ...     if not loop.last:
    ...         print('---')
    1 a
    ---
    2 b
    ---
    3 c

"""
__all__ = ['looper']
class looper:
    """
    Helper for looping (particularly in templates)

    Use this like::

        for loop, item in looper(seq):
            if loop.first:
                ...
    """
    def __init__(self, seq) -> None:
        ...
    
    def __iter__(self): # -> looper_iter:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class looper_iter:
    def __init__(self, seq) -> None:
        ...
    
    def __iter__(self): # -> Self@looper_iter:
        ...
    
    def __next__(self): # -> tuple[loop_pos, Unknown]:
        ...
    
    if sys.version < "3":
        next = ...


class loop_pos:
    def __init__(self, seq, pos) -> None:
        ...
    
    def __repr__(self): # -> LiteralString:
        ...
    
    def index(self): # -> Unknown:
        ...
    
    index = ...
    def number(self):
        ...
    
    number = ...
    def item(self):
        ...
    
    item = ...
    def __next__(self): # -> None:
        ...
    
    __next__ = ...
    if sys.version < "3":
        next = ...
    def previous(self): # -> None:
        ...
    
    previous = ...
    def odd(self): # -> bool:
        ...
    
    odd = ...
    def even(self):
        ...
    
    even = ...
    def first(self):
        ...
    
    first = ...
    def last(self):
        ...
    
    last = ...
    def length(self): # -> int:
        ...
    
    length = ...
    def first_group(self, getter=...): # -> bool | Any:
        """
        Returns true if this item is the start of a new group,
        where groups mean that some attribute has changed.  The getter
        can be None (the item itself changes), an attribute name like
        ``'.attr'``, a function, or a dict key or list index.
        """
        ...
    
    def last_group(self, getter=...): # -> bool | Any:
        """
        Returns true if this item is the end of a new group,
        where groups mean that some attribute has changed.  The getter
        can be None (the item itself changes), an attribute name like
        ``'.attr'``, a function, or a dict key or list index.
        """
        ...
    

