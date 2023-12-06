"""
This type stub file was generated by pyright.
"""

import sys

__all__ = ['PY3', 'b', 'basestring_', 'bytes', 'next', 'is_unicode', 'iteritems']
PY3 = ...
if sys.version_info[0] < 3:
    ...
else:
    def b(s): # -> bytes:
        ...
    
    def iteritems(d, **kw):
        ...
    
    next = ...
    basestring_ = ...
    bytes = bytes
text = str
def is_unicode(obj): # -> bool:
    ...

def coerce_text(v): # -> bytes | str:
    ...

