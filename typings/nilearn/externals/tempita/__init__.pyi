"""
This type stub file was generated by pyright.
"""

import os
import re
import sys
import tokenize
from urllib.parse import quote as url_quote
from io import StringIO
from html import escape as html_escape
from ._looper import looper
from .compat3 import PY3, basestring_, bytes, coerce_text, is_unicode, iteritems, next

"""
A small templating language

This implements a small templating language.  This language implements
if/elif/else, for/continue/break, expressions, and blocks of Python
code.  The syntax is::

  {{any expression (function calls etc)}}
  {{any expression | filter}}
  {{for x in y}}...{{endfor}}
  {{if x}}x{{elif y}}y{{else}}z{{endif}}
  {{py:x=1}}
  {{py:
  def foo(bar):
      return 'baz'
  }}
  {{default var = default_value}}
  {{# comment}}

You use this with the ``Template`` class or the ``sub`` shortcut.
The ``Template`` class takes the template string and the name of
the template (for errors) and a default namespace.  Then (like
``string.Template``) you can call the ``tmpl.substitute(**kw)``
method to make a substitution (or ``tmpl.substitute(a_dict)``).

``sub(content, **kw)`` substitutes the template immediately.  You
can use ``__name='tmpl.html'`` to set the name of the template.

If there are syntax errors ``TemplateError`` will be raised.
"""
__all__ = ['TemplateError', 'Template', 'sub', 'HTMLTemplate', 'sub_html', 'html', 'bunch']
in_re = ...
var_re = ...
class TemplateError(Exception):
    """Exception raised while parsing a template
    """
    def __init__(self, message, position, name=...) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    


class _TemplateContinue(Exception):
    ...


class _TemplateBreak(Exception):
    ...


def get_file_template(name, from_template):
    ...

class Template:
    default_namespace = ...
    default_encoding = ...
    default_inherit = ...
    def __init__(self, content, name=..., namespace=..., stacklevel=..., get_template=..., default_inherit=..., line_offset=..., delimiters=...) -> None:
        ...
    
    @classmethod
    def from_filename(cls, filename, namespace=..., encoding=..., default_inherit=..., get_template=...): # -> Self@Template:
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def substitute(self, *args, **kw): # -> LiteralString:
        ...
    


def sub(content, delimiters=..., **kw): # -> LiteralString:
    ...

def paste_script_template_renderer(content, vars, filename=...): # -> LiteralString:
    ...

class bunch(dict):
    def __init__(self, **kw) -> None:
        ...
    
    def __setattr__(self, name, value): # -> None:
        ...
    
    def __getattr__(self, name):
        ...
    
    def __getitem__(self, key):
        ...
    
    def __repr__(self): # -> str:
        ...
    


class html:
    def __init__(self, value) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    
    def __html__(self): # -> Unknown:
        ...
    
    def __repr__(self): # -> str:
        ...
    


def html_quote(value, force=...): # -> bytes | Literal['']:
    ...

def url(v): # -> str:
    ...

def attr(**kw): # -> html:
    ...

class HTMLTemplate(Template):
    default_namespace = ...


def sub_html(content, **kw): # -> LiteralString:
    ...

class TemplateDef:
    def __init__(self, template, func_name, func_signature, body, ns, pos, bound_self=...) -> None:
        ...
    
    def __repr__(self): # -> LiteralString:
        ...
    
    def __str__(self) -> str:
        ...
    
    def __call__(self, *args, **kw): # -> LiteralString:
        ...
    
    def __get__(self, obj, type=...): # -> Self@TemplateDef | TemplateDef:
        ...
    


class TemplateObject:
    def __init__(self, name) -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class TemplateObjectGetter:
    def __init__(self, template_obj) -> None:
        ...
    
    def __getattr__(self, attr): # -> Any | _Empty:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class _Empty:
    def __call__(self, *args, **kw): # -> Self@_Empty:
        ...
    
    def __str__(self) -> str:
        ...
    
    def __repr__(self): # -> Literal['Empty']:
        ...
    
    def __unicode__(self): # -> Literal['']:
        ...
    
    def __iter__(self): # -> Iterator[Any]:
        ...
    
    def __bool__(self): # -> Literal[False]:
        ...
    
    if sys.version < "3":
        __nonzero__ = ...


Empty = ...
def lex(s, name=..., trim_whitespace=..., line_offset=..., delimiters=...): # -> list[Unknown]:
    ...

statement_re = ...
single_statements = ...
trail_whitespace_re = ...
lead_whitespace_re = ...
def trim_lex(tokens):
    ...

def find_position(string, index, last_index, last_pos): # -> tuple[Unknown, Unknown]:
    """
    Given a string and index, return (line, column)
    """
    ...

def parse(s, name=..., line_offset=..., delimiters=...): # -> list[Unknown]:
    ...

def parse_expr(tokens, name, context=...): # -> tuple[Unknown, Unknown] | tuple[tuple[Literal['py'], Unknown, Unknown], Unknown] | tuple[tuple[Unknown, Unknown], Unknown] | tuple[tuple[Unknown | Literal['cond'], ...], Unknown] | tuple[tuple[Literal['for'], Unknown, tuple[Unknown, ...], Unknown, list[Unknown]], Unknown] | tuple[tuple[Literal['default'], Unknown, Unknown, Unknown], Unknown] | tuple[tuple[Literal['inherit'], Unknown, Unknown], Unknown] | tuple[tuple[Literal['def'], Unknown, Unknown, tuple[tuple[()], None, None, dict[Unknown, Unknown]] | Unknown, list[Unknown]], Unknown] | tuple[tuple[Literal['comment'], Unknown, Unknown], Unknown] | tuple[tuple[Literal['expr'], Unknown, Unknown], Unknown] | None:
    ...

def parse_cond(tokens, name, context): # -> tuple[tuple[Unknown | Literal['cond'], ...], Unknown] | None:
    ...

def parse_one_cond(tokens, name, context): # -> tuple[tuple[Literal['if'], Unknown, Unknown, list[Unknown]] | tuple[Literal['elif'], Unknown, Unknown, list[Unknown]] | tuple[Literal['else'], Unknown, None, list[Unknown]] | Unbound, Unknown] | None:
    ...

def parse_for(tokens, name, context): # -> tuple[tuple[Literal['for'], Unknown, tuple[Unknown, ...], Unknown, list[Unknown]], Unknown] | None:
    ...

def parse_default(tokens, name, context): # -> tuple[tuple[Literal['default'], Unknown, Unknown, Unknown], Unknown]:
    ...

def parse_inherit(tokens, name, context): # -> tuple[tuple[Literal['inherit'], Unknown, Unknown], Unknown]:
    ...

def parse_def(tokens, name, context): # -> tuple[tuple[Literal['def'], Unknown, Unknown, tuple[tuple[()], None, None, dict[Unknown, Unknown]] | Unknown, list[Unknown]], Unknown] | None:
    ...

def parse_signature(sig_text, name, pos):
    ...

def isolate_expression(string, start_pos, end_pos): # -> LiteralString:
    ...

_fill_command_usage = ...
def fill_command(args=...): # -> None:
    ...

if __name__ == '__main__':
    ...