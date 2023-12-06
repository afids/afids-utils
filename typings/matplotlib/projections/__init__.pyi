"""
This type stub file was generated by pyright.
"""

from .. import _docstring, axes
from .geo import AitoffAxes, HammerAxes, LambertAxes, MollweideAxes
from .polar import PolarAxes
from mpl_toolkits.mplot3d import Axes3D

"""
Non-separable transforms that map from data space to screen space.

Projections are defined as `~.axes.Axes` subclasses.  They include the
following elements:

- A transformation from data coordinates into display coordinates.

- An inverse of that transformation.  This is used, for example, to convert
  mouse positions from screen space back into data space.

- Transformations for the gridlines, ticks and ticklabels.  Custom projections
  will often need to place these elements in special locations, and Matplotlib
  has a facility to help with doing so.

- Setting up default values (overriding `~.axes.Axes.cla`), since the defaults
  for a rectilinear axes may not be appropriate.

- Defining the shape of the axes, for example, an elliptical axes, that will be
  used to draw the background of the plot and for clipping any data elements.

- Defining custom locators and formatters for the projection.  For example, in
  a geographic projection, it may be more convenient to display the grid in
  degrees, even if the data is in radians.

- Set up interactive panning and zooming.  This is left as an "advanced"
  feature left to the reader, but there is an example of this for polar plots
  in `matplotlib.projections.polar`.

- Any additional methods for additional convenience or features.

Once the projection axes is defined, it can be used in one of two ways:

- By defining the class attribute ``name``, the projection axes can be
  registered with `matplotlib.projections.register_projection` and subsequently
  simply invoked by name::

      fig.add_subplot(projection="my_proj_name")

- For more complex, parameterisable projections, a generic "projection" object
  may be defined which includes the method ``_as_mpl_axes``. ``_as_mpl_axes``
  should take no arguments and return the projection's axes subclass and a
  dictionary of additional arguments to pass to the subclass' ``__init__``
  method.  Subsequently a parameterised projection can be initialised with::

      fig.add_subplot(projection=MyProjection(param1=param1_value))

  where MyProjection is an object which implements a ``_as_mpl_axes`` method.

A full-fledged and heavily annotated example is in
:doc:`/gallery/misc/custom_projection`.  The polar plot functionality in
`matplotlib.projections.polar` may also be of interest.
"""
class ProjectionRegistry:
    """A mapping of registered projection names to projection classes."""
    def __init__(self) -> None:
        ...
    
    def register(self, *projections): # -> None:
        """Register a new set of projections."""
        ...
    
    def get_projection_class(self, name):
        """Get a projection class from its *name*."""
        ...
    
    def get_projection_names(self): # -> list[Unknown]:
        """Return the names of all projections currently registered."""
        ...
    


projection_registry = ...
def register_projection(cls): # -> None:
    ...

def get_projection_class(projection=...):
    """
    Get a projection class from its name.

    If *projection* is None, a standard rectilinear projection is returned.
    """
    ...

get_projection_names = ...
