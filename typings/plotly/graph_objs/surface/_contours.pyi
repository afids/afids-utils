"""
This type stub file was generated by pyright.
"""

from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType

class Contours(_BaseTraceHierarchyType):
    _parent_path_str = ...
    _path_str = ...
    _valid_props = ...
    @property
    def x(self): # -> tuple[Unknown, ...] | Contours | None:
        """
        The 'x' property is an instance of X
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.surface.contours.X`
          - A dict of string/value properties that will be passed
            to the X constructor

            Supported dict properties:

                color
                    Sets the color of the contour lines.
                end
                    Sets the end contour level value. Must be more
                    than `contours.start`
                highlight
                    Determines whether or not contour lines about
                    the x dimension are highlighted on hover.
                highlightcolor
                    Sets the color of the highlighted contour
                    lines.
                highlightwidth
                    Sets the width of the highlighted contour
                    lines.
                project
                    :class:`plotly.graph_objects.surface.contours.x
                    .Project` instance or dict with compatible
                    properties
                show
                    Determines whether or not contour lines about
                    the x dimension are drawn.
                size
                    Sets the step between each contour level. Must
                    be positive.
                start
                    Sets the starting contour level value. Must be
                    less than `contours.end`
                usecolormap
                    An alternate to "color". Determines whether or
                    not the contour lines are colored using the
                    trace "colorscale".
                width
                    Sets the width of the contour lines.

        Returns
        -------
        plotly.graph_objs.surface.contours.X
        """
        ...
    
    @x.setter
    def x(self, val): # -> None:
        ...
    
    @property
    def y(self): # -> tuple[Unknown, ...] | Contours | None:
        """
        The 'y' property is an instance of Y
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.surface.contours.Y`
          - A dict of string/value properties that will be passed
            to the Y constructor

            Supported dict properties:

                color
                    Sets the color of the contour lines.
                end
                    Sets the end contour level value. Must be more
                    than `contours.start`
                highlight
                    Determines whether or not contour lines about
                    the y dimension are highlighted on hover.
                highlightcolor
                    Sets the color of the highlighted contour
                    lines.
                highlightwidth
                    Sets the width of the highlighted contour
                    lines.
                project
                    :class:`plotly.graph_objects.surface.contours.y
                    .Project` instance or dict with compatible
                    properties
                show
                    Determines whether or not contour lines about
                    the y dimension are drawn.
                size
                    Sets the step between each contour level. Must
                    be positive.
                start
                    Sets the starting contour level value. Must be
                    less than `contours.end`
                usecolormap
                    An alternate to "color". Determines whether or
                    not the contour lines are colored using the
                    trace "colorscale".
                width
                    Sets the width of the contour lines.

        Returns
        -------
        plotly.graph_objs.surface.contours.Y
        """
        ...
    
    @y.setter
    def y(self, val): # -> None:
        ...
    
    @property
    def z(self): # -> tuple[Unknown, ...] | Contours | None:
        """
        The 'z' property is an instance of Z
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.surface.contours.Z`
          - A dict of string/value properties that will be passed
            to the Z constructor

            Supported dict properties:

                color
                    Sets the color of the contour lines.
                end
                    Sets the end contour level value. Must be more
                    than `contours.start`
                highlight
                    Determines whether or not contour lines about
                    the z dimension are highlighted on hover.
                highlightcolor
                    Sets the color of the highlighted contour
                    lines.
                highlightwidth
                    Sets the width of the highlighted contour
                    lines.
                project
                    :class:`plotly.graph_objects.surface.contours.z
                    .Project` instance or dict with compatible
                    properties
                show
                    Determines whether or not contour lines about
                    the z dimension are drawn.
                size
                    Sets the step between each contour level. Must
                    be positive.
                start
                    Sets the starting contour level value. Must be
                    less than `contours.end`
                usecolormap
                    An alternate to "color". Determines whether or
                    not the contour lines are colored using the
                    trace "colorscale".
                width
                    Sets the width of the contour lines.

        Returns
        -------
        plotly.graph_objs.surface.contours.Z
        """
        ...
    
    @z.setter
    def z(self, val): # -> None:
        ...
    
    def __init__(self, arg=..., x=..., y=..., z=..., **kwargs) -> None:
        """
        Construct a new Contours object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.surface.Contours`
        x
            :class:`plotly.graph_objects.surface.contours.X`
            instance or dict with compatible properties
        y
            :class:`plotly.graph_objects.surface.contours.Y`
            instance or dict with compatible properties
        z
            :class:`plotly.graph_objects.surface.contours.Z`
            instance or dict with compatible properties

        Returns
        -------
        Contours
        """
        ...
    

