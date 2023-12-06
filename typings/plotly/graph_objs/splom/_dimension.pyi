"""
This type stub file was generated by pyright.
"""

from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType

class Dimension(_BaseTraceHierarchyType):
    _parent_path_str = ...
    _path_str = ...
    _valid_props = ...
    @property
    def axis(self): # -> tuple[Unknown, ...] | Dimension | None:
        """
        The 'axis' property is an instance of Axis
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.splom.dimension.Axis`
          - A dict of string/value properties that will be passed
            to the Axis constructor

            Supported dict properties:

                matches
                    Determines whether or not the x & y axes
                    generated by this dimension match. Equivalent
                    to setting the `matches` axis attribute in the
                    layout with the correct axis id.
                type
                    Sets the axis type for this dimension's
                    generated x and y axes. Note that the axis
                    `type` values set in layout take precedence
                    over this attribute.

        Returns
        -------
        plotly.graph_objs.splom.dimension.Axis
        """
        ...
    
    @axis.setter
    def axis(self, val): # -> None:
        ...
    
    @property
    def label(self): # -> tuple[Unknown, ...] | Dimension | None:
        """
        Sets the label corresponding to this splom dimension.

        The 'label' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string

        Returns
        -------
        str
        """
        ...
    
    @label.setter
    def label(self, val): # -> None:
        ...
    
    @property
    def name(self): # -> tuple[Unknown, ...] | Dimension | None:
        """
        When used in a template, named items are created in the output
        figure in addition to any items the figure already has in this
        array. You can modify these items in the output figure by
        making your own item with `templateitemname` matching this
        `name` alongside your modifications (including `visible: false`
        or `enabled: false` to hide it). Has no effect outside of a
        template.

        The 'name' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string

        Returns
        -------
        str
        """
        ...
    
    @name.setter
    def name(self, val): # -> None:
        ...
    
    @property
    def templateitemname(self): # -> tuple[Unknown, ...] | Dimension | None:
        """
        Used to refer to a named item in this array in the template.
        Named items from the template will be created even without a
        matching item in the input figure, but you can modify one by
        making an item with `templateitemname` matching its `name`,
        alongside your modifications (including `visible: false` or
        `enabled: false` to hide it). If there is no template or no
        matching item, this item will be hidden unless you explicitly
        show it with `visible: true`.

        The 'templateitemname' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string

        Returns
        -------
        str
        """
        ...
    
    @templateitemname.setter
    def templateitemname(self, val): # -> None:
        ...
    
    @property
    def values(self): # -> tuple[Unknown, ...] | Dimension | None:
        """
        Sets the dimension values to be plotted.

        The 'values' property is an array that may be specified as a tuple,
        list, numpy array, or pandas Series

        Returns
        -------
        numpy.ndarray
        """
        ...
    
    @values.setter
    def values(self, val): # -> None:
        ...
    
    @property
    def valuessrc(self): # -> tuple[Unknown, ...] | Dimension | None:
        """
        Sets the source reference on Chart Studio Cloud for `values`.

        The 'valuessrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        ...
    
    @valuessrc.setter
    def valuessrc(self, val): # -> None:
        ...
    
    @property
    def visible(self): # -> tuple[Unknown, ...] | Dimension | None:
        """
        Determines whether or not this dimension is shown on the graph.
        Note that even visible false dimension contribute to the
        default grid generate by this splom trace.

        The 'visible' property must be specified as a bool
        (either True, or False)

        Returns
        -------
        bool
        """
        ...
    
    @visible.setter
    def visible(self, val): # -> None:
        ...
    
    def __init__(self, arg=..., axis=..., label=..., name=..., templateitemname=..., values=..., valuessrc=..., visible=..., **kwargs) -> None:
        """
        Construct a new Dimension object

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of
            :class:`plotly.graph_objs.splom.Dimension`
        axis
            :class:`plotly.graph_objects.splom.dimension.Axis`
            instance or dict with compatible properties
        label
            Sets the label corresponding to this splom dimension.
        name
            When used in a template, named items are created in the
            output figure in addition to any items the figure
            already has in this array. You can modify these items
            in the output figure by making your own item with
            `templateitemname` matching this `name` alongside your
            modifications (including `visible: false` or `enabled:
            false` to hide it). Has no effect outside of a
            template.
        templateitemname
            Used to refer to a named item in this array in the
            template. Named items from the template will be created
            even without a matching item in the input figure, but
            you can modify one by making an item with
            `templateitemname` matching its `name`, alongside your
            modifications (including `visible: false` or `enabled:
            false` to hide it). If there is no template or no
            matching item, this item will be hidden unless you
            explicitly show it with `visible: true`.
        values
            Sets the dimension values to be plotted.
        valuessrc
            Sets the source reference on Chart Studio Cloud for
            `values`.
        visible
            Determines whether or not this dimension is shown on
            the graph. Note that even visible false dimension
            contribute to the default grid generate by this splom
            trace.

        Returns
        -------
        Dimension
        """
        ...
    

