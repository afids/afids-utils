"""
This type stub file was generated by pyright.
"""

from nilearn.plotting.displays._axes import GlassBrainAxes
from nilearn.plotting.displays._slicers import OrthoSlicer

class OrthoProjector(OrthoSlicer):
    """A class to create linked axes for plotting orthogonal projections \
    of 3D maps.

    This visualization mode can be activated from
    :func:`~nilearn.plotting.plot_glass_brain`, by setting
    ``display_mode='ortho'``:

      .. code-block:: python

          from nilearn.datasets import load_mni152_template
          from nilearn.plotting import plot_glass_brain

          img = load_mni152_template()
          # display is an instance of the OrthoProjector class
          display = plot_glass_brain(img, display_mode="ortho")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
     The 3 axes used to plot each view ('x', 'y', and 'z').

    frame_axes : :class:`~matplotlib.axes.Axes`
     The axes framing the whole set of views.

    """
    _axes_class = GlassBrainAxes
    @classmethod
    def find_cut_coords(cls, img=..., threshold=..., cut_coords=...): # -> tuple[None, ...]:
        """Find the coordinates of the cut."""
        ...
    
    def draw_cross(self, cut_coords=..., **kwargs): # -> None:
        """Do nothing.

        It does not make sense to draw crosses for the position of
        the cuts since we are taking the max along one axis.
        """
        ...
    
    def add_graph(self, adjacency_matrix, node_coords, node_color=..., node_size=..., edge_cmap=..., edge_vmin=..., edge_vmax=..., edge_threshold=..., edge_kwargs=..., node_kwargs=..., colorbar=...): # -> None:
        """Plot undirected graph on each of the axes.

        Parameters
        ----------
        adjacency_matrix : :class:`numpy.ndarray` of shape ``(n, n)``
            Represents the edges strengths of the graph.
            The matrix can be symmetric which will result in
            an undirected graph, or not symmetric which will
            result in a directed graph.

        node_coords : :class:`numpy.ndarray` of shape ``(n, 3)``
            3D coordinates of the graph nodes in world space.

        node_color : color or sequence of colors, optional
            Color(s) of the nodes. Default='auto'.

        node_size : scalar or array_like, optional
            Size(s) of the nodes in points^2. Default=50.

        edge_cmap : :class:`~matplotlib.colors.Colormap`, optional
            Colormap used for representing the strength of the edges.
            Default=cm.bwr.

        edge_vmin, edge_vmax : :obj:`float`, optional
            - If not ``None``, either or both of these values will be used
              to as the minimum and maximum values to color edges.
            - If ``None`` are supplied, the maximum absolute value within the
              given threshold will be used as minimum (multiplied by -1) and
              maximum coloring levels.

        edge_threshold : :obj:`str` or :obj:`int` or :obj:`float`, optional
            - If it is a number only the edges with a value greater than
              ``edge_threshold`` will be shown.
            - If it is a string it must finish with a percent sign,
              e.g. "25.3%", and only the edges with a abs(value) above
              the given percentile will be shown.

        edge_kwargs : :obj:`dict`, optional
            Will be passed as kwargs for each edge
            :class:`~matplotlib.lines.Line2D`.

        node_kwargs : :obj:`dict`
            Will be passed as kwargs to the function
            :func:`~matplotlib.pyplot.scatter` which plots all the
            nodes at one.
        """
        ...
    


class XProjector(OrthoProjector):
    """The ``XProjector`` class enables sagittal visualization through 2D \
    projections with :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode can be activated by setting ``display_mode='x'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the XProjector class
        display = plot_glass_brain(img, display_mode="x")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting.

    frame_axes : :class:`~matplotlib.axes.Axes`
     The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.YProjector : Coronal view
    nilearn.plotting.displays.ZProjector : Axial view

    """
    _cut_displayed = ...
    _default_figsize = ...


class YProjector(OrthoProjector):
    """The ``YProjector`` class enables coronal visualization through 2D \
    projections with :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode can be activated by setting ``display_mode='y'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the YProjector class
        display = plot_glass_brain(img, display_mode="y")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting.

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.XProjector : Sagittal view
    nilearn.plotting.displays.ZProjector : Axial view

    """
    _cut_displayed = ...
    _default_figsize = ...


class ZProjector(OrthoProjector):
    """The ``ZProjector`` class enables axial visualization through 2D \
    projections with :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode can be activated by setting ``display_mode='z'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the ZProjector class
        display = plot_glass_brain(img, display_mode="z")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
     The axes used for plotting.

    frame_axes : :class:`~matplotlib.axes.Axes`
     The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.XProjector : Sagittal view
    nilearn.plotting.displays.YProjector : Coronal view

    """
    _cut_displayed = ...
    _default_figsize = ...


class XZProjector(OrthoProjector):
    """The ``XZProjector`` class enables to combine sagittal \
    and axial views \
    on the same figure through 2D projections with \
    :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode
    can be activated by setting ``display_mode='xz'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the XZProjector class
        display = plot_glass_brain(img, display_mode="xz")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
            The axes used for plotting in each direction ('x' and 'z' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
                 The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.YXProjector : Coronal + Sagittal views
    nilearn.plotting.displays.YZProjector : Coronal + Axial views

    """
    _cut_displayed = ...


class YXProjector(OrthoProjector):
    """The ``YXProjector`` class enables to combine coronal \
    and sagittal views \
    on the same figure through 2D projections with \
    :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode
    can be activated by setting ``display_mode='yx'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the YXProjector class
        display = plot_glass_brain(img, display_mode="yx")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
     The axes used for plotting in each direction ('x' and 'y' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
     The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.XZProjector : Sagittal + Axial views
    nilearn.plotting.displays.YZProjector : Coronal + Axial views

    """
    _cut_displayed = ...


class YZProjector(OrthoProjector):
    """The ``YZProjector`` class enables to combine coronal and axial views \
    on the same figure through 2D projections with \
    :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode
    can be activated by setting ``display_mode='yz'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the YZProjector class
        display = plot_glass_brain(img, display_mode="yz")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
           The axes used for plotting in each direction ('y' and 'z' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
                 The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.XZProjector : Sagittal + Axial views
    nilearn.plotting.displays.YXProjector : Coronal + Sagittal views

    """
    _cut_displayed = ...
    _default_figsize = ...


class LYRZProjector(OrthoProjector):
    """The ``LYRZProjector`` class enables ? visualization \
    on the same figure through 2D projections with \
    :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode
    can be activated by setting ``display_mode='lyrz'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the LYRZProjector class
        display = plot_glass_brain(img, display_mode="lyrz")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
     The axes used for plotting in each direction ('l', 'y', 'r',
        and 'z' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.LZRYProjector : ?? views

    """
    _cut_displayed = ...


class LZRYProjector(OrthoProjector):
    """The ``LZRYProjector`` class enables ? visualization \
    on the same figure through 2D projections with \
    :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode
    can be activated by setting ``display_mode='lzry'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the LZRYProjector class
        display = plot_glass_brain(img, display_mode="lzry")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting in each direction ('l', 'z', 'r',
        and 'y' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.LYRZProjector : ?? views

    """
    _cut_displayed = ...


class LZRProjector(OrthoProjector):
    """The ``LZRProjector`` class enables hemispheric sagittal visualization \
    on the same figure through 2D projections with \
    :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode
    can be activated by setting ``display_mode='lzr'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the LZRProjector class
        display = plot_glass_brain(img, display_mode="lzr")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting in each direction ('l', 'z' and 'r' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.LYRProjector : ?? views

    """
    _cut_displayed = ...


class LYRProjector(OrthoProjector):
    """The ``LYRProjector`` class enables ? visualization \
    on the same figure through 2D projections with \
    :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode
    can be activated by setting ``display_mode='lyr'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the LYRProjector class
        display = plot_glass_brain(img, display_mode="lyr")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting in each direction ('l', 'y' and 'r' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.LZRProjector : ?? views

    """
    _cut_displayed = ...


class LRProjector(OrthoProjector):
    """The ``LRProjector`` class enables left-right visualization \
    on the same figure through 2D projections with \
    :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode
    can be activated by setting ``display_mode='lr'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the LRProjector class
        display = plot_glass_brain(img, display_mode="lr")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting in each direction ('l', and 'r' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    """
    _cut_displayed = ...


class LProjector(OrthoProjector):
    """The ``LProjector`` class enables the visualization of left 2D \
    projection with :func:`~nilearn.plotting.plot_glass_brain`.

    This
    visualization mode can be activated by setting ``display_mode='l'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the LProjector class
        display = plot_glass_brain(img, display_mode="l")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting in each direction ('l' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.RProjector : right projection view

    """
    _cut_displayed = ...
    _default_figsize = ...


class RProjector(OrthoProjector):
    """The ``RProjector`` class enables the visualization of right 2D \
    projection with :func:`~nilearn.plotting.plot_glass_brain`.

    This visualization mode can be activated by setting ``display_mode='r'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_glass_brain

        img = load_mni152_template()
        # display is an instance of the RProjector class
        display = plot_glass_brain(img, display_mode="r")

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.GlassBrainAxes`
        The axes used for plotting in each direction ('r' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.LProjector : left projection view

    """
    _cut_displayed = ...
    _default_figsize = ...


PROJECTORS = ...
def get_projector(display_mode): # -> (img: Unknown, threshold: Unknown | None = None, cut_coords: Unknown | None = None, figure: Unknown | None = None, axes: Unknown | None = None, black_bg: bool = False, leave_space: bool = False, colorbar: bool = False, brain_color: Unknown = (0.5, 0.5, 0.5), **kwargs: Unknown) -> OrthoProjector:
    """Retrieve a projector from a given display mode.

    Parameters
    ----------
    display_mode : {"ortho", "xz", "yz", "yx", "x", "y",\
    "z", "lzry", "lyrz", "lyr", "lzr", "lr", "l", "r"}
        The desired display mode.

    Returns
    -------
    projector : :class:`~nilearn.plotting.displays.OrthoProjector`\
    or instance of derived classes

        The projector corresponding to the requested display mode:

            - "ortho": Returns an
              :class:`~nilearn.plotting.displays.OrthoProjector`.
            - "xz": Returns a
              :class:`~nilearn.plotting.displays.XZProjector`.
            - "yz": Returns a
              :class:`~nilearn.plotting.displays.YZProjector`.
            - "yx": Returns a
              :class:`~nilearn.plotting.displays.YXProjector`.
            - "x": Returns a
              :class:`~nilearn.plotting.displays.XProjector`.
            - "y": Returns a
              :class:`~nilearn.plotting.displays.YProjector`.
            - "z": Returns a
              :class:`~nilearn.plotting.displays.ZProjector`.
            - "lzry": Returns a
              :class:`~nilearn.plotting.displays.LZRYProjector`.
            - "lyrz": Returns a
              :class:`~nilearn.plotting.displays.LYRZProjector`.
            - "lyr": Returns a
              :class:`~nilearn.plotting.displays.LYRProjector`.
            - "lzr": Returns a
              :class:`~nilearn.plotting.displays.LZRProjector`.
            - "lr": Returns a
              :class:`~nilearn.plotting.displays.LRProjector`.
            - "l": Returns a
              :class:`~nilearn.plotting.displays.LProjector`.
            - "z": Returns a
              :class:`~nilearn.plotting.displays.RProjector`.

    """
    ...

