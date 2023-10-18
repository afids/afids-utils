"""
This type stub file was generated by pyright.
"""

from nilearn._utils.docs import fill_doc
from nilearn.plotting.displays import CutAxes

@fill_doc
class BaseSlicer:
    """BaseSlicer implementation which main purpose is to auto adjust \
    the axes size to the data with different layout of cuts.

    It creates 3 linked axes for plotting orthogonal cuts.

    Attributes
    ----------
    cut_coords : 3 :obj:`tuple` of :obj:`int`
        The cut position, in world space.

    frame_axes : :class:`matplotlib.axes.Axes`, optional
        The matplotlib axes that will be subdivided in 3.

    black_bg : :obj:`bool`, optional
        If ``True``, the background of the figure will be put to
        black. If you wish to save figures with a black background,
        you will need to pass ``facecolor='k', edgecolor='k'``
        to :func:`~matplotlib.pyplot.savefig`.
        Default=False.

    brain_color : :obj:`tuple`, optional
        The brain color to use as the background color (e.g., for
        transparent colorbars).
        Default=(0.5, 0.5, 0.5)
    """
    _default_figsize = ...
    _axes_class = CutAxes
    def __init__(self, cut_coords, axes=..., black_bg=..., brain_color=..., **kwargs) -> None:
        ...
    
    @property
    def brain_color(self):
        """Return brain color."""
        ...
    
    @property
    def black_bg(self): # -> bool:
        """Return black background."""
        ...
    
    @staticmethod
    def find_cut_coords(img=..., threshold=..., cut_coords=...):
        """Act as placeholder and is not implemented in the base class \
        and has to be implemented in derived classes."""
        ...
    
    @classmethod
    @fill_doc
    def init_with_figure(cls, img, threshold=..., cut_coords=..., figure=..., axes=..., black_bg=..., leave_space=..., colorbar=..., brain_color=..., **kwargs): # -> Self@BaseSlicer:
        """Initialize the slicer with an image.

        Parameters
        ----------
        %(img)s
        cut_coords : 3 :obj:`tuple` of :obj:`int`
            The cut position, in world space.

        axes : :class:`matplotlib.axes.Axes`, optional
            The axes that will be subdivided in 3.

        black_bg : :obj:`bool`, optional
            If ``True``, the background of the figure will be put to
            black. If you wish to save figures with a black background,
            you will need to pass ``facecolor='k', edgecolor='k'``
            to :func:`matplotlib.pyplot.savefig`.
            Default=False.

        brain_color : :obj:`tuple`, optional
            The brain color to use as the background color (e.g., for
            transparent colorbars).
            Default=(0.5, 0.5, 0.5).
        """
        ...
    
    def title(self, text, x=..., y=..., size=..., color=..., bgcolor=..., alpha=..., **kwargs): # -> None:
        """Write a title to the view.

        Parameters
        ----------
        text : :obj:`str`
            The text of the title.

        x : :obj:`float`, optional
            The horizontal position of the title on the frame in
            fraction of the frame width. Default=0.01.

        y : :obj:`float`, optional
            The vertical position of the title on the frame in
            fraction of the frame height. Default=0.99.

        size : :obj:`int`, optional
            The size of the title text. Default=15.

        color : matplotlib color specifier, optional
            The color of the font of the title.

        bgcolor : matplotlib color specifier, optional
            The color of the background of the title.

        alpha : :obj:`float`, optional
            The alpha value for the background. Default=1.

        kwargs :
            Extra keyword arguments are passed to matplotlib's text
            function.
        """
        ...
    
    @fill_doc
    def add_overlay(self, img, threshold=..., colorbar=..., cbar_tick_format=..., cbar_vmin=..., cbar_vmax=..., **kwargs): # -> None:
        """Plot a 3D map in all the views.

        Parameters
        ----------
        %(img)s
            If it is a masked array, only the non-masked part will be plotted.

        threshold : :obj:`int` or :obj:`float` or ``None``, optional
            Threshold to apply:

                - If ``None`` is given, the maps are not thresholded.
                - If a number is given, it is used to threshold the maps:
                  values below the threshold (in absolute value) are
                  plotted as transparent.

            Default=1e-6.

        cbar_tick_format: str, optional
            Controls how to format the tick labels of the colorbar.
            Ex: use "%%i" to display as integers.
            Default is '%%.2g' for scientific notation.

        colorbar : :obj:`bool`, optional
            If ``True``, display a colorbar on the right of the plots.
            Default=False.

        kwargs : :obj:`dict`
            Extra keyword arguments are passed to function
            :func:`~matplotlib.pyplot.imshow`.

        cbar_vmin : :obj:`float`, optional
            Minimal value for the colorbar. If None, the minimal value
            is computed based on the data.

        cbar_vmax : :obj:`float`, optional
            Maximal value for the colorbar. If None, the maximal value
            is computed based on the data.
        """
        ...
    
    @fill_doc
    def add_contours(self, img, threshold=..., filled=..., **kwargs): # -> None:
        """Contour a 3D map in all the views.

        Parameters
        ----------
        %(img)s
            Provides image to plot.

        threshold : :obj:`int` or :obj:`float` or ``None``, optional
            Threshold to apply:

                - If ``None`` is given, the maps are not thresholded.
                - If a number is given, it is used to threshold the maps,
                  values below the threshold (in absolute value) are plotted
                  as transparent.

            Default=1e-6.

        filled : :obj:`bool`, optional
            If ``filled=True``, contours are displayed with color fillings.
            Default=False.

        kwargs : :obj:`dict`
            Extra keyword arguments are passed to function
            :func:`~matplotlib.pyplot.contour`, or function
            :func:`~matplotlib.pyplot.contourf`.
            Useful, arguments are typical "levels", which is a
            list of values to use for plotting a contour or contour
            fillings (if ``filled=True``), and
            "colors", which is one color or a list of colors for
            these contours.

        Notes
        -----
        If colors are not specified, default coloring choices
        (from matplotlib) for contours and contour_fillings can be
        different.

        """
        ...
    
    @fill_doc
    def add_edges(self, img, color=...): # -> None:
        """Plot the edges of a 3D map in all the views.

        Parameters
        ----------
        %(img)s
            The 3D map to be plotted.
            If it is a masked array, only the non-masked part will be plotted.

        color : matplotlib color: :obj:`str` or (r, g, b) value
            The color used to display the edge map.
            Default='r'.
        """
        ...
    
    def add_markers(self, marker_coords, marker_color=..., marker_size=..., **kwargs): # -> None:
        """Add markers to the plot.

        Parameters
        ----------
        marker_coords : :class:`~numpy.ndarray` of shape ``(n_markers, 3)``
            Coordinates of the markers to plot. For each slice, only markers
            that are 2 millimeters away from the slice are plotted.

        marker_color : pyplot compatible color or :obj:`list` of\
        shape ``(n_markers,)``, optional
            List of colors for each marker
            that can be string or matplotlib colors.
            Default='r'.

        marker_size : :obj:`float` or :obj:`list` of :obj:`float` of\
        shape ``(n_markers,)``, optional
            Size in pixel for each marker. Default=30.
        """
        ...
    
    def annotate(self, left_right=..., positions=..., scalebar=..., size=..., scale_size=..., scale_units=..., scale_loc=..., decimals=..., **kwargs): # -> None:
        """Add annotations to the plot.

        Parameters
        ----------
        left_right : :obj:`bool`, optional
            If ``True``, annotations indicating which side
            is left and which side is right are drawn.
            Default=True.

        positions : :obj:`bool`, optional
            If ``True``, annotations indicating the
            positions of the cuts are drawn.
            Default=True.

        scalebar : :obj:`bool`, optional
            If ``True``, cuts are annotated with a reference scale bar.
            For finer control of the scale bar, please check out
            the ``draw_scale_bar`` method on the axes in "axes" attribute
            of this object.
            Default=False.

        size : :obj:`int`, optional
            The size of the text used. Default=12.

        scale_size : :obj:`int` or :obj:`float`, optional
            The length of the scalebar, in units of ``scale_units``.
            Default=5.0.

        scale_units : {'cm', 'mm'}, optional
            The units for the ``scalebar``. Default='cm'.

        scale_loc : :obj:`int`, optional
            The positioning for the scalebar. Default=4.
            Valid location codes are:

                - 1: "upper right"
                - 2: "upper left"
                - 3: "lower left"
                - 4: "lower right"
                - 5: "right"
                - 6: "center left"
                - 7: "center right"
                - 8: "lower center"
                - 9: "upper center"
                - 10: "center"

        decimals : :obj:`int`, optional
            Number of decimal places on slice position annotation. If zero,
            the slice position is integer without decimal point.
            Default=0.

        kwargs : :obj:`dict`
            Extra keyword arguments are passed to matplotlib's text
            function.
        """
        ...
    
    def close(self): # -> None:
        """Close the figure.

        This is necessary to avoid leaking memory.
        """
        ...
    
    def savefig(self, filename, dpi=...): # -> None:
        """Save the figure to a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file name to save to. Its extension determines the
            file type, typically '.png', '.svg' or '.pdf'.

        dpi : ``None`` or scalar, optional
            The resolution in dots per inch.
            Default=None.
        """
        ...
    


@fill_doc
class OrthoSlicer(BaseSlicer):
    """Class to create 3 linked axes for plotting orthogonal \
    cuts of 3D maps.

    This visualization mode can be activated
    from Nilearn plotting functions, like
    :func:`~nilearn.plotting.plot_img`, by setting
    ``display_mode='ortho'``:

     .. code-block:: python

         from nilearn.datasets import load_mni152_template
         from nilearn.plotting import plot_img

         img = load_mni152_template()
         # display is an instance of the OrthoSlicer class
         display = plot_img(img, display_mode="ortho")


    Attributes
    ----------
    cut_coords : :obj:`list`
        The cut coordinates.

    axes : :obj:`dict` of :class:`~matplotlib.axes.Axes`
        The 3 axes used to plot each view.

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    Notes
    -----
    The extent of the different axes are adjusted to fit the data
    best in the viewing area.

    See Also
    --------
    nilearn.plotting.displays.MosaicSlicer : Three cuts are performed \
    along multiple rows and columns.
    nilearn.plotting.displays.TiledSlicer : Three cuts are performed \
    and arranged in a 2x2 grid.

    """
    _cut_displayed = ...
    _axes_class = CutAxes
    _default_figsize = ...
    @classmethod
    @fill_doc
    def find_cut_coords(cls, img=..., threshold=..., cut_coords=...): # -> list[Any | int]:
        """Instantiate the slicer and find cut coordinates.

        Parameters
        ----------
        %(img)s
        threshold : :obj:`int` or :obj:`float` or ``None``, optional
            Threshold to apply:

                - If ``None`` is given, the maps are not thresholded.
                - If a number is given, it is used to threshold the maps,
                  values below the threshold (in absolute value) are plotted
                  as transparent.

            Default=None.

        cut_coords : 3 :obj:`tuple` of :obj:`int`
            The cut position, in world space.
        """
        ...
    
    def draw_cross(self, cut_coords=..., **kwargs): # -> None:
        """Draw a crossbar on the plot to show where the cut is performed.

        Parameters
        ----------
        cut_coords : 3-:obj:`tuple` of :obj:`float`, optional
            The position of the cross to draw. If ``None`` is passed, the
            ``OrthoSlicer``'s cut coordinates are used.

        kwargs : :obj:`dict`
            Extra keyword arguments are passed to function
            :func:`~matplotlib.pyplot.axhline`.
        """
        ...
    


class TiledSlicer(BaseSlicer):
    """A class to create 3 axes for plotting orthogonal \
    cuts of 3D maps, organized in a 2x2 grid.

    This visualization mode can be activated from Nilearn plotting functions,
    like :func:`~nilearn.plotting.plot_img`, by setting
    ``display_mode='tiled'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_img

        img = load_mni152_template()
        # display is an instance of the TiledSlicer class
        display = plot_img(img, display_mode="tiled")

    Attributes
    ----------
    cut_coords : :obj:`list`
        The cut coordinates.

    axes : :obj:`dict` of :class:`~matplotlib.axes.Axes`
        The 3 axes used to plot each view.

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    Notes
    -----
    The extent of the different axes are adjusted to fit the data
    best in the viewing area.

    See Also
    --------
    nilearn.plotting.displays.MosaicSlicer : Three cuts are performed \
    along multiple rows and columns.
    nilearn.plotting.displays.OrthoSlicer : Three cuts are performed \
       and arranged in a 2x2 grid.

    """
    _cut_displayed = ...
    _axes_class = CutAxes
    _default_figsize = ...
    @classmethod
    def find_cut_coords(cls, img=..., threshold=..., cut_coords=...): # -> list[Any | int]:
        """Instantiate the slicer and find cut coordinates.

        Parameters
        ----------
        img : 3D :class:`~nibabel.nifti1.Nifti1Image`
            The brain map.

        threshold : :obj:`float`, optional
            The lower threshold to the positive activation.
            If ``None``, the activation threshold is computed using the
            80% percentile of the absolute value of the map.

        cut_coords : :obj:`list` of :obj:`float`, optional
            xyz world coordinates of cuts.

        Returns
        -------
        cut_coords : :obj:`list` of :obj:`float`
            xyz world coordinates of cuts.
        """
        ...
    
    def draw_cross(self, cut_coords=..., **kwargs): # -> None:
        """Draw a crossbar on the plot to show where the cut is performed.

        Parameters
        ----------
        cut_coords : 3-:obj:`tuple` of :obj:`float`, optional
            The position of the cross to draw. If ``None`` is passed, the
            ``OrthoSlicer``'s cut coordinates are used.

        kwargs : :obj:`dict`
            Extra keyword arguments are passed to function
            :func:`~matplotlib.pyplot.axhline`.
        """
        ...
    


class BaseStackedSlicer(BaseSlicer):
    """A class to create linked axes for plotting stacked cuts of 2D maps.

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~matplotlib.axes.Axes`
        The axes used to plot each view.

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    Notes
    -----
    The extent of the different axes are adjusted to fit the data
    best in the viewing area.
    """
    @classmethod
    def find_cut_coords(cls, img=..., threshold=..., cut_coords=...): # -> NDArray[Any] | Literal[7]:
        """Instantiate the slicer and find cut coordinates.

        Parameters
        ----------
        img : 3D :class:`~nibabel.nifti1.Nifti1Image`
            The brain map.

        threshold : :obj:`float`, optional
            The lower threshold to the positive activation.
            If ``None``, the activation threshold is computed using the
            80% percentile of the absolute value of the map.

        cut_coords : :obj:`list` of :obj:`float`, optional
            xyz world coordinates of cuts.

        Returns
        -------
        cut_coords : :obj:`list` of :obj:`float`
            xyz world coordinates of cuts.
        """
        ...
    
    def draw_cross(self, cut_coords=..., **kwargs): # -> None:
        """Draw a crossbar on the plot to show where the cut is performed.

        Parameters
        ----------
        cut_coords : 3-:obj:`tuple` of :obj:`float`, optional
            The position of the cross to draw. If ``None`` is passed, the
            ``OrthoSlicer``'s cut coordinates are used.

        kwargs : :obj:`dict`
            Extra keyword arguments are passed to function
            :func:`matplotlib.pyplot.axhline`.
        """
        ...
    


class XSlicer(BaseStackedSlicer):
    """The ``XSlicer`` class enables sagittal visualization with \
    plotting functions of Nilearn like \
    :func:`nilearn.plotting.plot_img`.

    This visualization mode
    can be activated by setting ``display_mode='x'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_img

        img = load_mni152_template()
        # display is an instance of the XSlicer class
        display = plot_img(img, display_mode="x")

    Attributes
    ----------
    cut_coords : 1D :class:`~numpy.ndarray`
        The cut coordinates.

    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.CutAxes`
        The axes used for plotting.

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.YSlicer : Coronal view
    nilearn.plotting.displays.ZSlicer : Axial view

    """
    _direction = ...
    _default_figsize = ...


class YSlicer(BaseStackedSlicer):
    """The ``YSlicer`` class enables coronal visualization with \
    plotting functions of Nilearn like \
    :func:`nilearn.plotting.plot_img`.

    This visualization mode
    can be activated by setting ``display_mode='y'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_img

        img = load_mni152_template()
        # display is an instance of the YSlicer class
        display = plot_img(img, display_mode="y")

    Attributes
    ----------
    cut_coords : 1D :class:`~numpy.ndarray`
        The cut coordinates.

    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.CutAxes`
        The axes used for plotting.

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.XSlicer : Sagittal view
    nilearn.plotting.displays.ZSlicer : Axial view

    """
    _direction = ...
    _default_figsize = ...


class ZSlicer(BaseStackedSlicer):
    """The ``ZSlicer`` class enables axial visualization with \
    plotting functions of Nilearn like \
    :func:`nilearn.plotting.plot_img`.

    This visualization mode
    can be activated by setting ``display_mode='z'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_img

        img = load_mni152_template()
        # display is an instance of the ZSlicer class
        display = plot_img(img, display_mode="z")

    Attributes
    ----------
    cut_coords : 1D :class:`~numpy.ndarray`
        The cut coordinates.

    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.CutAxes`
        The axes used for plotting.

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.XSlicer : Sagittal view
    nilearn.plotting.displays.YSlicer : Coronal view

    """
    _direction = ...
    _default_figsize = ...


class XZSlicer(OrthoSlicer):
    """The ``XZSlicer`` class enables to combine sagittal and axial views \
    on the same figure with plotting functions of Nilearn like \
    :func:`nilearn.plotting.plot_img`.

    This visualization mode
    can be activated by setting ``display_mode='xz'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_img

        img = load_mni152_template()
        # display is an instance of the XZSlicer class
        display = plot_img(img, display_mode="xz")

    Attributes
    ----------
    cut_coords : :obj:`list` of :obj:`float`
        The cut coordinates.

    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.CutAxes`
        The axes used for plotting in each direction ('x' and 'z' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.YXSlicer : Coronal + Sagittal views
    nilearn.plotting.displays.YZSlicer : Coronal + Axial views

    """
    _cut_displayed = ...


class YXSlicer(OrthoSlicer):
    """The ``YXSlicer`` class enables to combine coronal and sagittal views \
    on the same figure with plotting functions of Nilearn like \
    :func:`nilearn.plotting.plot_img`.

    This visualization mode
    can be activated by setting ``display_mode='yx'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_img

        img = load_mni152_template()
        # display is an instance of the YXSlicer class
        display = plot_img(img, display_mode="yx")

    Attributes
    ----------
    cut_coords : :obj:`list` of :obj:`float`
        The cut coordinates.

    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.CutAxes`
        The axes used for plotting in each direction ('x' and 'y' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.XZSlicer : Sagittal + Axial views
    nilearn.plotting.displays.YZSlicer : Coronal + Axial views

    """
    _cut_displayed = ...


class YZSlicer(OrthoSlicer):
    """The ``YZSlicer`` class enables to combine coronal and axial views \
    on the same figure with plotting functions of Nilearn like \
    :func:`nilearn.plotting.plot_img`.

    This visualization mode
    can be activated by setting ``display_mode='yz'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_img

        img = load_mni152_template()
        # display is an instance of the YZSlicer class
        display = plot_img(img, display_mode="yz")

    Attributes
    ----------
    cut_coords : :obj:`list` of :obj:`float`
        The cut coordinates.

    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.CutAxes`
        The axes used for plotting in each direction ('y' and 'z' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.XZSlicer : Sagittal + Axial views
    nilearn.plotting.displays.YXSlicer : Coronal + Sagittal views

    """
    _cut_displayed = ...
    _default_figsize = ...


class MosaicSlicer(BaseSlicer):
    """A class to create 3 :class:`~matplotlib.axes.Axes` for \
    plotting cuts of 3D maps, in multiple rows and columns.

    This visualization mode can be activated from Nilearn plotting
    functions, like :func:`~nilearn.plotting.plot_img`, by setting
    ``display_mode='mosaic'``.

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_img

        img = load_mni152_template()
        # display is an instance of the MosaicSlicer class
        display = plot_img(img, display_mode="mosaic")

    Attributes
    ----------
    cut_coords : :obj:`dict` <:obj:`str`: 1D :class:`~numpy.ndarray`>
        The cut coordinates in a dictionary. The keys are the directions
        ('x', 'y', 'z'), and the values are arrays holding the cut
        coordinates.

    axes : :obj:`dict` of :class:`~matplotlib.axes.Axes`
        The 3 axes used to plot multiple views.

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.TiledSlicer : Three cuts are performed \
    in orthogonal directions.
    nilearn.plotting.displays.OrthoSlicer : Three cuts are performed \
    and arranged in a 2x2 grid.

    """
    _cut_displayed = ...
    _axes_class = CutAxes
    _default_figsize = ...
    @classmethod
    def find_cut_coords(cls, img=..., threshold=..., cut_coords=...): # -> dict[Unknown, Unknown]:
        """Instantiate the slicer and find cut coordinates for mosaic plotting.

        Parameters
        ----------
        img : 3D :class:`~nibabel.nifti1.Nifti1Image`, optional
            The brain image.

        threshold : :obj:`float`, optional
            The lower threshold to the positive activation. If ``None``,
            the activation threshold is computed using the 80% percentile of
            the absolute value of the map.

        cut_coords : :obj:`list` / :obj:`tuple` of 3 :obj:`float`,\
        :obj:`int`, optional
            xyz world coordinates of cuts. If ``cut_coords``
            are not provided, 7 coordinates of cuts are automatically
            calculated.

        Returns
        -------
        cut_coords : :obj:`dict`
            xyz world coordinates of cuts in a direction.
            Each key denotes the direction.
        """
        ...
    
    def draw_cross(self, cut_coords=..., **kwargs): # -> None:
        """Draw a crossbar on the plot to show where the cut is performed.

        Parameters
        ----------
        cut_coords : 3-:obj:`tuple` of :obj:`float`, optional
            The position of the cross to draw. If ``None`` is passed, the
            ``OrthoSlicer``'s cut coordinates are used.

        kwargs : :obj:`dict`
            Extra keyword arguments are passed to function
            :func:`matplotlib.pyplot.axhline`.
        """
        ...
    


SLICERS = ...
def get_slicer(display_mode): # -> ((img: Unknown, threshold: Unknown | None = None, cut_coords: Unknown | None = None, figure: Unknown | None = None, axes: Unknown | None = None, black_bg: bool = False, leave_space: bool = False, colorbar: bool = False, brain_color: Unknown = (0.5, 0.5, 0.5), **kwargs: Unknown) -> OrthoSlicer) | ((img: Unknown, threshold: Unknown | None = None, cut_coords: Unknown | None = None, figure: Unknown | None = None, axes: Unknown | None = None, black_bg: bool = False, leave_space: bool = False, colorbar: bool = False, brain_color: Unknown = (0.5, 0.5, 0.5), **kwargs: Unknown) -> TiledSlicer) | ((img: Unknown, threshold: Unknown | None = None, cut_coords: Unknown | None = None, figure: Unknown | None = None, axes: Unknown | None = None, black_bg: bool = False, leave_space: bool = False, colorbar: bool = False, brain_color: Unknown = (0.5, 0.5, 0.5), **kwargs: Unknown) -> MosaicSlicer) | ((img: Unknown, threshold: Unknown | None = None, cut_coords: Unknown | None = None, figure: Unknown | None = None, axes: Unknown | None = None, black_bg: bool = False, leave_space: bool = False, colorbar: bool = False, brain_color: Unknown = (0.5, 0.5, 0.5), **kwargs: Unknown) -> XSlicer) | ((img: Unknown, threshold: Unknown | None = None, cut_coords: Unknown | None = None, figure: Unknown | None = None, axes: Unknown | None = None, black_bg: bool = False, leave_space: bool = False, colorbar: bool = False, brain_color: Unknown = (0.5, 0.5, 0.5), **kwargs: Unknown) -> YSlicer) | ((img: Unknown, threshold: Unknown | None = None, cut_coords: Unknown | None = None, figure: Unknown | None = None, axes: Unknown | None = None, black_bg: bool = False, leave_space: bool = False, colorbar: bool = False, brain_color: Unknown = (0.5, 0.5, 0.5), **kwargs: Unknown) -> ZSlicer):
    """Retrieve a slicer from a given display mode.

    Parameters
    ----------
    display_mode : :obj:`str`
        The desired display mode.
        Possible options are:

            - "ortho": Three cuts are performed in orthogonal directions.
            - "tiled": Three cuts are performed and arranged in a 2x2 grid.
            - "mosaic": Three cuts are performed along multiple rows and
              columns.
            - "x": Sagittal
            - "y": Coronal
            - "z": Axial
            - "xz": Sagittal + Axial
            - "yz": Coronal + Axial
            - "yx": Coronal + Sagittal

    Returns
    -------
    slicer : An instance of one of the subclasses of\
    :class:`~nilearn.plotting.displays.BaseSlicer`

        The slicer corresponding to the requested display mode:

            - "ortho": Returns an
              :class:`~nilearn.plotting.displays.OrthoSlicer`.
            - "tiled": Returns a
              :class:`~nilearn.plotting.displays.TiledSlicer`.
            - "mosaic": Returns a
              :class:`~nilearn.plotting.displays.MosaicSlicer`.
            - "xz": Returns a
              :class:`~nilearn.plotting.displays.XZSlicer`.
            - "yz": Returns a
              :class:`~nilearn.plotting.displays.YZSlicer`.
            - "yx": Returns a
              :class:`~nilearn.plotting.displays.YZSlicer`.
            - "x": Returns a
              :class:`~nilearn.plotting.displays.XSlicer`.
            - "y": Returns a
              :class:`~nilearn.plotting.displays.YSlicer`.
            - "z": Returns a
              :class:`~nilearn.plotting.displays.ZSlicer`.

    """
    ...

