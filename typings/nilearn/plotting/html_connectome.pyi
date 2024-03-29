"""
This type stub file was generated by pyright.
"""

from nilearn.plotting.html_document import HTMLDocument

"""Handle plotting of connectomes in html."""
class ConnectomeView(HTMLDocument):
    ...


def view_connectome(adjacency_matrix, node_coords, edge_threshold=..., edge_cmap=..., symmetric_cmap=..., linewidth=..., node_color=..., node_size=..., colorbar=..., colorbar_height=..., colorbar_fontsize=..., title=..., title_fontsize=...): # -> ConnectomeView:
    """Insert a 3d plot of a connectome into an HTML page.

    Parameters
    ----------
    adjacency_matrix : ndarray, shape=(n_nodes, n_nodes)
        The weights of the edges.

    node_coords : ndarray, shape=(n_nodes, 3)
        The coordinates of the nodes in MNI space.

    node_color : color or sequence of colors, optional
        Color(s) of the nodes. Default='auto'.

    edge_threshold : str, number or None, optional
        If None, no thresholding.
        If it is a number only connections of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%", and only connections of amplitude above the
        given percentile will be shown.

    edge_cmap : str or matplotlib colormap, optional
        Colormap to use. Default=cm.bwr.

    symmetric_cmap : bool, optional
        Make colormap symmetric (ranging from -vmax to vmax).
        Default=True.

    linewidth : float, optional
        Width of the lines that show connections. Default=6.0.

    node_size : float, optional
        Size of the markers showing the seeds in pixels.
        Default=3.0.

    colorbar : bool, optional
        Add a colorbar. Default=True.

    colorbar_height : float, optional
        Height of the colorbar, relative to the figure height.
        Default=0.5.

    colorbar_fontsize : int, optional
        Fontsize of the colorbar tick labels. Default=25.

    title : str, optional
        Title for the plot.

    title_fontsize : int, optional
        Fontsize of the title. Default=25.

    Returns
    -------
    ConnectomeView : plot of the connectome.
        It can be saved as an html page or rendered (transparently) by the
        Jupyter notebook. Useful methods are :

        - 'resize' to resize the plot displayed in a Jupyter notebook
        - 'save_as_html' to save the plot to a file
        - 'open_in_browser' to save the plot and open it in a web browser.

    See Also
    --------
    nilearn.plotting.plot_connectome:
        projected views of a connectome in a glass brain.

    nilearn.plotting.view_markers:
        interactive plot of colored markers

    nilearn.plotting.view_surf, nilearn.plotting.view_img_on_surf:
        interactive view of statistical maps or surface atlases on the cortical
        surface.

    """
    ...

def view_markers(marker_coords, marker_color=..., marker_size=..., marker_labels=..., title=..., title_fontsize=...): # -> ConnectomeView:
    """Insert a 3d plot of markers in a brain into an HTML page.

    Parameters
    ----------
    marker_coords : ndarray, shape=(n_nodes, 3)
        The coordinates of the nodes in MNI space.

    marker_color : ndarray, shape=(n_nodes,), optional
        colors of the markers: list of strings, hex rgb or rgba strings, rgb
        triplets, or rgba triplets (i.e. formats accepted by matplotlib, see
        https://matplotlib.org/users/colors.html#specifying-colors)

    marker_size : float or array-like, optional
        Size of the markers showing the seeds in pixels. Default=5.0.

    marker_labels : list of str, shape=(n_nodes), optional
        Labels for the markers: list of strings

    title : str, optional
        Title for the plot.

    title_fontsize : int, optional
        Fontsize of the title. Default=25.

    Returns
    -------
    ConnectomeView : plot of the markers.
        It can be saved as an html page or rendered (transparently) by the
        Jupyter notebook. Useful methods are :

        - 'resize' to resize the plot displayed in a Jupyter notebook
        - 'save_as_html' to save the plot to a file
        - 'open_in_browser' to save the plot and open it in a web browser.

    See Also
    --------
    nilearn.plotting.plot_connectome:
        projected views of a connectome in a glass brain.

    nilearn.plotting.view_connectome:
        interactive plot of a connectome.

    nilearn.plotting.view_surf, nilearn.plotting.view_img_on_surf:
        interactive view of statistical maps or surface atlases on the cortical
        surface.

    """
    ...

