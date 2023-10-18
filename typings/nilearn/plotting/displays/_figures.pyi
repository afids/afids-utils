"""
This type stub file was generated by pyright.
"""

class SurfaceFigure:
    """Abstract class for surface figures.

    Parameters
    ----------
    figure : Figure instance or ``None``, optional
        Figure to be wrapped.

    output_file : :obj:`str` or ``None``, optional
        Path to output file.
    """
    def __init__(self, figure=..., output_file=...) -> None:
        ...
    
    def show(self):
        """Show the figure."""
        ...
    


class PlotlySurfaceFigure(SurfaceFigure):
    """Implementation of a surface figure obtained with `plotly` engine.

    Parameters
    ----------
    figure : Plotly figure instance or ``None``, optional
        Plotly figure instance to be used.

    output_file : :obj:`str` or ``None``, optional
        Output file path.

    Attributes
    ----------
    figure : Plotly figure instance
        Plotly figure. Use this attribute to access the underlying
        plotly figure for further customization and use plotly
        functionality.

    output_file : :obj:`str`
        Output file path.

    """
    def __init__(self, figure=..., output_file=...) -> None:
        ...
    
    def show(self, renderer=...): # -> None:
        """Show the figure.

        Parameters
        ----------
        renderer : :obj:`str`, optional
            Plotly renderer to be used.
            Default='browser'.
        """
        ...
    
    def savefig(self, output_file=...): # -> None:
        """Save the figure to file.

        Parameters
        ----------
        output_file : :obj:`str` or ``None``, optional
            Path to output file.
        """
        ...
    


