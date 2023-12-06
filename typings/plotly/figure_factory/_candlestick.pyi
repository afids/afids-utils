"""
This type stub file was generated by pyright.
"""

def make_increasing_candle(open, high, low, close, dates, **kwargs): # -> list[dict[str, str | list[Unknown] | int]]:
    """
    Makes boxplot trace for increasing candlesticks

    _make_increasing_candle() and _make_decreasing_candle separate the
    increasing traces from the decreasing traces so kwargs (such as
    color) can be passed separately to increasing or decreasing traces
    when direction is set to 'increasing' or 'decreasing' in
    FigureFactory.create_candlestick()

    :param (list) open: opening values
    :param (list) high: high values
    :param (list) low: low values
    :param (list) close: closing values
    :param (list) dates: list of datetime objects. Default: None
    :param kwargs: kwargs to be passed to increasing trace via
        plotly.graph_objs.Scatter.

    :rtype (list) candle_incr_data: list of the box trace for
        increasing candlesticks.
    """
    ...

def make_decreasing_candle(open, high, low, close, dates, **kwargs): # -> list[dict[str, str | list[Unknown] | int]]:
    """
    Makes boxplot trace for decreasing candlesticks

    :param (list) open: opening values
    :param (list) high: high values
    :param (list) low: low values
    :param (list) close: closing values
    :param (list) dates: list of datetime objects. Default: None
    :param kwargs: kwargs to be passed to decreasing trace via
        plotly.graph_objs.Scatter.

    :rtype (list) candle_decr_data: list of the box trace for
        decreasing candlesticks.
    """
    ...

def create_candlestick(open, high, low, close, dates=..., direction=..., **kwargs): # -> Figure:
    """
    **deprecated**, use instead the plotly.graph_objects trace
    :class:`plotly.graph_objects.Candlestick`

    :param (list) open: opening values
    :param (list) high: high values
    :param (list) low: low values
    :param (list) close: closing values
    :param (list) dates: list of datetime objects. Default: None
    :param (string) direction: direction can be 'increasing', 'decreasing',
        or 'both'. When the direction is 'increasing', the returned figure
        consists of all candlesticks where the close value is greater than
        the corresponding open value, and when the direction is
        'decreasing', the returned figure consists of all candlesticks
        where the close value is less than or equal to the corresponding
        open value. When the direction is 'both', both increasing and
        decreasing candlesticks are returned. Default: 'both'
    :param kwargs: kwargs passed through plotly.graph_objs.Scatter.
        These kwargs describe other attributes about the ohlc Scatter trace
        such as the color or the legend name. For more information on valid
        kwargs call help(plotly.graph_objs.Scatter)

    :rtype (dict): returns a representation of candlestick chart figure.

    Example 1: Simple candlestick chart from a Pandas DataFrame

    >>> from plotly.figure_factory import create_candlestick
    >>> from datetime import datetime
    >>> import pandas as pd

    >>> df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
    >>> fig = create_candlestick(df['AAPL.Open'], df['AAPL.High'], df['AAPL.Low'], df['AAPL.Close'],
    ...                          dates=df.index)
    >>> fig.show()

    Example 2: Customize the candlestick colors

    >>> from plotly.figure_factory import create_candlestick
    >>> from plotly.graph_objs import Line, Marker
    >>> from datetime import datetime

    >>> import pandas as pd
    >>> df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

    >>> # Make increasing candlesticks and customize their color and name
    >>> fig_increasing = create_candlestick(df['AAPL.Open'], df['AAPL.High'], df['AAPL.Low'], df['AAPL.Close'],
    ...     dates=df.index,
    ...     direction='increasing', name='AAPL',
    ...     marker=Marker(color='rgb(150, 200, 250)'),
    ...     line=Line(color='rgb(150, 200, 250)'))

    >>> # Make decreasing candlesticks and customize their color and name
    >>> fig_decreasing = create_candlestick(df['AAPL.Open'], df['AAPL.High'], df['AAPL.Low'], df['AAPL.Close'],
    ...     dates=df.index,
    ...     direction='decreasing',
    ...     marker=Marker(color='rgb(128, 128, 128)'),
    ...     line=Line(color='rgb(128, 128, 128)'))

    >>> # Initialize the figure
    >>> fig = fig_increasing

    >>> # Add decreasing data with .extend()
    >>> fig.add_trace(fig_decreasing['data']) # doctest: +SKIP
    >>> fig.show()

    Example 3: Candlestick chart with datetime objects

    >>> from plotly.figure_factory import create_candlestick

    >>> from datetime import datetime

    >>> # Add data
    >>> open_data = [33.0, 33.3, 33.5, 33.0, 34.1]
    >>> high_data = [33.1, 33.3, 33.6, 33.2, 34.8]
    >>> low_data = [32.7, 32.7, 32.8, 32.6, 32.8]
    >>> close_data = [33.0, 32.9, 33.3, 33.1, 33.1]
    >>> dates = [datetime(year=2013, month=10, day=10),
    ...          datetime(year=2013, month=11, day=10),
    ...          datetime(year=2013, month=12, day=10),
    ...          datetime(year=2014, month=1, day=10),
    ...          datetime(year=2014, month=2, day=10)]

    >>> # Create ohlc
    >>> fig = create_candlestick(open_data, high_data,
    ...     low_data, close_data, dates=dates)
    >>> fig.show()
    """
    ...

class _Candlestick:
    """
    Refer to FigureFactory.create_candlestick() for docstring.
    """
    def __init__(self, open, high, low, close, dates, **kwargs) -> None:
        ...
    
    def get_candle_increase(self): # -> tuple[list[Unknown], list[Unknown]]:
        """
        Separate increasing data from decreasing data.

        The data is increasing when close value > open value
        and decreasing when the close value <= open value.
        """
        ...
    
    def get_candle_decrease(self): # -> tuple[list[Unknown], list[Unknown]]:
        """
        Separate increasing data from decreasing data.

        The data is increasing when close value > open value
        and decreasing when the close value <= open value.
        """
        ...
    

