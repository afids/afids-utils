"""
This type stub file was generated by pyright.
"""

import abc
import contextlib
from matplotlib import _api

_log = ...
subprocess_creation_flags = ...
def adjusted_figsize(w, h, dpi, n): # -> tuple[Any | Unknown, Any | Unknown]:
    """
    Compute figure size so that pixels are a multiple of n.

    Parameters
    ----------
    w, h : float
        Size in inches.

    dpi : float
        The dpi.

    n : int
        The target multiple.

    Returns
    -------
    wnew, hnew : float
        The new figure size in inches.
    """
    ...

class MovieWriterRegistry:
    """Registry of available writer classes by human readable name."""
    def __init__(self) -> None:
        ...
    
    def register(self, name): # -> (writer_cls: Unknown) -> Unknown:
        """
        Decorator for registering a class under a name.

        Example use::

            @registry.register(name)
            class Foo:
                pass
        """
        ...
    
    def is_available(self, name): # -> Literal[False]:
        """
        Check if given writer is available by name.

        Parameters
        ----------
        name : str

        Returns
        -------
        bool
        """
        ...
    
    def __iter__(self): # -> Generator[Unknown, Any, None]:
        """Iterate over names of available writer class."""
        ...
    
    def list(self): # -> list[Unknown]:
        """Get a list of available MovieWriters."""
        ...
    
    def __getitem__(self, name):
        """Get an available writer class from its name."""
        ...
    


writers = ...
class AbstractMovieWriter(abc.ABC):
    """
    Abstract base class for writing movies, providing a way to grab frames by
    calling `~AbstractMovieWriter.grab_frame`.

    `setup` is called to start the process and `finish` is called afterwards.
    `saving` is provided as a context manager to facilitate this process as ::

        with moviewriter.saving(fig, outfile='myfile.mp4', dpi=100):
            # Iterate over frames
            moviewriter.grab_frame(**savefig_kwargs)

    The use of the context manager ensures that `setup` and `finish` are
    performed as necessary.

    An instance of a concrete subclass of this class can be given as the
    ``writer`` argument of `Animation.save()`.
    """
    def __init__(self, fps=..., metadata=..., codec=..., bitrate=...) -> None:
        ...
    
    @abc.abstractmethod
    def setup(self, fig, outfile, dpi=...): # -> None:
        """
        Setup for writing the movie file.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure object that contains the information for frames.
        outfile : str
            The filename of the resulting movie file.
        dpi : float, default: ``fig.dpi``
            The DPI (or resolution) for the file.  This controls the size
            in pixels of the resulting movie file.
        """
        ...
    
    @property
    def frame_size(self): # -> tuple[int, int]:
        """A tuple ``(width, height)`` in pixels of a movie frame."""
        ...
    
    @abc.abstractmethod
    def grab_frame(self, **savefig_kwargs): # -> None:
        """
        Grab the image information from the figure and save as a movie frame.

        All keyword arguments in *savefig_kwargs* are passed on to the
        `~.Figure.savefig` call that saves the figure.
        """
        ...
    
    @abc.abstractmethod
    def finish(self): # -> None:
        """Finish any processing for writing the movie."""
        ...
    
    @contextlib.contextmanager
    def saving(self, fig, outfile, dpi, *args, **kwargs): # -> Generator[Self@AbstractMovieWriter, Any, None]:
        """
        Context manager to facilitate writing the movie file.

        ``*args, **kw`` are any parameters that should be passed to `setup`.
        """
        ...
    


class MovieWriter(AbstractMovieWriter):
    """
    Base class for writing movies.

    This is a base class for MovieWriter subclasses that write a movie frame
    data to a pipe. You cannot instantiate this class directly.
    See examples for how to use its subclasses.

    Attributes
    ----------
    frame_format : str
        The format used in writing frame data, defaults to 'rgba'.
    fig : `~matplotlib.figure.Figure`
        The figure to capture data from.
        This must be provided by the subclasses.
    """
    supported_formats = ...
    def __init__(self, fps=..., codec=..., bitrate=..., extra_args=..., metadata=...) -> None:
        """
        Parameters
        ----------
        fps : int, default: 5
            Movie frame rate (per second).
        codec : str or None, default: :rc:`animation.codec`
            The codec to use.
        bitrate : int, default: :rc:`animation.bitrate`
            The bitrate of the movie, in kilobits per second.  Higher values
            means higher quality movies, but increase the file size.  A value
            of -1 lets the underlying movie encoder select the bitrate.
        extra_args : list of str or None, optional
            Extra command-line arguments passed to the underlying movie
            encoder.  The default, None, means to use
            :rc:`animation.[name-of-encoder]_args` for the builtin writers.
        metadata : dict[str, str], default: {}
            A dictionary of keys and values for metadata to include in the
            output file. Some keys that may be of use include:
            title, artist, genre, subject, copyright, srcform, comment.
        """
        ...
    
    def setup(self, fig, outfile, dpi=...): # -> None:
        ...
    
    def finish(self): # -> None:
        """Finish any processing for writing the movie."""
        ...
    
    def grab_frame(self, **savefig_kwargs): # -> None:
        ...
    
    @classmethod
    def bin_path(cls): # -> str:
        """
        Return the binary path to the commandline tool used by a specific
        subclass. This is a class method so that the tool can be looked for
        before making a particular MovieWriter subclass available.
        """
        ...
    
    @classmethod
    def isAvailable(cls): # -> bool:
        """Return whether a MovieWriter subclass is actually available."""
        ...
    


class FileMovieWriter(MovieWriter):
    """
    `MovieWriter` for writing to individual files and stitching at the end.

    This must be sub-classed to be useful.
    """
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def setup(self, fig, outfile, dpi=..., frame_prefix=...): # -> None:
        """
        Setup for writing the movie file.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure to grab the rendered frames from.
        outfile : str
            The filename of the resulting movie file.
        dpi : float, default: ``fig.dpi``
            The dpi of the output file. This, with the figure size,
            controls the size in pixels of the resulting movie file.
        frame_prefix : str, optional
            The filename prefix to use for temporary files.  If *None* (the
            default), files are written to a temporary directory which is
            deleted by `finish`; if not *None*, no temporary files are
            deleted.
        """
        ...
    
    def __del__(self): # -> None:
        ...
    
    @property
    def frame_format(self): # -> str:
        """
        Format (png, jpeg, etc.) to use for saving the frames, which can be
        decided by the individual subclasses.
        """
        ...
    
    @frame_format.setter
    def frame_format(self, frame_format): # -> None:
        ...
    
    def grab_frame(self, **savefig_kwargs): # -> None:
        ...
    
    def finish(self): # -> None:
        ...
    


@writers.register('pillow')
class PillowWriter(AbstractMovieWriter):
    @classmethod
    def isAvailable(cls): # -> Literal[True]:
        ...
    
    def setup(self, fig, outfile, dpi=...): # -> None:
        ...
    
    def grab_frame(self, **savefig_kwargs): # -> None:
        ...
    
    def finish(self): # -> None:
        ...
    


class FFMpegBase:
    """
    Mixin class for FFMpeg output.

    This is a base class for the concrete `FFMpegWriter` and `FFMpegFileWriter`
    classes.
    """
    _exec_key = ...
    _args_key = ...
    @property
    def output_args(self): # -> list[Unknown]:
        ...
    


@writers.register('ffmpeg')
class FFMpegWriter(FFMpegBase, MovieWriter):
    """
    Pipe-based ffmpeg writer.

    Frames are streamed directly to ffmpeg via a pipe and written in a single
    pass.
    """
    ...


@writers.register('ffmpeg_file')
class FFMpegFileWriter(FFMpegBase, FileMovieWriter):
    """
    File-based ffmpeg writer.

    Frames are written to temporary files on disk and then stitched
    together at the end.
    """
    supported_formats = ...


class ImageMagickBase:
    """
    Mixin class for ImageMagick output.

    This is a base class for the concrete `ImageMagickWriter` and
    `ImageMagickFileWriter` classes, which define an ``input_names`` attribute
    (or property) specifying the input names passed to ImageMagick.
    """
    _exec_key = ...
    _args_key = ...
    @_api.deprecated("3.6")
    @property
    def delay(self):
        ...
    
    @_api.deprecated("3.6")
    @property
    def output_args(self): # -> list[Unknown]:
        ...
    
    @classmethod
    def bin_path(cls):
        ...
    
    @classmethod
    def isAvailable(cls): # -> Literal[False]:
        ...
    


@writers.register('imagemagick')
class ImageMagickWriter(ImageMagickBase, MovieWriter):
    """
    Pipe-based animated gif writer.

    Frames are streamed directly to ImageMagick via a pipe and written
    in a single pass.
    """
    input_names = ...


@writers.register('imagemagick_file')
class ImageMagickFileWriter(ImageMagickBase, FileMovieWriter):
    """
    File-based animated gif writer.

    Frames are written to temporary files on disk and then stitched
    together at the end.
    """
    supported_formats = ...
    input_names = ...


@writers.register('html')
class HTMLWriter(FileMovieWriter):
    """Writer for JavaScript-based HTML movies."""
    supported_formats = ...
    @classmethod
    def isAvailable(cls): # -> Literal[True]:
        ...
    
    def __init__(self, fps=..., codec=..., bitrate=..., extra_args=..., metadata=..., embed_frames=..., default_mode=..., embed_limit=...) -> None:
        ...
    
    def setup(self, fig, outfile, dpi=..., frame_dir=...): # -> None:
        ...
    
    def grab_frame(self, **savefig_kwargs): # -> None:
        ...
    
    def finish(self): # -> None:
        ...
    


class Animation:
    """
    A base class for Animations.

    This class is not usable as is, and should be subclassed to provide needed
    behavior.

    .. note::

        You must store the created Animation in a variable that lives as long
        as the animation should run. Otherwise, the Animation object will be
        garbage-collected and the animation stops.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        The figure object used to get needed events, such as draw or resize.

    event_source : object, optional
        A class that can run a callback when desired events
        are generated, as well as be stopped and started.

        Examples include timers (see `TimedAnimation`) and file
        system notifications.

    blit : bool, default: False
        Whether blitting is used to optimize drawing.  If the backend does not
        support blitting, then this parameter has no effect.

    See Also
    --------
    FuncAnimation,  ArtistAnimation
    """
    def __init__(self, fig, event_source=..., blit=...) -> None:
        ...
    
    def __del__(self): # -> None:
        ...
    
    def save(self, filename, writer=..., fps=..., dpi=..., codec=..., bitrate=..., extra_args=..., metadata=..., extra_anim=..., savefig_kwargs=..., *, progress_callback=...):
        """
        Save the animation as a movie file by drawing every frame.

        Parameters
        ----------
        filename : str
            The output filename, e.g., :file:`mymovie.mp4`.

        writer : `MovieWriter` or str, default: :rc:`animation.writer`
            A `MovieWriter` instance to use or a key that identifies a
            class to use, such as 'ffmpeg'.

        fps : int, optional
            Movie frame rate (per second).  If not set, the frame rate from the
            animation's frame interval.

        dpi : float, default: :rc:`savefig.dpi`
            Controls the dots per inch for the movie frames.  Together with
            the figure's size in inches, this controls the size of the movie.

        codec : str, default: :rc:`animation.codec`.
            The video codec to use.  Not all codecs are supported by a given
            `MovieWriter`.

        bitrate : int, default: :rc:`animation.bitrate`
            The bitrate of the movie, in kilobits per second.  Higher values
            means higher quality movies, but increase the file size.  A value
            of -1 lets the underlying movie encoder select the bitrate.

        extra_args : list of str or None, optional
            Extra command-line arguments passed to the underlying movie
            encoder.  The default, None, means to use
            :rc:`animation.[name-of-encoder]_args` for the builtin writers.

        metadata : dict[str, str], default: {}
            Dictionary of keys and values for metadata to include in
            the output file. Some keys that may be of use include:
            title, artist, genre, subject, copyright, srcform, comment.

        extra_anim : list, default: []
            Additional `Animation` objects that should be included
            in the saved movie file. These need to be from the same
            `.Figure` instance. Also, animation frames will
            just be simply combined, so there should be a 1:1 correspondence
            between the frames from the different animations.

        savefig_kwargs : dict, default: {}
            Keyword arguments passed to each `~.Figure.savefig` call used to
            save the individual frames.

        progress_callback : function, optional
            A callback function that will be called for every frame to notify
            the saving progress. It must have the signature ::

                def func(current_frame: int, total_frames: int) -> Any

            where *current_frame* is the current frame number and
            *total_frames* is the total number of frames to be saved.
            *total_frames* is set to None, if the total number of frames can
            not be determined. Return values may exist but are ignored.

            Example code to write the progress to stdout::

                progress_callback = lambda i, n: print(f'Saving frame {i}/{n}')

        Notes
        -----
        *fps*, *codec*, *bitrate*, *extra_args* and *metadata* are used to
        construct a `.MovieWriter` instance and can only be passed if
        *writer* is a string.  If they are passed as non-*None* and *writer*
        is a `.MovieWriter`, a `RuntimeError` will be raised.
        """
        ...
    
    def new_frame_seq(self):
        """Return a new sequence of frame information."""
        ...
    
    def new_saved_frame_seq(self):
        """Return a new sequence of saved/cached frame information."""
        ...
    
    def to_html5_video(self, embed_limit=...): # -> str:
        """
        Convert the animation to an HTML5 ``<video>`` tag.

        This saves the animation as an h264 video, encoded in base64
        directly into the HTML5 video tag. This respects :rc:`animation.writer`
        and :rc:`animation.bitrate`. This also makes use of the
        *interval* to control the speed, and uses the *repeat*
        parameter to decide whether to loop.

        Parameters
        ----------
        embed_limit : float, optional
            Limit, in MB, of the returned animation. No animation is created
            if the limit is exceeded.
            Defaults to :rc:`animation.embed_limit` = 20.0.

        Returns
        -------
        str
            An HTML5 video tag with the animation embedded as base64 encoded
            h264 video.
            If the *embed_limit* is exceeded, this returns the string
            "Video too large to embed."
        """
        ...
    
    def to_jshtml(self, fps=..., embed_frames=..., default_mode=...): # -> str:
        """
        Generate HTML representation of the animation.

        Parameters
        ----------
        fps : int, optional
            Movie frame rate (per second). If not set, the frame rate from
            the animation's frame interval.
        embed_frames : bool, optional
        default_mode : str, optional
            What to do when the animation ends. Must be one of ``{'loop',
            'once', 'reflect'}``. Defaults to ``'loop'`` if the *repeat*
            parameter is True, otherwise ``'once'``.
        """
        ...
    
    def pause(self): # -> None:
        """Pause the animation."""
        ...
    
    def resume(self): # -> None:
        """Resume the animation."""
        ...
    


class TimedAnimation(Animation):
    """
    `Animation` subclass for time-based animation.

    A new frame is drawn every *interval* milliseconds.

    .. note::

        You must store the created Animation in a variable that lives as long
        as the animation should run. Otherwise, the Animation object will be
        garbage-collected and the animation stops.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        The figure object used to get needed events, such as draw or resize.
    interval : int, default: 200
        Delay between frames in milliseconds.
    repeat_delay : int, default: 0
        The delay in milliseconds between consecutive animation runs, if
        *repeat* is True.
    repeat : bool, default: True
        Whether the animation repeats when the sequence of frames is completed.
    blit : bool, default: False
        Whether blitting is used to optimize drawing.
    """
    def __init__(self, fig, interval=..., repeat_delay=..., repeat=..., event_source=..., *args, **kwargs) -> None:
        ...
    
    repeat = ...


class ArtistAnimation(TimedAnimation):
    """
    `TimedAnimation` subclass that creates an animation by using a fixed
    set of `.Artist` objects.

    Before creating an instance, all plotting should have taken place
    and the relevant artists saved.

    .. note::

        You must store the created Animation in a variable that lives as long
        as the animation should run. Otherwise, the Animation object will be
        garbage-collected and the animation stops.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        The figure object used to get needed events, such as draw or resize.
    artists : list
        Each list entry is a collection of `.Artist` objects that are made
        visible on the corresponding frame.  Other artists are made invisible.
    interval : int, default: 200
        Delay between frames in milliseconds.
    repeat_delay : int, default: 0
        The delay in milliseconds between consecutive animation runs, if
        *repeat* is True.
    repeat : bool, default: True
        Whether the animation repeats when the sequence of frames is completed.
    blit : bool, default: False
        Whether blitting is used to optimize drawing.
    """
    def __init__(self, fig, artists, *args, **kwargs) -> None:
        ...
    


class FuncAnimation(TimedAnimation):
    """
    `TimedAnimation` subclass that makes an animation by repeatedly calling
    a function *func*.

    .. note::

        You must store the created Animation in a variable that lives as long
        as the animation should run. Otherwise, the Animation object will be
        garbage-collected and the animation stops.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        The figure object used to get needed events, such as draw or resize.

    func : callable
        The function to call at each frame.  The first argument will
        be the next value in *frames*.   Any additional positional
        arguments can be supplied using `functools.partial` or via the *fargs*
        parameter.

        The required signature is::

            def func(frame, *fargs) -> iterable_of_artists

        It is often more convenient to provide the arguments using
        `functools.partial`. In this way it is also possible to pass keyword
        arguments. To pass a function with both positional and keyword
        arguments, set all arguments as keyword arguments, just leaving the
        *frame* argument unset::

            def func(frame, art, *, y=None):
                ...

            ani = FuncAnimation(fig, partial(func, art=ln, y='foo'))

        If ``blit == True``, *func* must return an iterable of all artists
        that were modified or created. This information is used by the blitting
        algorithm to determine which parts of the figure have to be updated.
        The return value is unused if ``blit == False`` and may be omitted in
        that case.

    frames : iterable, int, generator function, or None, optional
        Source of data to pass *func* and each frame of the animation

        - If an iterable, then simply use the values provided.  If the
          iterable has a length, it will override the *save_count* kwarg.

        - If an integer, then equivalent to passing ``range(frames)``

        - If a generator function, then must have the signature::

             def gen_function() -> obj

        - If *None*, then equivalent to passing ``itertools.count``.

        In all of these cases, the values in *frames* is simply passed through
        to the user-supplied *func* and thus can be of any type.

    init_func : callable, optional
        A function used to draw a clear frame. If not given, the results of
        drawing from the first item in the frames sequence will be used. This
        function will be called once before the first frame.

        The required signature is::

            def init_func() -> iterable_of_artists

        If ``blit == True``, *init_func* must return an iterable of artists
        to be re-drawn. This information is used by the blitting algorithm to
        determine which parts of the figure have to be updated.  The return
        value is unused if ``blit == False`` and may be omitted in that case.

    fargs : tuple or None, optional
        Additional arguments to pass to each call to *func*. Note: the use of
        `functools.partial` is preferred over *fargs*. See *func* for details.

    save_count : int, optional
        Fallback for the number of values from *frames* to cache. This is
        only used if the number of frames cannot be inferred from *frames*,
        i.e. when it's an iterator without length or a generator.

    interval : int, default: 200
        Delay between frames in milliseconds.

    repeat_delay : int, default: 0
        The delay in milliseconds between consecutive animation runs, if
        *repeat* is True.

    repeat : bool, default: True
        Whether the animation repeats when the sequence of frames is completed.

    blit : bool, default: False
        Whether blitting is used to optimize drawing.  Note: when using
        blitting, any animated artists will be drawn according to their zorder;
        however, they will be drawn on top of any previous artists, regardless
        of their zorder.

    cache_frame_data : bool, default: True
        Whether frame data is cached.  Disabling cache might be helpful when
        frames contain large objects.
    """
    def __init__(self, fig, func, frames=..., init_func=..., fargs=..., save_count=..., *, cache_frame_data=..., **kwargs) -> None:
        ...
    
    def new_frame_seq(self): # -> count[int] | Generator[Any, Unknown, None] | Iterator[Any] | Iterator[int]:
        ...
    
    def new_saved_frame_seq(self): # -> Iterator[Unknown] | Generator[Any, Any, None] | islice[Any]:
        ...
    
    save_count = ...


