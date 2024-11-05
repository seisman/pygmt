"""
Test the behavior of the Figure class.

Doesn't include the plotting commands which have their own test files.
"""

import importlib
from pathlib import Path

import numpy.testing as npt
import pytest
from pygmt import Figure
from pygmt.exceptions import GMTError, GMTInvalidInput
from pygmt.helpers import GMTTempFile

_HAS_IPYTHON = bool(importlib.util.find_spec("IPython"))
_HAS_RIOXARRAY = bool(importlib.util.find_spec("rioxarray"))


@pytest.mark.benchmark
def test_figure_repr():
    """
    Make sure that figure output's PNG and HTML printable representations look ok.
    """
    fig = Figure()
    fig.basemap(region=[0, 1, 2, 3], frame=True)
    # Check that correct PNG 8-byte file header is produced
    # https://en.wikipedia.org/wiki/Portable_Network_Graphics#File_header
    repr_png = fig._repr_png_()
    assert repr_png.hex().startswith("89504e470d0a1a0a")
    # Check that correct HTML image tags are produced
    repr_html = fig._repr_html_()
    assert repr_html.startswith('<img src="data:image/png;base64,')
    assert repr_html.endswith('" width="500px">')


@pytest.mark.skipif(not _HAS_IPYTHON, reason="run when IPython is installed")
def test_figure_show():
    """
    Test that show creates the correct file name and deletes the temp dir.
    """
    fig = Figure()
    fig.basemap(region="10/70/-300/800", projection="X3i/5i", frame="af")
    fig.show()


@pytest.mark.mpl_image_compare
def test_figure_shift_origin():
    """
    Test if fig.shift_origin works.
    """
    kwargs = {"region": [0, 3, 0, 5], "projection": "X3c/5c", "frame": 0}
    fig = Figure()
    # First call shift_origin without projection and region.
    # Test issue https://github.com/GenericMappingTools/pygmt/issues/514
    fig.shift_origin(xshift="2c", yshift="3c")
    fig.basemap(**kwargs)
    fig.shift_origin(xshift="4c")
    fig.basemap(**kwargs)
    fig.shift_origin(yshift="6c")
    fig.basemap(**kwargs)
    fig.shift_origin(xshift="-4c", yshift="6c")
    fig.basemap(**kwargs)
    return fig


def test_figure_show_invalid_method():
    """
    Test to check if an error is raised when an invalid method is passed to show.
    """
    fig = Figure()
    fig.basemap(region="10/70/-300/800", projection="X3i/5i", frame="af")
    with pytest.raises(GMTInvalidInput):
        fig.show(method="test")


@pytest.mark.skipif(_HAS_IPYTHON, reason="run without IPython installed")
def test_figure_show_notebook_error_without_ipython():
    """
    Test to check if an error is raised when display method is 'notebook', but IPython
    is not installed.
    """
    fig = Figure()
    fig.basemap(region=[0, 1, 2, 3], frame=True)
    with pytest.raises(GMTError):
        fig.show(method="notebook")


def test_figure_display_external():
    """
    Test to check that a figure can be displayed in an external window.
    """
    fig = Figure()
    fig.basemap(region=[0, 3, 6, 9], projection="X1c", frame=True)
    fig.show(method="external")


def test_figure_unsupported_xshift_yshift():
    """
    Raise an exception if X/Y/xshift/yshift is used.
    """
    fig = Figure()
    fig.basemap(region=[0, 1, 0, 1], projection="X1c/1c", frame=True)
    with pytest.raises(GMTInvalidInput):
        fig.plot(x=1, y=1, style="c3c", xshift="3c")
    with pytest.raises(GMTInvalidInput):
        fig.plot(x=1, y=1, style="c3c", X="3c")
    with pytest.raises(GMTInvalidInput):
        fig.plot(x=1, y=1, style="c3c", yshift="3c")
    with pytest.raises(GMTInvalidInput):
        fig.plot(x=1, y=1, style="c3c", Y="3c")
