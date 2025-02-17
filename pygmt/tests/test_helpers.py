"""
Test the helper functions/classes/etc used in wrapping GMT.
"""

import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import xarray as xr
from pygmt import Figure
from pygmt.exceptions import GMTInvalidInput
from pygmt.helpers import (
    GMTTempFile,
    args_in_kwargs,
    build_arg_list,
    kwargs_to_strings,
    launch_external_viewer,
    unique_name,
)
from pygmt.helpers.testing import load_static_earth_relief, skip_if_no


def test_kwargs_to_strings_fails():
    """
    Make sure it fails for invalid conversion types.
    """
    with pytest.raises(GMTInvalidInput):
        kwargs_to_strings(bla="blablabla")

@pytest.mark.parametrize(
    "outfile",
    [123, "", ".", "..", "path/to/dir/", "path\\to\\dir\\", Path(), Path("..")],
)
def test_build_arg_list_invalid_output(outfile):
    """
    Test that build_arg_list raises an exception when output file name is invalid.
    """
    with pytest.raises(GMTInvalidInput):
        build_arg_list({}, outfile=outfile)


def test_args_in_kwargs():
    """
    Test that args_in_kwargs function returns correct Boolean responses.
    """
    kwargs = {"A": 1, "B": 2, "C": 3}
    # Passing list of arguments with passing values in the beginning
    passing_args_1 = ["B", "C", "D"]
    assert args_in_kwargs(args=passing_args_1, kwargs=kwargs)
    # Passing list of arguments that starts with failing arguments
    passing_args_2 = ["D", "X", "C"]
    assert args_in_kwargs(args=passing_args_2, kwargs=kwargs)
    # Failing list of arguments
    failing_args = ["D", "E", "F"]
    assert not args_in_kwargs(args=failing_args, kwargs=kwargs)


def test_skip_if_no():
    """
    Test that the skip_if_no helper testing function returns a pytest.mask.skipif mark
    decorator.
    """
    # Check pytest.mark with a dependency that can be imported
    mark_decorator = skip_if_no(package="numpy")
    assert mark_decorator.args[0] is False

    # Check pytest.mark with a dependency that cannot be imported
    mark_decorator = skip_if_no(package="nullpackage")
    assert mark_decorator.args[0] is True
    assert mark_decorator.kwargs["reason"] == "Could not import 'nullpackage'"
    assert mark_decorator.markname == "skipif"
