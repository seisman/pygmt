"""
Utilities and common tasks for wrapping the GMT modules.
"""

import io
import os
import pathlib
import shutil
import string
import subprocess
import sys
import time
import webbrowser
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import xarray as xr
from pygmt.encodings import charset
from pygmt.exceptions import GMTInvalidInput

def _validate_data_input(
    data=None, x=None, y=None, z=None, required_z=False, required_data=True, kind=None
) -> None:
    """
    Check if the combination of data/x/y/z is valid.

    Examples
    --------
    >>> _validate_data_input(data="infile")
    >>> _validate_data_input(x=[1, 2, 3], y=[4, 5, 6])
    >>> _validate_data_input(x=[1, 2, 3], y=[4, 5, 6], z=[7, 8, 9])
    >>> _validate_data_input(data=None, required_data=False)
    >>> _validate_data_input()
    Traceback (most recent call last):
        ...
    pygmt.exceptions.GMTInvalidInput: No input data provided.
    >>> _validate_data_input(x=[1, 2, 3])
    Traceback (most recent call last):
        ...
    pygmt.exceptions.GMTInvalidInput: Must provide both x and y.
    >>> _validate_data_input(y=[4, 5, 6])
    Traceback (most recent call last):
        ...
    pygmt.exceptions.GMTInvalidInput: Must provide both x and y.
    >>> _validate_data_input(x=[1, 2, 3], y=[4, 5, 6], required_z=True)
    Traceback (most recent call last):
        ...
    pygmt.exceptions.GMTInvalidInput: Must provide x, y, and z.
    >>> import numpy as np
    >>> import pandas as pd
    >>> import xarray as xr
    >>> data = np.arange(8).reshape((4, 2))
    >>> _validate_data_input(data=data, required_z=True, kind="matrix")
    Traceback (most recent call last):
        ...
    pygmt.exceptions.GMTInvalidInput: data must provide x, y, and z columns.
    >>> _validate_data_input(
    ...     data=pd.DataFrame(data, columns=["x", "y"]),
    ...     required_z=True,
    ...     kind="vectors",
    ... )
    Traceback (most recent call last):
        ...
    pygmt.exceptions.GMTInvalidInput: data must provide x, y, and z columns.
    >>> _validate_data_input(
    ...     data=xr.Dataset(pd.DataFrame(data, columns=["x", "y"])),
    ...     required_z=True,
    ...     kind="vectors",
    ... )
    Traceback (most recent call last):
        ...
    pygmt.exceptions.GMTInvalidInput: data must provide x, y, and z columns.
    >>> _validate_data_input(data="infile", x=[1, 2, 3])
    Traceback (most recent call last):
        ...
    pygmt.exceptions.GMTInvalidInput: Too much data. Use either data or x/y/z.
    >>> _validate_data_input(data="infile", y=[4, 5, 6])
    Traceback (most recent call last):
        ...
    pygmt.exceptions.GMTInvalidInput: Too much data. Use either data or x/y/z.
    >>> _validate_data_input(data="infile", x=[1, 2, 3], y=[4, 5, 6])
    Traceback (most recent call last):
        ...
    pygmt.exceptions.GMTInvalidInput: Too much data. Use either data or x/y/z.
    >>> _validate_data_input(data="infile", z=[7, 8, 9])
    Traceback (most recent call last):
        ...
    pygmt.exceptions.GMTInvalidInput: Too much data. Use either data or x/y/z.

    Raises
    ------
    GMTInvalidInput
        If the data input is not valid.
    """
    if data is None:  # data is None
        if x is None and y is None:  # both x and y are None
            if required_data:  # data is not optional
                msg = "No input data provided."
                raise GMTInvalidInput(msg)
        elif x is None or y is None:  # either x or y is None
            msg = "Must provide both x and y."
            raise GMTInvalidInput(msg)
        if required_z and z is None:  # both x and y are not None, now check z
            msg = "Must provide x, y, and z."
            raise GMTInvalidInput(msg)
    else:  # data is not None
        if x is not None or y is not None or z is not None:
            msg = "Too much data. Use either data or x/y/z."
            raise GMTInvalidInput(msg)
        # check if data has the required z column
        if required_z:
            msg = "data must provide x, y, and z columns."
            if kind == "matrix" and data.shape[1] < 3:
                raise GMTInvalidInput(msg)
            if kind == "vectors":
                if hasattr(data, "shape") and (
                    (len(data.shape) == 1 and data.shape[0] < 3)
                    or (len(data.shape) > 1 and data.shape[1] < 3)
                ):  # np.ndarray or pd.DataFrame
                    raise GMTInvalidInput(msg)
                if hasattr(data, "data_vars") and len(data.data_vars) < 3:  # xr.Dataset
                    raise GMTInvalidInput(msg)


def build_arg_list(  # noqa: PLR0912
    kwdict: dict[str, Any],
    confdict: Mapping[str, Any] | None = None,
    infile: str | pathlib.PurePath | Sequence[str | pathlib.PurePath] | None = None,
    outfile: str | pathlib.PurePath | None = None,
) -> list[str]:
    r"""
    Convert keyword dictionaries and input/output files into a list of GMT arguments.

    Make sure all values in ``kwdict`` have been previously converted to a string
    representation using the ``kwargs_to_strings`` decorator. The only exceptions are
    ``True``, ``False`` and ``None``.

    Any remaining lists or tuples will be interpreted as multiple entries for the same
    parameter. For example, the kwargs entry ``"B": ["xa", "yaf"]`` will be
    converted to ``["-Bxa", "-Byaf"]``.

    Parameters
    ----------
    kwdict
        A dictionary containing parsed keyword arguments.
    confdict
        A dictionary containing configurable GMT parameters.
    infile
        The input file or a list of input files.
    outfile
        The output file.

    Returns
    -------
    args
        The list of command line arguments that will be passed to GMT modules. The
        keyword arguments are sorted alphabetically, followed by GMT configuration
        key-value pairs, with optional input file(s) at the beginning and optional
        output file at the end.

    Examples
    --------
    >>> build_arg_list(dict(A=True, B=False, C=None, D=0, E=200, F="", G="1/2/3/4"))
    ['-A', '-D0', '-E200', '-F', '-G1/2/3/4']
    >>> build_arg_list(dict(A="1/2/3/4", B=["xaf", "yaf", "WSen"], C=("1p", "2p")))
    ['-A1/2/3/4', '-BWSen', '-Bxaf', '-Byaf', '-C1p', '-C2p']
    >>> print(
    ...     build_arg_list(
    ...         dict(
    ...             B=["af", "WSne+tBlank Space"],
    ...             F='+t"Empty Spaces"',
    ...             l="'Void Space'",
    ...         )
    ...     )
    ... )
    ['-BWSne+tBlank Space', '-Baf', '-F+t"Empty Spaces"', "-l'Void Space'"]
    >>> print(
    ...     build_arg_list(
    ...         dict(A="0", B=True, C="rainbow"),
    ...         confdict=dict(FORMAT_DATE_MAP="o dd"),
    ...         infile="input.txt",
    ...         outfile="output.txt",
    ...     )
    ... )
    ['input.txt', '-A0', '-B', '-Crainbow', '--FORMAT_DATE_MAP=o dd', '->output.txt']
    >>> print(
    ...     build_arg_list(
    ...         dict(A="0", B=True),
    ...         confdict=dict(FORMAT_DATE_MAP="o dd"),
    ...         infile=["f1.txt", "f2.txt"],
    ...         outfile="out.txt",
    ...     )
    ... )
    ['f1.txt', 'f2.txt', '-A0', '-B', '--FORMAT_DATE_MAP=o dd', '->out.txt']
    >>> build_arg_list(dict(B="12ABāβ①②"))
    ['-B12AB\\340@~\\142@~@%34%\\254@%%@%34%\\255@%%', '--PS_CHAR_ENCODING=ISO-8859-4']
    >>> build_arg_list(dict(B="12ABāβ①②"), confdict=dict(PS_CHAR_ENCODING="ISO-8859-5"))
    ['-B12AB\\340@~\\142@~@%34%\\254@%%@%34%\\255@%%', '--PS_CHAR_ENCODING=ISO-8859-5']
    >>> print(build_arg_list(dict(R="1/2/3/4", J="X4i", watre=True)))
    Traceback (most recent call last):
      ...
    pygmt.exceptions.GMTInvalidInput: Unrecognized parameter 'watre'.
    """
    gmt_args = []
    for key, value in kwdict.items():
        if len(key) > 2:  # Raise an exception for unrecognized options
            msg = f"Unrecognized parameter '{key}'."
            raise GMTInvalidInput(msg)
        if value is None or value is False:  # Exclude arguments that are None or False
            pass
        elif value is True:
            gmt_args.append(f"-{key}")
        elif is_nonstr_iter(value):
            gmt_args.extend(f"-{key}{_value}" for _value in value)
        else:
            gmt_args.append(f"-{key}{value}")

    gmt_args = sorted(gmt_args)

    # Convert non-ASCII characters (if any) in the arguments to octal codes and set
    # --PS_CHAR_ENCODING=encoding if necessary
    if (encoding := _check_encoding("".join(gmt_args))) != "ascii":
        gmt_args = [non_ascii_to_octal(arg, encoding=encoding) for arg in gmt_args]
        if not (confdict and "PS_CHAR_ENCODING" in confdict):
            gmt_args.append(f"--PS_CHAR_ENCODING={encoding}")

    if confdict:
        gmt_args.extend(f"--{key}={value}" for key, value in confdict.items())

    if infile:  # infile can be a single file or a list of files
        if isinstance(infile, str | pathlib.PurePath):
            gmt_args = [str(infile), *gmt_args]
        else:
            gmt_args = [str(_file) for _file in infile] + gmt_args
    if outfile is not None:
        if (
            not isinstance(outfile, str | pathlib.PurePath)
            or str(outfile) in {"", ".", ".."}
            or str(outfile).endswith(("/", "\\"))
        ):
            msg = f"Invalid output file name '{outfile}'."
            raise GMTInvalidInput(msg)
        gmt_args.append(f"->{outfile}")
    return gmt_args


def is_nonstr_iter(value):
    """
    Check if the value is not a string but is iterable (list, tuple, array)

    Parameters
    ----------
    value
        What you want to check.

    Returns
    -------
    is_iterable : bool
        Whether it is a non-string iterable or not.

    Examples
    --------

    >>> is_nonstr_iter("abc")
    False
    >>> is_nonstr_iter(10)
    False
    >>> is_nonstr_iter(None)
    False
    >>> is_nonstr_iter([1, 2, 3])
    True
    >>> is_nonstr_iter((1, 2, 3))
    True
    >>> import numpy as np
    >>> is_nonstr_iter(np.array([1.0, 2.0, 3.0]))
    True
    >>> is_nonstr_iter(np.array(["abc", "def", "ghi"]))
    True
    """
    return isinstance(value, Iterable) and not isinstance(value, str)


def args_in_kwargs(args, kwargs):
    """
    Take a list and a dictionary, and determine if any entries in the list are keys in
    the dictionary.

    This function is used to determine if at least one of the required
    arguments is passed to raise a GMTInvalidInput Error.

    Parameters
    ----------
    args : list
        List of required arguments, using the GMT short-form aliases.

    kwargs : dict
        The dictionary of kwargs is the format returned by the _preprocess
        function of the BasePlotting class. The keys are the GMT
        short-form aliases of the parameters.

    Returns
    -------
    bool
        If one of the required arguments is in ``kwargs``.

    Examples
    --------

    >>> args_in_kwargs(args=["A", "B"], kwargs={"C": "xyz"})
    False
    >>> args_in_kwargs(args=["A", "B"], kwargs={"B": "af"})
    True
    >>> args_in_kwargs(args=["A", "B"], kwargs={"B": None})
    False
    >>> args_in_kwargs(args=["A", "B"], kwargs={"B": True})
    True
    >>> args_in_kwargs(args=["A", "B"], kwargs={"B": False})
    False
    >>> args_in_kwargs(args=["A", "B"], kwargs={"B": 0})
    True
    """
    return any(
        kwargs.get(arg) is not None and kwargs.get(arg) is not False for arg in args
    )
