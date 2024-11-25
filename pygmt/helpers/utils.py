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
from typing import Any, Literal

import xarray as xr
from pygmt.encodings import charset
from pygmt.exceptions import GMTInvalidInput

# Type hints for the list of encodings supported by PyGMT.
Encoding = Literal[
    "ascii",
    "ISOLatin1+",
    "ISO-8859-1",
    "ISO-8859-2",
    "ISO-8859-3",
    "ISO-8859-4",
    "ISO-8859-5",
    "ISO-8859-6",
    "ISO-8859-7",
    "ISO-8859-8",
    "ISO-8859-9",
    "ISO-8859-10",
    "ISO-8859-11",
    "ISO-8859-13",
    "ISO-8859-14",
    "ISO-8859-15",
    "ISO-8859-16",
]


def _validate_data_input(
    data=None, x=None, y=None, z=None, required_z=False, required_data=True, kind=None
):
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


def data_kind(
    data: Any, required: bool = True
) -> Literal[
    "arg", "empty", "file", "geojson", "grid", "image", "matrix", "stringio", "vectors"
]:
    r"""
    Check the kind of data that is provided to a module.

    The argument passed to the ``data`` parameter can have any data type. The
    following data kinds are recognized and returned as ``kind``:

    - ``"arg"``: ``data`` is ``None`` and ``required=False``, or bool, int, float,
      representing an optional argument, used for dealing with optional virtual files
    - ``"empty"`: ``data`` is ``None`` and ``required=True``. It means the data is given
      via a series of vectors like x/y/z
    - ``"file"``: a string or a :class:`pathlib.PurePath` object or a sequence of them,
      representing one or more file names
    - ``"geojson"``: a geo-like Python object that implements ``__geo_interface__``
      (e.g., geopandas.GeoDataFrame or shapely.geometry)
    - ``"grid"``: a :class:`xarray.DataArray` object that is not 3-D
    - ``"image"``: a 3-D :class:`xarray.DataArray` object
    - ``"stringio"``: a :class:`io.StringIO` object
    - ``"matrix"``: a 2-D array-like object that implements ``__array_interface__``
      (e.g., :class:`numpy.ndarray`)
    - ``"vectors"``: any unrecognized data. Common data types include, a
      :class:`pandas.DataFrame` object, a dictionary with array-like values, a 1-D/3-D
      :class:`numpy.ndarray` object, or array-like objects.

    Parameters
    ----------
    data
        The data to be passed to a GMT module.
    required
        Whether 'data' is required. Set to ``False`` when dealing with optional virtual
        files.

    Returns
    -------
    kind
        The data kind.

    Examples
    --------
    >>> import io
    >>> from pathlib import Path
    >>> import numpy as np
    >>> import pandas as pd
    >>> import xarray as xr

    The "arg" kind:

    >>> [data_kind(data=data, required=False) for data in (2, 2.0, True, False)]
    ['arg', 'arg', 'arg', 'arg']
    >>> data_kind(data=None, required=False)
    'arg'

    The "empty" kind:

    >>> data_kind(data=None, required=True)
    'empty'

    The "file" kind:

    >>> [data_kind(data=data) for data in ("file.txt", ("file1.txt", "file2.txt"))]
    ['file', 'file']
    >>> data_kind(data=Path("file.txt"))
    'file'
    >>> data_kind(data=(Path("file1.txt"), Path("file2.txt")))
    'file'

    The "grid" kind:

    >>> data_kind(data=xr.DataArray(np.random.rand(4, 3)))  # 2-D xarray.DataArray
    'grid'
    >>> data_kind(data=xr.DataArray(np.arange(12)))  # 1-D xarray.DataArray
    'grid'
    >>> data_kind(data=xr.DataArray(np.random.rand(2, 3, 4, 5)))  # 4-D xarray.DataArray
    'grid'

    The "image" kind:

    >>> data_kind(data=xr.DataArray(np.random.rand(3, 4, 5)))  # 3-D xarray.DataArray
    'image'

    The "stringio"`` kind:

    >>> data_kind(data=io.StringIO("TEXT1\nTEXT23\n"))
    'stringio'

    The "matrix"`` kind:

    >>> data_kind(data=np.arange(10).reshape((5, 2)))  # 2-D numpy.ndarray
    'matrix'

    The "vectors" kind:

    >>> data_kind(data=np.arange(10))  # 1-D numpy.ndarray
    'vectors'
    >>> data_kind(data=np.arange(60).reshape((3, 4, 5)))  # 3-D numpy.ndarray
    'vectors'
    >>> data_kind(xr.DataArray(np.arange(12), name="x").to_dataset())  # xarray.Dataset
    'vectors'
    >>> data_kind(data=[1, 2, 3])  # 1-D sequence
    'vectors'
    >>> data_kind(data=[[1, 2, 3], [4, 5, 6]])  # sequence of sequences
    'vectors'
    >>> data_kind(data={"x": [1, 2, 3], "y": [4, 5, 6]})  # dictionary
    'vectors'
    >>> data_kind(data=pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))  # pd.DataFrame
    'vectors'
    >>> data_kind(data=pd.Series([1, 2, 3], name="x"))  # pd.Series
    'vectors'
    """
    match data:
        case None if required:  # No data provided and required=True.
            kind = "empty"
        case str() | pathlib.PurePath():  # One file.
            kind = "file"
        case list() | tuple() if all(
            isinstance(_file, str | pathlib.PurePath) for _file in data
        ):  # A list/tuple of files.
            kind = "file"
        case io.StringIO():
            kind = "stringio"
        case (bool() | int() | float()) | None if not required:
            # An option argument, mainly for dealing with optional virtual files.
            kind = "arg"
        case xr.DataArray():
            # An xarray.DataArray object, representing either a grid or an image.
            kind = "image" if len(data.dims) == 3 else "grid"
        case x if hasattr(x, "__geo_interface__"):
            # Geo-like Python object that implements ``__geo_interface__`` (e.g.,
            # geopandas.GeoDataFrame or shapely.geometry).
            # Reference: https://gist.github.com/sgillies/2217756
            kind = "geojson"
        case x if hasattr(x, "__array_interface__") and data.ndim == 2:
            # 2-D Array-like objects that implements ``__array_interface__`` (e.g.,
            # numpy.ndarray).
            # Reference: https://numpy.org/doc/stable/reference/arrays.interface.html
            kind = "matrix"
        case _:  # Fall back to "vectors" if data is None and required=True.
            kind = "vectors"
    return kind  # type: ignore[return-value]


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
            raise GMTInvalidInput(f"Unrecognized parameter '{key}'.")
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
            raise GMTInvalidInput(f"Invalid output file name '{outfile}'.")
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
