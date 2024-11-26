"""
Functions to convert data types into ctypes friendly formats.
"""

import contextlib
import ctypes as ctp
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from packaging.version import Version
from pygmt.exceptions import GMTInvalidInput


def _to_numpy(data: Any) -> np.ndarray:
    """
    Convert an array-like object to a C contiguous NumPy array.

    The function aims to convert any array-like objects (e.g., Python lists or tuples,
    NumPy arrays with various dtypes, pandas.Series with NumPy/pandas/PyArrow dtypes,
    PyArrow arrays with various dtypes) to a NumPy array.

    The function is internally used in the ``vectors_to_arrays`` function, which is
    responsible for converting a sequence of vectors to a list of C contiguous NumPy
    arrays. Thus, the function uses the :numpy:func:`numpy.ascontiguousarray` function
    rather than the :numpy:func:`numpy.asarray`/:numpy::func:`numpy.asanyarray`
    functions, to ensure the returned NumPy array is C contiguous.

    Parameters
    ----------
    data
        The array-like object to convert.

    Returns
    -------
    array
        The C contiguous NumPy array.
    """
    # Mapping of unsupported dtypes to expected NumPy dtypes.
    dtypes: dict[str, type | str] = {
        # For string dtypes.
        "large_string": np.str_,  # pa.large_string and pa.large_utf8
        "string": np.str_,  # pa.string, pa.utf8, pd.StringDtype
        "string_view": np.str_,  # pa.string_view
        # For datetime dtypes.
        "date32[day][pyarrow]": "datetime64[D]",
        "date64[ms][pyarrow]": "datetime64[ms]",
    }

    if (
        hasattr(data, "isna")
        and data.isna().any()
        and Version(pd.__version__) < Version("2.2")
    ):
        # Workaround for dealing with pd.NA with pandas < 2.2.
        # Bug report at: https://github.com/GenericMappingTools/pygmt/issues/2844
        # Following SPEC0, pandas 2.1 will be dropped in 2025 Q3, so it's likely
        # we can remove the workaround in PyGMT v0.17.0.
        array = np.ascontiguousarray(data.astype(float))
    else:
        vec_dtype = str(getattr(data, "dtype", getattr(data, "type", "")))
        array = np.ascontiguousarray(data, dtype=dtypes.get(vec_dtype))

    # Check if a np.object_ array can be converted to np.str_.
    if array.dtype == np.object_:
        with contextlib.suppress(TypeError, ValueError):
            return np.ascontiguousarray(array, dtype=np.str_)
    return array


def vectors_to_arrays(vectors: Sequence[Any]) -> list[np.ndarray]:
    """
    Convert 1-D vectors (scalars, lists, or array-like) to C contiguous 1-D arrays.

    Arrays must be in C contiguous order for us to pass their memory pointers to GMT.
    If any are not, convert them to C order (which requires copying the memory). This
    usually happens when vectors are columns of a 2-D array or have been sliced.

    The returned arrays are guaranteed to be C contiguous and at least 1-D.

    Parameters
    ----------
    vectors
        The vectors that must be converted.

    Returns
    -------
    arrays
        List of converted numpy arrays.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> data = np.array([[1, 2], [3, 4], [5, 6]])
    >>> vectors = [data[:, 0], data[:, 1], pd.Series(data=[-1, -2, -3])]
    >>> all(i.flags.c_contiguous for i in vectors)
    False
    >>> all(isinstance(i, np.ndarray) for i in vectors)
    False
    >>> arrays = vectors_to_arrays(vectors)
    >>> all(i.flags.c_contiguous for i in arrays)
    True
    >>> all(isinstance(i, np.ndarray) for i in arrays)
    True
    >>> all(i.ndim == 1 for i in arrays)
    True
    """
    return [_to_numpy(vector) for vector in vectors]


def sequence_to_ctypes_array(
    sequence: Sequence[int | float] | np.ndarray | None, ctype, size: int
) -> ctp.Array | None:
    """
    Convert a sequence of numbers into a ctypes array variable.

    If the sequence is ``None``, returns ``None``. Otherwise, returns a ctypes array.
    The function only works for sequences of numbers. For converting a sequence of
    strings, use ``strings_to_ctypes_array`` instead.

    Parameters
    ----------
    sequence
        The sequence to convert. If ``None``, returns ``None``. Otherwise, it must be a
        sequence (e.g., list, tuple, numpy array).
    ctype
        The ctypes type of the array (e.g., ``ctypes.c_int``).
    size
        The size of the array. If the sequence is smaller than the size, the remaining
        elements will be filled with zeros. If the sequence is larger than the size, an
        exception will be raised.

    Returns
    -------
    ctypes_array
        The ctypes array variable.

    Examples
    --------
    >>> import ctypes as ctp
    >>> ctypes_array = sequence_to_ctypes_array([1, 2, 3], ctp.c_long, 3)
    >>> type(ctypes_array)
    <class 'pygmt.clib.conversion.c_long_Array_3'>
    >>> ctypes_array[:]
    [1, 2, 3]
    >>> ctypes_array = sequence_to_ctypes_array([1, 2], ctp.c_long, 5)
    >>> type(ctypes_array)
    <class 'pygmt.clib.conversion.c_long_Array_5'>
    >>> ctypes_array[:]
    [1, 2, 0, 0, 0]
    >>> ctypes_array = sequence_to_ctypes_array(None, ctp.c_long, 5)
    >>> print(ctypes_array)
    None
    >>> ctypes_array = sequence_to_ctypes_array([1, 2, 3, 4, 5, 6], ctp.c_long, 5)
    Traceback (most recent call last):
    ...
    IndexError: invalid index
    """
    if sequence is None:
        return None
    return (ctype * size)(*sequence)


def strings_to_ctypes_array(strings: Sequence[str] | np.ndarray) -> ctp.Array:
    """
    Convert a sequence (e.g., a list) of strings or numpy.ndarray of strings into a
    ctypes array.

    Parameters
    ----------
    strings
        A sequence of strings, or a numpy.ndarray of str dtype.

    Returns
    -------
    ctypes_array
        A ctypes array of strings.

    Examples
    --------
    >>> strings = ["first", "second", "third"]
    >>> ctypes_array = strings_to_ctypes_array(strings)
    >>> type(ctypes_array)
    <class 'pygmt.clib.conversion.c_char_p_Array_3'>
    >>> [s.decode() for s in ctypes_array]
    ['first', 'second', 'third']
    """
    return (ctp.c_char_p * len(strings))(*[s.encode() for s in strings])


def array_to_datetime(array: Sequence[Any] | np.ndarray) -> np.ndarray:
    """
    Convert a 1-D datetime array from various types into numpy.datetime64.

    If the input array is not in legal datetime formats, raise a ValueError exception.

    Parameters
    ----------
    array
        The input datetime array in various formats.

        Supported types:

        - str
        - numpy.datetime64
        - pandas.DateTimeIndex
        - datetime.datetime and datetime.date

    Returns
    -------
    array
        1-D datetime array in numpy.datetime64.

    Raises
    ------
    ValueError
        If the datetime string is invalid.

    Examples
    --------
    >>> import datetime
    >>> # numpy.datetime64 array
    >>> x = np.array(
    ...     ["2010-06-01", "2011-06-01T12", "2012-01-01T12:34:56"],
    ...     dtype="datetime64[ns]",
    ... )
    >>> array_to_datetime(x)
    array(['2010-06-01T00:00:00.000000000', '2011-06-01T12:00:00.000000000',
           '2012-01-01T12:34:56.000000000'], dtype='datetime64[ns]')

    >>> # pandas.DateTimeIndex array
    >>> import pandas as pd
    >>> x = pd.date_range("2013", freq="YS", periods=3)
    >>> array_to_datetime(x)
    array(['2013-01-01T00:00:00.000000000', '2014-01-01T00:00:00.000000000',
           '2015-01-01T00:00:00.000000000'], dtype='datetime64[ns]')

    >>> # Python's built-in date and datetime
    >>> x = [datetime.date(2018, 1, 1), datetime.datetime(2019, 1, 1)]
    >>> array_to_datetime(x)
    array(['2018-01-01T00:00:00.000000', '2019-01-01T00:00:00.000000'],
          dtype='datetime64[us]')

    >>> # Raw datetime strings in various format
    >>> x = [
    ...     "2018",
    ...     "2018-02",
    ...     "2018-03-01",
    ...     "2018-04-01T01:02:03",
    ... ]
    >>> array_to_datetime(x)
    array(['2018-01-01T00:00:00', '2018-02-01T00:00:00',
           '2018-03-01T00:00:00', '2018-04-01T01:02:03'],
          dtype='datetime64[s]')

    >>> # Mixed datetime types
    >>> x = [
    ...     "2018-01-01",
    ...     np.datetime64("2018-01-01"),
    ...     datetime.datetime(2018, 1, 1),
    ... ]
    >>> array_to_datetime(x)
    array(['2018-01-01T00:00:00.000000', '2018-01-01T00:00:00.000000',
           '2018-01-01T00:00:00.000000'], dtype='datetime64[us]')
    """
    return np.asarray(array, dtype=np.datetime64)
