"""
Defines the Session class to create and destroy a GMT API session and provides access to
the API functions.

Uses ctypes to wrap most of the core functions from the C API.
"""

import contextlib
import ctypes as ctp
import io
import sys
import warnings
from collections.abc import Callable, Generator, Sequence
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from pygmt.clib.conversion import (
    array_to_datetime,
    dataarray_to_matrix,
    sequence_to_ctypes_array,
    strings_to_ctypes_array,
    vectors_to_arrays,
)
from pygmt.clib.loading import get_gmt_version, load_libgmt
from pygmt.datatypes import _GMT_DATASET, _GMT_GRID, _GMT_IMAGE
from pygmt.exceptions import GMTCLibError, GMTCLibNoSessionError, GMTInvalidInput
from pygmt.helpers import (
    _validate_data_input,
    data_kind,
    tempfile_from_geojson,
    tempfile_from_image,
)

FAMILIES = [
    "GMT_IS_DATASET",  # Entity is a data table
    "GMT_IS_GRID",  # Entity is a grid
    "GMT_IS_IMAGE",  # Entity is a 1- or 3-band unsigned char image
    "GMT_IS_PALETTE",  # Entity is a color palette table
    "GMT_IS_POSTSCRIPT",  # Entity is a PostScript content struct
    "GMT_IS_MATRIX",  # Entity is a user matrix
    "GMT_IS_VECTOR",  # Entity is a set of user vectors
    "GMT_IS_CUBE",  # Entity is a 3-D data cube
]

VIAS = [
    "GMT_VIA_MATRIX",  # dataset is passed as a matrix
    "GMT_VIA_VECTOR",  # dataset is passed as a set of vectors
]

GEOMETRIES = [
    "GMT_IS_NONE",  # items without geometry (e.g., CPT)
    "GMT_IS_POINT",  # items are points
    "GMT_IS_LINE",  # items are lines
    "GMT_IS_POLY",  # items are polygons
    "GMT_IS_LP",  # items could be any one of LINE or POLY
    "GMT_IS_PLP",  # items could be any one of POINT, LINE, or POLY
    "GMT_IS_SURFACE",  # items are 2-D grid
    "GMT_IS_VOLUME",  # items are 3-D grid
    "GMT_IS_TEXT",  # Text strings which triggers ASCII text reading
]

METHODS = [
    "GMT_IS_DUPLICATE",  # tell GMT the data are read-only
    "GMT_IS_REFERENCE",  # tell GMT to duplicate the data
]

DIRECTIONS = ["GMT_IN", "GMT_OUT"]

MODES = ["GMT_CONTAINER_ONLY", "GMT_IS_OUTPUT"]
MODE_MODIFIERS = [
    "GMT_GRID_IS_CARTESIAN",
    "GMT_GRID_IS_GEO",
    "GMT_WITH_STRINGS",
]

REGISTRATIONS = ["GMT_GRID_PIXEL_REG", "GMT_GRID_NODE_REG"]

DTYPES = {
    np.int8: "GMT_CHAR",
    np.int16: "GMT_SHORT",
    np.int32: "GMT_INT",
    np.int64: "GMT_LONG",
    np.uint8: "GMT_UCHAR",
    np.uint16: "GMT_USHORT",
    np.uint32: "GMT_UINT",
    np.uint64: "GMT_ULONG",
    np.float32: "GMT_FLOAT",
    np.float64: "GMT_DOUBLE",
    np.str_: "GMT_TEXT",
    np.datetime64: "GMT_DATETIME",
    np.timedelta64: "GMT_LONG",
}
# Dictionary for storing the values of GMT constants.
GMT_CONSTANTS = {}

# Load the GMT library outside the Session class to avoid repeated loading.
_libgmt = load_libgmt()
__gmt_version__ = get_gmt_version(_libgmt)


class Session:
    def create_data(
        self,
        family,
        geometry,
        mode,
        dim=None,
        ranges=None,
        inc=None,
        registration="GMT_GRID_NODE_REG",
        pad=None,
    ):
        """
        Create an empty GMT data container.

        Parameters
        ----------
        family : str
            A valid GMT data family name (e.g., ``'GMT_IS_DATASET'``). See the
            ``FAMILIES`` attribute for valid names.
        geometry : str
            A valid GMT data geometry name (e.g., ``'GMT_IS_POINT'``). See the
            ``GEOMETRIES`` attribute for valid names.
        mode : str
            A valid GMT data mode (e.g., ``'GMT_IS_OUTPUT'``). See the
            ``MODES`` attribute for valid names.
        dim : list of 4 integers
            The dimensions of the dataset. See the documentation for the GMT C
            API function ``GMT_Create_Data`` (``src/gmt_api.c``) for the full
            range of options regarding 'dim'. If ``None``, will pass in the
            NULL pointer.
        ranges : list of 4 floats
            The dataset extent. Also a bit of a complicated argument. See the C
            function documentation. It's called ``range`` in the C function but
            it would conflict with the Python built-in ``range`` function.
        inc : list of 2 floats
            The increments between points of the dataset. See the C function
            documentation.
        registration : str
            The node registration (what the coordinates mean). Can be
            ``'GMT_GRID_PIXEL_REG'`` or ``'GMT_GRID_NODE_REG'``. Defaults to
            ``'GMT_GRID_NODE_REG'``.
        pad : int
            The grid padding. Defaults to ``GMT_PAD_DEFAULT``.

        Returns
        -------
        data_ptr : int
            A ctypes pointer (an integer) to the allocated ``GMT_Dataset``
            object.
        """
        c_create_data = self.get_libgmt_func(
            "GMT_Create_Data",
            argtypes=[
                ctp.c_void_p,  # API
                ctp.c_uint,  # family
                ctp.c_uint,  # geometry
                ctp.c_uint,  # mode
                ctp.POINTER(ctp.c_uint64),  # dim
                ctp.POINTER(ctp.c_double),  # range
                ctp.POINTER(ctp.c_double),  # inc
                ctp.c_uint,  # registration
                ctp.c_int,  # pad
                ctp.c_void_p,
            ],  # data
            restype=ctp.c_void_p,
        )

        family_int = self._parse_constant(family, valid=FAMILIES, valid_modifiers=VIAS)
        mode_int = self._parse_constant(
            mode,
            valid=MODES,
            valid_modifiers=MODE_MODIFIERS,
        )
        geometry_int = self._parse_constant(geometry, valid=GEOMETRIES)
        registration_int = self._parse_constant(registration, valid=REGISTRATIONS)

        # Convert dim, ranges, and inc to ctypes arrays if given (will be None
        # if not given to represent NULL pointers)
        dim = sequence_to_ctypes_array(dim, ctp.c_uint64, 4)
        ranges = sequence_to_ctypes_array(ranges, ctp.c_double, 4)
        inc = sequence_to_ctypes_array(inc, ctp.c_double, 2)

        # Use a NULL pointer (None) for existing data to indicate that the
        # container should be created empty. Fill it in later using put_vector
        # and put_matrix.
        data_ptr = c_create_data(
            self.session_pointer,
            family_int,
            geometry_int,
            mode_int,
            dim,
            ranges,
            inc,
            registration_int,
            self._parse_pad(family, pad),
            None,
        )

        if data_ptr is None:
            raise GMTCLibError("Failed to create an empty GMT data pointer.")

        return data_ptr

    def _parse_pad(self, family, pad):
        """
        Parse and return an appropriate value for pad if none is given.

        Pad is a bit tricky because, for matrix types, pad control the matrix ordering
        (row or column major). Using the default pad will set it to column major and
        mess things up with the numpy arrays.
        """
        if pad is None:
            pad = 0 if "MATRIX" in family else self["GMT_PAD_DEFAULT"]
        return pad

    def _parse_constant(self, constant, valid, valid_modifiers=None):
        """
        Parse a constant, convert it to an int, and validate it.

        The GMT C API takes certain defined constants, like ``'GMT_IS_GRID'``,
        that need to be validated and converted to integer values using
        :meth:`pygmt.clib.Session.__getitem__`.

        The constants can also take a modifier by appending another constant
        name, e.g. ``'GMT_IS_GRID|GMT_VIA_MATRIX'``. The two parts must be
        converted separately and their values are added.

        If valid modifiers are not given, then will assume that modifiers are
        not allowed. In this case, will raise a
        :class:`pygmt.exceptions.GMTInvalidInput` exception if given a
        modifier.

        Parameters
        ----------
        constant : str
            The name of a valid GMT API constant, with an optional modifier.
        valid : list of str
            A list of valid values for the constant. Will raise a
            :class:`pygmt.exceptions.GMTInvalidInput` exception if the given
            value is not on the list.
        """
        parts = constant.split("|")
        name = parts[0]
        nmodifiers = len(parts) - 1
        if nmodifiers > 1:
            raise GMTInvalidInput(
                f"Only one modifier is allowed in constants, {nmodifiers} given: '{constant}'"
            )
        if nmodifiers > 0 and valid_modifiers is None:
            raise GMTInvalidInput(
                "Constant modifiers are not allowed since valid values were not given: '{constant}'"
            )
        if name not in valid:
            raise GMTInvalidInput(
                f"Invalid constant argument '{name}'. Must be one of {valid}."
            )
        if (
            nmodifiers > 0
            and valid_modifiers is not None
            and parts[1] not in valid_modifiers
        ):
            raise GMTInvalidInput(
                f"Invalid constant modifier '{parts[1]}'. Must be one of {valid_modifiers}."
            )
        integer_value = sum(self[part] for part in parts)
        return integer_value

    def _check_dtype_and_dim(self, array, ndim):
        """
        Check that a numpy array has the given number of dimensions and is a valid data
        type.

        Parameters
        ----------
        array : numpy.ndarray
            The array to be tested.
        ndim : int
            The desired number of array dimensions.

        Returns
        -------
        gmt_type : int
            The GMT constant value representing this data type.

        Raises
        ------
        GMTInvalidInput
            If the array has the wrong number of dimensions or
            is an unsupported data type.

        Examples
        --------

        >>> import numpy as np
        >>> data = np.array([1, 2, 3], dtype="float64")
        >>> with Session() as ses:
        ...     gmttype = ses._check_dtype_and_dim(data, ndim=1)
        ...     gmttype == ses["GMT_DOUBLE"]
        True
        >>> data = np.ones((5, 2), dtype="float32")
        >>> with Session() as ses:
        ...     gmttype = ses._check_dtype_and_dim(data, ndim=2)
        ...     gmttype == ses["GMT_FLOAT"]
        True
        """
        # Check that the array has the given number of dimensions
        if array.ndim != ndim:
            raise GMTInvalidInput(
                f"Expected a numpy {ndim}-D array, got {array.ndim}-D."
            )

        # Check that the array has a valid/known data type
        if array.dtype.type not in DTYPES:
            try:
                if array.dtype.type is np.object_:
                    # Try to convert unknown object type to np.datetime64
                    array = array_to_datetime(array)
                else:
                    raise ValueError
            except ValueError as e:
                raise GMTInvalidInput(
                    f"Unsupported numpy data type '{array.dtype.type}'."
                ) from e
        return self[DTYPES[array.dtype.type]]

    def put_vector(self, dataset, column, vector):
        r"""
        Attach a numpy 1-D array as a column on a GMT dataset.

        Use this function to attach numpy array data to a GMT dataset and pass
        it to GMT modules. Wraps ``GMT_Put_Vector``.

        The dataset must be created by :meth:`pygmt.clib.Session.create_data`
        first. Use ``family='GMT_IS_DATASET|GMT_VIA_VECTOR'``.

        Not all numpy dtypes are supported, only: int8, int16, int32, int64,
        uint8, uint16, uint32, uint64, float32, float64, str\_, and datetime64.

        .. warning::
            The numpy array must be C contiguous in memory. If it comes from a
            column slice of a 2-D array, for example, you will have to make a
            copy. Use :func:`numpy.ascontiguousarray` to make sure your vector
            is contiguous (it won't copy if it already is).

        Parameters
        ----------
        dataset : :class:`ctypes.c_void_p`
            The ctypes void pointer to a ``GMT_Dataset``. Create it with
            :meth:`pygmt.clib.Session.create_data`.
        column : int
            The column number of this vector in the dataset (starting from 0).
        vector : numpy 1-D array
            The array that will be attached to the dataset. Must be a 1-D C
            contiguous array.

        Raises
        ------
        GMTCLibError
            If given invalid input or ``GMT_Put_Vector`` exits with
            status != 0.
        """
        c_put_vector = self.get_libgmt_func(
            "GMT_Put_Vector",
            argtypes=[ctp.c_void_p, ctp.c_void_p, ctp.c_uint, ctp.c_uint, ctp.c_void_p],
            restype=ctp.c_int,
        )

        gmt_type = self._check_dtype_and_dim(vector, ndim=1)
        if gmt_type in {self["GMT_TEXT"], self["GMT_DATETIME"]}:
            if gmt_type == self["GMT_DATETIME"]:
                vector = np.datetime_as_string(array_to_datetime(vector))
            vector_pointer = strings_to_ctypes_array(vector)
        else:
            vector_pointer = vector.ctypes.data_as(ctp.c_void_p)
        status = c_put_vector(
            self.session_pointer, dataset, column, gmt_type, vector_pointer
        )
        if status != 0:
            raise GMTCLibError(
                f"Failed to put vector of type {vector.dtype} "
                f"in column {column} of dataset."
            )

    def put_strings(self, dataset, family, strings):
        """
        Attach a numpy 1-D array of dtype str as a column on a GMT dataset.

        Use this function to attach string type numpy array data to a GMT
        dataset and pass it to GMT modules. Wraps ``GMT_Put_Strings``.

        The dataset must be created by :meth:`pygmt.clib.Session.create_data`
        first.

        .. warning::
            The numpy array must be C contiguous in memory. If it comes from a
            column slice of a 2-D array, for example, you will have to make a
            copy. Use :func:`numpy.ascontiguousarray` to make sure your vector
            is contiguous (it won't copy if it already is).

        Parameters
        ----------
        dataset : :class:`ctypes.c_void_p`
            The ctypes void pointer to a ``GMT_Dataset``. Create it with
            :meth:`pygmt.clib.Session.create_data`.
        family : str
            The family type of the dataset. Can be either ``GMT_IS_VECTOR`` or
            ``GMT_IS_MATRIX``.
        strings : numpy 1-D array
            The array that will be attached to the dataset. Must be a 1-D C
            contiguous array.

        Raises
        ------
        GMTCLibError
            If given invalid input or ``GMT_Put_Strings`` exits with
            status != 0.
        """
        c_put_strings = self.get_libgmt_func(
            "GMT_Put_Strings",
            argtypes=[
                ctp.c_void_p,
                ctp.c_uint,
                ctp.c_void_p,
                ctp.POINTER(ctp.c_char_p),
            ],
            restype=ctp.c_int,
        )

        family_int = self._parse_constant(
            family, valid=FAMILIES, valid_modifiers=METHODS
        )

        strings_pointer = strings_to_ctypes_array(strings)

        status = c_put_strings(
            self.session_pointer, family_int, dataset, strings_pointer
        )
        if status != 0:
            raise GMTCLibError(
                f"Failed to put strings of type {strings.dtype} into dataset"
            )

    def put_matrix(self, dataset, matrix, pad=0):
        """
        Attach a numpy 2-D array to a GMT dataset.

        Use this function to attach numpy array data to a GMT dataset and pass
        it to GMT modules. Wraps ``GMT_Put_Matrix``.

        The dataset must be created by :meth:`pygmt.clib.Session.create_data`
        first. Use ``|GMT_VIA_MATRIX'`` in the family.

        Not all numpy dtypes are supported, only: int8, int16, int32, int64,
        uint8, uint16, uint32, uint64, float32, and float64.

        .. warning::
            The numpy array must be C contiguous in memory. Use
            :func:`numpy.ascontiguousarray` to make sure your vector is
            contiguous (it won't copy if it already is).

        Parameters
        ----------
        dataset : :class:`ctypes.c_void_p`
            The ctypes void pointer to a ``GMT_Dataset``. Create it with
            :meth:`pygmt.clib.Session.create_data`.
        matrix : numpy 2-D array
            The array that will be attached to the dataset. Must be a 2-D C
            contiguous array.
        pad : int
            The amount of padding that should be added to the matrix. Use when
            creating grids for modules that require padding.

        Raises
        ------
        GMTCLibError
            If given invalid input or ``GMT_Put_Matrix`` exits with
            status != 0.
        """
        c_put_matrix = self.get_libgmt_func(
            "GMT_Put_Matrix",
            argtypes=[ctp.c_void_p, ctp.c_void_p, ctp.c_uint, ctp.c_int, ctp.c_void_p],
            restype=ctp.c_int,
        )

        gmt_type = self._check_dtype_and_dim(matrix, ndim=2)
        matrix_pointer = matrix.ctypes.data_as(ctp.c_void_p)
        status = c_put_matrix(
            self.session_pointer, dataset, gmt_type, pad, matrix_pointer
        )
        if status != 0:
            raise GMTCLibError(f"Failed to put matrix of type {matrix.dtype}.")

    def read_data(
        self,
        infile: str,
        kind: Literal["dataset", "grid", "image"],
        family: str | None = None,
        geometry: str | None = None,
        mode: str = "GMT_READ_NORMAL",
        region: Sequence[float] | None = None,
        data=None,
    ):
        """
        Read a data file into a GMT data container.

        Wraps ``GMT_Read_Data`` but only allows reading from a file. The function
        definition is different from the original C API function.

        Parameters
        ----------
        infile
            The input file name.
        kind
            The data kind of the input file. Valid values are ``"dataset"``, ``"grid"``
            and ``"image"``.
        family
            A valid GMT data family name (e.g., ``"GMT_IS_DATASET"``). See the
            ``FAMILIES`` attribute for valid names. If ``None``, will determine the data
            family from the ``kind`` parameter.
        geometry
            A valid GMT data geometry name (e.g., ``"GMT_IS_POINT"``). See the
            ``GEOMETRIES`` attribute for valid names. If ``None``, will determine the
            data geometry from the ``kind`` parameter.
        mode
            How the data is to be read from the file. This option varies depending on
            the given family. See the
            :gmt-docs:`GMT API documentation <devdocs/api.html#import-from-a-file-stream-or-handle>`
            for details. Default is ``GMT_READ_NORMAL`` which corresponds to the default
            read mode value of 0 in the ``GMT_enum_read`` enum.
        region
            Subregion of the data, in the form of [xmin, xmax, ymin, ymax, zmin, zmax].
            If ``None``, the whole data is read.
        data
            ``None`` or the pointer returned by this function after a first call. It's
            useful when reading grids/images/cubes in two steps (get a grid/image/cube
            structure with a header, then read the data).

        Returns
        -------
        Pointer to the data container, or ``None`` if there were errors.

        Raises
        ------
        GMTCLibError
            If the GMT API function fails to read the data.
        """  # noqa: W505
        c_read_data = self.get_libgmt_func(
            "GMT_Read_Data",
            argtypes=[
                ctp.c_void_p,  # V_API
                ctp.c_uint,  # family
                ctp.c_uint,  # method
                ctp.c_uint,  # geometry
                ctp.c_uint,  # mode
                ctp.POINTER(ctp.c_double),  # wesn
                ctp.c_char_p,  # infile
                ctp.c_void_p,  # data
            ],
            restype=ctp.c_void_p,  # data_ptr
        )

        # Determine the family, geometry and data container from kind
        _family, _geometry, dtype = {
            "dataset": ("GMT_IS_DATASET", "GMT_IS_PLP", _GMT_DATASET),
            "grid": ("GMT_IS_GRID", "GMT_IS_SURFACE", _GMT_GRID),
            "image": ("GMT_IS_IMAGE", "GMT_IS_SURFACE", _GMT_IMAGE),
        }[kind]
        if family is None:
            family = _family
        if geometry is None:
            geometry = _geometry

        data_ptr = c_read_data(
            self.session_pointer,
            self[family],
            self["GMT_IS_FILE"],  # Reading from a file
            self[geometry],
            self[mode],
            sequence_to_ctypes_array(region, ctp.c_double, 6),
            infile.encode(),
            data,
        )
        if data_ptr is None:
            raise GMTCLibError(f"Failed to read dataset from '{infile}'.")
        return ctp.cast(data_ptr, ctp.POINTER(dtype))

    def write_data(self, family, geometry, mode, wesn, output, data):
        """
        Write a GMT data container to a file.

        The data container should be created by
        :meth:`pygmt.clib.Session.create_data`.

        Wraps ``GMT_Write_Data`` but only allows writing to a file. So the
        ``method`` argument is omitted.

        Parameters
        ----------
        family : str
            A valid GMT data family name (e.g., ``'GMT_IS_DATASET'``). See the
            ``FAMILIES`` attribute for valid names. Don't use the
            ``GMT_VIA_VECTOR`` or ``GMT_VIA_MATRIX`` constructs for this. Use
            ``GMT_IS_VECTOR`` and ``GMT_IS_MATRIX`` instead.
        geometry : str
            A valid GMT data geometry name (e.g., ``'GMT_IS_POINT'``). See the
            ``GEOMETRIES`` attribute for valid names.
        mode : str
            How the data is to be written to the file. This option varies
            depending on the given family. See the GMT API documentation for
            details.
        wesn : list or numpy array
            [xmin, xmax, ymin, ymax, zmin, zmax] of the data. Must have 6
            elements.
        output : str
            The output file name.
        data : :class:`ctypes.c_void_p`
            Pointer to the data container created by
            :meth:`pygmt.clib.Session.create_data`.

        Raises
        ------
        GMTCLibError
            For invalid input arguments or if the GMT API functions returns a
            non-zero status code.
        """
        c_write_data = self.get_libgmt_func(
            "GMT_Write_Data",
            argtypes=[
                ctp.c_void_p,
                ctp.c_uint,
                ctp.c_uint,
                ctp.c_uint,
                ctp.c_uint,
                ctp.POINTER(ctp.c_double),
                ctp.c_char_p,
                ctp.c_void_p,
            ],
            restype=ctp.c_int,
        )

        family_int = self._parse_constant(family, valid=FAMILIES, valid_modifiers=VIAS)
        geometry_int = self._parse_constant(geometry, valid=GEOMETRIES)
        status = c_write_data(
            self.session_pointer,
            family_int,
            self["GMT_IS_FILE"],
            geometry_int,
            self[mode],
            sequence_to_ctypes_array(wesn, ctp.c_double, 6),
            output.encode(),
            data,
        )
        if status != 0:
            raise GMTCLibError(f"Failed to write dataset to '{output}'")

    @contextlib.contextmanager
    def virtualfile_from_vectors(self, *vectors):
        """
        Store 1-D arrays as columns of a table inside a virtual file.

        Use the virtual file name to pass in the data in your vectors to a GMT
        module.

        Context manager (use in a ``with`` block). Yields the virtual file name
        that you can pass as an argument to a GMT module call. Closes the
        virtual file upon exit of the ``with`` block.

        Use this instead of creating the data container and virtual file by
        hand with :meth:`pygmt.clib.Session.create_data`,
        :meth:`pygmt.clib.Session.put_vector`, and
        :meth:`pygmt.clib.Session.open_virtualfile`.

        If the arrays are C contiguous blocks of memory, they will be passed
        without copying to GMT. If they are not (e.g., they are columns of a
        2-D array), they will need to be copied to a contiguous block.

        Parameters
        ----------
        vectors : 1-D arrays
            The vectors that will be included in the array. All must be of the
            same size.

        Yields
        ------
        fname : str
            The name of virtual file. Pass this as a file name argument to a
            GMT module.

        Examples
        --------

        >>> from pygmt.helpers import GMTTempFile
        >>> import numpy as np
        >>> import pandas as pd
        >>> x = [1, 2, 3]
        >>> y = np.array([4, 5, 6])
        >>> z = pd.Series([7, 8, 9])
        >>> with Session() as ses:
        ...     with ses.virtualfile_from_vectors(x, y, z) as fin:
        ...         # Send the output to a file so that we can read it
        ...         with GMTTempFile() as fout:
        ...             ses.call_module("info", [fin, f"->{fout.name}"])
        ...             print(fout.read().strip())
        <vector memory>: N = 3 <1/3> <4/6> <7/9>
        """
        # Conversion to a C-contiguous array needs to be done here and not in
        # put_vector or put_strings because we need to maintain a reference to
        # the copy while it is being used by the C API. Otherwise, the array
        # would be garbage collected and the memory freed. Creating it in this
        # context manager guarantees that the copy will be around until the
        # virtual file is closed. The conversion is implicit in
        # vectors_to_arrays.
        arrays = vectors_to_arrays(vectors)

        columns = len(arrays)
        # Find arrays that are of string dtype from column 3 onwards
        # Assumes that first 2 columns contains coordinates like longitude
        # latitude, or datetime string types.
        for col, array in enumerate(arrays[2:]):
            if pd.api.types.is_string_dtype(array.dtype):
                columns = col + 2
                break

        rows = len(arrays[0])
        if not all(len(i) == rows for i in arrays):
            raise GMTInvalidInput("All arrays must have same size.")

        family = "GMT_IS_DATASET|GMT_VIA_VECTOR"
        geometry = "GMT_IS_POINT"

        dataset = self.create_data(
            family, geometry, mode="GMT_CONTAINER_ONLY", dim=[columns, rows, 1, 0]
        )

        # Use put_vector for columns with numerical type data
        for col, array in enumerate(arrays[:columns]):
            self.put_vector(dataset, column=col, vector=array)

        # Use put_strings for last column(s) with string type data
        # Have to use modifier "GMT_IS_DUPLICATE" to duplicate the strings
        string_arrays = arrays[columns:]
        if string_arrays:
            if len(string_arrays) == 1:
                strings = string_arrays[0]
            elif len(string_arrays) > 1:
                strings = np.array(
                    [" ".join(vals) for vals in zip(*string_arrays, strict=True)]
                )
            strings = np.asanyarray(a=strings, dtype=str)
            self.put_strings(
                dataset, family="GMT_IS_VECTOR|GMT_IS_DUPLICATE", strings=strings
            )

        with self.open_virtualfile(
            family, geometry, "GMT_IN|GMT_IS_REFERENCE", dataset
        ) as vfile:
            yield vfile

    @contextlib.contextmanager
    def virtualfile_from_matrix(self, matrix):
        """
        Store a 2-D array as a table inside a virtual file.

        Use the virtual file name to pass in the data in your matrix to a GMT
        module.

        Context manager (use in a ``with`` block). Yields the virtual file name
        that you can pass as an argument to a GMT module call. Closes the
        virtual file upon exit of the ``with`` block.

        The virtual file will contain the array as a ``GMT_MATRIX`` pretending
        to be a ``GMT_DATASET``.

        **Not meant for creating ``GMT_GRID``**. The grid requires more
        metadata than just the data matrix. Use
        :meth:`pygmt.clib.Session.virtualfile_from_grid` instead.

        Use this instead of creating the data container and virtual file by
        hand with :meth:`pygmt.clib.Session.create_data`,
        :meth:`pygmt.clib.Session.put_matrix`, and
        :meth:`pygmt.clib.Session.open_virtualfile`

        The matrix must be C contiguous in memory. If it is not (e.g., it is a
        slice of a larger array), the array will be copied to make sure it is.

        Parameters
        ----------
        matrix : 2-D array
            The matrix that will be included in the GMT data container.

        Yields
        ------
        fname : str
            The name of virtual file. Pass this as a file name argument to a
            GMT module.

        Examples
        --------

        >>> from pygmt.helpers import GMTTempFile
        >>> import numpy as np
        >>> data = np.arange(12).reshape((4, 3))
        >>> print(data)
        [[ 0  1  2]
         [ 3  4  5]
         [ 6  7  8]
         [ 9 10 11]]
        >>> with Session() as ses:
        ...     with ses.virtualfile_from_matrix(data) as fin:
        ...         # Send the output to a file so that we can read it
        ...         with GMTTempFile() as fout:
        ...             ses.call_module("info", [fin, f"->{fout.name}"])
        ...             print(fout.read().strip())
        <matrix memory>: N = 4 <0/9> <1/10> <2/11>
        """
        # Conversion to a C-contiguous array needs to be done here and not in
        # put_matrix because we need to maintain a reference to the copy while
        # it is being used by the C API. Otherwise, the array would be garbage
        # collected and the memory freed. Creating it in this context manager
        # guarantees that the copy will be around until the virtual file is
        # closed.
        matrix = np.ascontiguousarray(matrix)
        rows, columns = matrix.shape

        family = "GMT_IS_DATASET|GMT_VIA_MATRIX"
        geometry = "GMT_IS_POINT"

        dataset = self.create_data(
            family, geometry, mode="GMT_CONTAINER_ONLY", dim=[columns, rows, 1, 0]
        )

        self.put_matrix(dataset, matrix)

        with self.open_virtualfile(
            family, geometry, "GMT_IN|GMT_IS_REFERENCE", dataset
        ) as vfile:
            yield vfile

    @contextlib.contextmanager
    def virtualfile_from_grid(self, grid):
        """
        Store a grid in a virtual file.

        Use the virtual file name to pass in the data in your grid to a GMT
        module. Grids must be :class:`xarray.DataArray` instances.

        Context manager (use in a ``with`` block). Yields the virtual file name
        that you can pass as an argument to a GMT module call. Closes the
        virtual file upon exit of the ``with`` block.

        The virtual file will contain the grid as a ``GMT_MATRIX`` with extra
        metadata.

        Use this instead of creating a data container and virtual file by hand
        with :meth:`pygmt.clib.Session.create_data`,
        :meth:`pygmt.clib.Session.put_matrix`, and
        :meth:`pygmt.clib.Session.open_virtualfile`.

        The grid data matrix must be C contiguous in memory. If it is not
        (e.g., it is a slice of a larger array), the array will be copied to
        make sure it is.

        Parameters
        ----------
        grid : :class:`xarray.DataArray`
            The grid that will be included in the virtual file.

        Yields
        ------
        fname : str
            The name of virtual file. Pass this as a file name argument to a
            GMT module.

        Examples
        --------

        >>> from pygmt.helpers.testing import load_static_earth_relief
        >>> from pygmt.helpers import GMTTempFile
        >>> data = load_static_earth_relief()
        >>> print(data.shape)
        (14, 8)
        >>> print(data.lon.values.min(), data.lon.values.max())
        -54.5 -47.5
        >>> print(data.lat.values.min(), data.lat.values.max())
        -23.5 -10.5
        >>> print(data.values.min(), data.values.max())
        190.0 981.0
        >>> with Session() as ses:
        ...     with ses.virtualfile_from_grid(data) as fin:
        ...         # Send the output to a file so that we can read it
        ...         with GMTTempFile() as fout:
        ...             ses.call_module(
        ...                 "grdinfo", [fin, "-L0", "-Cn", f"->{fout.name}"]
        ...             )
        ...             print(fout.read().strip())
        -55 -47 -24 -10 190 981 1 1 8 14 1 1
        >>> # The output is: w e s n z0 z1 dx dy n_columns n_rows reg gtype
        """
        _gtype = {0: "GMT_GRID_IS_CARTESIAN", 1: "GMT_GRID_IS_GEO"}[grid.gmt.gtype]
        _reg = {0: "GMT_GRID_NODE_REG", 1: "GMT_GRID_PIXEL_REG"}[grid.gmt.registration]

        # Conversion to a C-contiguous array needs to be done here and not in
        # put_matrix because we need to maintain a reference to the copy while
        # it is being used by the C API. Otherwise, the array would be garbage
        # collected and the memory freed. Creating it in this context manager
        # guarantees that the copy will be around until the virtual file is
        # closed. The conversion is implicit in dataarray_to_matrix.
        matrix, region, inc = dataarray_to_matrix(grid)

        family = "GMT_IS_GRID|GMT_VIA_MATRIX"
        geometry = "GMT_IS_SURFACE"
        gmt_grid = self.create_data(
            family,
            geometry,
            mode=f"GMT_CONTAINER_ONLY|{_gtype}",
            ranges=region,
            inc=inc,
            registration=_reg,
        )
        self.put_matrix(gmt_grid, matrix)
        args = (family, geometry, "GMT_IN|GMT_IS_REFERENCE", gmt_grid)
        with self.open_virtualfile(*args) as vfile:
            yield vfile

    def virtualfile_in(  # noqa: PLR0912
        self,
        check_kind=None,
        data=None,
        x=None,
        y=None,
        z=None,
        extra_arrays=None,
        required_z=False,
        required_data=True,
    ):
        """
        Store any data inside a virtual file.

        This convenience function automatically detects the kind of data passed
        into it, and produces a virtualfile that can be passed into GMT later
        on.

        Parameters
        ----------
        check_kind : str or None
            Used to validate the type of data that can be passed in. Choose
            from 'raster', 'vector', or None. Default is None (no validation).
        data : str or pathlib.Path or xarray.DataArray or {table-like} or None
            Any raster or vector data format. This could be a file name or
            path, a raster grid, a vector matrix/arrays, or other supported
            data input.
        x/y/z : 1-D arrays or None
            x, y, and z columns as numpy arrays.
        extra_arrays : list of 1-D arrays
            Optional. A list of numpy arrays in addition to x, y, and z.
            All of these arrays must be of the same size as the x/y/z arrays.
        required_z : bool
            State whether the 'z' column is required.
        required_data : bool
            Set to True when 'data' is required, or False when dealing with
            optional virtual files. [Default is True].

        Returns
        -------
        file_context : contextlib._GeneratorContextManager
            The virtual file stored inside a context manager. Access the file
            name of this virtualfile using ``with file_context as fname: ...``.

        Examples
        --------
        >>> from pygmt.helpers import GMTTempFile
        >>> import xarray as xr
        >>> data = xr.Dataset(
        ...     coords=dict(index=[0, 1, 2]),
        ...     data_vars=dict(
        ...         x=("index", [9, 8, 7]),
        ...         y=("index", [6, 5, 4]),
        ...         z=("index", [3, 2, 1]),
        ...     ),
        ... )
        >>> with Session() as ses:
        ...     with ses.virtualfile_in(check_kind="vector", data=data) as fin:
        ...         # Send the output to a file so that we can read it
        ...         with GMTTempFile() as fout:
        ...             ses.call_module("info", [fin, f"->{fout.name}"])
        ...             print(fout.read().strip())
        <vector memory>: N = 3 <7/9> <4/6> <1/3>
        """
        kind = data_kind(data, required=required_data)
        _validate_data_input(
            data=data,
            x=x,
            y=y,
            z=z,
            required_z=required_z,
            required_data=required_data,
            kind=kind,
        )

        if check_kind:
            valid_kinds = ("file", "arg") if required_data is False else ("file",)
            if check_kind == "raster":
                valid_kinds += ("grid", "image")
            elif check_kind == "vector":
                valid_kinds += ("empty", "matrix", "vectors", "geojson")
            if kind not in valid_kinds:
                raise GMTInvalidInput(
                    f"Unrecognized data type for {check_kind}: {type(data)}"
                )

        # Decide which virtualfile_from_ function to use
        _virtualfile_from = {
            "arg": contextlib.nullcontext,
            "empty": self.virtualfile_from_vectors,
            "file": contextlib.nullcontext,
            "geojson": tempfile_from_geojson,
            "grid": self.virtualfile_from_grid,
            "image": tempfile_from_image,
            "stringio": self.virtualfile_from_stringio,
            "matrix": self.virtualfile_from_matrix,
            "vectors": self.virtualfile_from_vectors,
        }[kind]

        # Ensure the data is an iterable (Python list or tuple).
        match kind:
            case "arg" | "file" | "geojson" | "grid" | "image" | "stringio":
                _data = (data,)
                if kind == "image" and data.dtype != "uint8":
                    msg = (
                        f"Input image has dtype: {data.dtype} which is unsupported, "
                        "and may result in an incorrect output. Please recast image "
                        "to a uint8 dtype and/or scale to 0-255 range, e.g. "
                        "using a histogram equalization function like "
                        "skimage.exposure.equalize_hist."
                    )
                    warnings.warn(message=msg, category=RuntimeWarning, stacklevel=2)
            case "empty":  # data is None, so data must be given via x/y/z.
                _data = [x, y]
                if z is not None:
                    _data.append(z)
                if extra_arrays:
                    _data.extend(extra_arrays)
            case "vectors":
                if hasattr(data, "items") and not hasattr(data, "to_frame"):
                    # pandas.DataFrame or xarray.Dataset types.
                    # pandas.Series will be handled below like a 1-D numpy.ndarray.
                    _data = [array for _, array in data.items()]
                else:
                    # Python list, tuple, numpy.ndarray, and pandas.Series types
                    _data = np.atleast_2d(np.asanyarray(data).T)
            case "matrix":
                # GMT can only accept a 2-D matrix which are signed integer (i),
                # unsigned integer (u) or floating point (f) types. For other data
                # types, we need to use virtualfile_from_vectors instead, which turns
                # the matrix into a list of vectors and allows for better handling of
                # non-integer/float type inputs (e.g. for string or datetime data types)
                _data = (data,)
                if data.dtype.kind not in "iuf":
                    _virtualfile_from = self.virtualfile_from_vectors
                    _data = data.T

        # Finally create the virtualfile from the data, to be passed into GMT
        file_context = _virtualfile_from(*_data)
        return file_context

    def virtualfile_from_data(
        self,
        check_kind=None,
        data=None,
        x=None,
        y=None,
        z=None,
        extra_arrays=None,
        required_z=False,
        required_data=True,
    ):
        """
        Store any data inside a virtual file.

        .. deprecated: 0.13.0

           Will be removed in v0.15.0. Use :meth:`pygmt.clib.Session.virtualfile_in`
           instead.
        """
        msg = (
            "API function 'Session.virtualfile_from_datae()' has been deprecated since "
            "v0.13.0 and will be removed in v0.15.0. Use 'Session.virtualfile_in()' "
            "instead."
        )
        warnings.warn(msg, category=FutureWarning, stacklevel=2)
        return self.virtualfile_in(
            check_kind=check_kind,
            data=data,
            x=x,
            y=y,
            z=z,
            extra_arrays=extra_arrays,
            required_z=required_z,
            required_data=required_data,
        )
