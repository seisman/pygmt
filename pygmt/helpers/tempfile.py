"""
Utilities for dealing with temporary file management.
"""

import io
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from packaging.version import Version


@contextmanager
def tempfile_from_geojson(geojson):
    """
    Saves any geo-like Python object which implements ``__geo_interface__`` (e.g. a
    geopandas.GeoDataFrame or shapely.geometry) to a temporary OGR_GMT text file.

    Parameters
    ----------
    geojson : geopandas.GeoDataFrame
        A geopandas GeoDataFrame, or any geo-like Python object which
        implements __geo_interface__, i.e. a GeoJSON.

    Yields
    ------
    tmpfilename : str
        A temporary OGR_GMT format file holding the geographical data.
        E.g. '1a2b3c4d5e6.gmt'.
    """
    with GMTTempFile(suffix=".gmt") as tmpfile:
        import geopandas as gpd

        Path(tmpfile.name).unlink()  # Ensure file is deleted first
        ogrgmt_kwargs = {"filename": tmpfile.name, "driver": "OGR_GMT", "mode": "w"}
        try:
            # OGR_GMT only supports 32-bit integers. We need to map int/int64
            # types to int32/float types depending on if the column has an
            # 32-bit integer overflow issue. Related issues:
            # https://github.com/geopandas/geopandas/issues/967#issuecomment-842877704
            # https://github.com/GenericMappingTools/pygmt/issues/2497
            int32_info = np.iinfo(np.int32)
            if Version(gpd.__version__).major < 1:  # GeoPandas v0.x
                # The default engine 'fiona' supports the 'schema' parameter.
                if geojson.index.name is None:
                    geojson.index.name = "index"
                geojson = geojson.reset_index(drop=False)
                schema = gpd.io.file.infer_schema(geojson)
                for col, dtype in schema["properties"].items():
                    if dtype in {"int", "int64"}:
                        overflow = (
                            geojson[col].max() > int32_info.max
                            or geojson[col].min() < int32_info.min
                        )
                        schema["properties"][col] = "float" if overflow else "int32"
                        geojson[col] = geojson[col].astype(schema["properties"][col])
                ogrgmt_kwargs["schema"] = schema
            else:  # GeoPandas v1.x.
                # The default engine "pyogrio" doesn't support the 'schema' parameter
                # but we can change the dtype directly.
                for col in geojson.columns:
                    if geojson[col].dtype.name in {"int", "int64", "Int64"}:
                        overflow = (
                            geojson[col].max() > int32_info.max
                            or geojson[col].min() < int32_info.min
                        )
                        dtype = "float" if overflow else "int32"
                        geojson[col] = geojson[col].astype(dtype)
            # Using geopandas.to_file to directly export to OGR_GMT format
            geojson.to_file(**ogrgmt_kwargs)
        except AttributeError:
            # Other 'geo' formats which implement __geo_interface__
            import json

            jsontext = json.dumps(geojson.__geo_interface__)
            gpd.read_file(filename=io.StringIO(jsontext)).to_file(**ogrgmt_kwargs)

        yield tmpfile.name


@contextmanager
def tempfile_from_image(image):
    """
    Saves a 3-band :class:`xarray.DataArray` to a temporary GeoTIFF file via rioxarray.

    Parameters
    ----------
    image : xarray.DataArray
        An xarray.DataArray with three dimensions, having a shape like
        (3, Y, X).

    Yields
    ------
    tmpfilename : str
        A temporary GeoTIFF file holding the image data. E.g. '1a2b3c4d5.tif'.
    """
    with GMTTempFile(suffix=".tif") as tmpfile:
        Path(tmpfile.name).unlink()  # Ensure file is deleted first
        try:
            image.rio.to_raster(raster_path=tmpfile.name)
        except AttributeError as e:  # object has no attribute 'rio'
            msg = (
                "Package `rioxarray` is required to be installed to use this function. "
                "Please use `python -m pip install rioxarray` or "
                "`mamba install -c conda-forge rioxarray` "
                "to install the package."
            )
            raise ImportError(msg) from e
        yield tmpfile.name
