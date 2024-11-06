"""
Define the Figure class that handles all plotting.
"""

import base64
from pathlib import Path, PurePath
from typing import Literal

try:
    import IPython

    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False

from pygmt.exceptions import GMTError, GMTInvalidInput
from pygmt.helpers import launch_external_viewer

# A registry of all figures that have had "show" called in this session.
# This is needed for the sphinx-gallery scraper in pygmt/sphinx_gallery.py
SHOWED_FIGURES = []
# Configurations for figure display.
SHOW_CONFIG = {
    "method": _get_default_display_method(),  # The image preview display method.
}


class Figure:
    def _preview(self, fmt: str, dpi: int, as_bytes: bool = False, **kwargs):
        """
        Grab a preview of the figure.

        Parameters
        ----------
        fmt
            The image format. Can be any extension that :meth:`pygmt.Figure.savefig`
            recognizes.
        dpi
            The image resolution (dots per inch).
        as_bytes
            If ``True``, will load the binary contents of the image as a bytes object,
            and return that instead of the file name.

        Returns
        -------
        preview
            If ``as_bytes = False``, this is the file name of the preview image file.
            Otherwise, it is the file content loaded as a bytes object.
        """
        fname = Path(self._preview_dir.name) / f"{self._name}.{fmt}"
        self.savefig(fname, dpi=dpi, **kwargs)
        if as_bytes:
            return fname.read_bytes()
        return fname

    def _repr_png_(self):
        """
        Show a PNG preview if the object is returned in an interactive shell.

        For the Jupyter notebook or IPython Qt console.
        """
        png = self._preview(fmt="png", dpi=70, anti_alias=True, as_bytes=True)
        return png

    def _repr_html_(self):
        """
        Show the PNG image embedded in HTML with a controlled width.

        Looks better than the raw PNG.
        """
        raw_png = self._preview(fmt="png", dpi=300, anti_alias=True, as_bytes=True)
        base64_png = base64.encodebytes(raw_png)
        html = '<img src="data:image/png;base64,{image}" width="{width}px">'
        return html.format(image=base64_png.decode("utf-8"), width=500)

    from pygmt.src import (  # type: ignore[misc]
        basemap,
        coast,
        colorbar,
        contour,
        grdcontour,
        grdimage,
        grdview,
        histogram,
        image,
        inset,
        legend,
        logo,
        meca,
        plot,
        plot3d,
        psconvert,
        rose,
        set_panel,
        shift_origin,
        solar,
        subplot,
        ternary,
        text,
        tilemap,
        timestamp,
        velo,
        wiggle,
    )
