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
    def show(
        self,
        method: Literal["external", "notebook", "none", None] = None,
        dpi: int = 300,
        width: int = 500,
        waiting: float = 0.5,
        **kwargs,
    ):
        """
        Display a preview of the figure.

        Inserts the preview in the Jupyter notebook output if available, otherwise opens
        it in the default viewer for your operating system (falls back to the default
        web browser).

        Use :func:`pygmt.set_display` to select the default display method
        (``"notebook"``, ``"external"``, ``"none"`` or ``None``).

        The ``method`` parameter allows to override the default display method for the
        current figure. The parameters ``dpi`` and ``width`` can be used to control the
        resolution and dimension of the figure in the notebook.

        The external viewer can be disabled by setting the environment variable
        :term:`PYGMT_USE_EXTERNAL_DISPLAY` to ``"false"``. This is useful when running
        tests and building the documentation to avoid popping up windows.

        The external viewer does not block the current process, thus it's necessary to
        suspend the execution of the current process for a short while after launching
        the external viewer, so that the preview image won't be deleted before the
        external viewer tries to open it. Set the ``waiting`` parameter to a larger
        number if the image viewer on your computer is slow to open the figure.

        Parameters
        ----------
        method
            The method to display the current image preview. Choose from:

            - ``"external"``: External PDF preview using the default PDF viewer
            - ``"notebook"``: Inline PNG preview in the current notebook
            - ``"none"``: Disable image preview
            - ``None``: Reset to the default display method

            The default display method is ``"external"`` in Python consoles and
            ``"notebook"`` in Jupyter notebooks, but can be changed by
            :func:`pygmt.set_display`.

        dpi
            The image resolution (dots per inch) in Jupyter notebooks.
        width
            The image width (in pixels) in Jupyter notebooks.
        waiting
            Suspend the execution of the current process for a given number of seconds
            after launching an external viewer. Only works if ``method="external"``.
        **kwargs : dict
            Additional keyword arguments passed to :meth:`pygmt.Figure.psconvert`. Valid
            parameters are ``gs_path``, ``gs_option``, ``resize``, ``bb_style``, and
            ``verbose``.
        """
        # Module level variable to know which figures had their show method called.
        # Needed for the sphinx-gallery scraper.
        SHOWED_FIGURES.append(self)

        # Set the display method
        if method is None:
            method = SHOW_CONFIG["method"]

        match method:
            case "notebook":
                if not _HAS_IPYTHON:
                    msg = (
                        "Notebook display is selected, but IPython is not available. "
                        "Make sure you have IPython installed, "
                        "or run the script in a Jupyter notebook."
                    )
                    raise ImportError(msg)
                png = self._preview(
                    fmt="png", dpi=dpi, anti_alias=True, as_bytes=True, **kwargs
                )
                IPython.display.display(IPython.display.Image(data=png, width=width))
            case "external":
                pdf = self._preview(
                    fmt="pdf", dpi=dpi, anti_alias=False, as_bytes=False, **kwargs
                )
                launch_external_viewer(pdf, waiting=waiting)
            case "none":
                pass  # Do nothing
            case _:
                msg = (
                    f"Invalid display method '{method}'. "
                    "Valid values are 'external', 'notebook', 'none' or None."
                )
                raise GMTInvalidInput(msg)

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
