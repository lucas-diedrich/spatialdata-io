from collections.abc import Mapping
from typing import Any

import dask.array as da
import numpy as np
from dask import delayed
from numpy.typing import NDArray
from pylibCZIrw import czi as pyczi
from spatialdata.models import Image2DModel

from ._utils import _assemble_delayed, _chunk_factory, _create_tiles


@delayed
def _get_img(
    slide: pyczi.CziReader,
    coords: tuple[int, int],
    size: tuple[int, int],
    channel: int,
) -> NDArray:
    """Return numpy array of slide region

    Parameters
    ----------
    slide
        WSI
    coords
        Upper left corner (x, y) to read
    size
        Size of tile
    channel
        Channel of image

    Returns
    -------
    np.array
        Image in (c, y, x) format and RGBA channels
    """
    # czi returns an np.ndarray
    # Shape VIHT*Z*C*X*Y* (*: Obligatory) https://zeiss.github.io/libczi/imagedocumentconcept.html#autotoc_md7
    # X/Y: 2D plane
    # Z: Z-stack
    # C: Channels
    # T: Time point
    # M is used in order to enumerate all tiles in a plane i.e all planes in a given plane shall have an M-index,
    # M-index starts counting from zero to the number of tiles on that plane
    # S: Tag-like- tags images of similar interest
    img = slide.read(
        plane={"C": channel},
        roi=(
            coords[0],  # xmin (x)
            coords[1],  # ymin (y)
            size[0],  # width (w)
            size[1],  # height (h)
        ),
    )

    # Return image in (c, y, x) format
    return np.array(img).T


def read_czi(path: str, chunk_size: tuple[int, int] = (10000, 10000), **kwargs: Mapping[str, Any]) -> Image2DModel:
    """Read .czi to Image2DModel

    Uses the CZI API to read .czi Carl Zeiss image format to spatialdata Image format

    Parameters
    ----------
    path
        Path to file
    chunk_size
        Size of the individual regions that are read into memory during the process
    kwargs
        Keyword arguments passed to Image2DModel.parse

    Returns
    -------
    Image2DModel
    """
    with pyczi.open_czi(path) as czidoc_r:
        # Read dimensions
        dimensions = czidoc_r.total_bounding_rectangle
        try:
            channels = dimensions["C"]
        except KeyError:
            raise KeyError("No channel information in .czi image")

        # Define coordinates for chunkwise loading of the slide
        chunk_coords, xmax, ymax = _create_tiles(dimensions=dimensions, tile_size=chunk_size)

        chunks = [
            _chunk_factory(
                _get_img,
                slide=czidoc_r,
                coords=chunk_coords,
                size=chunk_size,
                channel=c,
            )
            for c in range(*channels)
        ]

    array_ = _assemble_delayed(chunks)
    array = da.from_delayed(array_, shape=(4, ymax, xmax), dtype=np.uint8).rechunk(chunks=(4, *chunk_size[::-1]))

    return Image2DModel.parse(array, dims="cyx", c_coords="rgba", chunks=chunk_size, **kwargs)
