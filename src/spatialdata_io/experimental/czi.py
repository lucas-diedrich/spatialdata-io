from collections.abc import Mapping
from enum import Enum
from typing import Any, Optional

import dask.array as da
import numpy as np
from numpy.typing import NDArray
from pylibCZIrw import czi as pyczi
from spatialdata.models import Image2DModel

from ._utils import _assemble_delayed, _chunk_factory, _create_tiles


class CZIPixelType(Enum):
    """Features of CZI pixel types

    Stores dimensionality, data type, and channel names of CZI pixel types
    as class for simplified access.
    Documented pixel types https://zeiss.github.io/libczi/accessors.html
    """

    Gray8 = (1, np.uint8, None)
    Gray16 = (1, np.uint16, None)
    Gray32Float = (1, np.float32, None)
    Bgr24 = (3, np.uint8, "rgb")
    Bgr48 = (3, np.uint16, "rgb")
    Bgr96Float = (3, np.float32, "rgb")
    Invalid = (np.nan, np.nan, np.nan)

    def __init__(self, dimensionality: int, dtype: type, c_coords: Optional[str]) -> None:
        self.dimensionality = dimensionality
        self.dtype = dtype
        self.c_coords = c_coords


def _get_img(
    slide: pyczi.CziReader,
    coords: tuple[int, int],
    size: tuple[int, int],
    channel: int = 0,
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


def read_czi(
    path: str,
    chunk_size: tuple[int, int] = (10000, 10000),
    channel: int = 0,
    timepoint: int = 0,
    z_stack: int = 0,
    **kwargs: Mapping[str, Any],
) -> Image2DModel:
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
    channel
        If multiple channels are available, select these channels
    timepoint
        If timeseries, select the given index (defaults to 0 [first])
    z_stack
        If z_stack, defaults to the given stack/index (defaults to 0 [first])

    Returns
    -------
    Image2DModel
    """
    czidoc_r = pyczi.CziReader(path)

    # Read dimensions
    xmin, ymin, width, height = czidoc_r.total_bounding_rectangle

    # We need to know the pixel type to infer the dimensionality of the image
    pixel_type = czidoc_r.get_channel_pixel_type(channel)

    # Define coordinates for chunkwise loading of the slide
    chunk_coords = _create_tiles(dimensions=(width, height), tile_size=chunk_size, min_coordinates=(xmin, ymin))

    # TODO Currently, only 1 channel [RGB/grayscale] is read - it might be complicated to read multiple channels
    # if there are both brightfield (RGB) + fluorescence/grayscale images in the same file
    chunks = _chunk_factory(_get_img, slide=czidoc_r, coords=chunk_coords, channel=channel)

    array_ = _assemble_delayed(chunks)

    array = da.from_delayed(
        array_,
        shape=(CZIPixelType[pixel_type].dimensionality, width, height),
        dtype=CZIPixelType[pixel_type].dtype,
    )

    return Image2DModel.parse(
        array,
        dims="cyx",
        c_coords=CZIPixelType[pixel_type].c_coords,
        **kwargs,
    )
