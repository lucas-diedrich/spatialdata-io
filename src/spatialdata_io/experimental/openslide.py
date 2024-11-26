"""Reader for Whole Slide Images"""

import dask.array as da
import numpy as np
import openslide
from dask import delayed
from numpy.typing import NDArray
from spatialdata.models import Image2DModel

from ._utils import _assemble_delayed, _chunk_factory, _create_tiles


@delayed
def _get_img(
    slide: openslide.ImageSlide,
    coords: tuple[int, int],
    size: tuple[int, int],
    level: int,
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
    level
        Level in pyramidal image format

    Returns
    -------
    np.array
        Image in (c, y, x) format and RGBA channels
    """
    # Openslide returns a PILLOW image in RGBA format
    # Shape (x, y, c)
    img = slide.read_region(coords, level=level, size=size)

    # Return image in (c, y, x) format
    return np.array(img).T


def read_openslide(path: str, chunk_size: tuple[int, int] = (10000, 10000), pyramidal: bool = True) -> Image2DModel:
    """Read WSI to Image2DModel

    Uses openslide to read multiple pathology slide representations and parse them
    to a lazy dask array. Currently supported formats

    [tested]
    - .mirax (Mirax format)

    [in principle supported by openslide]
    - Aperio (.svs, .tif)
    - DICOM (.dcm)
    - Hamamatsu (.ndpi, .vms, .vmu)
    - Leica (.scn)
    - MIRAX (.mrxs)
    - Philips (.tiff)
    - Sakura (.svslide)
    - Trestle (.tif)
    - Ventana (.bif, .tif)
    - Zeiss (.czi)
    - Generic tiled TIFF (.tif)

    Parameters
    ----------
    path
        Path to file
    chunk_size
        Size of the individual regions that are read into memory during the process
    pyramidal
        Whether to create a pyramidal image with same scales as original image

    Returns
    -------
    Image2DModel
    """
    slide = openslide.OpenSlide(path)

    # Image is represented as pyramid. Read highest resolution
    dimensions = slide.dimensions

    # Openslide represents scales in format (level[0], level[1], ...)
    # Each scale factor is represented relative to top level
    # Get downsamples in format that can be passed to Image2DModel
    scale_factors = None
    if pyramidal:
        scale_factors = [
            int(slide.level_downsamples[i] / slide.level_downsamples[i - 1])
            for i in range(1, len(slide.level_downsamples))
        ]

    # Define coordinates for chunkwise loading of the slide
    chunk_coords = _create_tiles(dimensions=dimensions, tile_size=chunk_size, min_coordinates=(0, 0))

    chunks = _chunk_factory(_get_img, slide=slide, coords=chunk_coords, level=0)

    # Assemble into a single dask array
    array_ = _assemble_delayed(chunks)
    array = da.from_delayed(array_, shape=(4, *dimensions[::-1]), dtype=np.uint8).rechunk()

    return Image2DModel.parse(
        array,
        dims="cyx",
        c_coords="rgba",
        scale_factors=scale_factors,
        chunks=chunk_size,
    )
