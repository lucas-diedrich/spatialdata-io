"""Reader for Whole Slide Images"""

import openslide
import dask.array as da
from dask import delayed
import numpy as np
from spatialdata.models import Image2DModel


def _create_tiles(
    dimensions: tuple[int, int], tile_size: tuple[int, int]
) -> tuple[np.ndarray, int, int]:
    """Tiling of WSI in digestible, rectangular chunks

    Parameters
    ----------
    dimensions
        Size of WSI in (x, y)
    tile_size
        Size of individual tiles in (x, y)

    Returns
    -------
    tuple[np.ndarray, int, int]
        - Array in the shape (n_tile_x, n_tile_y, 2) where the last dimension
        indicates the upper left corner of each tile (x, y).
        - maximum value in x coordinates
        - maximum value in y coordinates
    """

    n_tile_x = int(np.ceil(dimensions[0] / tile_size[0]))
    n_tile_y = int(np.ceil(dimensions[1] / tile_size[1]))

    xmax = int(n_tile_x * tile_size[0])
    ymax = int(n_tile_y * tile_size[1])

    # Get all grid points
    x, y = np.meshgrid(
        np.arange(0, xmax, tile_size[0]),
        np.arange(0, ymax, tile_size[1]),
    )

    tile_coords = np.stack([x, y], axis=-1)

    return tile_coords, xmax, ymax


@delayed
def _get_img(
    slide: openslide.ImageSlide,
    coords: tuple[int, int],
    size: tuple[int, int],
    level: int,
) -> np.array:
    """Return numpy array of slide region

    Parameters
    ----------
    """
    # Openslide returns a PILLOW image in RGBA format
    # Shape (x, y, c)
    img = slide.read_region(coords, level=level, size=size)

    # Return image in (c, y, x) format
    return np.array(img).T


@delayed
def _assemble_delayed(chunks: list[list]):
    """Assemble chunks (delayed)"""
    return da.block(chunks)


def read_wsi(
    path: str, chunk_size=(10000, 10000), pyramidal: bool = True
) -> Image2DModel:
    """Read WSI to Image2DModel

    Uses openslide to read multiple pathology slide representations and parse them
    to a lazy dask array. Currently supported formats

    [tested]
    - .czi (Carl Zeiss proprietary format)
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
    dask.array
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
    chunk_coords, xmax, ymax = _create_tiles(
        dimensions=dimensions, tile_size=chunk_size
    )

    # Collect each delayed chunk as item in list of list
    # Inner list becomes dim=-1 (rows)
    # Outer list becomes dim=-2 (cols)
    # see dask.array.block
    chunks = [
        [
            _get_img(
                slide=slide, coords=chunk_coords[row, col], size=chunk_size, level=0
            )
            for row in range(chunk_coords.shape[0])
        ]
        for col in range(chunk_coords.shape[1])
    ]

    # Delayed
    array_ = _assemble_delayed(chunks)
    array = da.from_delayed(array_, shape=(4, ymax, xmax), dtype=np.uint8).rechunk(
        chunks=(4, *chunk_size[::-1])
    )

    return Image2DModel.parse(
        array,
        dims="cyx",
        c_coords="rgba",
        scale_factors=scale_factors,
        chunks=chunk_size,
    )
