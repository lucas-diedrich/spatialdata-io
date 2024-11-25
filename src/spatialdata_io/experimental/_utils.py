from collections.abc import Mapping
from typing import Any, Callable

import dask.array as da
import numpy as np
from dask import delayed
from numpy.typing import NDArray


def _create_tiles(dimensions: tuple[int, int], tile_size: tuple[int, int]) -> tuple[NDArray, int, int]:
    """Tiling of WSI in digestible, rectangular chunks

    Parameters
    ----------
    dimensions
        Size of WSI in (x, y)
    tile_size
        Size of individual tiles in (x, y)

    Returns
    -------
    tuple[NDArray, int, int]
        - Array in the shape (n_tile_x, n_tile_y, 2) where the last dimension indicates the upper left corner of each tile (x, y).
        - maximum value in x coordinates
        - maximum value in y coordinates

    """
    n_tile_row = int(np.ceil(dimensions[0] / tile_size[0]))
    n_tile_col = int(np.ceil(dimensions[1] / tile_size[1]))

    row_max = int(n_tile_row * tile_size[0])
    col_max = int(n_tile_col * tile_size[1])

    # Get all grid points
    cols, rows = np.meshgrid(np.arange(0, row_max, tile_size[0]), np.arange(0, col_max, tile_size[1]))

    tile_coords = np.stack([rows.T, cols.T], axis=-1)

    return tile_coords, row_max, col_max


@delayed
def _chunk_factory(
    func: Callable[..., NDArray],
    slide: Any,
    coords: NDArray,
    size: tuple[int, int],
    **func_kwargs: Mapping[str, Any],
) -> list[list[NDArray]]:
    """Abstract factory method to tile a large microscopy image.

    Parameters
    ----------
    func
        Function to retrieve a rectangular tile from the slide image
    slide
        Slide image in format compatible with func
    coords
        Coordinates of the upper left corner of the image in formt (n_row_x, n_row_y, 2)
        where the last dimension indicates the position in format (x, y)
    func_kwargs
        Additional keyword arguments passed to func
    """
    func_kwargs = func_kwargs if func_kwargs else {}

    # Collect each delayed chunk as item in list of list
    # Inner list becomes dim=-1 (rows)
    # Outer list becomes dim=-2 (cols)
    # see dask.array.block
    chunks = [
        [func(slide, coords=coords[x, y], size=size, **func_kwargs) for x in range(coords.shape[0])]
        for y in range(coords.shape[1])
    ]

    return chunks


@delayed
def _assemble_delayed(chunks: list[list[NDArray]]) -> NDArray:
    """Assemble chunks (delayed)"""
    return da.block(chunks)
