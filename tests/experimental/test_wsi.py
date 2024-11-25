from spatialdata_io.experimental.wsi import read_wsi, _create_tiles
import openslide
import numpy as np
import pytest


@pytest.mark.parametrize(
    ("dimensions", "tile_size", "output_shape", "output_max"),
    [
        ((10, 10), (1, 1), (10, 10, 2), (10, 10)),
        ((10, 10), (2, 1), (5, 10, 2), (10, 10)),
        ((10, 10), (1, 2), (10, 5, 2), (10, 10)),
        ((5, 5), (6, 6), (1, 1, 2), (6, 6)),
    ],
)
def test__create_tiles(
    dimensions: tuple[int, int],
    tile_size: tuple[int, int],
    output_shape: tuple[int, int],
    output_max,
):
    array, xmax, ymax = _create_tiles(dimensions, tile_size)
    assert array.shape == output_shape
    assert (xmax == output_max[0]) & (ymax == output_max[1])


@pytest.mark.parametrize(
    ("dataset", "xmin", "xmax", "ymin", "ymax"),
    [
        # Use weird coords as most of the WSI is empty
        # ("DigitalSlide_A6M_7S_1.mrxs", 75000, 76000, 118000, 119000),
        ("Mirax2.2-4-PNG.mrxs", 0, 0, 1000, 1000),
    ],
)
def test_read_wsi_pyramid(
    dataset: str, xmin: int, xmax: int, ymin: int, ymax: int
) -> None:
    """Test whether image can be loaded"""
    path = f"./data/wsi/mirax/{dataset}"
    image_model = read_wsi(path, pyramidal=True)

    # Get a subset of the image
    test_image = image_model.scale0.image[:, ymin:ymax, xmin:xmax].to_numpy()

    # Read image directly with openslide
    slide = openslide.OpenSlide(path)
    ref_image = np.array(
        slide.read_region((xmin, ymin), level=0, size=(xmax - xmin, ymax - ymin))
    )

    assert (test_image == ref_image.T).all()


@pytest.mark.parametrize(
    "chunk_size",
    [(1000, 1000), (10000, 10000), (10000, 5000)],
)
def test_read_wsi_chunksize(chunk_size: bool) -> None:
    """Test whether chunking works"""
    path = "./data/wsi/mirax/Mirax2.2-4-PNG.mrxs"
    image_model = read_wsi(path, chunk_size=chunk_size, pyramidal=False)

    assert all(
        [
            ((x == chunk_size[0]) & (y == chunk_size[1]))
            for x, y in zip(image_model.chunksizes["x"], image_model.chunksizes["y"])
        ]
    )
