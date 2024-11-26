import numpy as np
import openslide
import pytest

from spatialdata_io.experimental.openslide import read_openslide


@pytest.mark.parametrize(
    ("dataset", "xmin", "ymin", "xmax", "ymax"),
    [
        ("Mirax2.2-4-PNG.mrxs", 0, 0, 1000, 1000),
        # Asymmetric
        ("Mirax2.2-4-PNG.mrxs", 0, 0, 500, 1000),
    ],
)
def test_read_openslide(dataset: str, xmin: int, xmax: int, ymin: int, ymax: int) -> None:
    """Test whether image can be loaded"""
    path = f"./data/wsi/mirax/{dataset}"
    image_model = read_openslide(path, pyramidal=True)

    # Get a subset of the image
    test_image = image_model.scale0.image[:, ymin:ymax, xmin:xmax].transpose("y", "x", "c").to_numpy()

    # Read image directly with openslide
    slide = openslide.OpenSlide(path)
    ref_image = np.array(slide.read_region((xmin, ymin), level=0, size=(xmax - xmin, ymax - ymin)))

    assert (test_image == ref_image).all()


@pytest.mark.parametrize(
    "chunk_size",
    [(1000, 1000), (10000, 10000), (10000, 5000)],
)
def test_read_openslide_chunksize(chunk_size: tuple[int, int]) -> None:
    """Test whether chunking works"""
    path = "./data/wsi/mirax/Mirax2.2-4-PNG.mrxs"
    image_model = read_openslide(path, chunk_size=chunk_size, pyramidal=False)

    assert all(
        (
            ((x == chunk_size[0]) & (y == chunk_size[1]))
            for x, y in zip(image_model.chunksizes["x"], image_model.chunksizes["y"])
        )
    )
