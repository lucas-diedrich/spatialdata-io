from __future__ import annotations

import json
import os
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from dask_image.imread import imread
from imageio import imread as imread2
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations import (
    Affine,
    Identity,
    Scale,
    Sequence,
    set_transformation,
)
from xarray import DataArray

from spatialdata_io._constants._constants import VisiumHDKeys
from spatialdata_io._docs import inject_docs


@inject_docs(vx=VisiumHDKeys)
def visium_hd(
    path: str | Path,
    dataset_id: str,
    filtered_counts_file: bool = True,
    bin_size: int | list[int] | None = None,
    fullres_image_file: str | Path | None = None,
    load_all_images: bool = False,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    anndata_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """
    Read *10x Genomics* Visium HD formatted dataset.

    .. seealso::

        - `Space Ranger output
            <https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview>`_.

    Parameters
    ----------
    path
        Path to directory containing the *10x Genomics* Visium HD output.
    dataset_id
        Unique identifier of the dataset.
    filtered_counts_file
        It sets the value of `counts_file` to ``{vx.FILTERED_COUNTS_FILE!r}`` (when `True`) or to
        ``{vx.RAW_COUNTS_FILE!r}`` (when `False`).
    bin_size
        When specified, load the data of a specific bin size, or a list of bin sizes. By default, it loads all the
        available bin sizes.
    fullres_image_file
        Path to the full-resolution image. By default the image is searched in the ``{vx.MICROSCOPE_IMAGE!r}``
        directory.
    load_all_images
        If `False`, load only the full resolution, high resolution and low resolution images. If `True`, also the
        following images: ``{vx.IMAGE_CYTASSIST!r}``.
    imread_kwargs
        Keyword arguments for :func:`imageio.imread`.
    image_models_kwargs
        Keyword arguments for :class:`spatialdata.models.Image2DModel`.
    anndata_kwargs
        Keyword arguments for :func:`anndata.read_h5ad`.

    Returns
    -------
    SpatialData object for the Visium HD data.
    """
    path = Path(path)
    tables = {}
    shapes = {}
    images: dict[str, Any] = {}

    def load_image(path: Path, suffix: str, scale_factors: list[int] | None = None) -> None:
        _load_image(
            path=path,
            images=images,
            suffix=suffix,
            dataset_id=dataset_id,
            imread_kwargs=imread_kwargs,
            image_models_kwargs=image_models_kwargs,
            scale_factors=scale_factors,
        )

    metadata, hd_layout = _parse_metadata(path)
    transform_matrices = _get_transform_matrices(metadata, hd_layout)
    file_format = hd_layout[VisiumHDKeys.FILE_FORMAT]
    if file_format != "1.0":
        warnings.warn(
            f"File format {file_format} is not supported. A more recent file format may be supported in a newer version"
            f"of the spatialdata-io package.",
            UserWarning,
            stacklevel=2,
        )

    path_bins = path / VisiumHDKeys.BINNED
    all_bin_sizes = sorted(
        [bin_size for bin_size in os.listdir(path_bins) if os.path.isdir(os.path.join(path_bins, bin_size))]
    )
    if bin_size is None:
        bin_sizes = all_bin_sizes
    elif isinstance(bin_size, int) or isinstance(bin_size, list) and len(bin_size) == 0:
        if f"square_{bin_size:03}um" not in all_bin_sizes:
            warnings.warn(
                f"Requested bin size {bin_size} not found (available {all_bin_sizes}). Using all available bins.",
                UserWarning,
                stacklevel=2,
            )
            bin_sizes = all_bin_sizes
        else:
            bin_sizes = [f"square_{bin_size:03}um"]

    # iterate over the given bins and load the data
    for bin_size_str in bin_sizes:
        path_bin = path_bins / bin_size_str
        counts_file = VisiumHDKeys.FILTERED_COUNTS_FILE if filtered_counts_file else VisiumHDKeys.RAW_COUNTS_FILE
        adata = sc.read_10x_h5(
            path_bin / counts_file,
            gex_only=False,
            **anndata_kwargs,
        )

        path_bin_spatial = path_bin / VisiumHDKeys.SPATIAL

        with open(path_bin_spatial / VisiumHDKeys.SCALEFACTORS_FILE) as file:
            scalefactors = json.load(file)

        # consistency check
        found_bin_size = re.search(r"\d{3}", bin_size_str)
        assert found_bin_size is not None
        assert float(found_bin_size.group()) == scalefactors[VisiumHDKeys.SCALEFACTORS_BIN_SIZE_UM]
        assert np.isclose(
            scalefactors[VisiumHDKeys.SCALEFACTORS_BIN_SIZE_UM]
            / scalefactors[VisiumHDKeys.SCALEFACTORS_SPOT_DIAMETER_FULLRES],
            scalefactors[VisiumHDKeys.SCALEFACTORS_MICRONS_PER_PIXEL],
        )

        tissue_positions_file = path_bin_spatial / VisiumHDKeys.TISSUE_POSITIONS_FILE

        # read coordinates and set up adata.obs and adata.obsm
        coords = pd.read_parquet(tissue_positions_file)
        assert all(
            coords.columns.values
            == [
                VisiumHDKeys.BARCODE,
                VisiumHDKeys.IN_TISSUE,
                VisiumHDKeys.ARRAY_ROW,
                VisiumHDKeys.ARRAY_COL,
                VisiumHDKeys.LOCATIONS_Y,
                VisiumHDKeys.LOCATIONS_X,
            ]
        )
        coords.set_index(VisiumHDKeys.BARCODE, inplace=True, drop=True)
        coords_filtered = coords.loc[adata.obs.index]
        adata.obs = pd.merge(adata.obs, coords_filtered, how="left", left_index=True, right_index=True)
        # compatibility to legacy squidpy
        adata.obsm["spatial"] = adata.obs[[VisiumHDKeys.LOCATIONS_X, VisiumHDKeys.LOCATIONS_Y]].values
        # dropping the spatial coordinates (will be stored in shapes)
        adata.obs.drop(
            columns=[
                VisiumHDKeys.LOCATIONS_X,
                VisiumHDKeys.LOCATIONS_Y,
            ],
            inplace=True,
        )
        adata.obs[VisiumHDKeys.INSTANCE_KEY] = np.arange(len(adata))

        # scaling
        transform_original = Identity()
        transform_lowres = Scale(
            np.array(
                [
                    scalefactors[VisiumHDKeys.SCALEFACTORS_LOWRES],
                    scalefactors[VisiumHDKeys.SCALEFACTORS_LOWRES],
                ]
            ),
            axes=("x", "y"),
        )
        transform_hires = Scale(
            np.array(
                [
                    scalefactors[VisiumHDKeys.SCALEFACTORS_HIRES],
                    scalefactors[VisiumHDKeys.SCALEFACTORS_HIRES],
                ]
            ),
            axes=("x", "y"),
        )
        # parse shapes
        circles = ShapesModel.parse(
            adata.obsm["spatial"],
            geometry=0,
            radius=scalefactors[VisiumHDKeys.SCALEFACTORS_SPOT_DIAMETER_FULLRES] / 2.0,
            index=adata.obs[VisiumHDKeys.INSTANCE_KEY].copy(),
            transformations={
                "global": transform_original,
                "downscaled_hires": transform_hires,
                "downscaled_lowres": transform_lowres,
            },
        )
        shapes_name = dataset_id + "_" + bin_size_str
        shapes[shapes_name] = circles

        # parse table
        adata.obs[VisiumHDKeys.REGION_KEY] = shapes_name
        adata.obs[VisiumHDKeys.REGION_KEY] = adata.obs[VisiumHDKeys.REGION_KEY].astype("category")
        tables[bin_size_str] = TableModel.parse(
            adata,
            region=shapes_name,
            region_key=str(VisiumHDKeys.REGION_KEY),
            instance_key=str(VisiumHDKeys.INSTANCE_KEY),
        )

    # read full resolution image
    if fullres_image_file is not None:
        fullres_image_file = Path(fullres_image_file)
    else:
        path_fullres = path / VisiumHDKeys.MICROSCOPE_IMAGE
        if path_fullres.exists():
            fullres_image_filenames = [
                f for f in os.listdir(path_fullres) if os.path.isfile(os.path.join(path_fullres, f))
            ]
            if len(fullres_image_filenames) > 1:
                warnings.warn(
                    f"Multiple files found in {path_fullres}, using the first one: {fullres_image_filenames[0]}. Please"
                    " specify the path to the full resolution image manually using the `fullres_image_file` argument.",
                    UserWarning,
                    stacklevel=2,
                )
            fullres_image_filename = fullres_image_filenames[0]
            fullres_image_file = path_fullres / fullres_image_filename

    if fullres_image_file is not None:
        load_image(
            path=fullres_image_file,
            suffix="_full_image",
            scale_factors=[2, 2, 2, 2],
        )

    # hires image
    load_image(
        path=path / VisiumHDKeys.IMAGE_HIRES_FILE,
        suffix="_hires_image",
    )
    set_transformation(
        images[dataset_id + "_hires_image"],
        {"downscaled_hires": Identity()},
        set_all=True,
    )

    # lowres image
    load_image(
        path=path / VisiumHDKeys.IMAGE_LOWRES_FILE,
        suffix="_lowres_image",
    )
    set_transformation(
        images[dataset_id + "_lowres_image"],
        {"downscaled_lowres": Identity()},
        set_all=True,
    )

    # cytassist image
    if load_all_images:
        load_image(
            path=path / VisiumHDKeys.IMAGE_CYTASSIST,
            suffix="_cytassist_image",
        )
    image = images[dataset_id + "_cytassist_image"]
    affine0 = transform_matrices["cytassist_colrow_to_spot_colrow"]
    affine1 = transform_matrices["spot_colrow_to_microscope_colrow"]
    set_transformation(image, Sequence([affine0, affine1]), "global")

    sdata = SpatialData(tables=tables, images=images, shapes=shapes)

    return sdata


def _load_image(
    path: Path,
    images: dict[str, SpatialImage | MultiscaleSpatialImage],
    suffix: str,
    dataset_id: str,
    imread_kwargs: Mapping[str, Any],
    image_models_kwargs: Mapping[str, Any],
    scale_factors: list[int] | None,
) -> None:
    if path.exists():
        if path.suffix != ".btf":
            data = imread(path, **imread_kwargs)
            if len(data.shape) == 4:
                # this happens for the cytassist, hires and lowres images; the umi image doesn't need processing
                data = data.squeeze().transpose(2, 0, 1)
        else:
            if "MAX_IMAGE_PIXELS" in imread_kwargs:
                from PIL import Image as ImagePIL

                ImagePIL.MAX_IMAGE_PIXELS = dict(imread_kwargs).pop("MAX_IMAGE_PIXELS")
            # dask_image doesn't recognize .btf automatically and imageio v3 throws error due to pixel limit -> use imageio v2
            data = imread2(path, **imread_kwargs).squeeze().transpose(2, 0, 1)
        image = DataArray(data, dims=("c", "y", "x"))
        parsed = Image2DModel.parse(image, scale_factors=scale_factors, **image_models_kwargs)
        images[dataset_id + suffix] = parsed
    else:
        warnings.warn(f"File {path} does not exist, skipping it.", UserWarning, stacklevel=2)
    return None


def _get_affine(coefficients: list[int]) -> Affine:
    matrix = np.array(coefficients).reshape(3, 3)
    # the last row doesn't match with machine precision, let's check the matrix it's still close to a homogeneous
    # matrix, and fix this
    assert np.allclose(matrix[2], [0, 0, 1], atol=1e-2), matrix
    matrix[2] = [0, 0, 1]
    return Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))


def _parse_metadata(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with h5py.File(path / VisiumHDKeys.FEATURE_SLICE_FILE, "r") as f5:
        metadata = json.loads(dict(f5.attrs)[VisiumHDKeys.METADATA_JSON])
        hd_layout = json.loads(metadata[VisiumHDKeys.HD_LAYOUT_JSON])
    return metadata, hd_layout


def _get_transform_matrices(metadata: dict[str, Any], hd_layout: dict[str, Any]) -> dict[str, Affine]:
    transform_matrices = {}

    for key in [
        VisiumHDKeys.CYTASSIST_COLROW_TO_SPOT_COLROW,
        VisiumHDKeys.SPOT_COLROW_TO_CYTASSIST_COLROW,
        VisiumHDKeys.MICROSCOPE_COLROW_TO_SPOT_COLROW,
        VisiumHDKeys.SPOT_COLROW_TO_MICROSCOPE_COLROW,
    ]:
        data = metadata[VisiumHDKeys.TRANSFORM_MATRICES][key]
        transform_matrices[key.value] = _get_affine(data)

    return transform_matrices
