from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Union

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from dask_image.imread import imread
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, ShapesModel, TableModel

from spatialdata_io._docs import inject_docs
from spatialdata_io.readers._utils._utils import _initialize_raster_models_kwargs

__all__ = ["stereoseq"]


@inject_docs(xx=StereoseqKeys)
def stereoseq(
    path: str | Path,
    dataset_id: Union[str, None] = None,
    read_square_bin: bool = True,
    optional_tif: bool = False,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    labels_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """
    Read *Stereo-seq* formatted dataset.

    Parameters
    ----------
    path
        Path to the directory containing the data.
    dataset_id
        Dataset identifier. If not given will be determined automatically.
    read_square_bin
        If True, will read the square bin ``{xx.GEF_FILE!r}`` file and build corresponding points element.
    optional_tif
        If True, will read ``{xx.TISSUE_TIF!r}`` files.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.
    labels_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Labels2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    image_models_kwargs, labels_models_kwargs = _initialize_raster_models_kwargs(
        image_models_kwargs, labels_models_kwargs
    )
    path = Path(path)

    if dataset_id is None:
        dataset_id_path = path / StereoseqKeys.COUNT_DIRECTORY
        gef_files = [filename for filename in os.listdir(dataset_id_path) if filename.endswith(StereoseqKeys.GEF_FILE)]

        if gef_files:
            first_gef_file = gef_files[0]
            square_bin = str(first_gef_file.split(".")[0]) + StereoseqKeys.GEF_FILE
        else:
            raise ValueError(f"No {StereoseqKeys.GEF_FILE!r} files found in {dataset_id_path!r}")
    else:
        square_bin = dataset_id + StereoseqKeys.GEF_FILE

    image_patterns = [
        re.compile(r".*" + re.escape(StereoseqKeys.MASK_TIF)),
        re.compile(r".*" + re.escape(StereoseqKeys.REGIST_TIF)),
    ]
    if optional_tif:
        image_patterns.append(
            re.compile(r".*" + re.escape(StereoseqKeys.TISSUE_TIF)),
        )

    gef_patterns = [
        re.compile(r".*" + re.escape(StereoseqKeys.RAW_GEF)),
        re.compile(r".*" + re.escape(StereoseqKeys.CELLBIN_GEF)),
        re.compile(r".*" + re.escape(StereoseqKeys.TISSUECUT_GEF)),
        re.compile(r".*" + re.escape(square_bin)),
    ]

    image_filenames = [
        i for i in os.listdir(path / StereoseqKeys.REGISTER) if any(pattern.match(i) for pattern in image_patterns)
    ]
    cell_mask_file = [x for x in image_filenames if (f"{StereoseqKeys.MASK_TIF}" in x)]
    image_filenames = [x for x in image_filenames if x not in cell_mask_file]

    cellbin_gef_filename = [
        i for i in os.listdir(path / StereoseqKeys.CELLCUT) if any(pattern.match(i) for pattern in gef_patterns)
    ]
    squarebin_gef_filename = [
        i for i in os.listdir(path / StereoseqKeys.TISSUECUT) if any(pattern.match(i) for pattern in gef_patterns)
    ]

    table_pattern = re.compile(r".*" + re.escape(StereoseqKeys.CELL_CLUSTER_H5AD))
    table_filename = [i for i in os.listdir(path / StereoseqKeys.CELLCLUSTER) if table_pattern.match(i)]

    # create table using .h5ad and cellbin.gef
    adata = ad.read_h5ad(path / StereoseqKeys.CELLCLUSTER / table_filename[0])
    path_cellbin = path / StereoseqKeys.CELLCUT / cellbin_gef_filename[0]
    cellbin_gef = h5py.File(str(path_cellbin), "r")

    # add cell info to obs
    obs = pd.DataFrame(cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.CELL_DATASET][:])
    obs.columns = [
        StereoseqKeys.CELL_ID,
        StereoseqKeys.COORD_X,
        StereoseqKeys.COORD_Y,
        StereoseqKeys.OFFSET,
        StereoseqKeys.GENE_COUNT,
        StereoseqKeys.EXP_COUNT,
        StereoseqKeys.DNBCOUNT,
        StereoseqKeys.CELL_AREA,
        StereoseqKeys.CELL_TYPE_ID,
        StereoseqKeys.CLUSTER_ID,
    ]
    cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.CELL_DATASET][:]

    obsm_spatial = obs[[StereoseqKeys.COORD_X, StereoseqKeys.COORD_Y]].to_numpy()
    obs = obs.drop([StereoseqKeys.COORD_X, StereoseqKeys.COORD_Y], axis=1)
    obs[StereoseqKeys.CELL_EXON] = cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.CELL_EXON][:]

    # add gene info to var
    var = pd.DataFrame(
        cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.FEATURE_KEY][:],
        columns=[
            StereoseqKeys.GENE_NAME,
            StereoseqKeys.OFFSET,
            StereoseqKeys.CELL_COUNT,
            StereoseqKeys.EXP_COUNT,
            StereoseqKeys.MAX_MID_COUNT,
        ],
    )
    var[StereoseqKeys.GENE_NAME] = var[StereoseqKeys.GENE_NAME].str.decode("utf-8")
    var[StereoseqKeys.GENE_EXON] = cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.GENE_EXON][:]

    # add cell border to obsm
    cell_coordinates = [x for i in range(1, 33) for x in ("x_" + str(i), "y_" + str(i))]
    n_cells, n_coords, xy = cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.CELL_BORDER][:].shape
    arr = cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.CELL_BORDER][:]
    new_arr = arr.reshape(n_cells, n_coords * xy)
    obsm = pd.DataFrame(new_arr, columns=cell_coordinates)

    obs.index = adata.obs.index
    adata.obs = pd.merge(adata.obs, obs, left_index=True, right_index=True)
    var.index = adata.var.index
    adata.var = pd.merge(adata.var, var, left_index=True, right_index=True)
    obsm.index = adata.obs.index
    adata.obsm[StereoseqKeys.CELL_BORDER] = obsm
    adata.obsm[StereoseqKeys.SPATIAL_KEY] = obsm_spatial

    # add region and instance_id to obs for the TableModel
    adata.obs[StereoseqKeys.REGION_KEY] = StereoseqKeys.REGION
    adata.obs[StereoseqKeys.REGION_KEY] = adata.obs[StereoseqKeys.REGION_KEY].astype("category")
    adata.obs[StereoseqKeys.INSTANCE_KEY] = adata.obs.index

    # add all leftover columns in cellbin which don't fit .obs or .var to uns
    cell_exp = cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.CELL_EXP]
    gene_exp = cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.GENE_EXP]
    cellbin_uns = {
        StereoseqKeys.GENE_ID: cell_exp[StereoseqKeys.GENE_ID][:],
        StereoseqKeys.CELL_EXP: cell_exp[StereoseqKeys.COUNT][:],
        StereoseqKeys.GENE_EXP_EXON: cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.CELL_EXP_EXON][:],
        StereoseqKeys.CELL_ID: gene_exp[StereoseqKeys.CELL_ID][:],
        StereoseqKeys.GENE_EXP: gene_exp[StereoseqKeys.COUNT][:],
        StereoseqKeys.GENE_EXP_EXON: cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.GENE_EXP_EXON][:],
    }
    cellbin_uns_df = pd.DataFrame(cellbin_uns)

    adata.uns["cellBin_cell_gene_exon_exp"] = cellbin_uns_df
    adata.uns["cellBin_blockIndex"] = cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.BLOCK_INDEX][:]
    adata.uns["cellBin_blockSize"] = cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.BLOCK_SIZE][:]
    adata.uns["cellBin_cellTypeList"] = cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.CELL_TYPE_LIST][:]

    # add cellbin attrs to uns
    cellbin_attrs = {}
    for i in cellbin_gef.attrs.keys():
        cellbin_attrs[i] = cellbin_gef.attrs[i]
    adata.uns["cellBin_attrs"] = cellbin_attrs

    images = {
        f"{name}": Image2DModel.parse(
            imread(path / StereoseqKeys.REGISTER / name, **imread_kwargs), dims=("c", "y", "x"), **image_models_kwargs
        )
        for name in image_filenames
    }

    labels = {
        f"{cell_mask_name}": Labels2DModel.parse(
            imread(path / StereoseqKeys.REGISTER / cell_mask_name, **imread_kwargs).squeeze(),
            dims=("y", "x"),
            **labels_models_kwargs,
        )
        for cell_mask_name in cell_mask_file
    }

    table = TableModel.parse(
        adata,
        region=StereoseqKeys.REGION.value,
        region_key=StereoseqKeys.REGION_KEY.value,
        instance_key=StereoseqKeys.INSTANCE_KEY.value,
    )
    tables = {StereoseqKeys.REGION: table}

    radii = np.sqrt(adata.obs[StereoseqKeys.CELL_AREA].to_numpy() / np.pi)
    shapes = {
        StereoseqKeys.REGION: ShapesModel.parse(
            adata.obsm[StereoseqKeys.SPATIAL_KEY], geometry=0, radius=radii, index=adata.obs[StereoseqKeys.INSTANCE_KEY]
        )
    }
    shapes[StereoseqKeys.REGION].index.name = None

    if read_square_bin:
        # create points model using SquareBin.gef
        path_squarebin = path / StereoseqKeys.TISSUECUT / squarebin_gef_filename[0]
        squarebin_gef = h5py.File(str(path_squarebin), "r")

        dict(squarebin_gef[StereoseqKeys.GENE_EXP]["bin1"]["expression"].attrs)
        for i in squarebin_gef[StereoseqKeys.GENE_EXP].keys():
            # get gene info
            arr = squarebin_gef[StereoseqKeys.GENE_EXP][i][StereoseqKeys.FEATURE_KEY][:]
            df_gene = pd.DataFrame(arr, columns=[StereoseqKeys.FEATURE_KEY, StereoseqKeys.OFFSET, StereoseqKeys.COUNT])
            df_gene[StereoseqKeys.FEATURE_KEY] = df_gene[StereoseqKeys.FEATURE_KEY].str.decode("utf-8")

            # create df for points model
            arr = squarebin_gef[StereoseqKeys.GENE_EXP][i][StereoseqKeys.EXPRESSION][:]
            df_points = pd.DataFrame(arr, columns=[StereoseqKeys.COORD_X, StereoseqKeys.COORD_Y, StereoseqKeys.COUNT])
            df_points[StereoseqKeys.EXON] = squarebin_gef[StereoseqKeys.GENE_EXP][i][StereoseqKeys.EXON][:]

            # check that the column 'offset' is redundant with 'count'
            assert np.array_equal(df_gene["offset"], np.insert(np.cumsum(df_gene["count"]), 0, 0)[:-1])
            # unroll gene names by count such that there exists a mapping between coordinate counts and gene names
            df_points[StereoseqKeys.FEATURE_KEY] = [
                name for name, cell_count in zip(df_gene.gene, df_gene["count"]) for _ in range(cell_count)
            ]
            df_points[StereoseqKeys.FEATURE_KEY] = df_points[StereoseqKeys.FEATURE_KEY].astype("category")
            # TODO: contruct sparse table
            # TODO: contruct shapes object

            # add more gene info to var
            # df_gene = df_gene.rename(columns={"counts": "count"})
            # df_gene = df_gene.set_index(StereoseqKeys.FEATURE_KEY)
            # df_gene.index.name = None
            # df_gene = df_gene.add_suffix("_" + str(i))
            # adata.var = pd.concat([adata.var, df_gene], axis=1)
            # df_points[[StereoseqKeys.COORD_X, StereoseqKeys.COORD_Y]].drop_duplicates()
        # points = {
        #     f"transcripts_{bin}": PointsModel.parse(
        #         df,
        #         coordinates={"x": StereoseqKeys.COORD_X, "y": StereoseqKeys.COORD_Y},
        #         feature_key=StereoseqKeys.FEATURE_KEY,
        #     )
        #     for bin, df in df_by_bin.items()
        # }
    sdata = SpatialData(images=images, labels=labels, tables=tables, shapes=shapes)

    return sdata
