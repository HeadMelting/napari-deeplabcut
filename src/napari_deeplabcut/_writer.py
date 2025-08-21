import os
from itertools import groupby
from pathlib import Path

import pandas as pd
import yaml
import numpy as np
from napari.layers import Shapes
from napari_builtins.io import napari_write_shapes
from skimage.io import imsave
from skimage.util import img_as_ubyte

from napari_deeplabcut import misc
from napari_deeplabcut._reader import _load_config


def _write_config(config_path: str, params: dict):
    with open(config_path, "w") as file:
        yaml.safe_dump(params, file)


def _form_df(points_data, metadata):
    temp = pd.DataFrame(points_data[:, -1:0:-1], columns=["x", "y"])
    properties = metadata["properties"]
    meta = metadata["metadata"]
    temp["bodyparts"] = properties["label"]
    temp["individuals"] = properties["id"]
    temp["inds"] = points_data[:, 0].astype(int)
    # Attach visibility if present, default to 2 (visible)
    visibility = properties.get("visibility")
    if visibility is None:
        visibility = np.full(len(temp), 2, dtype=int)
    temp["visibility"] = visibility
    temp["scorer"] = meta["header"].scorer
    df = temp.set_index(["scorer", "individuals", "bodyparts", "inds"]).stack()
    df.index.set_names("coords", level=-1, inplace=True)
    df = df.unstack(["scorer", "individuals", "bodyparts", "coords"])
    df.index.name = None
    if not properties["id"][0]:
        df = df.droplevel("individuals", axis=1)
    # Expand target columns to include any new coords (e.g., visibility),
    # while respecting whether the header has an 'individuals' level.
    header = meta["header"]
    header_coords = header.coords or ["x", "y"]
    df_coords = list(pd.unique(df.columns.get_level_values("coords")))
    coords_all = list(dict.fromkeys(list(header_coords) + df_coords))
    if "individuals" in header.columns.names:
        target_columns = pd.MultiIndex.from_product(
            [[header.scorer], header.individuals, header.bodyparts, coords_all],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
    else:
        target_columns = pd.MultiIndex.from_product(
            [[header.scorer], header.bodyparts, coords_all],
            names=["scorer", "bodyparts", "coords"],
        )
    df = df.reindex(target_columns, axis=1)
    # Fill unannotated rows with NaNs
    # df = df.reindex(range(len(meta['paths'])))
    # df.index = meta['paths']
    if meta["paths"]:
        df.index = [meta["paths"][i] for i in df.index]
    misc.guarantee_multiindex_rows(df)
    return df


def write_hdf(filename, data, metadata):
    file, _ = os.path.splitext(filename)  # FIXME Unused currently
    df = _form_df(data, metadata)
    meta = metadata["metadata"]
    name = metadata["name"]
    root = meta["root"]
    if "machine" in name:  # We are attempting to save refined model predictions
        header = misc.DLCHeader(df.columns)
        gt_file = ""
        for file in os.listdir(root):
            if file.startswith("CollectedData") and file.endswith("h5"):
                gt_file = file
                break
        if gt_file:  # Refined predictions must be merged into the existing data
            df_gt = pd.read_hdf(os.path.join(root, gt_file))
            new_scorer = df_gt.columns.get_level_values("scorer")[0]
            header.scorer = new_scorer
            df.columns = header.columns
            df = pd.concat((df, df_gt))
            df = df[~df.index.duplicated(keep="first")]
            name = os.path.splitext(gt_file)[0]
        else:
            # Let us fetch the config.yaml file to get the scorer name...
            project_folder = Path(root).parents[1]
            config = _load_config(str(project_folder / "config.yaml"))
            new_scorer = config["scorer"]
            header.scorer = new_scorer
            df.columns = header.columns
            name = f"CollectedData_{new_scorer}"
    df.sort_index(inplace=True)
    # Drop rows that have no annotations for all coords (avoid empty CSV)
    # Keep a row if any of x, y, or visibility is not NaN for any keypoint
    cols_coords = df.columns.get_level_values("coords")
    keep_mask = (
        df.loc[:, cols_coords == "x"].notna().any(axis=1)
        | df.loc[:, cols_coords == "y"].notna().any(axis=1)
        | df.loc[:, cols_coords == "visibility"].notna().any(axis=1)
    )
    if keep_mask.any():
        df = df.loc[keep_mask]
    filename = name + ".h5"
    path = os.path.join(root, filename)
    df.to_hdf(path, key="keypoints", mode="w")
    df.to_csv(path.replace(".h5", ".csv"))
    return filename


def _write_image(data, output_path, plugin=None):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imsave(
        output_path,
        img_as_ubyte(data).squeeze(),
        plugin=plugin,
        check_contrast=False,
    )


def write_masks(foldername, data, metadata):
    folder, _ = os.path.splitext(foldername)
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, "{}_obj_{}.png")
    shapes = Shapes(data, shape_type="polygon")
    meta = metadata["metadata"]
    frame_inds = [int(array[0, 0]) for array in data]
    shape_inds = []
    for _, group in groupby(frame_inds):
        shape_inds += range(sum(1 for _ in group))
    masks = shapes.to_masks(mask_shape=meta["shape"][1:])
    for n, mask in enumerate(masks):
        image_name = os.path.basename(meta["paths"][frame_inds[n]])
        output_path = filename.format(os.path.splitext(image_name)[0], shape_inds[n])
        _write_image(mask, output_path)
    napari_write_shapes(os.path.join(folder, "vertices.csv"), data, metadata)
    return folder
