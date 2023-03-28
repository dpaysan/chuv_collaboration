# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.spatial as ss
from scipy.spatial.distance import squareform, pdist
from sklearn import mixture
from sklearn.neighbors import NearestNeighbors


def get_positive_cells_batch(dataset, img_names):
    postive_ids = []

    for i in range(len(img_names)):
        img_subset = dataset[dataset["image"] == img_names[i]]
        postive_ids.extend(get_positive_cells(img_subset))

    return postive_ids


def get_positive_cells(dataset, feature="int_mean", id_to_return="nuc_id"):
    dat = np.array(dataset[feature]).reshape(-1, 1)

    positive_cells = dataset[assign_cell_status(dat)][id_to_return].tolist()

    return positive_cells


def assign_cell_status(dat):
    thresh = get_two_component_threshold(dat)

    status = dat > thresh

    return status


def get_two_component_threshold(data):
    gm = mixture.GaussianMixture(n_components=2, random_state=0).fit(data)
    threshold = np.mean(gm.means_)

    return threshold


def get_cell_distances(dataset, id_col="nuc_id", range_norm=True):
    image_ids = dataset["image"].unique()
    grouped = dataset.groupby(dataset.image)
    cell_dists = []
    for i in range(len(image_ids)):
        data = grouped.get_group(image_ids[i])
        ids = np.array(data.loc[:, id_col])
        cell_dist_df = pd.DataFrame(
            squareform(pdist(data.loc[:, ["spat_centroid_y", "spat_centroid_x"]])), index=ids, columns=ids
        )
        if range_norm:
            cell_dist_df = (cell_dist_df - np.min(cell_dist_df))/(np.max(cell_dist_df) - np.min(cell_dist_df))

        cell_dists.append(cell_dist_df)
    return image_ids, cell_dists


