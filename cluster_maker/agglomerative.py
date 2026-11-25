###
## cluster_maker
## Evgenii Kondratenko - University of Bath
## November 2026
###

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering

VALID_LINKAGES = ("ward", "complete", "average", "single")


def _check_input(X: np.ndarray, linkage: str) -> None:
    """
    Validate the inputs shared by the agglomerative functions.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if linkage not in VALID_LINKAGES:
        raise ValueError(f"Unknown linkage '{linkage}'. Use one of {VALID_LINKAGES}.")


def agglomerative_clustering(
    X: np.ndarray,
    k: int,
    linkage: str = "single",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hierarchical (agglomerative) clustering using scikit-learn's
    AgglomerativeClustering.

    Unlike k-means, agglomerative clustering makes no assumption that clusters
    are round: with single linkage it can recover elongated or curved shapes.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : int
        Number of clusters.
    linkage : {"ward", "complete", "average", "single"}, default "single"
        Criterion used to merge clusters.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
        Mean of the points in each cluster (the algorithm has no centroids of
        its own; these keep the same interface as kmeans and sklearn_kmeans).
    """
    _check_input(X, linkage)
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if k > X.shape[0]:
        raise ValueError("k cannot be larger than the number of samples.")

    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X)
    centroids = np.vstack([X[labels == c].mean(axis=0) for c in range(k)])
    return labels, centroids


def merge_distances(
    X: np.ndarray,
    linkage: str = "single",
) -> np.ndarray:
    """
    Distances at which clusters are merged when the full hierarchy is built.

    A large jump between consecutive merge distances means two well-separated
    groups were joined, so the number of clusters just before that jump is a
    natural choice of k (equivalent to cutting the dendrogram at its largest gap).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    linkage : {"ward", "complete", "average", "single"}, default "single"

    Returns
    -------
    distances : ndarray of shape (n_samples - 1,)
        Merge distances in merge order: distances[i] is the distance of the
        merge that leaves n_samples - i - 1 clusters.
    """
    _check_input(X, linkage)
    if X.shape[0] < 2:
        raise ValueError("At least two samples are required.")

    model = AgglomerativeClustering(n_clusters=1, linkage=linkage, compute_distances=True)
    model.fit(X)
    return model.distances_
