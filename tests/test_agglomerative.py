###
## cluster_maker - tests for agglomerative.py
## Evgenii Kondratenko - University of Bath
## November 2026
###

import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cluster_maker import (
    agglomerative_clustering,
    merge_distances,
    run_clustering,
)


def make_two_rings(n_per_ring: int = 100, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """
    Two concentric rings (radius 1 and 3): non-round clusters where k-means fails.
    The first n_per_ring rows are the inner ring, the rest the outer ring.
    """
    rng = np.random.RandomState(seed)
    rings = []
    for radius in (1.0, 3.0):
        angles = rng.uniform(0, 2 * np.pi, n_per_ring)
        ring = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        rings.append(ring + rng.normal(scale=noise, size=ring.shape))
    return np.vstack(rings)


class TestAgglomerative(unittest.TestCase):
    def setUp(self):
        self.n = 100
        self.X = make_two_rings(self.n)

    # Detects: wrong output shapes or centroids that are not the cluster means,
    # which would break compute_inertia and plot_clusters_2d downstream.
    def test_output_shapes_and_centroids(self):
        labels, centroids = agglomerative_clustering(self.X, k=2, linkage="single")

        self.assertEqual(labels.shape, (self.X.shape[0],))
        self.assertEqual(centroids.shape, (2, 2))
        for c in range(2):
            self.assertTrue(np.allclose(centroids[c], self.X[labels == c].mean(axis=0)))

    # Detects: the module failing on non-round clusters (its whole purpose).
    # Single linkage must put each ring entirely in its own cluster.
    def test_single_linkage_recovers_rings(self):
        labels, _ = agglomerative_clustering(self.X, k=2, linkage="single")

        inner, outer = labels[: self.n], labels[self.n:]
        self.assertEqual(len(np.unique(inner)), 1)
        self.assertEqual(len(np.unique(outer)), 1)
        self.assertNotEqual(inner[0], outer[0])

    # Detects: merge distances of the wrong length, or a gap criterion that does
    # not point to the true number of groups (k = 2 rings).
    def test_merge_distances_largest_gap_at_true_k(self):
        d = np.sort(merge_distances(self.X, linkage="single"))
        n = len(d) + 1
        self.assertEqual(len(d), self.X.shape[0] - 1)

        gaps = {k: d[n - k] - d[n - k - 1] for k in range(2, 6)}
        self.assertEqual(max(gaps, key=gaps.get), 2)

    # Detects: invalid input being accepted silently instead of raising the same
    # error types as the rest of the package.
    def test_invalid_input_raises(self):
        with self.assertRaises(TypeError):
            agglomerative_clustering(self.X.tolist(), k=2)
        with self.assertRaises(ValueError):
            agglomerative_clustering(self.X, k=0)
        with self.assertRaises(ValueError):
            agglomerative_clustering(self.X, k=self.X.shape[0] + 1)
        with self.assertRaises(ValueError):
            agglomerative_clustering(self.X, k=2, linkage="median")

    # Detects: broken integration with the high-level pipeline (run_clustering).
    def test_run_clustering_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rings.csv")
            pd.DataFrame(self.X, columns=["x", "y"]).to_csv(path, index=False)

            result = run_clustering(
                input_path=path,
                feature_cols=["x", "y"],
                algorithm="agglomerative",
                k=2,
                linkage="single",
            )
            plt.close(result["fig_cluster"])

        self.assertEqual(result["labels"].shape, (self.X.shape[0],))
        self.assertEqual(result["centroids"].shape, (2, 2))
        self.assertIn("cluster", result["data"].columns)


if __name__ == "__main__":
    unittest.main()
