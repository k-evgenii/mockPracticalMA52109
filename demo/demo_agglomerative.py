###
## cluster_maker: demo for agglomerative (hierarchical) clustering
## Evgenii Kondratenko - University of Bath
## November 2026
##
## This script shows the effectiveness of the agglomerative module on the
## dataset 'difficult_dataset.csv' in folder 'data/'. The raw scatter plot
## shows concentric rings: clusters that are not round, where clustering on
## the raw coordinates fails.
##
## Usage:
##     python demo/demo_agglomerative.py data/difficult_dataset.csv
##
## Method:
##  1. Explore: descriptive statistics, the raw x-y scatter plot and a
##     histogram of each point's distance from the centre (radius). The
##     histogram shows one peak per ring, separated by sparse valleys.
##  2. Choose k: agglomerative clustering (ward linkage) on the radius, with k
##     taken from the largest gap in merge distances (the dendrogram's own
##     criterion). Silhouette scores on the radius are reported alongside.
##  3. Final clustering with the chosen k, shown in the original x-y plane
##     (package plot_clusters_2d).
##  4. Comparison at the chosen k: k-means and agglomerative clustering
##     (ward, average, single) on the raw coordinates against agglomerative
##     clustering on the radius. On the raw coordinates single linkage chains
##     the rings together through the stray points between them, and the other
##     methods cut the rings into sectors. Silhouette (on the raw coordinates)
##     is printed with the caveat that it favours round clusters.
##  5. Conclusion: a printed summary of the evidence.
##
## Outputs (all in 'difficult_dataset_demo_output/'):
##  - difficult_dataset_raw.png
##        Raw data: concentric rings, before clustering.
##  - difficult_dataset_radius_hist.png
##        Distribution of the distance from the centre: one peak per ring.
##  - difficult_dataset_merge_gaps.png
##        Gap in merge distance for each k (ward on radius); the highest bar
##        is the chosen k.
##  - difficult_dataset_metrics.csv
##        Merge gap, silhouette and cluster sizes for every k.
##  - difficult_dataset_final_k{k}.png
##        Final clustering in the x-y plane: one colour per ring.
##  - difficult_dataset_clustered_final_k{k}.csv
##        The input data with its radius and cluster label.
##  - difficult_dataset_method_comparison.png
##        Raw-coordinate methods against agglomerative clustering on the
##        radius, all at the chosen k.
###

from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import (
    calculate_descriptive_statistics,
    calculate_correlation,
    select_features,
    standardise_features,
    agglomerative_clustering,
    merge_distances,
    silhouette_score_sklearn,
    plot_clusters_2d,
    export_to_csv,
    run_clustering,
)

OUTPUT_DIR = "difficult_dataset_demo_output"
K_VALUES = list(range(2, 7))
LINKAGE = "ward"
RANDOM_STATE = 42
RAW_METHODS = [
    ("sklearn_kmeans", None),
    ("agglomerative", "ward"),
    ("agglomerative", "average"),
    ("agglomerative", "single"),
]


def load_data(input_path: str) -> tuple[pd.DataFrame, List[str]]:
    """
    Load the CSV file and select its first two numeric columns as features.

    Parameters
    ----------
    input_path : str

    Returns
    -------
    df : pandas.DataFrame
    feature_cols : list of str
        Names of the two feature columns.
    """
    df = pd.read_csv(input_path)
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(numeric_cols) < 2:
        print("Error: The input CSV must have at least two numeric columns.")
        sys.exit(1)
    return df, numeric_cols[:2]


def compute_radius(X: np.ndarray) -> np.ndarray:
    """
    Distance of each point from the centre (mean) of the data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)

    Returns
    -------
    R : ndarray of shape (n_samples, 1)
        Radius as a single-feature array, ready for clustering.
    """
    centred = X - X.mean(axis=0)
    return np.sqrt((centred ** 2).sum(axis=1)).reshape(-1, 1)


def merge_gaps(X: np.ndarray, linkage: str) -> dict:
    """
    Gap in merge distance for each k in K_VALUES.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    linkage : str

    Returns
    -------
    gaps : dict
        Mapping k -> distance of the merge leaving k-1 clusters minus the
        distance of the merge leaving k clusters.
    """
    d = np.sort(merge_distances(X, linkage=linkage))
    n = len(d) + 1
    return {k: float(d[n - k] - d[n - k - 1]) for k in K_VALUES}


def method_name(linkage: Optional[str], on_radius: bool = False) -> str:
    """
    Human-readable name of a clustering method.
    """
    if on_radius:
        return f"agglomerative ({linkage}) on radius"
    return "k-means on x, y" if linkage is None else f"agglomerative ({linkage}) on x, y"


def explore_data(df: pd.DataFrame, feature_cols: List[str], R: np.ndarray, base: str) -> None:
    """
    Step 1: print descriptive statistics and correlation, and save the raw
    x-y scatter plot and the radius histogram.
    """
    print("\n=== Descriptive statistics ===")
    print(calculate_descriptive_statistics(df[feature_cols]).to_string())

    print("\n=== Correlation matrix ===")
    print(calculate_correlation(df[feature_cols]).round(3).to_string())

    X_raw = df[feature_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots()
    ax.scatter(X_raw[:, 0], X_raw[:, 1], s=8, alpha=0.7)
    ax.set_xlabel(feature_cols[0])
    ax.set_ylabel(feature_cols[1])
    ax.set_title("Raw data")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{base}_raw.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(R.ravel(), bins=40)
    ax.set_xlabel("Distance from centre (radius)")
    ax.set_ylabel("Number of points")
    ax.set_title("Radius distribution: one peak per ring")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{base}_radius_hist.png"), dpi=150)
    plt.close(fig)


def choose_k(R: np.ndarray, base: str) -> tuple[int, int, pd.DataFrame]:
    """
    Step 2: agglomerative clustering on the radius for every k in K_VALUES.

    Parameters
    ----------
    R : ndarray of shape (n_samples, 1)
        Radius of each point.
    base : str
        Prefix for output file names.

    Returns
    -------
    merge_k : int
        k with the largest merge gap.
    silhouette_k : int
        k with the highest silhouette score on the radius.
    metrics_df : pandas.DataFrame
        One row per k: k, merge_gap, silhouette, min_cluster_size, max_cluster_size.
    """
    gaps = merge_gaps(R, LINKAGE)

    rows = []
    for k in K_VALUES:
        labels, _ = agglomerative_clustering(R, k=k, linkage=LINKAGE)
        sizes = np.bincount(labels, minlength=k)
        rows.append({
            "k": k,
            "merge_gap": round(gaps[k], 4),
            "silhouette": round(silhouette_score_sklearn(R, labels), 4),
            "min_cluster_size": int(sizes.min()),
            "max_cluster_size": int(sizes.max()),
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, f"{base}_metrics.csv"), index=False)
    print(f"\n=== Agglomerative ({LINKAGE}) on radius: metrics across k ===")
    print(metrics_df.to_string(index=False))

    plt.figure()
    plt.bar(list(gaps.keys()), list(gaps.values()))
    plt.xlabel("k")
    plt.ylabel("Gap in merge distance")
    plt.title(f"Merge-distance gap for each k ({LINKAGE} linkage on radius)")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base}_merge_gaps.png"), dpi=150)
    plt.close()

    ks = metrics_df["k"].to_numpy()
    merge_k = max(gaps, key=lambda k: gaps[k])
    silhouette_k = int(ks[np.argmax(metrics_df["silhouette"].to_numpy())])
    return merge_k, silhouette_k, metrics_df


def final_clustering(
    df: pd.DataFrame,
    feature_cols: List[str],
    R: np.ndarray,
    k: int,
    base: str,
) -> np.ndarray:
    """
    Step 3: agglomerative clustering on the radius with the chosen k. Saves
    the clusters in the x-y plane (package plot_clusters_2d) and the labelled CSV.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    """
    labels, _ = agglomerative_clustering(R, k=k, linkage=LINKAGE)

    X_raw = df[feature_cols].to_numpy(dtype=float)
    fig, ax = plot_clusters_2d(
        X_raw,
        labels,
        title=f"Agglomerative ({LINKAGE}) on radius, k = {k}",
    )
    ax.set_xlabel(feature_cols[0])
    ax.set_ylabel(feature_cols[1])
    ax.set_aspect("equal")
    fig.savefig(os.path.join(OUTPUT_DIR, f"{base}_final_k{k}.png"), dpi=150)
    plt.close(fig)

    out = df.copy()
    out["radius"] = R.ravel()
    out["cluster"] = labels
    export_to_csv(out, os.path.join(OUTPUT_DIR, f"{base}_clustered_final_k{k}.csv"))

    print(f"\n=== Final clustering (agglomerative {LINKAGE} on radius, k = {k}) ===")
    print(f"Cluster sizes: {np.bincount(labels, minlength=k).tolist()}")
    return labels


def compare_methods(
    input_path: str,
    feature_cols: List[str],
    X_raw: np.ndarray,
    X_std: np.ndarray,
    radius_labels: np.ndarray,
    k: int,
    base: str,
) -> pd.DataFrame:
    """
    Step 4: at the chosen k, cluster the raw coordinates with every method in
    RAW_METHODS (via run_clustering) and plot them next to the radius result.

    Returns
    -------
    summary : pandas.DataFrame
        One row per method: method, silhouette (on standardised x, y),
        largest_cluster_share.
    """
    panels = []
    for algorithm, linkage in RAW_METHODS:
        result = run_clustering(
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm=algorithm,
            k=k,
            standardise=True,
            random_state=RANDOM_STATE,
            linkage=linkage or "single",
        )
        plt.close(result["fig_cluster"])
        panels.append((method_name(linkage), result["labels"]))
    panels.append((method_name(LINKAGE, on_radius=True), radius_labels))

    rows = []
    for name, labels in panels:
        rows.append({
            "method": name,
            "silhouette": round(silhouette_score_sklearn(X_std, labels), 4),
            "largest_cluster_share": round(np.bincount(labels).max() / len(labels), 2),
        })

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()
    for ax, (name, labels) in zip(axes, panels):
        ax.scatter(X_raw[:, 0], X_raw[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel(feature_cols[0])
        ax.set_ylabel(feature_cols[1])
        ax.set_aspect("equal")
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f"All methods at k = {k}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{base}_method_comparison.png"), dpi=150)
    plt.close(fig)

    summary = pd.DataFrame(rows)
    print(f"\n=== Method comparison at k = {k} ===")
    print(summary.to_string(index=False))
    print("Note: silhouette favours round, compact clusters, so on the raw coordinates")
    print("it can rate methods that cut the rings into sectors above the method that")
    print("recovers them. Judge the methods on the comparison plot.")
    return summary


def main(args: List[str]) -> None:
    """
    Run the full demo on the CSV file given on the command line.
    """
    if len(args) != 1:
        print("Usage: python demo/demo_agglomerative.py <input_csv>")
        sys.exit(1)

    input_path = args[0]
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]

    df, feature_cols = load_data(input_path)
    print(f"Loaded {len(df)} rows with features: {feature_cols}")
    X_raw = df[feature_cols].to_numpy(dtype=float)
    X_std = standardise_features(select_features(df, feature_cols).to_numpy(dtype=float))
    R = compute_radius(X_raw)

    # Step 1: explore
    explore_data(df, feature_cols, R, base)

    # Step 2: choose k
    merge_k, silhouette_k, _ = choose_k(R, base)
    chosen_k = merge_k
    print(f"\nLargest merge gap at k = {merge_k}")
    print(f"Highest silhouette (on radius) at k = {silhouette_k}")
    print(f"Chosen k = {chosen_k} (largest merge gap)")

    # Step 3: final clustering
    labels = final_clustering(df, feature_cols, R, chosen_k, base)

    # Step 4: comparison
    summary = compare_methods(input_path, feature_cols, X_raw, X_std, labels, chosen_k, base)

    # Step 5: conclusion
    single_share = float(summary.loc[summary["method"] == method_name("single"), "largest_cluster_share"].iloc[0])
    print("\n=== Conclusion ===")
    print(f"Agglomerative ({LINKAGE}) clustering on the radius finds k = {chosen_k} rings")
    print(f"  (largest merge gap; silhouette on radius best at k = {silhouette_k}"
          f"{', agrees' if silhouette_k == chosen_k else ', differs'})")
    print(f"  cluster sizes: {np.bincount(labels, minlength=chosen_k).tolist()}")
    print(f"On the raw coordinates, single linkage puts {single_share:.0%} of the points in one")
    print("cluster (it chains the rings together through the stray points between them),")
    print("and k-means, ward and average linkage cut the rings into sectors.")
    print("Clustering the radius, chosen after seeing the rings in the raw plot, lets the")
    print("agglomerative module recover each ring as one cluster.")

    print("\nDemo completed.")
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
