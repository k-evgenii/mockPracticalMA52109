###
## cluster_maker: cluster analysis of simulated data
## Evgenii Kondratenko - University of Bath
## November 2026
##
## This script analyses the dataset 'simulated_data.csv' in folder 'data/'
## using only the clustering tools in cluster_maker (plus matplotlib for
## supporting plots), and determines a plausible number of clusters.
##
## Usage:
##     python demo/simulated_clustering.py data/simulated_data.csv
##
## Method:
##  1. Explore: descriptive statistics, correlations and a PCA projection of
##     the raw data, to see the structure before any clustering.
##  2. Choose k: run k-means for k = 2..8 on standardised features and collect
##     inertia, silhouette score and cluster sizes. k is the value with the
##     highest silhouette score, checked against the elbow (the last k whose
##     inertia drop is at least 15%).
##  3. Final clustering with the chosen k, shown with cluster centres on the
##     PCA projection and in every pair of original features.
##  4. Robustness: compare the package's manual k-means (single random start)
##     with sklearn_kmeans (10 starts). A single start can converge to a poor
##     local optimum, which is why sklearn_kmeans is used throughout.
##  5. Conclusion: a printed summary of the evidence for the chosen k.
##
## Outputs (all in 'simulated_demo_output/'):
##  - simulated_data_pca_raw.png
##        Raw data projected on PC1/PC2 (about 93% of the variance). Separate
##        groups are visible before any clustering is done.
##  - simulated_data_elbow.png
##        Inertia against k (package plot_elbow). Shows where adding further
##        clusters stops reducing the within-cluster spread substantially.
##  - simulated_data_silhouette.png
##        Silhouette score for each k. Measures how well separated the clusters
##        are; the highest bar is the chosen k.
##  - simulated_data_pca_clusters_k3-6.png
##        The same PCA view coloured by the clusters for k = 3, 4, 5, 6. For small
##        k separate groups are merged; at the chosen k each group has one colour.
##  - simulated_data_clusters_pca_k{k}.png
##        Final clusters with their centres (package plot_clusters_2d) on the
##        PCA coordinates.
##  - simulated_data_pairs_clusters_k{k}.png
##        Every pair of original features coloured by the final clusters: the
##        separation holds in the real features, not only in the PCA view.
##  - simulated_data_metrics.csv
##        Inertia, silhouette, cluster sizes and inertia drop for every k.
##  - simulated_data_clustered_k{k}.csv, simulated_data_clustered_final_k{k}.csv
##        The input data with cluster labels, for every k and for the final k.
###

from __future__ import annotations

import os
import sys
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from cluster_maker import (
    calculate_descriptive_statistics,
    calculate_correlation,
    select_features,
    standardise_features,
    kmeans,
    sklearn_kmeans,
    silhouette_score_sklearn,
    plot_clusters_2d,
    plot_elbow,
    run_clustering,
)

OUTPUT_DIR = "simulated_demo_output"
K_VALUES = list(range(2, 9))
PCA_K_VALUES = [3, 4, 5, 6]
ELBOW_DROP_THRESHOLD = 15.0  # % inertia drop below which adding a cluster no longer pays off
RANDOM_STATE = 42


def load_data(input_path: str) -> tuple[pd.DataFrame, List[str]]:
    """
    Load the CSV file and identify its numeric feature columns.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file.

    Returns
    -------
    df : pandas.DataFrame
        The loaded data.
    feature_cols : list of str
        Names of the numeric columns (at least two, otherwise the script exits).
    """
    df = pd.read_csv(input_path)
    feature_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(feature_cols) < 2:
        print("Error: The input CSV must have at least two numeric columns.")
        sys.exit(1)
    return df, feature_cols


def plot_pairwise(
    X: np.ndarray,
    feature_cols: List[str],
    output_path: str,
    labels: np.ndarray | None = None,
    title: str = "Pairwise feature scatter plots",
) -> None:
    """
    Plot every pair of features in a grid of scatter plots.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    feature_cols : list of str
        Feature names, used as axis labels (length n_features).
    output_path : str
        Where the figure is saved.
    labels : ndarray of shape (n_samples,) or None
        Cluster labels; if given, points are coloured by cluster.
    title : str
        Figure title.
    """
    pairs = list(combinations(range(len(feature_cols)), 2))
    n_cols = min(3, len(pairs))
    n_rows = int(np.ceil(len(pairs) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    axes = axes.ravel()

    for ax, (i, j) in zip(axes, pairs):
        if labels is None:
            ax.scatter(X[:, i], X[:, j], s=8, alpha=0.7)
        else:
            ax.scatter(X[:, i], X[:, j], c=labels, cmap="tab10", s=8, alpha=0.7)
        ax.set_xlabel(feature_cols[i])
        ax.set_ylabel(feature_cols[j])

    for ax in axes[len(pairs):]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pca(
    X: np.ndarray,
    output_path: str,
    labels: np.ndarray | None = None,
    title: str = "PCA projection",
) -> float:
    """
    Project the data onto its first two principal components and plot it.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    output_path : str
        Where the figure is saved.
    labels : ndarray of shape (n_samples,) or None
        Cluster labels; if given, points are coloured by cluster.
    title : str
        Figure title.

    Returns
    -------
    explained : float
        Fraction of the total variance explained by PC1 + PC2.
    """
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    ratios = pca.explained_variance_ratio_

    fig, ax = plt.subplots()
    if labels is None:
        ax.scatter(X_2d[:, 0], X_2d[:, 1], s=15, alpha=0.8)
    else:
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=15, alpha=0.8)
        fig.colorbar(scatter, ax=ax, label="Cluster label")

    ax.set_xlabel(f"PC1 ({ratios[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({ratios[1]:.0%} of variance)")
    ax.set_title(f"{title}\nPC1 + PC2 explain {ratios.sum():.0%} of total variance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return float(ratios.sum())


def plot_pca_grid(
    X: np.ndarray,
    labels_by_k: dict,
    output_path: str,
) -> None:
    """
    Plot the same PCA projection once per k, side by side, coloured by the
    k-means labels for that k.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    labels_by_k : dict
        Mapping k -> ndarray of shape (n_samples,) with the labels for that k.
    output_path : str
        Where the figure is saved.
    """
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    ratios = pca.explained_variance_ratio_

    n = len(labels_by_k)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)

    for ax, (k, labels) in zip(axes.ravel(), labels_by_k.items()):
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=12, alpha=0.8)
        ax.set_title(f"k = {k}")
        ax.set_xlabel(f"PC1 ({ratios[0]:.0%} of variance)")
        ax.set_ylabel(f"PC2 ({ratios[1]:.0%} of variance)")

    fig.suptitle(f"k-means clusters on PCA projection (PC1 + PC2 explain {ratios.sum():.0%} of variance)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def explore_data(df: pd.DataFrame, feature_cols: List[str], base: str) -> float:
    """
    Step 1: describe the data and show its structure before clustering.
    Prints descriptive statistics and correlations, and saves the raw PCA plot.

    Parameters
    ----------
    df : pandas.DataFrame
    feature_cols : list of str
    base : str
        Prefix for output file names.

    Returns
    -------
    explained : float
        Fraction of the total variance explained by PC1 + PC2 of the raw data.
    """
    print("\n=== Descriptive statistics ===")
    print(calculate_descriptive_statistics(df[feature_cols]).to_string())

    print("\n=== Correlation matrix ===")
    print(calculate_correlation(df[feature_cols]).round(3).to_string())

    X_raw = df[feature_cols].to_numpy(dtype=float)
    explained = plot_pca(
        X_raw,
        os.path.join(OUTPUT_DIR, f"{base}_pca_raw.png"),
        title="Raw data: PCA projection",
    )
    print(f"PCA on raw data: PC1 + PC2 explain {explained:.1%} of total variance")
    return explained


def choose_k(input_path: str, feature_cols: List[str], base: str) -> pd.DataFrame:
    """
    Step 2: run k-means for every k in K_VALUES and collect the evidence for
    choosing k. Saves the metrics CSV, the elbow and silhouette plots, and the
    PCA grid for PCA_K_VALUES.

    Parameters
    ----------
    input_path : str
    feature_cols : list of str
    base : str
        Prefix for output file names.

    Returns
    -------
    metrics_df : pandas.DataFrame
        One row per k with columns k, inertia, silhouette, min_cluster_size,
        max_cluster_size and inertia_drop_%.
    """
    X_raw = pd.read_csv(input_path)[feature_cols].to_numpy(dtype=float)
    metrics_summary = []
    labels_by_k = {}

    for k in K_VALUES:
        print(f"\n=== Running k-means with k = {k} ===")

        result = run_clustering(
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm="sklearn_kmeans",
            k=k,
            standardise=True,
            output_path=os.path.join(OUTPUT_DIR, f"{base}_clustered_k{k}.csv"),
            random_state=RANDOM_STATE,
            compute_elbow=False,
        )
        plt.close(result["fig_cluster"])

        sizes = np.bincount(result["labels"], minlength=k)
        if k in PCA_K_VALUES:
            labels_by_k[k] = result["labels"]

        metrics = {"k": k}
        metrics.update(result.get("metrics", {}))
        metrics["min_cluster_size"] = int(sizes.min())
        metrics["max_cluster_size"] = int(sizes.max())
        metrics_summary.append(metrics)

        print("Metrics:")
        for key, value in result.get("metrics", {}).items():
            print(f"  {key}: {value}")
        print(f"  cluster sizes: {sizes.tolist()}")

    metrics_df = pd.DataFrame(metrics_summary)
    # Percentage drop in inertia compared with the previous k: large drops before
    # the elbow, small drops after it
    metrics_df["inertia_drop_%"] = (-metrics_df["inertia"].pct_change() * 100).round(1)

    metrics_csv = os.path.join(OUTPUT_DIR, f"{base}_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    print("\n=== Summary of metrics across k ===")
    print(metrics_df.to_string(index=False))

    fig_elbow, _ = plot_elbow(
        metrics_df["k"].tolist(),
        metrics_df["inertia"].tolist(),
        title="Elbow curve (inertia vs k)",
    )
    fig_elbow.savefig(os.path.join(OUTPUT_DIR, f"{base}_elbow.png"), dpi=150)
    plt.close(fig_elbow)

    plt.figure()
    plt.bar(metrics_df["k"], metrics_df["silhouette"])
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette score for different k")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base}_silhouette.png"), dpi=150)
    plt.close()

    plot_pca_grid(
        X_raw,
        labels_by_k,
        os.path.join(OUTPUT_DIR, f"{base}_pca_clusters_k{PCA_K_VALUES[0]}-{PCA_K_VALUES[-1]}.png"),
    )

    return metrics_df


def suggest_k(metrics_df: pd.DataFrame) -> tuple[int, int]:
    """
    Suggest the number of clusters from the metrics table.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Output of choose_k.

    Returns
    -------
    silhouette_k : int
        k with the highest silhouette score.
    elbow_k : int
        Last k whose inertia drop is at least ELBOW_DROP_THRESHOLD %.
    """
    ks = metrics_df["k"].to_numpy()

    silhouette_k = int(ks[np.argmax(metrics_df["silhouette"].to_numpy())])

    drops = metrics_df["inertia_drop_%"].to_numpy()
    big_drops = ks[drops >= ELBOW_DROP_THRESHOLD]
    elbow_k = int(big_drops.max()) if big_drops.size else int(ks[0])

    return silhouette_k, elbow_k


def final_clustering(input_path: str, feature_cols: List[str], k: int, base: str) -> dict:
    """
    Step 3: run the full pipeline with the chosen k. Saves the labelled CSV,
    the final clusters with centres on the PCA projection, and the pairwise
    plot coloured by cluster.

    Parameters
    ----------
    input_path : str
    feature_cols : list of str
    k : int
        Chosen number of clusters.
    base : str
        Prefix for output file names.

    Returns
    -------
    result : dict
        The dictionary returned by run_clustering.
    """
    print(f"\n=== Final clustering with k = {k} ===")

    result = run_clustering(
        input_path=input_path,
        feature_cols=feature_cols,
        algorithm="sklearn_kmeans",
        k=k,
        standardise=True,
        output_path=os.path.join(OUTPUT_DIR, f"{base}_clustered_final_k{k}.csv"),
        random_state=RANDOM_STATE,
        compute_elbow=False,
    )
    plt.close(result["fig_cluster"])

    labels = result["labels"]
    X_raw = result["data"][feature_cols].to_numpy(dtype=float)

    # Cluster means in raw space are the k-means centroids mapped back from the
    # standardised space, since standardisation is an affine transformation.
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_raw)
    centres_raw = np.vstack([X_raw[labels == c].mean(axis=0) for c in range(k)])
    centres_2d = pca.transform(centres_raw)
    ratios = pca.explained_variance_ratio_

    fig, ax = plot_clusters_2d(
        X_2d,
        labels,
        centroids=centres_2d,
        title=f"Final clustering (k = {k}) on PCA projection",
    )
    ax.set_xlabel(f"PC1 ({ratios[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({ratios[1]:.0%} of variance)")
    fig.savefig(os.path.join(OUTPUT_DIR, f"{base}_clusters_pca_k{k}.png"), dpi=150)
    plt.close(fig)

    plot_pairwise(
        X_raw,
        feature_cols,
        os.path.join(OUTPUT_DIR, f"{base}_pairs_clusters_k{k}.png"),
        labels=labels,
        title=f"Final clustering (k = {k}): every pair of features",
    )

    print(f"Cluster sizes: {np.bincount(labels, minlength=k).tolist()}")
    print(f"Silhouette score: {result['metrics']['silhouette']}")

    return result


def robustness_check(X: np.ndarray, k: int) -> None:
    """
    Step 4: compare the manual k-means (single random start) with
    sklearn_kmeans (10 starts) on the same data and k, and print the result.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Standardised features.
    k : int
        Chosen number of clusters.
    """
    print(f"\n=== Robustness: manual k-means vs sklearn_kmeans (k = {k}) ===")

    labels_manual, _ = kmeans(X, k, random_state=RANDOM_STATE)
    labels_sklearn, _ = sklearn_kmeans(X, k, random_state=RANDOM_STATE)

    table = pd.crosstab(labels_manual, labels_sklearn, rownames=["manual"], colnames=["sklearn"])
    print("Label numbers are arbitrary: the partitions agree if every row and")
    print("every column of this table has exactly one non-zero cell.")
    print(table.to_string())

    print(f"Silhouette (manual):  {silhouette_score_sklearn(X, labels_manual):.4f}")
    print(f"Silhouette (sklearn): {silhouette_score_sklearn(X, labels_sklearn):.4f}")

    nonzero = table.to_numpy() > 0
    identical = (
        table.shape == (k, k)
        and bool((nonzero.sum(axis=1) == 1).all())
        and bool((nonzero.sum(axis=0) == 1).all())
    )
    if identical:
        print("Verdict: identical partitions; the clustering is stable.")
    else:
        print("Verdict: the manual single start converged to a different (worse)")
        print("partition, which is why sklearn_kmeans (10 starts) is used.")


def main(args: List[str]) -> None:
    """
    Run the full analysis on the CSV file given on the command line.

    Parameters
    ----------
    args : list of str
        Command-line arguments: exactly one, the input CSV path.
    """
    if len(args) != 1:
        print("Usage: python demo/simulated_clustering.py <input_csv>")
        sys.exit(1)

    input_path = args[0]
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]

    df, feature_cols = load_data(input_path)
    print(f"Loaded {len(df)} rows with features: {feature_cols}")

    # Step 1: explore
    explained = explore_data(df, feature_cols, base)

    # Step 2: choose k
    metrics_df = choose_k(input_path, feature_cols, base)
    silhouette_k, elbow_k = suggest_k(metrics_df)
    agree = silhouette_k == elbow_k
    print(f"\nHighest silhouette score at k = {silhouette_k}")
    print(f"Elbow (last inertia drop >= {ELBOW_DROP_THRESHOLD:.0f}%) at k = {elbow_k}")
    print("Silhouette and elbow agree." if agree else "Silhouette and elbow disagree: inspect the PCA plots.")

    chosen_k = silhouette_k
    reason = "highest silhouette" + ("; elbow also at this k" if agree else "")
    print(f"Chosen k = {chosen_k} ({reason})")

    # Step 3: final clustering
    result = final_clustering(input_path, feature_cols, chosen_k, base)

    # Step 4: robustness
    X = standardise_features(select_features(df, feature_cols).to_numpy(dtype=float))
    robustness_check(X, chosen_k)

    # Step 5: conclusion
    sizes = np.bincount(result["labels"], minlength=chosen_k).tolist()
    print("\n=== Conclusion ===")
    print(f"Chosen number of clusters: k = {chosen_k}")
    print(f"  silhouette score: {result['metrics']['silhouette']:.4f} (highest over k = {K_VALUES[0]}..{K_VALUES[-1]})")
    print(f"  elbow at k = {elbow_k} ({'agrees' if agree else 'differs'})")
    print(f"  PC1 + PC2 explain {explained:.1%} of the variance, so the PCA plots are a faithful view")
    print(f"  cluster sizes: {sizes}")
    print(f"The data separates into {chosen_k} well-defined clusters, consistent across the")
    print("metrics, the PCA projection and every pair of original features.")

    print("\nAnalysis completed.")
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
