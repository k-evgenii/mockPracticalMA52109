# Explanation – Task 2 (`demo/cluster_plot.py`)

## 1. What was wrong with the original script, and how it was fixed
There were two mistakes:

### Mistake 1 – the number of clusters was capped at 3
```python
k = min(k, 3),
```
The loop iterates over `k = 2, 3, 4, 5`, but `min(k, 3)` turned these values into `2, 3, 3, 3`.

This produced misleading plots: the "k = 4" and "k = 5"
graphs showed only three clusters, and their metrics were identical to those for k = 3.

**Fix:** the loop variable is passed straight to `run_clustering` (`k=k`).

### Mistake 2 – the silhouette plot was never produced
```python
if "silhouette_score" in metrics_df.columns:
    ...
    plt.bar(metrics_df["k"], metrics_df["silhouette_score"])
```
`run_clustering` (in `cluster_maker/interface.py`) returns its metrics under the keys
`"inertia"` and `"silhouette"`, as shown in docstring. 

The demo looked for a column
called `"silhouette_score"`, which never exists, so silhouette bar chart was silently skipped.

**Fix:** the demo now uses the key `"silhouette"`, both in the condition and in the bar chart.

The fix was made in the demo, because `interface.py` matches documentation.

## 2. What the corrected demo script now does

Run with:
```bash
python demo/cluster_plot.py data/demo_data.csv
```

1. Checks that exactly one argument (the input CSV path) is given and that the file exists.
2. Reads the CSV and selects the first two numeric columns as features.
3. For each k in 2, 3, 4, 5, calls `run_clustering`, which standardises the features, runs the
   package's own k-means implementation (`random_state=42` for reproducibility), and computes
   the inertia and silhouette score.
4. For each k it saves, in `demo_output/`:
   - `demo_data_clustered_k{k}.csv` – the input data with an added `cluster` column;
   - `demo_data_k{k}.png` – a 2D scatter plot of the clusters with their centroids.
5. Prints the metrics for each k, and writes a summary table `demo_data_metrics.csv`
   (one row per k: `k`, `inertia`, `silhouette`).
6. Saves `demo_data_silhouette.png`, a bar chart of the silhouette score against k.

## 3. Overview of the `cluster_maker` package

`cluster_maker` is a package covering a full clustering workflow from data generation/loading data, to visulisation and exporting of clustering results.

| Module | Purpose |
|---|---|
| `dataframe_builder.py` | Creates synthetic datasets: `define_dataframe_structure` defines cluster centres, and `simulate_data` generates Gaussian points around them (with a `true_cluster` column). |
| `data_analyser.py` | Basic exploratory analysis: descriptive statistics and the correlation matrix of numeric columns. |
| `preprocessing.py` | Prepares the data for clustering: `select_features` picks the feature columns and checks that they exist and are numeric; `standardise_features` scales each feature to zero mean and unit variance, so that no feature dominates the distance calculations. |
| `algorithms.py` | Clustering algorithms: a manual k-means implementation (`init_centroids`, `assign_clusters`, `update_centroids`, `kmeans`) and `sklearn_kmeans`, a thin wrapper around scikit-learn's `KMeans` with the same interface. |
| `evaluation.py` | Measures clustering quality: inertia, silhouette score, and an elbow curve (inertia over a range of k). |
| `plotting_clustered.py` | visualises a 2D cluster scatter plot with centroids coloured to match their clusters, and an elbow plot. |
| `data_exporter.py` | Writes results to disk, as CSV or as a formatted text table. |
| `interface.py` | The high-level entry point `run_clustering`, which chains the whole pipeline and returns all results in a single dictionary. |
