import json
import time
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer, StandardScaler

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_DIR = CURRENT_DIR.parent if CURRENT_DIR.name == "evaluation" else CURRENT_DIR

EMBEDDINGS_PATH = PROJECT_DIR / "data" / "processed" / "embeddings.npy"
PAPER_IDS_PATH = PROJECT_DIR / "data" / "processed" / "paper_ids.npy"

OUTPUT_DIR = CURRENT_DIR / "eval_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42
GRID_SAMPLE_SIZE = 12000
FINAL_COMPARISON_SIZE = 12000
METRIC_SAMPLE_SIZE = 5000

KMEANS_GRID = {
    "n_clusters": [5, 8, 10, 12, 15, 20, 25, 30],
    "init": ["k-means++"],
    "n_init": [10],
    "max_iter": [300],
}

DBSCAN_GRID = {
    "eps": [0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.10],
    "min_samples": [5, 10, 20, 40],
    "metric": ["euclidean"],
}

AGGLOMERATIVE_GRID = {
    "n_clusters": [5, 8, 10, 12, 15, 20, 25, 30],
    "linkage": ["ward", "average", "complete"],
    "metric": ["euclidean", "cosine"],
}


def load_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings and paper IDs from .npy files."""
    embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False)
    paper_ids = np.load(PAPER_IDS_PATH, allow_pickle=True)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Paper IDs shape: {paper_ids.shape}")
    if embeddings.shape[0] != paper_ids.shape[0]:
        raise ValueError("embeddings.npy and paper_ids.npy row counts do not match.")
    return embeddings, paper_ids


def clean_and_preprocess(embeddings: np.ndarray, paper_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clean embeddings, standardize them, and apply L2 normalization."""
    embeddings = embeddings.astype("float32")
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    valid_mask = np.linalg.norm(embeddings, axis=1) > 0
    embeddings = embeddings[valid_mask]
    paper_ids = paper_ids[valid_mask]

    embeddings = StandardScaler().fit_transform(embeddings)
    embeddings = Normalizer(norm="l2").fit_transform(embeddings)
    return embeddings, paper_ids


def sample_rows(X: np.ndarray, n_rows: int | None) -> np.ndarray:
    """Return a sampled matrix, or all rows when n_rows is None."""
    if n_rows is None or n_rows >= X.shape[0]:
        return X
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(X.shape[0], size=n_rows, replace=False)
    return X[idx]


def make_param_grid(grid: dict) -> list[dict]:
    """Expand parameter grid into a list of dictionaries."""
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in product(*grid.values())]


def normalize(values: list, higher_is_better: bool) -> list[float]:
    """Min-max normalize values to [0, 1]."""
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    if np.all(np.isnan(arr)):
        return [0.0] * len(values)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    out = np.ones_like(arr) * 0.5 if mn == mx else (arr - mn) / (mx - mn)
    if not higher_is_better:
        out = 1 - out
    return [float(x) for x in np.nan_to_num(out, nan=0.0)]


def fit_clustering(method: str, X: np.ndarray, params: dict) -> tuple[np.ndarray, float | None]:
    """Fit one clustering model and return labels plus K-Means SSE when available."""
    if method == "K-Means":
        model = KMeans(
            n_clusters=params["n_clusters"],
            init=params.get("init", "k-means++"),
            n_init=params.get("n_init", 10),
            max_iter=params.get("max_iter", 300),
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(X)
        return labels, float(model.inertia_)

    if method == "DBSCAN":
        model = DBSCAN(
            eps=params["eps"],
            min_samples=params["min_samples"],
            metric=params.get("metric", "euclidean"),
            n_jobs=-1,
        )
        return model.fit_predict(X), None

    if method == "Agglomerative":
        model = AgglomerativeClustering(
            n_clusters=params["n_clusters"],
            linkage=params["linkage"],
            metric=params["metric"],
        )
        return model.fit_predict(X), None

    raise ValueError(f"Unknown method: {method}")


def evaluate_labels(X: np.ndarray, labels: np.ndarray) -> dict:
    """Evaluate cluster labels using silhouette, cluster count, and noise ratio."""
    labels = np.asarray(labels)
    unique_labels = set(labels)

    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_ratio = float(np.mean(labels == -1))

    result = {
        "n_clusters_found": int(n_clusters),
        "noise_ratio": noise_ratio,
        "silhouette_score": None,
        "status": "valid",
    }

    if n_clusters < 2:
        result["status"] = "invalid_less_than_2_clusters"
        return result

    valid_mask = labels != -1
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    if len(np.unique(labels_valid)) < 2:
        result["status"] = "invalid_after_removing_noise"
        return result

    if METRIC_SAMPLE_SIZE and METRIC_SAMPLE_SIZE < X_valid.shape[0]:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(X_valid.shape[0], size=METRIC_SAMPLE_SIZE, replace=False)
        X_valid = X_valid[idx]
        labels_valid = labels_valid[idx]

    try:
        result["silhouette_score"] = float(silhouette_score(X_valid, labels_valid))
    except Exception as exc:
        result["status"] = f"silhouette_failed: {exc}"

    return result


def run_grid_search(method: str, X: np.ndarray, grid: dict) -> pd.DataFrame:
    """Run grid search for one clustering method."""
    rows = []
    print(f"\nGrid search: {method}")

    for params in make_param_grid(grid):
        if method == "Agglomerative" and params["linkage"] == "ward" and params["metric"] != "euclidean":
            continue

        start = time.time()
        try:
            labels, inertia = fit_clustering(method, X, params)
            row = {
                "method": method,
                "parameters": params,
                "runtime_seconds": time.time() - start,
                "inertia_sse": inertia,
                **evaluate_labels(X, labels),
            }
        except Exception as exc:
            row = {
                "method": method,
                "parameters": params,
                "runtime_seconds": time.time() - start,
                "inertia_sse": None,
                "n_clusters_found": None,
                "noise_ratio": None,
                "silhouette_score": None,
                "status": f"failed: {exc}",
            }

        print(row)
        rows.append(row)

    return pd.DataFrame(rows)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized criteria and final ranking score."""
    df = df.copy()
    df["silhouette_norm"] = normalize(df["silhouette_score"].tolist(), True)
    df["noise_norm"] = normalize(df["noise_ratio"].tolist(), False)
    df["runtime_norm"] = normalize(df["runtime_seconds"].tolist(), False)
    df["final_score_numeric"] = (
        0.60 * df["silhouette_norm"]
        + 0.25 * df["noise_norm"]
        + 0.15 * df["runtime_norm"]
    )
    df.loc[df["status"] != "valid", "final_score_numeric"] = -1
    df["rank"] = df["final_score_numeric"].rank(ascending=False, method="dense").astype(int)
    return df.sort_values(["rank", "method", "runtime_seconds"]).reset_index(drop=True)


def best_params_per_method(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Select the best parameter set for each method."""
    valid = grid_df[grid_df["status"] == "valid"].copy()
    if valid.empty:
        raise ValueError("No valid grid-search results.")
    return valid.sort_values("final_score_numeric", ascending=False).groupby("method").head(1)


def final_comparison(best_df: pd.DataFrame, X: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Apply each best model again and build the final truth table."""
    rows = []

    for _, best in best_df.iterrows():
        start = time.time()
        method = best["method"]
        params = best["parameters"]

        try:
            labels, inertia = fit_clustering(method, X, params)
            row = {
                "method": method,
                "best_parameters_from_grid_search": params,
                "runtime_seconds": time.time() - start,
                "inertia_sse": inertia,
                **evaluate_labels(X, labels),
            }
        except Exception as exc:
            row = {
                "method": method,
                "best_parameters_from_grid_search": params,
                "runtime_seconds": time.time() - start,
                "inertia_sse": None,
                "n_clusters_found": None,
                "noise_ratio": None,
                "silhouette_score": None,
                "status": f"failed_final_fit: {exc}",
            }

        print(row)
        rows.append(row)

    df = add_scores(pd.DataFrame(rows))
    max_sil = df["silhouette_score"].max()
    min_noise = df["noise_ratio"].min()
    min_runtime = df["runtime_seconds"].min()
    max_score = df["final_score_numeric"].max()

    table = {}
    for _, row in df.iterrows():
        flags = {
            "max_silhouette_score": bool(row["silhouette_score"] == max_sil),
            "min_noise_ratio": bool(row["noise_ratio"] == min_noise),
            "min_runtime_seconds": bool(row["runtime_seconds"] == min_runtime),
            "max_final_score": bool(row["final_score_numeric"] == max_score),
        }
        table[row["method"]] = {
            "best_parameters_from_grid_search": row["best_parameters_from_grid_search"],
            "n_clusters_found": None if pd.isna(row["n_clusters_found"]) else int(row["n_clusters_found"]),
            "noise_ratio": None if pd.isna(row["noise_ratio"]) else float(row["noise_ratio"]),
            "noise_norm": float(row["noise_norm"]),
            "silhouette_score": None if pd.isna(row["silhouette_score"]) else float(row["silhouette_score"]),
            "silhouette_norm": float(row["silhouette_norm"]),
            "runtime_seconds": float(row["runtime_seconds"]),
            "runtime_norm": float(row["runtime_norm"]),
            "inertia_sse": None if pd.isna(row["inertia_sse"]) else float(row["inertia_sse"]),
            "status": row["status"],
            **flags,
            "score": f"{sum(flags.values())} / 4",
            "final_score_numeric": float(row["final_score_numeric"]),
            "is_selected_best_method": bool(row["final_score_numeric"] == max_score),
        }

    return table, df


def save_outputs(table: dict, df: pd.DataFrame) -> None:
    """Save JSON and CSV outputs."""
    with open(OUTPUT_DIR / "clustering_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    df.to_csv(OUTPUT_DIR / "clustering_eval_results.csv", index=False)


def plot_and_save(table: dict) -> None:
    """Save and display clustering plots."""
    methods = list(table.keys())
    plots = [
        ("silhouette_score", "Clustering Silhouette Score", "clustering_silhouette_score.png"),
        ("noise_ratio", "Clustering Noise Ratio", "clustering_noise_ratio.png"),
        ("runtime_seconds", "Clustering Runtime", "clustering_runtime_seconds.png"),
        ("final_score_numeric", "Clustering Final Score", "clustering_final_score.png"),
    ]

    for metric, title, filename in plots:
        values = [table[m][metric] or 0 for m in methods]
        plt.figure(figsize=(8, 5))
        plt.bar(methods, values)
        plt.title(title)
        plt.xlabel("Method")
        plt.ylabel(metric)
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
        plt.show()


def main() -> None:
    """Run the full clustering evaluation pipeline."""
    embeddings, paper_ids = load_arrays()
    X, paper_ids = clean_and_preprocess(embeddings, paper_ids)

    X_grid = sample_rows(X, GRID_SAMPLE_SIZE)
    X_final = sample_rows(X, FINAL_COMPARISON_SIZE)

    grid_df = pd.concat(
        [
            run_grid_search("K-Means", X_grid, KMEANS_GRID),
            run_grid_search("DBSCAN", X_grid, DBSCAN_GRID),
            run_grid_search("Agglomerative", X_grid, AGGLOMERATIVE_GRID),
        ],
        ignore_index=True,
    )

    grid_df = add_scores(grid_df)
    best_df = best_params_per_method(grid_df)
    table, comparison_df = final_comparison(best_df, X_final)

    print(json.dumps(table, indent=2, ensure_ascii=False))
    save_outputs(table, comparison_df)
    plot_and_save(table)


if __name__ == "__main__":
    main()
