import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError as exc:
    raise ImportError("Install UMAP first: pip install umap-learn") from exc

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_DIR = CURRENT_DIR.parent if CURRENT_DIR.name == "evaluation" else CURRENT_DIR

EMBEDDINGS_PATH = PROJECT_DIR / "data" / "processed" / "embeddings.npy"
PAPER_IDS_PATH = PROJECT_DIR / "data" / "processed" / "paper_ids.npy"

OUTPUT_DIR = CURRENT_DIR / "eval_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
EVALUATION_SAMPLE_SIZE = 12000

N_COMPONENTS_LIST = [2, 5, 10, 25, 50]

UMAP_PARAMS = {
    "n_neighbors": 15,
    "min_dist": 0.0,
    "metric": "euclidean",
}


PREFER_SVD_IF_WITHIN_RELATIVE_MARGIN = 0.01




def load_arrays(embeddings_path: Path, paper_ids_path: Path):
    """Load embeddings and paper IDs from .npy files."""
    embeddings = np.load(embeddings_path, allow_pickle=False)
    paper_ids = np.load(paper_ids_path, allow_pickle=True)

    print("=" * 90)
    print("Loaded input files")
    print("=" * 90)
    print(f"Embeddings path: {embeddings_path}")
    print(f"Paper IDs path:   {paper_ids_path}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    print(f"Paper IDs shape:  {paper_ids.shape}")

    if embeddings.shape[0] != paper_ids.shape[0]:
        raise ValueError("embeddings.npy and paper_ids.npy have different row counts.")

    return embeddings, paper_ids


def clean_embeddings(embeddings, paper_ids):
    """Replace invalid values and remove zero-vector rows."""
    embeddings = embeddings.astype("float32")
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    valid_mask = np.linalg.norm(embeddings, axis=1) > 0
    print(f"Removed zero-vector rows: {int(np.sum(~valid_mask))}")

    return embeddings[valid_mask], paper_ids[valid_mask]


def preprocess_embeddings(embeddings):
    """Standardize embeddings before dimensionality reduction."""
    scaled = StandardScaler().fit_transform(embeddings)
    print(f"Scaled embeddings shape: {scaled.shape}")
    return scaled


def sample_rows(X, paper_ids, sample_size):
    """Return a random sample or all rows if sample_size is None."""
    if sample_size is None or sample_size >= X.shape[0]:
        idx = np.arange(X.shape[0])
    else:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)

    return X[idx], paper_ids[idx]




def normalize_group(values, higher_is_better=True):
    """Normalize values inside one n_components group."""
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)

    if np.all(np.isnan(arr)):
        return [0.0 for _ in values]

    mn, mx = np.nanmin(arr), np.nanmax(arr)
    norm = np.ones_like(arr) * 0.5 if mn == mx else (arr - mn) / (mx - mn)

    if not higher_is_better:
        norm = 1.0 - norm

    return [float(x) for x in np.nan_to_num(norm, nan=0.0)]


def add_group_scores(group_df):
    """Add normalized metrics and compression score inside one component group."""
    group_df = group_df.copy()

    group_df["explained_variance_norm"] = normalize_group(
        group_df["cumulative_explained_variance"].tolist(), True
    )
    group_df["reconstruction_mse_norm"] = normalize_group(
        group_df["reconstruction_mse"].tolist(), False
    )
    group_df["runtime_norm"] = normalize_group(
        group_df["runtime_seconds"].tolist(), False
    )

    group_df["compression_score_numeric"] = (
        0.45 * group_df["explained_variance_norm"]
        + 0.40 * group_df["reconstruction_mse_norm"]
        + 0.15 * group_df["runtime_norm"]
    )

    group_df.loc[group_df["excluded_from_compression_score"], "compression_score_numeric"] = None
    group_df.loc[group_df["status"] != "valid", "compression_score_numeric"] = None

    return group_df


def pca_eigen_covariance(X, n_components):
    """PCA using covariance matrix and eigen decomposition."""
    start = time.time()

    covariance_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    components = eigenvectors[:, :n_components]
    X_reduced = X @ components
    X_reconstructed = X_reduced @ components.T

    return {
        "method": "PCA_Eigen_Covariance",
        "method_type": "linear",
        "parameters": {},
        "n_components": n_components,
        "cumulative_explained_variance": float(np.sum(eigenvalues[:n_components]) / np.sum(eigenvalues)),
        "reconstruction_mse": float(mean_squared_error(X, X_reconstructed)),
        "runtime_seconds": time.time() - start,
        "can_visualize_2d": n_components == 2,
        "excluded_from_compression_score": False,
        "status": "valid",
        "notes": "Manual PCA using covariance matrix and eigen decomposition.",
    }


def pca_svd(X, n_components):
    """PCA using SVD. This is usually the practical method for large data."""
    start = time.time()

    model = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=RANDOM_STATE,
        iterated_power=7,
    )

    X_reduced = model.fit_transform(X)
    X_reconstructed = model.inverse_transform(X_reduced)

    return {
        "method": "PCA_SVD",
        "method_type": "linear",
        "parameters": {"svd_solver": "randomized", "iterated_power": 7},
        "n_components": n_components,
        "cumulative_explained_variance": float(np.sum(model.explained_variance_ratio_)),
        "reconstruction_mse": float(mean_squared_error(X, X_reconstructed)),
        "runtime_seconds": time.time() - start,
        "can_visualize_2d": n_components == 2,
        "excluded_from_compression_score": False,
        "status": "valid",
        "notes": "PCA using SVD; practical and stable for high-dimensional data.",
    }


def truncated_svd(X, n_components):
    """SVD-based dimensionality reduction."""
    start = time.time()

    model = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    X_reduced = model.fit_transform(X)
    X_reconstructed = model.inverse_transform(X_reduced)

    return {
        "method": "TruncatedSVD",
        "method_type": "linear",
        "parameters": {},
        "n_components": n_components,
        "cumulative_explained_variance": float(np.sum(model.explained_variance_ratio_)),
        "reconstruction_mse": float(mean_squared_error(X, X_reconstructed)),
        "runtime_seconds": time.time() - start,
        "can_visualize_2d": n_components == 2,
        "excluded_from_compression_score": False,
        "status": "valid",
        "notes": "SVD-based dimensionality reduction.",
    }


def umap_reduction(X, n_components):
    """UMAP as an extra non-linear visualization method."""
    start = time.time()

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=UMAP_PARAMS["n_neighbors"],
        min_dist=UMAP_PARAMS["min_dist"],
        metric=UMAP_PARAMS["metric"],
        random_state=RANDOM_STATE,
        low_memory=True,
    )

    reducer.fit_transform(X)

    return {
        "method": "UMAP",
        "method_type": "non-linear",
        "parameters": UMAP_PARAMS,
        "n_components": n_components,
        "cumulative_explained_variance": None,
        "reconstruction_mse": None,
        "runtime_seconds": time.time() - start,
        "can_visualize_2d": n_components == 2,
        "excluded_from_compression_score": True,
        "status": "valid",
        "notes": "UMAP is extra. It does not provide PCA-style explained variance or reconstruction MSE.",
    }

def evaluate_component_count(X, n_components):
    """Evaluate all methods once for one n_components value."""
    print("\n" + "=" * 90)
    print(f"Evaluating n_components = {n_components}")
    print("=" * 90)

    rows = []

    for func in [pca_eigen_covariance, pca_svd, truncated_svd, umap_reduction]:
        try:
            row = func(X, n_components)
        except Exception as exc:
            row = {
                "method": func.__name__,
                "method_type": None,
                "parameters": {},
                "n_components": n_components,
                "cumulative_explained_variance": None,
                "reconstruction_mse": None,
                "runtime_seconds": None,
                "can_visualize_2d": n_components == 2,
                "excluded_from_compression_score": True,
                "status": f"failed: {exc}",
                "notes": "Method failed.",
            }

        print(row)
        rows.append(row)

    return add_group_scores(pd.DataFrame(rows))


def evaluate_all_components(X):
    """Evaluate all methods grouped by the same number of components."""
    max_components = min(X.shape[0] - 1, X.shape[1])
    groups = []

    for n_components in N_COMPONENTS_LIST:
        if n_components <= max_components:
            groups.append(evaluate_component_count(X, n_components))

    return pd.concat(groups, ignore_index=True)


def choose_best_for_compression(df):
    """Choose best compression method, preferring PCA_SVD if it is within 1% of the best."""
    scored = df[df["compression_score_numeric"].notna()].copy()
    best_row = scored.sort_values("compression_score_numeric", ascending=False).iloc[0]

    svd_rows = scored[scored["method"] == "PCA_SVD"]
    if not svd_rows.empty:
        best_svd = svd_rows.sort_values("compression_score_numeric", ascending=False).iloc[0]
        gap = (best_row["compression_score_numeric"] - best_svd["compression_score_numeric"]) / max(best_row["compression_score_numeric"], 1e-12)

        if gap <= PREFER_SVD_IF_WITHIN_RELATIVE_MARGIN:
            return f"PCA_SVD_components_{int(best_svd['n_components'])}", best_svd

    return f"{best_row['method']}_components_{int(best_row['n_components'])}", best_row


def choose_best_for_2d_visualization(df):
    """Choose best 2D method. UMAP is preferred for visualization if available."""
    two_d = df[(df["n_components"] == 2) & (df["status"] == "valid")].copy()
    umap_2d = two_d[two_d["method"] == "UMAP"]

    if not umap_2d.empty:
        row = umap_2d.iloc[0]
        return "UMAP_components_2", row

    row = two_d.sort_values("runtime_seconds").iloc[0]
    return f"{row['method']}_components_2", row


def build_truth_table(df):
    """Build a clean JSON grouped by n_components."""
    best_compression_key, best_compression_row = choose_best_for_compression(df)
    best_visual_key, best_visual_row = choose_best_for_2d_visualization(df)

    output = {
        "summary": {
            "best_for_compression": best_compression_key,
            "best_for_2d_visualization": best_visual_key,
            "selected_compression_reason": (
                "PCA_SVD is selected when it is within 1% of the best score, "
                "because SVD is more practical and stable for large high-dimensional data."
            ),
            "selected_visualization_reason": (
                "UMAP is selected for 2D visualization because it is the added non-linear visualization method."
            ),
        },
        "by_n_components": {},
    }

    for n_components, group in df.groupby("n_components"):
        group = group.copy()

        max_explained = group["cumulative_explained_variance"].max()
        min_mse = group["reconstruction_mse"].min()
        min_runtime = group["runtime_seconds"].min()
        max_score = group["compression_score_numeric"].max()

        component_key = str(int(n_components))
        output["by_n_components"][component_key] = {}

        for _, row in group.iterrows():
            flags = {
                "max_cumulative_explained_variance_in_same_components": (
                    False if pd.isna(row["cumulative_explained_variance"])
                    else bool(row["cumulative_explained_variance"] == max_explained)
                ),
                "min_reconstruction_mse_in_same_components": (
                    False if pd.isna(row["reconstruction_mse"])
                    else bool(row["reconstruction_mse"] == min_mse)
                ),
                "min_runtime_seconds_in_same_components": (
                    False if pd.isna(row["runtime_seconds"])
                    else bool(row["runtime_seconds"] == min_runtime)
                ),
                "max_compression_score_in_same_components": (
                    False if pd.isna(row["compression_score_numeric"])
                    else bool(row["compression_score_numeric"] == max_score)
                ),
            }

            score_count = sum(flags.values())

            output["by_n_components"][component_key][row["method"]] = {
                "method": row["method"],
                "method_type": row["method_type"],
                "parameters": row["parameters"],
                "n_components": int(row["n_components"]),
                "cumulative_explained_variance": None if pd.isna(row["cumulative_explained_variance"]) else float(row["cumulative_explained_variance"]),
                "explained_variance_norm": float(row["explained_variance_norm"]),
                "reconstruction_mse": None if pd.isna(row["reconstruction_mse"]) else float(row["reconstruction_mse"]),
                "reconstruction_mse_norm": float(row["reconstruction_mse_norm"]),
                "runtime_seconds": None if pd.isna(row["runtime_seconds"]) else float(row["runtime_seconds"]),
                "runtime_norm": float(row["runtime_norm"]),
                "compression_score_numeric": None if pd.isna(row["compression_score_numeric"]) else float(row["compression_score_numeric"]),
                "can_visualize_2d": bool(row["can_visualize_2d"]),
                "excluded_from_compression_score": bool(row["excluded_from_compression_score"]),
                "status": row["status"],
                "notes": row["notes"],
                **flags,
                "score": f"{score_count} / 4",
                "is_selected_best_for_compression": (
                    f"{row['method']}_components_{int(row['n_components'])}" == best_compression_key
                ),
                "is_selected_best_for_2d_visualization": (
                    row["method"] == best_visual_row["method"] and int(row["n_components"]) == 2
                ),
            }

    return output


def save_results(truth_table, df):
    """Save JSON and CSV results."""
    json_path = OUTPUT_DIR / "dimensionality_reduction_eval_result.json"
    csv_path = OUTPUT_DIR / "dimensionality_reduction_eval_result.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(truth_table, f, indent=2, ensure_ascii=False)

    df.to_csv(csv_path, index=False)

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


def plot_metric_by_components(df, metric, title, filename):
    """Plot a metric grouped by n_components and save it."""
    plot_df = df[df[metric].notna()].copy()

    pivot = plot_df.pivot_table(
        index="n_components",
        columns="method",
        values=metric,
        aggfunc="first",
    )

    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel("Number of components")
    ax.set_ylabel(metric)
    ax.grid(axis="y")

    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved plot: {output_path}")


def save_plots(df):
    """Save clean plots comparing methods at the same component counts."""
    plot_metric_by_components(
        df,
        "cumulative_explained_variance",
        "Explained Variance by Number of Components",
        "dim_reduction_explained_variance_by_components.png",
    )
    plot_metric_by_components(
        df,
        "reconstruction_mse",
        "Reconstruction MSE by Number of Components",
        "dim_reduction_reconstruction_mse_by_components.png",
    )
    plot_metric_by_components(
        df,
        "runtime_seconds",
        "Runtime by Number of Components",
        "dim_reduction_runtime_by_components.png",
    )
    plot_metric_by_components(
        df,
        "compression_score_numeric",
        "Compression Score by Number of Components",
        "dim_reduction_score_by_components.png",
    )


def main():
    """Run the full clean dimensionality reduction evaluation."""
    embeddings, paper_ids = load_arrays(EMBEDDINGS_PATH, PAPER_IDS_PATH)
    embeddings, paper_ids = clean_embeddings(embeddings, paper_ids)
    X = preprocess_embeddings(embeddings)
    X_eval, _ = sample_rows(X, paper_ids, EVALUATION_SAMPLE_SIZE)

    results_df = evaluate_all_components(X_eval)
    truth_table = build_truth_table(results_df)

    print("\n" + "=" * 90)
    print("Clean Dimensionality Reduction Evaluation")
    print("=" * 90)
    print(json.dumps(truth_table, indent=2, ensure_ascii=False))

    save_results(truth_table, results_df)
    save_plots(results_df)


if __name__ == "__main__":
    main()
