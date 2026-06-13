from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
FEATURE_DIR = BASE_DIR / "artifacts" / "feature_matrices"
PLOT_DIR = BASE_DIR / "artifacts" / "plots" / "week7"
CLUSTER_DIR = BASE_DIR / "artifacts" / "clustering"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
CLUSTER_DIR.mkdir(parents=True, exist_ok=True)

MATRIX_PATH = FEATURE_DIR / "catalog_feature_matrix.npz"
METADATA_PATH = FEATURE_DIR / "catalog_feature_metadata.csv"
VALIDATION_PATH = CLUSTER_DIR / "week7_kmeans_validation.csv"
ASSIGNMENTS_PATH = CLUSTER_DIR / "week7_cluster_assignments.csv"


def main() -> None:
    matrix = load_npz(MATRIX_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    reducer = TruncatedSVD(n_components=min(50, matrix.shape[1] - 1), random_state=42)
    reduced = reducer.fit_transform(matrix)
    reduced = StandardScaler().fit_transform(reduced)

    rows = []
    for k in range(4, 21, 2):
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = model.fit_predict(reduced)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(
                    reduced,
                    labels,
                    sample_size=min(5000, len(labels)),
                    random_state=42,
                ),
                "davies_bouldin": davies_bouldin_score(reduced, labels),
                "inertia": model.inertia_,
            }
        )

    validation = pd.DataFrame(rows)
    validation.to_csv(VALIDATION_PATH, index=False)

    best_k = int(
        validation.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True])
        .iloc[0]["k"]
    )
    best_model = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    metadata["cluster"] = best_model.fit_predict(reduced)
    metadata.to_csv(ASSIGNMENTS_PATH, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(validation["k"], validation["silhouette"], marker="o")
    axes[0].set_title("K-means Silhouette Sweep")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Silhouette")
    axes[0].grid(alpha=0.3)
    axes[1].plot(validation["k"], validation["inertia"], marker="o", color="#F28E2B")
    axes[1].set_title("K-means Inertia Sweep")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Inertia")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "script_kmeans_sweep.png", dpi=160)
    plt.close(fig)

    summary = {
        "best_k": best_k,
        "rows_clustered": int(len(metadata)),
        "validation_path": str(VALIDATION_PATH),
        "assignments_path": str(ASSIGNMENTS_PATH),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
