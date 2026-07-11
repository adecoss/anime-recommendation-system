from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURE_DIR = BASE_DIR / "artifacts" / "feature_matrices"
PLOT_DIR = BASE_DIR / "artifacts" / "plots" / "representation"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MATRIX_PATH = FEATURE_DIR / "catalog_feature_matrix.npz"
SUMMARY_PATH = FEATURE_DIR / "representation_dimensionality_summary.json"


def main() -> None:
    matrix = load_npz(MATRIX_PATH)
    n_components = min(120, matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(matrix)
    retained = np.cumsum(svd.explained_variance_ratio_)

    summary = {
        "method": "TruncatedSVD",
        "input_matrix": str(MATRIX_PATH),
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "components_tested": int(n_components),
        "retained_energy": float(retained[-1]),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(retained) + 1), retained, linewidth=2)
    ax.set_title("SVD Retained Energy on Catalog Feature Matrix")
    ax.set_xlabel("Components")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "script_svd_retained_energy.png", dpi=160)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
