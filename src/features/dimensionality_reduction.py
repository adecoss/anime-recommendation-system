from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = BASE_DIR / "artifacts"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

print("Loading feature matrices...")

numeric_df = pd.read_csv(
    EMBEDDINGS_DIR / "numeric_features.csv"
)

user_item_matrix = load_npz(
    EMBEDDINGS_DIR / "user_item_matrix.npz"
)

anime_index = pd.read_csv(
    EMBEDDINGS_DIR / "anime_index.csv"
)

print(f"Numeric matrix: {numeric_df.shape}")
print(f"Sparse matrix: {user_item_matrix.shape}")

# =========================================================
# PCA
# =========================================================

print("Running PCA...")

pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(numeric_df)

explained_variance = np.cumsum(
    pca.explained_variance_ratio_
)

print(
    f"PCA components retained: {pca.n_components_}"
)

print(
    f"Explained variance: {explained_variance[-1]:.4f}"
)

# =========================================================
# SAVE PCA EMBEDDINGS
# =========================================================

pca_df = pd.DataFrame(X_pca)

pca_df.to_csv(
    EMBEDDINGS_DIR / "pca_embeddings.csv",
    index=False
)

joblib.dump(
    pca,
    EMBEDDINGS_DIR / "pca_model.pkl"
)
# =========================================================
# PCA VARIANCE PLOT
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    explained_variance,
    marker="o"
)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")

plt.grid(True)

plt.savefig(
    PLOTS_DIR / "pca_explained_variance.png",
    bbox_inches="tight"
)

plt.close()

# =========================================================
# TRUNCATED SVD
# =========================================================

print("Running Truncated SVD...")

N_COMPONENTS = 100

svd = TruncatedSVD(
    n_components=N_COMPONENTS,
    random_state=42
)

X_svd = svd.fit_transform(user_item_matrix)

svd_variance = np.cumsum(
    svd.explained_variance_ratio_
)

print(
    f"SVD latent matrix shape: {X_svd.shape}"
)

print(
    f"SVD retained variance: {svd_variance[-1]:.4f}"
)

# =========================================================
# SAVE SVD EMBEDDINGS
# =========================================================

svd_df = pd.DataFrame(X_svd)

svd_df.to_csv(
    EMBEDDINGS_DIR / "svd_user_embeddings.csv",
    index=False
)

joblib.dump(
    svd,
    EMBEDDINGS_DIR / "svd_model.pkl"
)

# =========================================================
# SINGULAR VALUE DECAY PLOT
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    svd.singular_values_,
    marker="o"
)

plt.xlabel("Latent Dimension")
plt.ylabel("Singular Value")
plt.title("Singular Value Decay")

plt.grid(True)

plt.savefig(
    PLOTS_DIR / "singular_value_decay.png",
    bbox_inches="tight"
)

plt.close()

# =========================================================
# t-SNE VISUALIZATION
# =========================================================

print("Running t-SNE visualization...")

TSNE_SAMPLE = 5000

sample_size = min(
    TSNE_SAMPLE,
    X_svd.shape[0]
)

sample_indices = np.random.choice(
    X_svd.shape[0],
    size=sample_size,
    replace=False
)

X_sample = X_svd[sample_indices]

print(f"t-SNE sample size: {sample_size}")

# t-SNE can be slow

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init="pca",
    learning_rate="auto"
)

X_tsne = tsne.fit_transform(X_sample)

# =========================================================
# SAVE t-SNE EMBEDDINGS
# =========================================================

tsne_df = pd.DataFrame({
    "x": X_tsne[:, 0],
    "y": X_tsne[:, 1]
})

tsne_df.to_csv(
    EMBEDDINGS_DIR / "tsne_embeddings.csv",
    index=False
)

# =========================================================
# t-SNE PLOT
# =========================================================

plt.figure(figsize=(12, 8))

plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    s=5,
    alpha=0.6
)

plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("t-SNE Visualization of User Embeddings")

plt.savefig(
    PLOTS_DIR / "tsne_visualization.png",
    bbox_inches="tight"
)

plt.close()


# =========================================================
# SAVE SUMMARY METRICS
# =========================================================

metrics = {
    "pca_components": int(pca.n_components_),
    "pca_variance": float(explained_variance[-1]),
    "svd_components": int(N_COMPONENTS),
    "svd_variance": float(svd_variance[-1]),
    "tsne_sample_size": int(sample_size)
}

joblib.dump(
    metrics,
    EMBEDDINGS_DIR / "dimensionality_metrics.pkl"
)

print("Dimensionality reduction pipeline complete")