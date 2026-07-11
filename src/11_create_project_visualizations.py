from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET = BASE_DIR / "data" / "processed" / "anime_dataset.csv"
RATINGS = BASE_DIR / "data" / "processed" / "current_user_ratings.csv"
EDA_DIR = BASE_DIR / "artifacts" / "plots" / "eda"
REPRESENTATION_DIR = BASE_DIR / "artifacts" / "plots" / "representation"
SEGMENTATION_DIR = BASE_DIR / "artifacts" / "plots" / "segmentation"
RECOMMENDER_DIR = BASE_DIR / "artifacts" / "plots" / "recommender"
DOC_FIG_DIR = BASE_DIR / "reports" / "document" / "figures"

for directory in [EDA_DIR, REPRESENTATION_DIR, SEGMENTATION_DIR, RECOMMENDER_DIR, DOC_FIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def split_pipe(value) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def parse_recommendations(value) -> list[tuple[int, int]]:
    edges = []
    for part in split_pipe(value):
        if ":" not in part:
            continue
        target, votes = part.rsplit(":", 1)
        try:
            edges.append((int(target), int(float(votes))))
        except ValueError:
            continue
    return edges


def parse_relations(value) -> list[tuple[str, int]]:
    edges = []
    for part in split_pipe(value):
        if ":" not in part:
            continue
        relation, target = part.rsplit(":", 1)
        try:
            edges.append((relation.strip(), int(target)))
        except ValueError:
            continue
    return edges


def add_bar_labels(ax, fmt="{:.0f}") -> None:
    for patch in ax.patches:
        width = patch.get_width()
        if width <= 0:
            continue
        ax.text(width, patch.get_y() + patch.get_height() / 2, " " + fmt.format(width), va="center", fontsize=8)


df = pd.read_csv(DATASET)
df = df.rename(columns={"explicit_genres": "explicit_tags", "explicit_genre_weights": "explicit_tag_weights"})
for col in ["genres", "tags", "explicit_tags", "studios", "demographics", "type", "rating", "season"]:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].fillna("").astype(str)
if "duration_minutes" not in df.columns:
    df["duration_minutes"] = pd.to_numeric(df.get("duration"), errors="coerce")
if "season_final" not in df.columns:
    df["season_final"] = df["season"].replace({"nan": "", "None": ""}).fillna("")

id_to_title = dict(zip(df["mal_id"].astype(int), df["title"].fillna("").astype(str)))
id_to_members = dict(zip(df["mal_id"].astype(int), pd.to_numeric(df["members"], errors="coerce").fillna(0)))
valid_ids = set(id_to_title)

recommendation_edges: list[tuple[int, int, int]] = []
relation_edges: list[tuple[int, int, str]] = []
for row in df.itertuples(index=False):
    source = int(row.mal_id)
    for target, votes in parse_recommendations(getattr(row, "recommendations", "")):
        if target in valid_ids:
            recommendation_edges.append((source, target, votes))
    for relation, target in parse_relations(getattr(row, "relations", "")):
        if target in valid_ids:
            relation_edges.append((source, target, relation))

# EDA: centrality summaries.
in_votes = defaultdict(int)
out_votes = defaultdict(int)
for source, target, votes in recommendation_edges:
    out_votes[source] += votes
    in_votes[target] += votes
top_rec = pd.DataFrame(
    [{"title": id_to_title[k], "weighted_in_votes": v} for k, v in in_votes.items()]
).sort_values("weighted_in_votes", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(top_rec["title"][::-1], top_rec["weighted_in_votes"][::-1], color="#4C78A8")
ax.set_title("Top Anime by Incoming Recommendation Weight")
ax.set_xlabel("Incoming recommendation votes")
add_bar_labels(ax)
save(EDA_DIR / "top_recommendation_indegree.png")

rel_degree = Counter()
for source, target, _ in relation_edges:
    rel_degree[source] += 1
    rel_degree[target] += 1
top_rel = pd.DataFrame([{"title": id_to_title[k], "relation_degree": v} for k, v in rel_degree.items()]).sort_values(
    "relation_degree", ascending=False
).head(15)
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(top_rel["title"][::-1], top_rel["relation_degree"][::-1], color="#9C755F")
ax.set_title("Top Anime by Relation Graph Degree")
ax.set_xlabel("Relation degree")
add_bar_labels(ax)
save(EDA_DIR / "top_relation_degree.png")

# EDA: compact network views among popular titles.
top_ids = set(df.sort_values("popularity").head(120)["mal_id"].astype(int))

rec_graph = nx.DiGraph()
for anime_id in top_ids:
    rec_graph.add_node(anime_id)
for source, target, votes in recommendation_edges:
    if source in top_ids and target in top_ids and votes >= 2:
        rec_graph.add_edge(source, target, weight=votes)
if rec_graph.number_of_edges() > 0:
    pos = nx.spring_layout(rec_graph, seed=42, k=0.8, weight="weight")
    sizes = [40 + np.log1p(id_to_members.get(n, 0)) * 18 for n in rec_graph.nodes()]
    edge_widths = [0.3 + np.log1p(rec_graph[u][v]["weight"]) * 0.35 for u, v in rec_graph.edges()]
    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_edges(rec_graph, pos, ax=ax, alpha=0.25, width=edge_widths, arrows=True, arrowsize=7, edge_color="#4C78A8")
    nx.draw_networkx_nodes(rec_graph, pos, ax=ax, node_size=sizes, node_color="#F28E2B", alpha=0.85, linewidths=0.4, edgecolors="white")
    label_nodes = sorted(rec_graph.nodes(), key=lambda n: in_votes.get(n, 0), reverse=True)[:12]
    nx.draw_networkx_labels(rec_graph, pos, labels={n: id_to_title[n][:24] for n in label_nodes}, font_size=7, ax=ax)
    ax.set_title("Recommendation Network Among Top-Popularity Anime")
    ax.axis("off")
    save(EDA_DIR / "recommendation_network_top_popular.png")

navigation_relations = {"Prequel", "Sequel", "Side Story", "Parent Story", "Alternative Version", "Summary"}
relation_colors = {
    "Sequel": "#4C78A8",
    "Prequel": "#F28E2B",
    "Side Story": "#59A14F",
    "Parent Story": "#E15759",
    "Alternative Version": "#B07AA1",
    "Summary": "#9C755F",
}
relation_graph = nx.Graph()
relation_digraph = nx.DiGraph()
for source, target, relation in relation_edges:
    if relation not in navigation_relations:
        continue
    relation_graph.add_edge(source, target, relation=relation)
    relation_digraph.add_edge(source, target, relation=relation)

franchise_root = 5081
if franchise_root in relation_graph:
    component_nodes = nx.node_connected_component(relation_graph, franchise_root)
    franchise_graph = relation_digraph.subgraph(component_nodes).copy()
    levels = nx.single_source_shortest_path_length(relation_graph.subgraph(component_nodes), franchise_root)
    for node, level in levels.items():
        franchise_graph.nodes[node]["level"] = level
    pos = nx.multipartite_layout(franchise_graph, subset_key="level", align="vertical", scale=2.8)

    watch_order_ids = [5081, 9260, 31757, 31758, 11597, 15689, 17074, 21855, 28025, 31181, 32268, 35247, 36999, 57864]
    watch_order_rank = {anime_id: rank + 1 for rank, anime_id in enumerate(watch_order_ids)}
    node_colors = ["#F28E2B" if node in watch_order_rank else "#BAB0AC" for node in franchise_graph.nodes()]
    node_sizes = [650 if node == franchise_root else 430 if node in watch_order_rank else 300 for node in franchise_graph.nodes()]

    fig, ax = plt.subplots(figsize=(13, 8))
    for relation, color in relation_colors.items():
        edges = [(u, v) for u, v, d in franchise_graph.edges(data=True) if d.get("relation") == relation]
        nx.draw_networkx_edges(
            franchise_graph,
            pos,
            edgelist=edges,
            ax=ax,
            alpha=0.7,
            width=1.5,
            edge_color=color,
            arrows=True,
            arrowsize=13,
            connectionstyle="arc3,rad=0.05",
            label=relation,
        )
    nx.draw_networkx_nodes(franchise_graph, pos, node_color=node_colors, node_size=node_sizes, edgecolors="white", linewidths=0.8, ax=ax)
    labels = {
        node: (f"{watch_order_rank[node]}. " if node in watch_order_rank else "") + id_to_title.get(node, str(node))[:32]
        for node in franchise_graph.nodes()
    }
    nx.draw_networkx_labels(franchise_graph, pos, labels=labels, font_size=7, ax=ax)
    edge_labels = {(u, v): d.get("relation", "") for u, v, d in franchise_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(franchise_graph, pos, edge_labels=edge_labels, font_size=6, rotate=False, ax=ax)
    ax.set_title("Bakemonogatari Relation Tree: Franchise Connectivity, Not Final Watch Order")
    legend_handles = [Line2D([0], [0], color=color, lw=2, label=relation) for relation, color in relation_colors.items()]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8, frameon=False)
    ax.axis("off")
    save(EDA_DIR / "bakemonogatari_relation_tree.png")

if RATINGS.exists():
    user_rating_counts = Counter()
    anime_rating_counts = Counter()
    for chunk in pd.read_csv(RATINGS, chunksize=1_000_000):
        if "userID" in chunk:
            user_rating_counts.update(chunk["userID"].dropna().astype(int).tolist())
        if "animeID" in chunk:
            anime_rating_counts.update(chunk["animeID"].dropna().astype(int).tolist())

    quantity_bins = [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, np.inf]
    quantity_labels = ["1", "2-5", "6-10", "11-25", "26-50", "51-100", "101-250", "251-500", "501-1000", "1000+"]
    user_quantity = pd.cut(pd.Series(user_rating_counts.values()), quantity_bins, labels=quantity_labels, include_lowest=True).value_counts(sort=False)
    anime_quantity = pd.cut(pd.Series(anime_rating_counts.values()), quantity_bins, labels=quantity_labels, include_lowest=True).value_counts(sort=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].bar(user_quantity.index.astype(str), user_quantity.values, color="#59A14F")
    axes[0].set_title("Ratings per User")
    axes[0].set_xlabel("Rating count bucket")
    axes[0].set_ylabel("Users")
    axes[0].tick_params(axis="x", rotation=45)
    for x, y in enumerate(user_quantity.values):
        axes[0].text(x, y, f"{int(y):,}", ha="center", va="bottom", fontsize=7, rotation=45)

    axes[1].bar(anime_quantity.index.astype(str), anime_quantity.values, color="#F28E2B")
    axes[1].set_title("Ratings per Anime")
    axes[1].set_xlabel("Rating count bucket")
    axes[1].set_ylabel("Anime")
    axes[1].tick_params(axis="x", rotation=45)
    for x, y in enumerate(anime_quantity.values):
        axes[1].text(x, y, f"{int(y):,}", ha="center", va="bottom", fontsize=7, rotation=45)

    save(EDA_DIR / "rating_quantity_distribution.png")

# Representation: representation scatter views.
numeric_cols = [
    "score",
    "scored_by",
    "rank",
    "popularity",
    "members",
    "favorites",
    "episodes",
    "duration_minutes",
    "total_watch_minutes",
    "aired_year",
    "aired_month",
]
numeric_cols = [col for col in numeric_cols if col in df.columns]
X_numeric = sparse.csr_matrix(StandardScaler().fit_transform(df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)))
cat_specs = {
    "genres": df["genres"].apply(split_pipe).tolist(),
    "demographics": df["demographics"].apply(split_pipe).tolist(),
    "type": df["type"].apply(lambda x: [x] if x else []).tolist(),
    "rating": df["rating"].apply(lambda x: [x] if x else []).tolist(),
    "season": df["season_final"].apply(lambda x: [x] if x else []).tolist(),
}
cat_mats = []
for values in cat_specs.values():
    cat_mats.append(sparse.csr_matrix(MultiLabelBinarizer().fit_transform(values)))
X_cat = sparse.hstack(cat_mats, format="csr")
text = df["genres"] + " " + df["tags"] + " " + df["explicit_tags"] + " " + df["studios"] + " " + df["demographics"]
X_text_small = TfidfVectorizer(max_features=350, min_df=3, max_df=0.85, stop_words="english").fit_transform(text)
X_text_large = TfidfVectorizer(max_features=3500, min_df=3, max_df=0.85, stop_words="english").fit_transform(text)
X_dense_mixed = sparse.hstack([X_numeric, X_cat, X_text_small], format="csr").toarray()
X_sparse_full = sparse.hstack([X_numeric, X_cat, X_text_large], format="csr")

pca = PCA(n_components=min(12, X_dense_mixed.shape[1]), random_state=42)
X_pca = pca.fit_transform(X_dense_mixed)
type_codes, type_uniques = pd.factorize(df["type"])
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=type_codes, cmap="tab10", s=8, alpha=0.45, linewidths=0)
handles, _ = scatter.legend_elements(num=len(type_uniques))
ax.legend(handles, type_uniques, title="type", loc="best", fontsize=8)
ax.set_title("PCA Projection of Dense Mixed Catalog Matrix")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.grid(alpha=0.2)
save(REPRESENTATION_DIR / "pca_2d_by_type.png")

svd = TruncatedSVD(n_components=50, random_state=42)
X_svd = svd.fit_transform(X_sparse_full)
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_svd[:, 0], X_svd[:, 1], c=np.log1p(pd.to_numeric(df["members"], errors="coerce").fillna(0)), cmap="viridis", s=8, alpha=0.45, linewidths=0)
fig.colorbar(scatter, ax=ax, label="log(1 + members)")
ax.set_title("SVD Projection of Sparse Text/Tag Catalog Matrix")
ax.set_xlabel("SVD1")
ax.set_ylabel("SVD2")
ax.grid(alpha=0.2)
save(REPRESENTATION_DIR / "svd_2d_by_members.png")

sample_n = min(6000, len(df))
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(df), size=sample_n, replace=False)
tsne = TSNE(n_components=2, perplexity=35, learning_rate="auto", init="pca", random_state=42)
X_tsne = tsne.fit_transform(X_svd[sample_idx, : min(50, X_svd.shape[1])])
sample_types = df.iloc[sample_idx]["type"].fillna("Unknown")
sample_type_codes, sample_type_names = pd.factorize(sample_types)
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=sample_type_codes, cmap="tab10", s=10, alpha=0.6, linewidths=0)
handles, _ = scatter.legend_elements(num=len(sample_type_names))
ax.legend(handles, sample_type_names, title="type", loc="best", fontsize=8)
ax.set_title("t-SNE on SVD Catalog Embeddings (Sample)")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.grid(alpha=0.2)
save(REPRESENTATION_DIR / "tsne_svd_catalog_sample.png")

# Segmentation: cluster scatter and profile heatmap.
X_cluster_scaled = StandardScaler().fit_transform(X_svd)
k_values = list(range(4, 21, 2))
k_rows = []
for k in k_values:
    labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X_cluster_scaled)
    sil = silhouette_score(X_cluster_scaled, labels, sample_size=min(5000, len(labels)), random_state=42)
    k_rows.append((k, sil))
best_k = max(k_rows, key=lambda item: item[1])[0]
labels = KMeans(n_clusters=best_k, random_state=42, n_init=20).fit_predict(X_cluster_scaled)
df["cluster"] = labels

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_svd[:, 0], X_svd[:, 1], c=labels, cmap="tab20", s=8, alpha=0.5, linewidths=0)
fig.colorbar(scatter, ax=ax, label="K-means cluster")
ax.set_title(f"SVD Space Colored by K-means Cluster (k={best_k})")
ax.set_xlabel("SVD1")
ax.set_ylabel("SVD2")
ax.grid(alpha=0.2)
save(SEGMENTATION_DIR / "kmeans_svd_cluster_scatter.png")

top_genres = Counter()
for value in df["genres"]:
    top_genres.update(split_pipe(value))
top_genres = [name for name, _ in top_genres.most_common(12)]
heat = []
cluster_ids = sorted(df["cluster"].unique())
for cluster_id in cluster_ids:
    subset = df[df["cluster"] == cluster_id]
    denominator = max(len(subset), 1)
    counts = Counter()
    for value in subset["genres"]:
        counts.update(split_pipe(value))
    heat.append([counts[genre] / denominator for genre in top_genres])
heat = np.array(heat)
fig, ax = plt.subplots(figsize=(11, max(5, len(cluster_ids) * 0.35)))
im = ax.imshow(heat, aspect="auto", cmap="YlGnBu")
ax.set_xticks(range(len(top_genres)), labels=top_genres, rotation=45, ha="right")
ax.set_yticks(range(len(cluster_ids)), labels=cluster_ids)
ax.set_xlabel("Top catalog genres")
ax.set_ylabel("Cluster")
ax.set_title("Cluster Genre Profile Heatmap")
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        if heat[i, j] >= 0.15:
            ax.text(j, i, f"{heat[i, j]:.0%}", ha="center", va="center", fontsize=6)
fig.colorbar(im, ax=ax, label="Share of cluster")
save(SEGMENTATION_DIR / "cluster_genre_profile_heatmap.png")

sample_n_density = min(6000, len(df))
density_idx = rng.choice(len(df), size=sample_n_density, replace=False)
X_db = X_cluster_scaled[density_idx]
optics_rows = []
for min_samples in [10, 20, 35, 50]:
    for xi in [0.03, 0.05, 0.08]:
        model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=0.01, n_jobs=-1)
        optics_labels = model.fit_predict(X_db)
        clusters = len(set(optics_labels)) - (1 if -1 in optics_labels else 0)
        noise_pct = float((optics_labels == -1).mean() * 100)
        optics_rows.append({"min_samples": min_samples, "xi": xi, "clusters": clusters, "noise_pct": noise_pct})
optics_results = pd.DataFrame(optics_rows)
pivot = optics_results.pivot(index="min_samples", columns="xi", values="clusters")
fig, ax = plt.subplots(figsize=(7.5, 4.5))
im = ax.imshow(pivot.values, cmap="Purples")
ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
ax.set_xlabel("xi")
ax.set_ylabel("min_samples")
ax.set_title("OPTICS Cluster Count Sweep")
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax.text(j, i, int(pivot.values[i, j]), ha="center", va="center", color="black")
fig.colorbar(im, ax=ax, label="clusters")
save(SEGMENTATION_DIR / "optics_parameter_sweep.png")

# Copy selected figures into the Overleaf-compatible figure folder.
for source in [
    EDA_DIR / "recommendation_network_top_popular.png",
    EDA_DIR / "bakemonogatari_relation_tree.png",
    EDA_DIR / "rating_quantity_distribution.png",
    EDA_DIR / "top_recommendation_indegree.png",
    EDA_DIR / "top_relation_degree.png",
    REPRESENTATION_DIR / "pca_mixed_catalog_explained_variance.png",
    REPRESENTATION_DIR / "svd_sparse_catalog_energy.png",
    REPRESENTATION_DIR / "pca_2d_by_type.png",
    REPRESENTATION_DIR / "svd_2d_by_members.png",
    REPRESENTATION_DIR / "tsne_svd_catalog_sample.png",
    SEGMENTATION_DIR / "kmeans_parameter_sweep.png",
    SEGMENTATION_DIR / "dbscan_parameter_sweep.png",
    SEGMENTATION_DIR / "optics_parameter_sweep.png",
    SEGMENTATION_DIR / "kmeans_cluster_sizes.png",
    SEGMENTATION_DIR / "kmeans_svd_cluster_scatter.png",
    SEGMENTATION_DIR / "cluster_genre_profile_heatmap.png",
    RECOMMENDER_DIR / "metric_comparison.png",
    RECOMMENDER_DIR / "median_rank.png",
    RECOMMENDER_DIR / "user_level_distribution.png",
    RECOMMENDER_DIR / "hit_at_10_by_user_level.png",
]:
    if source.exists():
        (DOC_FIG_DIR / source.name).write_bytes(source.read_bytes())

print("dataset_rows", len(df))
print("recommendation_edges", len(recommendation_edges))
print("relation_edges", len(relation_edges))
print("bakemonogatari_relation_nodes", len(component_nodes) if franchise_root in relation_graph else 0)
print("segmentation_best_k", best_k)
print("segmentation_best_k_silhouette", max(score for _, score in k_rows))
print("pca_12_retained", float(pca.explained_variance_ratio_.sum()))
print("svd_50_retained", float(svd.explained_variance_ratio_.sum()))
