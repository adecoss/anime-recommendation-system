from pathlib import Path
import community as community_louvain
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pickle

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
GRAPH_DIR = ARTIFACTS_DIR / "graph"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD DATASET
# =========================================================

anime_df = pd.read_csv(
    PROCESSED_DIR / "anime_dataset.csv"
)

print(f"Loaded anime dataset: {anime_df.shape}")
# =========================================================
# CREATE GRAPH
# =========================================================

print("Building recommendation graph...")

G = nx.DiGraph()

# =========================================================
# ADD NODES
# =========================================================

for _, row in anime_df.iterrows():

    mal_id = row["mal_id"]

    G.add_node(
        mal_id,
        title=row.get("title", "Unknown"),
        score=row.get("score", 0),
        popularity=row.get("popularity", 0),
        genres=row.get("genres", "")
    )

# =========================================================
# ADD RECOMMENDATION EDGES
# =========================================================
edge_count = 0

for _, row in anime_df.iterrows():

    source_id = row["mal_id"]

    recommendations = row.get("recommendations")

    if pd.isna(recommendations):
        continue

    recommendations = str(recommendations)

    if not recommendations.strip():
        continue

    recs = recommendations.split("|")

    for rec in recs:

        try:

            target_id, votes = rec.split(":")

            target_id = int(target_id)
            votes = int(votes)

            G.add_edge(
                source_id,
                target_id,
                weight=votes
            )

            edge_count += 1

        except Exception:
            continue

print(f"Graph nodes: {G.number_of_nodes()}")
print(f"Graph edges: {G.number_of_edges()}")
# =========================================================
# BASIC GRAPH STATISTICS
# =========================================================

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

density = nx.density(G)

print(f"Graph density: {density:.8f}")

# =========================================================
# DEGREE CENTRALITY
# =========================================================

print("Computing degree centrality...")

in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())

# =========================================================
# PAGERANK
# =========================================================

print("Computing PageRank...")

pagerank_scores = nx.pagerank(
    G,
    alpha=0.85,
    weight="weight"
)

# =========================================================
# COMMUNITY DETECTION
# =========================================================

print("Running community detection...")

# Louvain works on undirected graphs
G_undirected = G.to_undirected()

partition = community_louvain.best_partition(
    G_undirected
)

num_communities = len(
    set(partition.values())
)

print(f"Detected communities: {num_communities}")

# =========================================================
# CREATE METRICS DATAFRAME
# =========================================================

metrics_df = pd.DataFrame({
    "mal_id": list(G.nodes()),
    "title": [
        G.nodes[n].get("title", "")
        for n in G.nodes()
    ],
    "in_degree": [
        in_degree.get(n, 0)
        for n in G.nodes()
    ],
    "out_degree": [
        out_degree.get(n, 0)
        for n in G.nodes()
    ],
    "pagerank": [
        pagerank_scores.get(n, 0)
        for n in G.nodes()
    ],
    "community": [
        partition.get(n, -1)
        for n in G.nodes()
    ]
})

# =========================================================
# SORT BY PAGERANK
# =========================================================

metrics_df = metrics_df.sort_values(
    "pagerank",
    ascending=False
)

# =========================================================
# SAVE METRICS
# =========================================================

metrics_df.to_csv(
    GRAPH_DIR / "anime_graph_metrics.csv",
    index=False
)

# =========================================================
# SAVE GRAPH
# =========================================================

with open(
    GRAPH_DIR / "recommendation_graph.gpickle",
    "wb"
) as f:

    pickle.dump(G, f)

# =========================================================
# TOP PAGERANK PLOT
# =========================================================

top_n = 20

top_df = metrics_df.head(top_n)

plt.figure(figsize=(14, 8))

plt.barh(
    top_df["title"][::-1],
    top_df["pagerank"][::-1]
)

plt.xlabel("PageRank Score")
plt.ylabel("Anime")
plt.title("Top Anime by Recommendation Graph PageRank")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "top_pagerank_anime.png",
    bbox_inches="tight"
)

plt.close()

# =========================================================
# COMMUNITY DISTRIBUTION PLOT
# =========================================================

community_sizes = (
    metrics_df["community"]
    .value_counts()
    .head(20)
)

plt.figure(figsize=(12, 6))

plt.bar(
    community_sizes.index.astype(str),
    community_sizes.values
)

plt.xlabel("Community ID")
plt.ylabel("Number of Anime")
plt.title("Largest Anime Communities")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "community_distribution.png",
    bbox_inches="tight"
)

plt.close()

# =========================================================
# SAVE SUMMARY
# =========================================================

summary = {
    "nodes": int(num_nodes),
    "edges": int(num_edges),
    "density": float(density),
    "communities": int(num_communities)
}

joblib.dump(
    summary,
    GRAPH_DIR / "graph_summary.pkl"
)

print("Graph intelligence pipeline complete")