from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
ANIME_PATH = BASE_DIR / "data" / "processed" / "anime_dataset.csv"
GRAPH_DIR = BASE_DIR / "artifacts" / "graph_exports"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def parse_recommendations(value: object) -> list[tuple[int, int]]:
    if pd.isna(value) or str(value).strip() == "":
        return []

    edges = []
    for item in str(value).split("|"):
        target, _, weight = item.partition(":")
        try:
            edges.append((int(target), int(weight)))
        except ValueError:
            continue
    return edges


def main() -> None:
    anime_df = pd.read_csv(ANIME_PATH)
    graph = nx.DiGraph()

    for _, row in anime_df.iterrows():
        graph.add_node(
            int(row["mal_id"]),
            title=row.get("title", ""),
            score=row.get("score"),
            members=row.get("members"),
            genres=row.get("genres", ""),
        )

    for _, row in anime_df.iterrows():
        source = int(row["mal_id"])
        for target, weight in parse_recommendations(row.get("recommendations")):
            if target in graph:
                graph.add_edge(source, target, weight=weight)

    pagerank = nx.pagerank(graph, alpha=0.85, weight="weight")
    metrics = pd.DataFrame(
        {
            "mal_id": list(graph.nodes),
            "title": [graph.nodes[node].get("title", "") for node in graph.nodes],
            "in_degree": [graph.in_degree(node) for node in graph.nodes],
            "out_degree": [graph.out_degree(node) for node in graph.nodes],
            "pagerank": [pagerank.get(node, 0.0) for node in graph.nodes],
        }
    ).sort_values("pagerank", ascending=False)

    metrics.to_csv(GRAPH_DIR / "recommendation_graph_metrics.csv", index=False)
    summary = {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "density": float(nx.density(graph)),
        "metrics_path": str(GRAPH_DIR / "recommendation_graph_metrics.csv"),
    }
    (GRAPH_DIR / "recommendation_graph_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
