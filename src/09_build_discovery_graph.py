from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


BASE_DIR = Path(__file__).resolve().parents[1]
ANIME_PATH = BASE_DIR / "data" / "processed" / "anime_dataset.csv"
GRAPH_DIR = BASE_DIR / "artifacts" / "graph"
PLOT_DIR = BASE_DIR / "artifacts" / "plots" / "graph"
REPORT_PATH = BASE_DIR / "reports" / "graph_discovery_report.md"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


RELATION_WEIGHTS = {
    "Sequel": 10.0,
    "Prequel": 10.0,
    "Parent Story": 9.0,
    "Full Story": 9.0,
    "Side Story": 6.0,
    "Alternative Version": 5.0,
    "Alternative Setting": 5.0,
    "Summary": 3.0,
    "Other": 2.0,
}


def project_path(value: object) -> object:
    if not isinstance(value, (str, Path)):
        return value
    text = str(value)
    try:
        path = Path(text)
        if path.is_absolute():
            return str(path.relative_to(BASE_DIR))
    except (ValueError, OSError):
        pass
    base_text = str(BASE_DIR)
    if text.startswith(base_text):
        return text[len(base_text) :].lstrip("\\/")
    return text


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def parse_recommendations(value: object) -> list[tuple[int, float]]:
    if pd.isna(value) or str(value).strip() == "":
        return []

    edges = []
    for item in str(value).split("|"):
        target, _, weight = item.partition(":")
        try:
            edges.append((int(float(target)), float(weight or 1)))
        except ValueError:
            continue
    return edges


def parse_relations(value: object) -> list[tuple[int, str, float]]:
    if pd.isna(value) or str(value).strip() == "":
        return []

    edges = []
    for item in str(value).split("|"):
        target, _, relation = item.partition(":")
        relation = relation.strip() or "Other"
        try:
            target_id = int(float(target))
        except ValueError:
            continue
        edges.append((target_id, relation, RELATION_WEIGHTS.get(relation, 2.0)))
    return edges


def add_or_update_edge(graph: nx.DiGraph, source: int, target: int, weight: float, edge_type: str, label: str) -> None:
    if source == target or source not in graph or target not in graph:
        return
    if graph.has_edge(source, target):
        graph[source][target]["weight"] += float(weight)
        graph[source][target]["edge_types"] = "|".join(
            sorted(set(str(graph[source][target].get("edge_types", "")).split("|")) | {edge_type})
        ).strip("|")
        labels = sorted(set(str(graph[source][target].get("labels", "")).split("|")) | {label})
        graph[source][target]["labels"] = "|".join(label for label in labels if label)
    else:
        graph.add_edge(source, target, weight=float(weight), edge_types=edge_type, labels=label)


def load_catalog() -> pd.DataFrame:
    df = pd.read_csv(ANIME_PATH)
    numeric_cols = ["mal_id", "score", "members", "popularity", "favorites"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["mal_id"].notna()].copy()
    df["mal_id"] = df["mal_id"].astype(int)
    return df


def build_graphs(anime_df: pd.DataFrame) -> dict[str, nx.DiGraph]:
    graphs = {
        "recommendation": nx.DiGraph(graph_kind="recommendation"),
        "relation": nx.DiGraph(graph_kind="relation"),
        "combined": nx.DiGraph(graph_kind="combined"),
    }

    for graph in graphs.values():
        for _, row in anime_df.iterrows():
            graph.add_node(
                int(row["mal_id"]),
                title=row.get("title", ""),
                type=row.get("type", ""),
                score=row.get("score"),
                members=row.get("members"),
                popularity=row.get("popularity"),
                genres=row.get("genres", ""),
                demographics=row.get("demographics", ""),
            )

    for _, row in anime_df.iterrows():
        source = int(row["mal_id"])
        for target, weight in parse_recommendations(row.get("recommendations")):
            add_or_update_edge(graphs["recommendation"], source, target, weight, "recommendation", "recommendation")
            add_or_update_edge(graphs["combined"], source, target, weight, "recommendation", "recommendation")

        for target, relation, weight in parse_relations(row.get("relations")):
            add_or_update_edge(graphs["relation"], source, target, weight, "relation", relation)
            add_or_update_edge(graphs["combined"], source, target, weight, "relation", relation)

    return graphs


def graph_summary(graph: nx.DiGraph, name: str) -> dict:
    weak_components = list(nx.weakly_connected_components(graph))
    strong_components = list(nx.strongly_connected_components(graph))
    weak_sizes = sorted((len(component) for component in weak_components), reverse=True)
    isolates = list(nx.isolates(graph))
    return {
        "graph": name,
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "density": float(nx.density(graph)),
        "weak_components": int(len(weak_components)),
        "strong_components": int(len(strong_components)),
        "largest_weak_component": int(weak_sizes[0]) if weak_sizes else 0,
        "largest_weak_component_pct": float(weak_sizes[0] / graph.number_of_nodes()) if weak_sizes else 0.0,
        "isolated_nodes": int(len(isolates)),
        "isolated_pct": float(len(isolates) / max(graph.number_of_nodes(), 1)),
        "mean_out_degree": float(np.mean([degree for _, degree in graph.out_degree()])) if graph.number_of_nodes() else 0.0,
        "mean_in_degree": float(np.mean([degree for _, degree in graph.in_degree()])) if graph.number_of_nodes() else 0.0,
    }


def graph_metrics(graph: nx.DiGraph, anime_df: pd.DataFrame) -> pd.DataFrame:
    pagerank = nx.pagerank(graph, alpha=0.85, weight="weight") if graph.number_of_edges() else {node: 0.0 for node in graph.nodes}
    hub, authority = nx.hits(graph, max_iter=300, normalized=True) if graph.number_of_edges() else ({}, {})
    rows = []
    for node in graph.nodes:
        rows.append(
            {
                "mal_id": node,
                "title": graph.nodes[node].get("title", ""),
                "type": graph.nodes[node].get("type", ""),
                "score": graph.nodes[node].get("score"),
                "members": graph.nodes[node].get("members"),
                "popularity": graph.nodes[node].get("popularity"),
                "genres": graph.nodes[node].get("genres", ""),
                "in_degree": int(graph.in_degree(node)),
                "out_degree": int(graph.out_degree(node)),
                "weighted_in_degree": float(graph.in_degree(node, weight="weight")),
                "weighted_out_degree": float(graph.out_degree(node, weight="weight")),
                "pagerank": float(pagerank.get(node, 0.0)),
                "hub_score": float(hub.get(node, 0.0)),
                "authority_score": float(authority.get(node, 0.0)),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics["pagerank_rank"] = metrics["pagerank"].rank(ascending=False, method="min").astype(int)
    metrics["members_rank"] = metrics["members"].rank(ascending=False, method="min")
    metrics["score_rank"] = metrics["score"].rank(ascending=False, method="min")
    return metrics.sort_values("pagerank", ascending=False)


def ranking_comparison(metrics: pd.DataFrame, graph_name: str, mylist_path: Path) -> pd.DataFrame:
    columns = ["comparison", "metric", "value", "interpretation"]
    rows = []
    compare_cols = ["pagerank", "members", "score", "weighted_in_degree"]
    valid = metrics[compare_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if not valid.empty:
        rows.extend(
            [
                {
                    "comparison": f"{graph_name}_pagerank_vs_members",
                    "metric": "spearman",
                    "value": float(valid["pagerank"].corr(valid["members"], method="spearman")),
                    "interpretation": "Graph prestige compared with MAL popularity.",
                },
                {
                    "comparison": f"{graph_name}_pagerank_vs_score",
                    "metric": "spearman",
                    "value": float(valid["pagerank"].corr(valid["score"], method="spearman")),
                    "interpretation": "Graph prestige compared with average catalog score.",
                },
                {
                    "comparison": f"{graph_name}_pagerank_vs_weighted_in_degree",
                    "metric": "spearman",
                    "value": float(valid["pagerank"].corr(valid["weighted_in_degree"], method="spearman")),
                    "interpretation": "PageRank compared with direct incoming recommendation weight.",
                },
            ]
        )

    mylist = safe_read_csv(mylist_path)
    if not mylist.empty:
        if {"mal_id", "mylist_hybrid_score"}.issubset(mylist.columns):
            graph_top = set(metrics.head(50)["mal_id"].astype(int))
            model_top = set(mylist.sort_values("mylist_hybrid_score", ascending=False).head(50)["mal_id"].astype(int))
            rows.append(
                {
                    "comparison": f"{graph_name}_top50_vs_mylist_hybrid_top50",
                    "metric": "overlap",
                    "value": float(len(graph_top & model_top) / max(len(graph_top | model_top), 1)),
                    "interpretation": "Global graph prestige is compared with the personalized MyList hybrid example.",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def validity_checks(summaries: pd.DataFrame, combined: nx.DiGraph, rec_metrics: pd.DataFrame, combined_metrics: pd.DataFrame) -> pd.DataFrame:
    rec_top = set(rec_metrics.head(50)["mal_id"].astype(int))
    combined_top = set(combined_metrics.head(50)["mal_id"].astype(int))
    relation_edges = sum(1 for _, _, data in combined.edges(data=True) if "relation" in str(data.get("edge_types", "")))
    recommendation_edges = sum(1 for _, _, data in combined.edges(data=True) if "recommendation" in str(data.get("edge_types", "")))
    return pd.DataFrame(
        [
            {
                "check": "combined_graph_has_edges",
                "value": int(combined.number_of_edges()),
                "status": "pass" if combined.number_of_edges() > 0 else "fail",
                "interpretation": "The graph should encode real recommendation/relation structure, not just isolated catalog rows.",
            },
            {
                "check": "largest_component_share",
                "value": float(summaries.loc[summaries["graph"].eq("combined"), "largest_weak_component_pct"].iloc[0]),
                "status": "inspect",
                "interpretation": "Large component means many titles are navigable; isolated nodes remain cold-start content.",
            },
            {
                "check": "isolated_node_share",
                "value": float(summaries.loc[summaries["graph"].eq("combined"), "isolated_pct"].iloc[0]),
                "status": "inspect",
                "interpretation": "High isolation limits graph recommendations for obscure entries.",
            },
            {
                "check": "edge_type_balance",
                "value": f"{recommendation_edges} recommendation edges; {relation_edges} relation edges",
                "status": "inspect",
                "interpretation": "Recommendation edges measure similarity; relation edges measure franchise/navigation paths.",
            },
            {
                "check": "definition_sensitivity_top50_overlap",
                "value": float(len(rec_top & combined_top) / max(len(rec_top | combined_top), 1)),
                "status": "inspect",
                "interpretation": "Overlap between recommendation-only and combined PageRank shows sensitivity to graph definition.",
            },
        ]
    )


def save_plots(summary_df: pd.DataFrame, metrics: pd.DataFrame) -> None:
    top = metrics.head(20).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top["title"], top["pagerank"], color="#4C78A8")
    ax.set_title("Graph discovery Combined Graph: Top PageRank Anime")
    ax.set_xlabel("PageRank")
    for bar in bars:
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {bar.get_width():.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "top_pagerank.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(np.log1p(metrics["members"].fillna(0)), metrics["pagerank"], alpha=0.35, s=12, color="#59A14F")
    ax.set_title("PageRank vs MAL Popularity")
    ax.set_xlabel("log(1 + members)")
    ax.set_ylabel("PageRank")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pagerank_vs_popularity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(summary_df["graph"], summary_df["largest_weak_component_pct"], color="#B07AA1")
    ax.set_title("Largest Weak Component Share by Graph Definition")
    ax.set_ylabel("Share of nodes")
    ax.set_ylim(0, 1)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "component_share.png", dpi=180)
    plt.close(fig)


def to_markdown(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in view.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(summary_df: pd.DataFrame, metrics: pd.DataFrame, comparison: pd.DataFrame, checks: pd.DataFrame) -> None:
    top_cols = ["mal_id", "title", "type", "pagerank", "weighted_in_degree", "in_degree", "members", "score"]
    report = f"""# Graph discovery Graph Analytics and Centrality Report

## Graph Definition

The graph grain is one anime catalog entry per node, keyed by MAL id. The graph is directed and weighted.

Recommendation edges point from a source anime to a target anime that users or source systems mark as similar. Their weights are recommendation vote/count signals. Relation edges also point from the source entry to the related target entry, with higher weights for prerequisite-style relations such as sequel/prequel/parent story and lower weights for looser relations such as summary or other.

Three graph definitions are regenerated from the same processed catalog:

- `recommendation`: similarity/recommendation edges only.
- `relation`: franchise/navigation edges only.
- `combined`: recommendation and relation edges together.

## Graph Report

{to_markdown(summary_df)}

The combined graph is the main Graph discovery graph because the recommender needs both types of structure: similarity edges help discovery, while relation edges prevent bad sequencing such as recommending a later season before the parent story.

## Centrality Results

{to_markdown(metrics[top_cols], max_rows=15)}

PageRank is interpreted as graph prestige, not anime quality. A high PageRank title is central because many weighted paths point toward it. This often overlaps with popularity, but it is not identical: a title can be popular without being a structural bridge, and a franchise hub can be central because many sequels, summaries, or similar shows point through it.

## Comparison To Non-Graph Ranking

{to_markdown(comparison)}

This comparison addresses the Graph discovery requirement to compare graph ranking against popularity or a model-based ranking. Popularity captures crowd scale; PageRank captures structural position in the anime relation/recommendation network. The personalized Recommender MyList hybrid, when present, is included only as a local profile comparison.

## Validity Checks

{to_markdown(checks)}

The main limitation is that graph centrality is not a personalized preference score. It is useful as a reranker, guardrail, and navigation layer, especially for franchise order and entry-point discovery. It should not replace collaborative ranking for users with enough interaction history.

## Reproducible Artifacts

- `artifacts/graph/combined_graph_metrics.csv`
- `artifacts/graph/graph_summary.csv`
- `artifacts/graph/graph_validity_checks.csv`
- `artifacts/graph/graph_ranking_comparison.csv`
- `artifacts/plots/graph/top_pagerank.png`
- `artifacts/plots/graph/pagerank_vs_popularity.png`
- `artifacts/plots/graph/component_share.png`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    anime_df = load_catalog()
    graphs = build_graphs(anime_df)

    summaries = pd.DataFrame([graph_summary(graph, name) for name, graph in graphs.items()])
    metrics_by_graph = {name: graph_metrics(graph, anime_df) for name, graph in graphs.items()}
    combined_metrics = metrics_by_graph["combined"]
    recommendation_metrics = metrics_by_graph["recommendation"]
    comparison = ranking_comparison(
        combined_metrics,
        "combined",
        BASE_DIR / "artifacts" / "recommendation" / "mylist_guarded_recommendation_example.csv",
    )
    checks = validity_checks(summaries, graphs["combined"], recommendation_metrics, combined_metrics)

    for name, metrics in metrics_by_graph.items():
        write_csv_atomic(metrics, GRAPH_DIR / f"{name}_graph_metrics.csv")
    write_csv_atomic(summaries, GRAPH_DIR / "graph_summary.csv")
    write_csv_atomic(comparison, GRAPH_DIR / "graph_ranking_comparison.csv")
    write_csv_atomic(checks, GRAPH_DIR / "graph_validity_checks.csv")
    write_csv_atomic(combined_metrics.head(100), GRAPH_DIR / "top_graph_titles.csv")

    save_plots(summaries, combined_metrics)
    write_report(summaries, combined_metrics, comparison, checks)

    summary_payload = {
        "graph_definition": "directed weighted anime graph with recommendation and relation edges",
        "main_graph": "combined",
        "outputs": {
            "summary": project_path(GRAPH_DIR / "graph_summary.csv"),
            "combined_metrics": project_path(GRAPH_DIR / "combined_graph_metrics.csv"),
            "comparison": project_path(GRAPH_DIR / "graph_ranking_comparison.csv"),
            "validity_checks": project_path(GRAPH_DIR / "graph_validity_checks.csv"),
            "report": project_path(REPORT_PATH),
        },
        "summaries": summaries.to_dict(orient="records"),
    }
    print(json.dumps(summary_payload, indent=2))
    print("Top combined-graph PageRank titles:")
    print(combined_metrics[["mal_id", "title", "pagerank", "weighted_in_degree", "members", "score"]].head(20))


if __name__ == "__main__":
    main()
