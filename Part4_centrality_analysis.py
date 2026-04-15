"""
Part 4: Static Network Analysis — Centrality-based identification of key users (subreddits).

Dataset: Reddit Hyperlink Network (SNAP)
    - soc-redditHyperlinks-title.tsv
    - soc-redditHyperlinks-body.tsv

Computes centrality measures (In-Degree, Out-Degree, Betweenness, PageRank, Eigenvector)
on the directed subreddit hyperlink graph, identifies the most influential subreddits,
and exports results for cross-validation with Part 5 (Information Diffusion).

Reference: Freeman, L.C. (1979). "Centrality in social networks: Conceptual clarification."
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

OUTPUT_DIR = "results/p4"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── 1. Load Reddit Hyperlink Network ────────────────────────────────────────

def load_reddit_graph():
    """
    Load both title and body hyperlink TSVs, merge into a single directed graph.
    Edge weight = number of hyperlinks between two subreddits.
    Also store average sentiment per edge.
    """
    frames = []
    for fname in ["dataset/soc-redditHyperlinks-title.tsv", "dataset/soc-redditHyperlinks-body.tsv"]:
        if os.path.exists(fname):
            df = pd.read_csv(fname, sep="\t", usecols=[0, 1, 3, 4],
                             names=["source", "target", "timestamp", "sentiment"],
                             header=0)
            frames.append(df)
            print(f"  Loaded {fname}: {len(df)} edges")
        else:
            print(f"  WARNING: {fname} not found, skipping")

    if not frames:
        raise FileNotFoundError("No Reddit hyperlink TSV files found in current directory")

    raw = pd.concat(frames, ignore_index=True)
    print(f"  Total raw edges (title + body): {len(raw)}")

    # Aggregate: count edges and average sentiment per (source, target) pair
    agg = raw.groupby(["source", "target"]).agg(
        weight=("sentiment", "count"),
        avg_sentiment=("sentiment", "mean")
    ).reset_index()

    G = nx.DiGraph()
    for _, row in agg.iterrows():
        G.add_edge(row["source"], row["target"],
                   weight=int(row["weight"]),
                   sentiment=round(row["avg_sentiment"], 4))

    print(f"\nConstructed directed graph: {G.number_of_nodes()} subreddits, "
          f"{G.number_of_edges()} unique directed edges")
    return G, raw


# ─── 2. Compute Centrality Measures ──────────────────────────────────────────

def compute_centralities(G):
    """
    Compute centrality measures appropriate for a directed network.
    - In-degree centrality: how many subreddits link TO this one (popularity/authority)
    - Out-degree centrality: how many subreddits this one links TO (activity/influence)
    - Betweenness centrality: bridge nodes connecting different communities
    - PageRank: recursive importance (accounts for who links to you)
    - Eigenvector centrality: importance based on connections to other important nodes
    """
    print("\nComputing centrality measures...")
    N = G.number_of_nodes()

    print("  • In-degree centrality")
    in_degree = nx.in_degree_centrality(G)

    print("  • Out-degree centrality")
    out_degree = nx.out_degree_centrality(G)

    print(f"  • Betweenness centrality (sampled k={min(500, N)} nodes)")
    betweenness = nx.betweenness_centrality(G, k=min(500, N), seed=42)

    print("  • PageRank (α=0.85)")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=300)

    print("  • Eigenvector centrality")
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
    except nx.PowerIterationFailedConvergence:
        print("    (Did not converge — using in-degree as proxy)")
        eigenvector = in_degree

    nodes = list(G.nodes())
    df = pd.DataFrame({
        "subreddit": nodes,
        "in_degree": [in_degree[n] for n in nodes],
        "out_degree": [out_degree[n] for n in nodes],
        "betweenness": [betweenness[n] for n in nodes],
        "pagerank": [pagerank[n] for n in nodes],
        "eigenvector": [eigenvector[n] for n in nodes],
    })
    return df


# ─── 3. Rank & Identify Top-K Influential Subreddits ─────────────────────────

def top_k_nodes(df, k=20):
    """Return top-k subreddits per centrality measure and a consensus ranking."""
    measures = ["in_degree", "out_degree", "betweenness", "pagerank", "eigenvector"]
    rankings = {}
    for col in measures:
        rankings[col] = df.nlargest(k, col)[["subreddit", col]].reset_index(drop=True)

    # Consensus: average of per-measure rank (lower rank = more central)
    for col in measures:
        df[f"rank_{col}"] = df[col].rank(ascending=False)
    rank_cols = [c for c in df.columns if c.startswith("rank_")]
    df["avg_rank"] = df[rank_cols].mean(axis=1)

    consensus = df.nsmallest(k, "avg_rank")[
        ["subreddit", "avg_rank"] + measures
    ].reset_index(drop=True)
    return rankings, consensus


# ─── 4. Correlation Analysis ─────────────────────────────────────────────────

def correlation_analysis(df):
    """Pairwise Spearman correlation between centrality measures."""
    cols = ["in_degree", "out_degree", "betweenness", "pagerank", "eigenvector"]
    corr = df[cols].corr(method="spearman")
    print("\nSpearman Correlation Matrix:")
    print(corr.round(3).to_string())
    return corr


# ─── 5. Visualization ────────────────────────────────────────────────────────

def plot_centrality_distributions(df):
    """Log-scale histogram of each centrality measure."""
    cols = ["in_degree", "out_degree", "betweenness", "pagerank", "eigenvector"]
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
    for ax, col in zip(axes, cols):
        vals = df[col]
        vals_pos = vals[vals > 0]
        ax.hist(vals_pos, bins=50, edgecolor="black", alpha=0.7, log=True)
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count (log)")
    plt.suptitle("Centrality Score Distributions (Reddit Hyperlink Network)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "centrality_distributions.png"), dpi=150)
    plt.close()
    print("Saved centrality_distributions.png")


def plot_correlation_heatmap(corr):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    labels = [c.replace("_", "\n") for c in corr.columns]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im)
    plt.title("Spearman Correlation of Centrality Measures")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "centrality_correlation.png"), dpi=150)
    plt.close()
    print("Saved centrality_correlation.png")


def plot_top_k_bar(consensus, k=20):
    """Horizontal bar chart of top-k subreddits by PageRank."""
    top = consensus.nlargest(k, "pagerank").sort_values("pagerank")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["subreddit"], top["pagerank"], color="#42A5F5")
    ax.set_xlabel("PageRank Score")
    ax.set_title(f"Top-{k} Most Central Subreddits (by PageRank)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_subreddits_pagerank.png"), dpi=150)
    plt.close()
    print("Saved top_subreddits_pagerank.png")


# ─── 6. Export for Part 5 ────────────────────────────────────────────────────

def export_for_diffusion(G, consensus, k=20):
    """Save graph and seed sets so Part 5 can directly consume them."""
    # Save edgelist (directed)
    nx.write_edgelist(G, os.path.join(OUTPUT_DIR, "reddit_graph.edgelist"), data=False)

    # Top-k central seeds (consensus ranking)
    seeds_top = consensus["subreddit"].tolist()[:k]

    # Random baseline seed set of same size
    rng = np.random.default_rng(42)
    all_nodes = list(G.nodes())
    seeds_random = rng.choice(all_nodes, size=k, replace=False).tolist()

    seed_data = {"top_central": seeds_top, "random_baseline": seeds_random}
    with open(os.path.join(OUTPUT_DIR, "seed_nodes.json"), "w") as f:
        json.dump(seed_data, f, indent=2)

    print(f"\nExported directed edgelist and seed nodes (top-{k} central + random baseline)")
    return seeds_top, seeds_random


# ─── 7. Basic Network Stats ──────────────────────────────────────────────────

def print_network_stats(G):
    print("\n=== Network Statistics ===")
    print(f"  Nodes (subreddits):     {G.number_of_nodes()}")
    print(f"  Edges (hyperlinks):     {G.number_of_edges()}")
    print(f"  Density:                {nx.density(G):.6f}")
    print(f"  Is weakly connected:    {nx.is_weakly_connected(G)}")
    wcc = list(nx.weakly_connected_components(G))
    print(f"  Weakly connected comp.: {len(wcc)}")
    largest_wcc = max(wcc, key=len)
    print(f"  Largest WCC size:       {len(largest_wcc)} "
          f"({100*len(largest_wcc)/G.number_of_nodes():.1f}%)")
    if nx.is_weakly_connected(G):
        # Average shortest path on a sample (full is too expensive)
        sub_nodes = list(largest_wcc)[:500]
        subG = G.subgraph(sub_nodes)
        if nx.is_weakly_connected(subG):
            avg_path = nx.average_shortest_path_length(subG.to_undirected())
            print(f"  Avg shortest path (sample 500): {avg_path:.2f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading Reddit Hyperlink Network...")
    G, raw_df = load_reddit_graph()
    print_network_stats(G)

    df = compute_centralities(G)
    df.to_csv(os.path.join(OUTPUT_DIR, "centrality_scores.csv"), index=False)
    print(f"\nSaved centrality scores for {len(df)} subreddits")

    rankings, consensus = top_k_nodes(df, k=20)

    print("\n=== Top-20 Subreddits (Consensus Ranking) ===")
    display_cols = ["subreddit", "avg_rank", "in_degree", "out_degree",
                    "betweenness", "pagerank", "eigenvector"]
    print(consensus[display_cols].to_string(index=False))

    print("\n--- Top-10 by Individual Measures ---")
    for measure, top_df in rankings.items():
        print(f"\n  {measure}:")
        for i, row in top_df.head(10).iterrows():
            print(f"    {row['subreddit']:30s}  {row[measure]:.6f}")

    corr = correlation_analysis(df)

    plot_centrality_distributions(df)
    plot_correlation_heatmap(corr)
    plot_top_k_bar(consensus, k=20)

    export_for_diffusion(G, consensus, k=20)

    print("\n✓ Part 4 complete. Results in ./results/p4")


if __name__ == "__main__":
    main()
