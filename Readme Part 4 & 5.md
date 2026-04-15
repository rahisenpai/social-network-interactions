# Social Network Analysis: Reddit Hyperlink Network

Structural analysis and diffusion modeling on the [Reddit Hyperlink Network](https://snap.stanford.edu/data/soc-RedditHyperlinks.html) (SNAP). The dataset contains ~858K directed hyperlinks between ~55K subreddits from Jan 2014 to April 2017.

## Project Structure

```
├── part4_centrality_analysis.py      # Static centrality analysis
├── part5_information_diffusion.py    # Information diffusion (IC & LT models)
├── soc-redditHyperlinks-title.tsv    # Dataset (title hyperlinks)
├── soc-redditHyperlinks-body.tsv     # Dataset (body hyperlinks)
├── requirements.txt
└── results/                          # Generated outputs (plots, CSVs, edgelists)
```

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset from [SNAP](https://snap.stanford.edu/data/soc-RedditHyperlinks.html) and place both TSV files in the project root.

## Part 4: Static Centrality Analysis

Computes five centrality measures on the directed subreddit graph:

| Measure | What it captures |
|---|---|
| In-degree | Popularity — how many subreddits link *to* this one |
| Out-degree | Activity — how many subreddits this one links *to* |
| Betweenness | Bridge role — sits on shortest paths between others |
| PageRank | Recursive importance — linked by other important nodes |
| Eigenvector | Influence — connected to other well-connected nodes |

```bash
python part4_centrality_analysis.py
```

### Key Findings

- Top subreddits by consensus ranking: askreddit, iama, funny, gaming, todayilearned
- subredditdrama and bestof dominate out-degree and betweenness (meta/bridge subreddits)
- In-degree ↔ PageRank correlation is very high (ρ = 0.96); out-degree is largely independent (ρ ~ 0.2)

## Part 5: Information Diffusion

Simulates information spread using two classical models seeded from Part 4's high-centrality nodes vs a random baseline:

- **Independent Cascade (IC):** each active node tries to activate each out-neighbor with probability *p*
- **Linear Threshold (LT):** a node activates when the fraction of its active in-neighbors exceeds a random threshold

```bash
python part5_information_diffusion.py
```

### Key Results

| Model | Central Seeds | Random Seeds | Improvement |
|---|---|---|---|
| IC (p=0.01) | 302 activated | 29 activated | +942% |
| LT | 7,247 activated | 53 activated | +13,583% |

**Per-centrality seed comparison (mean spread):**

| Centrality | IC | LT |
|---|---|---|
| In-degree | 170 | 2,269 |
| Out-degree | 472 | 10,278 |
| Betweenness | 306 | 7,498 |
| PageRank | 196 | 2,749 |
| Eigenvector | 186 | 2,132 |

Out-degree seeds produce the largest cascades (they have the most outgoing edges to propagate through).

### Cross-Validation

Spearman correlation between centrality scores and diffusion activation frequency:

| Measure | ρ |
|---|---|
| Eigenvector | 0.810 |
| In-degree | 0.745 |
| PageRank | 0.709 |
| Betweenness | 0.626 |
| Out-degree | 0.293 |

Eigenvector centrality is the strongest predictor of which subreddits get activated during diffusion.

## Generated Outputs

| File | Description |
|---|---|
| `centrality_scores.csv` | Centrality scores for all 67,180 subreddits |
| `diffusion_per_centrality.csv` | Spread statistics per centrality-based seed selection |
| `centrality_distributions.png` | Histograms of centrality score distributions |
| `centrality_correlation.png` | Spearman correlation heatmap |
| `top_subreddits_pagerank.png` | Top-20 subreddits by PageRank |
| `diffusion_*_comparison.png` | Box plots: central vs random seed spread |
| `diffusion_*_cascade.png` | Cascade growth curves over time |
| `diffusion_per_centrality.png` | Bar chart: spread by centrality measure |
| `cross_validation_centrality_diffusion.png` | Scatter plots: centrality vs activation frequency |

## References

- Freeman, L.C. (1979). "Centrality in social networks: Conceptual clarification." *Social Networks*, 1, 215–239.
- Kempe, D., Kleinberg, J., & Tardos, É. (2003). "Maximizing the Spread of Influence through a Social Network." *KDD*.
- Kumar, S., Hamilton, W.L., Leskovec, J., & Jurafsky, D. (2018). "Community Interaction and Conflict on the Web." *WWW*.
