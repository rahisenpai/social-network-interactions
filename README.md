## Analysing and Predicting Interactions in Large Social Networks

Dataset: [Reddit Hyperlink Network](https://snap.stanford.edu/data/soc-RedditHyperlinks.html)


## 📊 Part 4: Centrality Analysis

**Goal:**  
Identify the most structurally important subreddits in a directed Reddit hyperlink network.

**Approach:**
- Construct a **directed graph** (~55K nodes, ~858K edges)
- Compute 5 centrality measures:
  - **In-degree** → popularity  
  - **Out-degree** → activity  
  - **Betweenness** → bridge roles  
  - **PageRank** → recursive importance  
  - **Eigenvector** → influence  

**Key Insights:**
- ⭐ Top hubs: *askreddit, iama, funny, gaming*
- 🔗 *subredditdrama* & *bestof* act as bridge/meta communities
- 📈 Strong correlation: **In-degree ↔ PageRank (ρ ≈ 0.96)**
- 📉 Out-degree is largely independent → captures a different role

**Outputs:**
- `centrality_scores.csv`
- Distribution plots and correlation heatmap
- Top subreddit rankings


---

## 🚀 Part 5: Information Diffusion

**Goal:**  
Simulate how information spreads across the network using structural properties.

**Models Used:**
- **Independent Cascade (IC):** probabilistic activation of out-neighbors  
- **Linear Threshold (LT):** activation based on influence from in-neighbors  

**Experiment Setup:**
- Compare:
  - 🔥 High-centrality seeds (from Part 4)
  - 🎲 Random seed baseline


### 📊 Key Results

| Model | Central Seeds | Random Seeds | Improvement |
|------|--------------|-------------|-------------|
| IC (p=0.01) | 302 | 29 | +942% |
| LT | 7247 | 53 | +13,583% |

➡️ High-centrality nodes significantly amplify diffusion.


### 🧠 Insights

- 🚀 **Out-degree** → best for maximizing spread
- 🎯 **Eigenvector centrality** → best predictor of activation
- 🔗 Network structure strongly influences diffusion behavior


### 📈 Cross-Validation

| Measure | Spearman ρ |
|--------|------------|
| Eigenvector | 0.810 |
| In-degree | 0.745 |
| PageRank | 0.709 |
| Betweenness | 0.626 |
| Out-degree | 0.293 |

➡️ Influence depends on *quality of connections*, not just quantity.


### 📦 Outputs

- Diffusion comparison plots (central vs random)
- Cascade growth curves
- Per-centrality spread analysis
- Centrality vs diffusion correlation plots


---

## ⚡ Takeaway

High-centrality subreddits are not just structurally important —  
they are **key drivers of large-scale information diffusion**.