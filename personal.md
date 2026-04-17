# Project Report: Static Centrality Analysis & Information Diffusion on the Reddit Hyperlink Network

---

## 1. Project Overview

This project analyzes the Reddit Hyperlink Network from SNAP (Stanford Network Analysis Project). The dataset captures how subreddits link to each other through posts — when a post in one subreddit contains a hyperlink to another subreddit, that creates a directed edge in our network.

We perform two complementary analyses:

- **Part 4 — Static Centrality Analysis:** We compute centrality measures to identify the most structurally important (influential) subreddits in the network.
- **Part 5 — Information Diffusion:** We simulate how information spreads through the network using two classical diffusion models, and we use the centrality results from Part 4 as seed nodes to validate that high-centrality nodes are indeed better at spreading information.

The core hypothesis is: **subreddits with higher centrality scores should play a disproportionately large role in information diffusion.** Our results strongly confirm this.

---

## 2. Dataset Description

| Property | Value |
|---|---|
| Source | SNAP: Reddit Hyperlink Network |
| Files | `soc-redditHyperlinks-title.tsv`, `soc-redditHyperlinks-body.tsv` |
| Total raw edges | ~858,488 hyperlinks |
| Unique directed edges | 339,643 |
| Nodes (subreddits) | 67,180 |
| Time span | January 2014 – April 2017 |
| Edge attributes | Timestamp, sentiment label (-1 negative, +1 neutral/positive), text property vector (86 LIWC + sentiment features) |
| Graph type | Directed, signed, temporal, attributed |

Each row in the TSV represents a hyperlink from a source subreddit to a target subreddit. We merged both files (title hyperlinks + body hyperlinks) and aggregated multiple edges between the same pair into a single weighted edge.

### Network Statistics
- Density: 0.000075 (very sparse — typical of real social networks)
- Weakly connected components: 712
- Largest weakly connected component: 65,648 nodes (97.7% of the network)
- The network is almost entirely connected, meaning most subreddits can reach most others through some chain of hyperlinks.

---

## 3. Code Structure Overview

### Part 4: `part4_centrality_analysis.py`

The script has 7 logical sections:

1. **`load_reddit_graph()`** — Reads both TSV files, parses source/target/sentiment columns, aggregates duplicate edges (counting occurrences as weight, averaging sentiment), and builds a NetworkX DiGraph.

2. **`compute_centralities(G)`** — Computes 5 centrality measures for every node in the graph. Each measure captures a different notion of "importance" (explained in detail in Section 4 below).

3. **`top_k_nodes(df, k=20)`** — Ranks all subreddits by each centrality measure individually, then computes a consensus ranking by averaging each node's rank across all 5 measures. A subreddit that ranks high on multiple measures gets a low average rank (= most important overall).

4. **`correlation_analysis(df)`** — Computes the Spearman rank correlation between all pairs of centrality measures. This tells us which measures agree with each other and which capture different structural properties.

5. **Visualization functions** — Generates histograms of centrality distributions, a correlation heatmap, and a bar chart of top subreddits.

6. **`export_for_diffusion()`** — Saves the graph as an edgelist and exports two seed sets (top-20 central nodes + 20 random nodes) as JSON, so Part 5 can consume them directly.

7. **`print_network_stats()`** — Reports basic graph properties (nodes, edges, density, connected components).

### Part 5: `part5_information_diffusion.py`

The script has 7 logical sections:

1. **`load_graph_and_seeds()`** — Loads the directed graph and seed node sets exported by Part 4.

2. **`independent_cascade(G, seeds, p)`** — Implements the IC diffusion model. Each newly activated node gets ONE chance to activate each of its out-neighbors with probability p. The process runs until no new activations occur.

3. **`linear_threshold(G, seeds)`** — Implements the LT diffusion model. Each node has a random threshold. A node activates when the weighted fraction of its active in-neighbors exceeds its threshold. Edge weight = 1/in_degree(node).

4. **`run_simulations()`** — Monte Carlo wrapper that runs a diffusion model N times and collects statistics (mean, std, min, max, full history of activations per time step).

5. **`per_centrality_comparison()`** — For each of the 5 centrality measures, selects the top-20 nodes as seeds and runs both IC and LT. This reveals which centrality measure produces the best seed set for maximizing spread.

6. **Visualization functions** — Box plots (central vs random spread), cascade growth curves (activated nodes over time), and bar charts (spread per centrality measure).

7. **`cross_validate_centrality_and_diffusion()`** — The key validation step. Runs 100 IC simulations with random seeds, counts how often each node gets activated, then correlates that activation frequency with each centrality score using Spearman correlation. High correlation = centrality predicts diffusion participation.

---

## 4. Part 4: Centrality Measures — What They Are and Why They Matter

### 4.1 In-Degree Centrality

**Formula:** `C_in(v) = (number of edges pointing TO v) / (N - 1)`

**What it means:** How many other subreddits link to this subreddit. A subreddit with high in-degree is "popular" — many communities reference it.

**Example from our results:** `askreddit` has the highest in-degree (0.081), meaning ~8.1% of all other subreddits have at least one hyperlink pointing to it. This makes sense — askreddit is one of the most referenced subreddits on the platform.

**Top 5 by in-degree:** askreddit, iama, pics, funny, videos

### 4.2 Out-Degree Centrality

**Formula:** `C_out(v) = (number of edges going FROM v) / (N - 1)`

**What it means:** How many other subreddits this subreddit links to. High out-degree = the subreddit actively references many other communities. These are often "meta" subreddits that curate or comment on content from elsewhere.

**Example from our results:** `bestof` has the highest out-degree (0.046), followed by `subredditdrama` (0.045). These are meta-subreddits whose entire purpose is to link to interesting content in other subreddits.

**Top 5 by out-degree:** bestof, subredditdrama, titlegore, drama, hailcorporate

**Key insight:** The subreddits with high in-degree (popular destinations) are completely different from those with high out-degree (active linkers). This is why out-degree has low correlation with other measures (ρ ~ 0.2).

### 4.3 Betweenness Centrality

**Formula:** `C_B(v) = Σ (σ_st(v) / σ_st)` for all pairs s,t where σ_st is the number of shortest paths from s to t, and σ_st(v) is the number of those paths passing through v.

**What it means:** How often a subreddit sits on the shortest path between two other subreddits. High betweenness = the node is a "bridge" connecting different parts of the network. If you removed it, communication between some communities would take longer paths.

**Example from our results:** `subredditdrama` has the highest betweenness (0.038). This makes perfect sense — it's a meta-subreddit that connects drama from many unrelated communities, acting as a bridge between otherwise disconnected clusters.

**Top 5 by betweenness:** subredditdrama, bestof, askreddit, iama, funny

**Note:** We used sampled betweenness (k=500 random source nodes) because exact betweenness on 67K nodes would be too slow. This is a standard approximation (Brandes, 2001).

### 4.4 PageRank

**Formula:** `PR(v) = (1-α)/N + α × Σ PR(u)/out_degree(u)` for all u that link to v. (α = 0.85 damping factor)

**What it means:** Originally designed by Google to rank web pages. A subreddit has high PageRank if it's linked to by other subreddits that themselves have high PageRank. It's a recursive measure of importance — being linked by important nodes matters more than being linked by obscure ones.

**Example from our results:** `askreddit` has the highest PageRank (0.022). Not only do many subreddits link to it (high in-degree), but the subreddits linking to it are themselves important.

**Top 5 by PageRank:** askreddit, iama, pics, funny, videos

**Correlation with in-degree:** Very high (ρ = 0.96). This is expected — in a network where most nodes have similar "quality" of incoming links, PageRank largely tracks in-degree.

### 4.5 Eigenvector Centrality

**Formula:** Based on the leading eigenvector of the adjacency matrix. `x_v = (1/λ) × Σ x_u` for all neighbors u of v.

**What it means:** Similar to PageRank but without the damping factor. A node is important if it's connected to other important nodes. It captures "prestige" — being in the inner circle of well-connected subreddits.

**Example from our results:** `askreddit` again tops the list (0.224), followed by `iama` (0.215). These subreddits are not just popular — they're connected to other popular subreddits, forming a tightly-knit core.

**Correlation with in-degree:** Very high (ρ = 0.95). With closeness to PageRank also very high (ρ = 0.93).

---

## 5. Part 4 Results: Image-by-Image Explanation

### 5.1 `centrality_distributions.png` — Centrality Score Distributions

**What it shows:** Five histograms (one per centrality measure) showing how centrality scores are distributed across all 67,180 subreddits. The y-axis is log-scaled.

**What to observe:**
- All distributions are heavily right-skewed (power-law-like). The vast majority of subreddits have very low centrality, while a tiny number have extremely high scores.
- This is characteristic of real-world social networks — they follow a "rich get richer" pattern (preferential attachment). A few hub subreddits dominate the network.
- The in-degree and PageRank distributions look very similar (because they're highly correlated at ρ = 0.96).
- Out-degree has a different shape because it captures a different structural role (active linkers vs popular destinations).

**Significance:** This power-law distribution confirms that the Reddit hyperlink network has a scale-free structure, consistent with the Barabási-Albert model of network growth. A small number of subreddits act as hubs.

### 5.2 `centrality_correlation.png` — Spearman Correlation Heatmap

**What it shows:** A 5×5 matrix showing how strongly each pair of centrality measures agrees on the ranking of subreddits. Values range from -1 (perfect disagreement) to +1 (perfect agreement).

**Key values:**
| Pair | ρ | Interpretation |
|---|---|---|
| In-degree ↔ PageRank | 0.96 | Almost identical rankings — being linked by many = being important |
| In-degree ↔ Eigenvector | 0.95 | Same story — popular subreddits are connected to other popular ones |
| PageRank ↔ Eigenvector | 0.93 | These three measures form a cluster capturing "popularity/prestige" |
| Betweenness ↔ In-degree | 0.67 | Moderate — bridges are somewhat popular but not always |
| Out-degree ↔ anything else | 0.18–0.29 | Very low — out-degree captures something fundamentally different |

**Significance:** This tells us there are essentially two independent structural roles in the Reddit network:
1. **Popular/prestigious hubs** (captured by in-degree, PageRank, eigenvector) — subreddits everyone links TO
2. **Active linkers/bridges** (captured by out-degree, partially betweenness) — subreddits that link OUT to many others

Betweenness sits in between because bridge nodes need both incoming and outgoing connections.

### 5.3 `top_subreddits_pagerank.png` — Top-20 Subreddits by PageRank

**What it shows:** A horizontal bar chart of the 20 most important subreddits ranked by PageRank score.

**What to observe:**
- `askreddit` dominates with PageRank = 0.022, followed by `iama` at 0.018
- The top subreddits are all major default/popular subreddits: askreddit, iama, pics, funny, videos, todayilearned, worldnews, gaming, news
- These are the "hubs" of Reddit — the subreddits that the rest of the platform revolves around
- The scores drop off sharply after the top 5, again reflecting the power-law distribution

**Significance:** PageRank identifies the subreddits that would cause the most disruption to information flow if removed. These are the structural backbone of the Reddit hyperlink network.

---

## 6. Part 5: Information Diffusion Models — What They Are and Why They Matter

### 6.1 Independent Cascade (IC) Model

**How it works:**
1. Start with a set of "seed" nodes (initially activated subreddits)
2. At each time step, every newly activated node gets ONE chance to activate each of its out-neighbors (subreddits it links to) with probability p = 0.01
3. If the activation attempt fails, that edge never gets another chance
4. The process continues until no new activations occur

**Analogy:** Think of it like a rumor spreading. If subreddit A links to subreddit B, there's a 1% chance that content/information from A reaches B. Each link only gets one shot.

**Why p = 0.01:** The Reddit network is sparse (density = 0.000075) but has high-degree hubs. A low propagation probability prevents unrealistic total saturation while still allowing meaningful cascades from well-connected seeds.

### 6.2 Linear Threshold (LT) Model

**How it works:**
1. Each node picks a random threshold θ ∈ [0, 1]
2. Start with seed nodes activated
3. At each time step, an inactive node v checks: what fraction of my in-neighbors (subreddits that link TO me) are already active?
4. If that fraction ≥ θ_v, node v activates
5. Edge weight = 1/in_degree(v), so each active in-neighbor contributes equally

**Analogy:** Think of it like peer pressure. A subreddit "adopts" information when enough of the subreddits linking to it have already adopted it. The threshold represents how resistant each subreddit is to influence.

**Key difference from IC:** In IC, each edge independently tries once. In LT, it's about cumulative social pressure — the more of your neighbors are active, the more likely you are to activate.

### 6.3 Why Two Models?

IC and LT capture different mechanisms of information spread:
- IC = **independent, probabilistic** spread (like a virus — each contact has a fixed chance)
- LT = **threshold-based, cumulative** spread (like social influence — you need enough peers to convince you)

Using both gives us confidence that our findings aren't an artifact of one particular model.

### 6.4 Monte Carlo Simulation

Both models involve randomness (IC: random activation attempts; LT: random thresholds). So we run each simulation 50 times and report the mean spread ± standard deviation. This gives us statistically robust results.

---

## 7. Part 5 Results: Image-by-Image Explanation

### 7.1 `diffusion_independent_cascade_comparison.png` — IC: Central vs Random Seeds

**What it shows:** A box plot comparing the total number of activated subreddits when seeding from the top-20 central subreddits vs 20 randomly chosen subreddits, under the Independent Cascade model.

**Key numbers:**
- Central seeds: mean spread = ~302 subreddits (0.4% of network)
- Random seeds: mean spread = ~29 subreddits (0.04% of network)
- Improvement: **+942%** — central seeds activate ~10x more subreddits

**What to observe:**
- The green box (central seeds) is dramatically higher than the orange box (random seeds)
- The central seeds box is also wider, showing more variance — this is because the cascade can sometimes hit a chain of well-connected nodes and spread further
- Random seeds barely spread at all because most random subreddits have very few outgoing links

**Significance:** Even with a conservative propagation probability (p=0.01), structurally central subreddits can trigger cascades that are an order of magnitude larger than random starting points.

### 7.2 `diffusion_linear_threshold_comparison.png` — LT: Central vs Random Seeds

**What it shows:** Same comparison but for the Linear Threshold model.

**Key numbers:**
- Central seeds: mean spread = ~7,247 subreddits (10.8% of network)
- Random seeds: mean spread = ~53 subreddits (0.1% of network)
- Improvement: **+13,583%** — central seeds activate ~137x more subreddits

**What to observe:**
- The difference is even more dramatic than IC
- Central seeds can activate over 10% of the entire network
- Random seeds barely move the needle

**Why LT shows bigger differences than IC:** In the LT model, activating a hub node (like askreddit) means that every subreddit that links to askreddit now has one more active in-neighbor pushing them toward their threshold. This creates a cascading chain reaction. In IC, each edge only gets one independent chance, so the hub advantage is less amplified.

**Significance:** This is the strongest evidence that centrality matters for diffusion. The LT model, which captures cumulative social influence, shows that starting from central nodes can trigger massive cascading adoption.

### 7.3 `diffusion_independent_cascade_cascade.png` — IC: Cascade Growth Over Time

**What it shows:** A line plot showing the average number of activated subreddits at each time step, for central seeds (green) vs random seeds (orange).

**What to observe:**
- Central seeds (green line) rise steeply in the first 2-3 time steps, then plateau
- Random seeds (orange line) barely rise at all
- Most of the spread happens in the first few steps — this is typical of IC where each edge only gets one chance
- The cascade "burns out" quickly because failed activation attempts are permanent

**Significance:** Information cascades in the IC model are fast and short-lived. The initial seed set determines almost everything — if you start from hubs, you get a big burst; if you start from periphery, the cascade dies immediately.

### 7.4 `diffusion_linear_threshold_cascade.png` — LT: Cascade Growth Over Time

**What it shows:** Same as above but for the Linear Threshold model.

**What to observe:**
- Central seeds (green line) show a more gradual, sustained growth over many time steps
- The curve keeps climbing for 5-10+ steps before plateauing
- Random seeds (orange line) plateau almost immediately

**Why LT cascades last longer:** In LT, even if a node doesn't activate in step 1, it might activate in step 5 when enough of its neighbors have accumulated. This creates a "snowball effect" that IC doesn't have.

**Significance:** The LT model reveals that central nodes don't just cause a one-time burst — they trigger sustained, growing cascades that build momentum over time.

### 7.5 `diffusion_per_centrality.png` — Spread by Centrality-Based Seed Selection

**What it shows:** A grouped bar chart comparing the mean spread when using the top-20 nodes from each centrality measure as seeds. Blue bars = IC model, purple bars = LT model.

**Key numbers:**

| Centrality | IC Mean Spread | LT Mean Spread |
|---|---|---|
| In-degree | 192 | 2,269 |
| Out-degree | 473 | 10,278 |
| Betweenness | 306 | 7,498 |
| PageRank | 206 | 2,749 |
| Eigenvector | 162 | 2,132 |

**What to observe:**
- **Out-degree seeds produce the largest cascades** in both models (473 IC, 10,278 LT)
- Betweenness is second-best (306 IC, 7,498 LT)
- In-degree, PageRank, and eigenvector produce similar, smaller cascades

**Why out-degree wins for seeding:** This is a crucial insight. Out-degree measures how many subreddits a node links TO — i.e., how many outgoing edges it has. In diffusion, information flows along outgoing edges. So a node with high out-degree has more "channels" to spread information through. Subreddits like `bestof` and `subredditdrama` link to hundreds of other subreddits, giving them maximum reach as seed nodes.

**Why in-degree/PageRank/eigenvector are weaker seeds:** These measure how many subreddits link TO the node. Being popular (many incoming links) doesn't help you spread information — you need outgoing links for that. `askreddit` is the most linked-to subreddit, but it doesn't link out to many others, so it's a poor seed for diffusion.

**Significance:** This reveals an important distinction: **the most "important" nodes (by popularity) are not necessarily the best seeds for information spread.** The best seeds are the most active linkers. This has practical implications for viral marketing, misinformation containment, etc.

### 7.6 `cross_validation_centrality_diffusion.png` — Centrality vs Activation Frequency

**What it shows:** Five scatter plots, one per centrality measure. X-axis = centrality score, Y-axis = how often that subreddit gets activated across 100 random IC simulations. The Spearman correlation ρ is shown in each title.

**Key correlations:**

| Measure | ρ (Spearman) | Interpretation |
|---|---|---|
| Eigenvector | 0.810 | Strong — prestigious nodes get activated most often |
| In-degree | 0.745 | Strong — popular nodes are frequently reached |
| PageRank | 0.709 | Strong — recursively important nodes are frequently reached |
| Betweenness | 0.626 | Moderate — bridge nodes are somewhat more reachable |
| Out-degree | 0.293 | Weak — active linkers aren't necessarily easy to reach |

**What to observe:**
- Eigenvector centrality has the highest correlation (ρ = 0.81) — nodes that are connected to other well-connected nodes are the most likely to be reached by any random cascade
- Out-degree has the lowest correlation (ρ = 0.29) — being an active linker doesn't make you easy to reach (it makes you good at spreading, not receiving)
- The scatter plots show a clear positive trend for in-degree, PageRank, and eigenvector, but a much noisier relationship for out-degree

**Important distinction from Section 7.5:** Section 7.5 asked "which nodes are best at SPREADING information?" (answer: out-degree). This section asks "which nodes are most likely to RECEIVE information?" (answer: eigenvector/in-degree). These are different questions with different answers.

**Significance:** This cross-validation confirms that centrality scores computed purely from network structure (Part 4) are strong predictors of dynamic diffusion behavior (Part 5). The network's static topology determines its dynamic information flow.

---

## 8. How Parts 4 and 5 Validate Each Other

The central claim of this project is that **static structural properties (centrality) predict dynamic behavior (diffusion)**. Here's how the results validate each other:

1. **Central seeds spread more:** Both IC (+942%) and LT (+13,583%) show that top-centrality nodes trigger dramatically larger cascades than random nodes. This directly confirms that centrality identifies structurally important nodes.

2. **Centrality correlates with activation frequency:** The cross-validation shows strong Spearman correlations (up to ρ = 0.81) between centrality scores and how often nodes participate in diffusion. Nodes that Part 4 identifies as "important" are the same nodes that Part 5 shows are most active in information spread.

3. **Different centralities capture different roles:**
   - Out-degree/betweenness → best for SEEDING spread (active linkers/bridges)
   - Eigenvector/in-degree/PageRank → best for PREDICTING who gets reached (popular/prestigious nodes)
   - This is consistent: nodes that link out a lot are good spreaders; nodes that are linked to a lot are easy to reach.

4. **The consensus ranking combines both roles:** The top-20 consensus list (which averages all 5 measures) includes both types — popular hubs like askreddit AND active linkers like subredditdrama. These are the nodes that are important from multiple structural perspectives.

---

## 9. Potential Presentation Q&A

**Q: Why did you choose the Reddit Hyperlink Network?**
A: It's a real-world directed social network with rich attributes (timestamps, sentiment, text features). The directed nature is important because information flows in a specific direction (from source to target subreddit). It's also large enough (67K nodes, 340K edges) to produce meaningful results but small enough to compute centrality measures in reasonable time.

**Q: Why did you use 5 different centrality measures instead of just one?**
A: Each measure captures a different notion of "importance." In-degree measures popularity, out-degree measures activity, betweenness measures bridging, PageRank measures recursive importance, and eigenvector measures prestige. Using all five gives a more complete picture. Our correlation analysis shows that some are redundant (in-degree ≈ PageRank at ρ=0.96) while others capture genuinely different properties (out-degree vs everything else at ρ~0.2).

**Q: Why is the propagation probability p=0.01 in the IC model?**
A: The network is sparse (density = 0.000075) but has high-degree hubs. A higher p would cause unrealistic total saturation from any seed set. p=0.01 produces meaningful but not trivial cascades, allowing us to distinguish between good and bad seed sets.

**Q: Why do out-degree seeds spread more than in-degree seeds?**
A: In diffusion models, information flows along outgoing edges. A node with high out-degree has more channels to push information through. High in-degree means many nodes link TO you, but that doesn't help you spread — it helps you receive. This is a key insight: popularity ≠ influence in directed networks.

**Q: Why does the LT model show much bigger differences than IC?**
A: LT has a cumulative "snowball" effect. When a hub activates, it pushes all its neighbors closer to their thresholds. This creates chain reactions that amplify the initial advantage of central seeds. IC doesn't have this — each edge independently succeeds or fails with no memory.

**Q: What does the cross-validation actually prove?**
A: It proves that centrality (a static, structural property computed without any simulation) can predict diffusion behavior (a dynamic process). Specifically, eigenvector centrality predicts activation frequency with ρ = 0.81. This means you can identify which subreddits will be most involved in information spread just by looking at the network structure, without running any simulations.

**Q: What are the practical applications?**
A: (1) Viral marketing: seed campaigns from high out-degree/betweenness nodes for maximum reach. (2) Misinformation containment: monitor high-centrality nodes as early warning systems. (3) Community health: subreddits with high betweenness (like subredditdrama) are bridges where conflict can spread between communities. (4) Platform design: understanding which subreddits are structural bottlenecks helps in content moderation and recommendation.

**Q: What is the Spearman correlation and why use it instead of Pearson?**
A: Spearman correlation measures the monotonic relationship between two variables based on their ranks, not their actual values. We use it because centrality distributions are heavily skewed (power-law), so Pearson correlation (which assumes linear relationships) would be misleading. Spearman is robust to non-linear but monotonic relationships.

**Q: Why Monte Carlo simulation? Why not just run the model once?**
A: Both IC and LT involve randomness (IC: random activation attempts; LT: random thresholds). A single run could be an outlier. By running 50 simulations and averaging, we get statistically robust estimates of the expected spread. The standard deviation tells us how variable the results are.

**Q: What is the consensus ranking and how is it computed?**
A: For each centrality measure, we rank all 67,180 subreddits from 1 (highest) to 67,180 (lowest). Then for each subreddit, we average its 5 ranks. A subreddit that ranks in the top 10 on all measures gets an average rank near 10. A subreddit that's #1 on one measure but #50,000 on another gets a mediocre average. This identifies nodes that are consistently important across multiple structural dimensions.

**Q: What libraries did you use?**
A: NetworkX for graph construction and centrality computation, NumPy for numerical operations, Pandas for data manipulation, and Matplotlib for visualization. The diffusion models (IC and LT) are implemented from scratch — no external diffusion library needed.

---

## 10. References

1. Freeman, L.C. (1979). "Centrality in social networks: Conceptual clarification." *Social Networks*, 1, 215–239.
2. Kempe, D., Kleinberg, J., & Tardos, É. (2003). "Maximizing the Spread of Influence through a Social Network." *KDD*.
3. Kumar, S., Hamilton, W.L., Leskovec, J., & Jurafsky, D. (2018). "Community Interaction and Conflict on the Web." *WWW*.
4. Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). "The PageRank Citation Ranking: Bringing Order to the Web." Stanford InfoLab.
5. Brandes, U. (2001). "A Faster Algorithm for Betweenness Centrality." *Journal of Mathematical Sociology*, 25(2), 163–177.