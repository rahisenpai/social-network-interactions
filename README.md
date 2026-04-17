# Analysis & Prediction of Interactions in Large Social Networks

This project explores the structural and dynamic properties of the [Reddit Hyperlink Network](https://snap.stanford.edu/data/soc-RedditHyperlinks.html), a directed, signed, and temporal network representing connections between various subreddits over a 2.5-year period (January 2014 – April 2017). By leveraging network science principles, the study analyzes how these communities interact, cluster, and propagate information, while exploring classical and advanced models for link prediction.  
  
Exploratory Data Analysis can be found in `eda.ipynb`.

## Key Deliverables

More details about all the deliverables can be found in `report.pdf`.

### Classical Link Prediction
This section replicates foundational research by Liben-Nowell and Kleinberg to predict future edges based on network topology.
- Methodology: Evaluation of proximity heuristics like Common Neighbors, Jaccard’s Coefficient, and Adamic/Adar.
- Key Finding: Topological features effectively predict future links, with Adamic/Adar achieving a "Factor over Random" of 5.61x.

### Scaling Link Prediction with Node Embeddings
The project improves upon classical methods by representing subreddits as dense vectors in latent space.
- Methodology: Compared Logistic Regression (LR) using various edge feature constructions (Hadamard, Concatenation) against a Graph Convolutional Network (GCN).
- Key Finding: LR-Hadamard achieved the highest AUC of 0.9615, outperforming both structural baselines and the GCN.

### Community Detection and Analysis
Using the Louvain algorithm, the project identifies how subreddits naturally cluster into thematic groups.
- Results: Detected 789 communities with a modularity score of 0.4196, indicating strong structural organization.
- Sentiment Insights: Interactions within communities are 91.4% positive, compared to 88.7% for cross-community links, supporting the homophily hypothesis.

### Centrality Analysis
Identifies structurally significant subreddits using multiple centrality measures.
- Key Metrics: In-degree (popularity), out-degree (activity), betweenness (bridge role), PageRank, and eigenvector centrality.
- Findings: A small number of hubs like askreddit and iama dominate the network. Correlation analysis reveals that popular nodes (high in-degree) differ from bridge nodes (high out-degree).

### Information Diffusion Modeling
Simulates how information spreads using Independent Cascade (IC) and Linear Threshold (LT) models.
- Key Insight: High-centrality seeds produce significantly larger cascades than random seeds (up to +13,583% in the LT model).
- Validation: Confirms that static network structure directly predicts dynamic diffusion behavior.

### Temporal Analysis and Prediction with TGN
Implements a Temporal Graph Network (TGN) to process continuous-time events and evolving graph structures.
- Performance: The TGN achieved an impressive AUC-ROC of 0.9766 and Avg-Prec of 0.9756.
- Longitudinal Study: Over time, the network saw an 80% increase in nodes, though overall density decreased, suggesting the network becomes increasingly sparse as it scales.
