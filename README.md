# ResearchIQ — Academic Research Intelligence Engine

> An unsupervised data mining platform that transforms 1.9 million ArXiv papers
> into an interactive intelligence system for discovering knowledge patterns,
> research trends, and collaboration networks.

**An-Najah National University · Faculty of Information Technology and Artificial Intelligence**
**Course:** Data Mining 10672349 · **Supervisor:** Dr.-Ing. Ahmed Abualia

---

## Team

| Name | 
|---|
| Yazan Ashour |
| Mohammad Amad |
| Mohammad Ali Jabir |

---

## Project Overview

ResearchIQ answers questions that no researcher can answer manually:

- *What research topics are exploding right now?*
- *Which papers should I read next given my interests?*
- *Who are the most influential researchers bridging different fields?*
- *Where are the biggest knowledge gaps in the literature?*
- *Which concepts always appear together across thousands of papers?*

The system is fully **unsupervised** — no labels, no ground truth, no human-defined
categories are used as input at any stage. Every pattern is discovered from raw text.

---

 **Live App:**
 https://research-iq.streamlit.app/

## Live Demo



https://github.com/user-attachments/assets/4a626b2d-ae1f-4ff5-827f-2afdb6f47d80


---

## Project Structure

```
ResearchIQ/
├── data/
│   ├── prosecced/                      # Processed pipeline outputs
│   │   ├── embeddings.npy              # (40000 × 728) sentence embeddings
│   │   ├── paper_ids.npy               # Aligned paper ID array
│   │   ├── papers_clean.parquet        # Cleaned corpus — NO category labels
│   │   ├── papers_timeseries.parquet   # Full corpus for time series (1.9M)
│   │   └── ground_truth_labels.parquet # ArXiv categories — evaluation only
│   └── raw/
│       └── .gitkeep                    # Raw JSON not tracked (4 GB)
│
├── data-processing/
│   ├── load_and_clean.py               # Step 1 — stream, filter, clean ArXiv JSON
│   ├── embed.py                        # Step 2 — sentence-transformer embeddings
│   ├── load_for_timeseries.py          # Step 3a — full corpus loader for time series
│   └── probe.py                        # Raw dataset structural probe
│
├── evaluation/
│   ├── clustering_eval.py              # Silhouette, Davies-Bouldin, ARI
│   └── dimensionality_reduction.py     # UMAP quality evaluation
│
├── results/
│   ├── eval_results/                   # Evaluation metric outputs
│   ├── time_series/                    # 10 ARIMA figures + text report
│   ├── analysis_report.txt             # Full pipeline analysis report
│   ├── association_rules.csv           # 660 mined association rules
│   ├── cluster_distribution.png        # Cluster size visualisation
│   ├── clustering_results.json         # Cluster metrics and parameters
│   ├── dimensionality_reduction.json   # UMAP evaluation results
│   ├── papers_2d_projection.csv        # UMAP 2D coordinates (40000 papers)
│   ├── papers_with_clusters.parquet    # Clustered papers with all features
│   ├── pipeline_summary.json           # Full pipeline run summary
│   ├── recommendation_results.json     # Recommender system outputs
│   ├── reduced_embeddings_pca50.npy    # PCA 50D reduced embeddings
│   └── research_map_umap2d.png         # UMAP 2D knowledge map visualisation
│
├── app.py                              # Streamlit application entry point
├── recommendation_pipeline.py          # Full recommender + clustering pipeline
├── time_series.py                      # ARIMA time series analysis
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** [ArXiv Research Papers — Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)

The raw dataset (`arxiv-metadata-oai-snapshot.json`, ~4 GB) is not tracked in
this repository due to its size. Download it from Kaggle and place it at:

```
data/raw/arxiv-metadata-oai-snapshot.json
```

**Dataset facts:**

| Property | Value |
|---|---|
| Raw papers available | 3.4 million+ |
| Papers used (clustering) | 40000 |
| Papers used (time series) | 1.9 million |
| Fields covered | CS, Math, Physics, Astrophysics, HEP, Quantum, Condensed Matter, Statistics |
| Year range | 2015 – 2026 |
| Category labels | Saved separately — **never used as model input** |

The `categories` field from ArXiv is saved to `ground_truth_labels.parquet`
and used **only** for post-hoc evaluation (Adjusted Rand Index). It is never
fed into any unsupervised model at any stage of the pipeline.

---

## Setup

### Requirements

- Python 3.10+
- 8 GB RAM minimum (16 GB recommended for the full pipeline)
- GPU optional — Step 2 (embedding) runs significantly faster on Google Colab free GPU

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Yazan-Ashour/research-iq-datamining-project.git
cd research-iq-datamining-project

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

Run each step in order. Steps 1 and 2 only need to run once — all outputs are
saved to disk and reused by later steps.

### Step 1 — Load and clean

```bash
python data-processing/load_and_clean.py
```

Streams the ArXiv JSON file line by line, samples up to 5,000 papers per
category group for years 2015–2024, applies abstract length and year filters,
strips all category labels from the feature set, and saves two files:

- `data/prosecced/papers_clean.parquet` — clean corpus, no category labels
- `data/prosecced/ground_truth_labels.parquet` — labels saved separately for evaluation only

### Step 2 — Generate embeddings

```bash
python data-processing/embed.py
```

Encodes each paper's title and abstract using the `Alibaba-NLP/gte-modernbert-base`
sentence-transformer model (728 dimensions, L2-normalised).

> **Tip:** Run this step on Google Colab free GPU (~2 minutes) rather than CPU
> (~60 minutes). Upload `papers_clean.parquet`, run the script, then download
> `embeddings.npy` and `paper_ids.npy`.

Outputs:
- `data/prosecced/embeddings.npy` — (4000 × 728) float32 array
- `data/prosecced/paper_ids.npy` — aligned paper ID array

### Step 3 — Run the recommendation pipeline

```bash
python recommendation_pipeline.py
```

Executes the full unsupervised pipeline in one script:
UMAP dimensionality reduction → HDBSCAN clustering → recommender system →
association rule mining. Saves all results to `results/`.

### Step 4 — Time series analysis

```bash
python time_series.py
```

Loads 1.9 million papers from the full ArXiv corpus, applies year-stratified
resampling to correct for sampling bias, runs ARIMA forecasting per category,
IQR-based anomaly detection, and keyword trend analysis. Saves 10 figures and
a full text report to `results/time_series/`.

### Step 5 — Launch the app

```bash
streamlit run app.py
```

Opens the interactive ResearchIQ dashboard at `http://localhost:8501`.

---

## Data Mining Techniques

### Mandatory techniques

| Technique | Implementation | Result |
|---|---|---|
| **Clustering** | HDBSCAN on UMAP 5D projection of sentence embeddings | 5 clusters · silhouette 0.475 · Davies-Bouldin 0.644 |
| **Dimensionality reduction** | UMAP 5D for clustering · UMAP 2D for visualisation · PCA 50D intermediate | Adjusted Rand Index 0.470 vs ArXiv categories |

### Bonus techniques (all 4 implemented — requirement was ≥ 2)

| Technique | Implementation | Result |
|---|---|---|
| **Recommender system** | TF-IDF ranking + cosine similarity on embeddings + association rule boosting | Multi-strategy ranked results per query |
| **Time series analysis** | `auto_arima` per category on normalised monthly share · IQR anomaly detection | Overall MAPE 5.86% · 4 anomalies detected · 12-month forecast |
| **Association rule mining** | Apriori on keyword + author transactions · joint TF-IDF and Apriori grid search | 660 rules · highest lift 655 |

---

## Cluster Summary

Clusters are discovered fully unsupervised from paper embeddings.
Cluster names are derived automatically from the `paper_keywords` column
(top 3 most frequent terms per cluster). The ArXiv category field is never
used to guide or label the clustering.

| Cluster | Auto-derived name | Papers | Top keywords |
|---|---|---|---|
| 0 | High Energy Physics | 3,985 | higgs, quark, neutrino |
| 1 | CS / ML / Networks | 10,864 | network, learning, algorithm |
| 2 | Astrophysics | 6,395 | galaxies, stars, radio |
| 3 | Mathematics | 7,384 | algebra, group, equation |
| 4 | Quantum / Condensed Matter | 11,369 | quantum, entanglement, spin |

### Cluster evaluation

| Metric | Score | Interpretation |
|---|---|---|
| Silhouette score | **0.475** | Strong — above the 0.40 meaningful threshold |
| Davies-Bouldin index | **0.644** | Good — below 1.0 |
| Adjusted Rand Index | **0.470** | Expected < 1.0 — clusters are finer-grained than the 10 official ArXiv categories |

---

## Key Findings

**CS is dominating scientific publishing.**
Computer Science grew from 5% of ArXiv submissions in 2015 to nearly 50% by
2025 — a compound annual growth rate of +18% on normalised share. No other
field comes close.

**Mathematics is the only stationary field.**
Its ARIMA model selected d=0, meaning the series is stationary — the only
category whose relative share is not trending in either direction.

**October spikes, January dips — every year.**
Anomaly detection found 4 monthly outliers, all calendar-driven: October
submission surges before major conference deadlines, January slowdowns after
the holiday period.

**"Model" is now universal scientific vocabulary.**
The word "model" appears on average 1.3 times per abstract in 2025, up from
0.7 in 2015 — rising across every field, not just CS.

**Association rules reveal concept dependencies.**
Top rule: `monte → monte carlo` (lift 655, confidence 1.00).
More meaningfully: `quantum discord → discord` (lift 553) and
`holes → black holes` (lift 447) surface how subfield-specific
terminology clusters tightly in the literature.

**CS vocabulary is infiltrating every field.**
The words "model", "data", and "learning" are rising in astro-ph, math, and
physics abstracts — not just in CS — showing that CS methodology is spreading
across disciplines, not merely growing in isolation.

---

## App Pages

| Page | Description | Techniques demonstrated |
|---|---|---|
| **Knowledge Map** | Interactive UMAP scatter plot of 40000 papers as coloured dots with cluster filters and keyword search | Clustering · Dimensionality reduction |
| **Trend Explorer** | Annual publication trends, keyword heatmap, ARIMA forecast charts, anomaly markers | Time series · TF-IDF |
| **Paper Recommender** | Search by topic or author — returns ranked similar papers with similarity scores | Recommender system · Embeddings |
| **Association Rules** | 660 rules with confidence/lift scatter plot and interactive filters | Association rule mining |
| **Author Network** | Co-authorship force-directed graph with PageRank rankings and bridge author detection | Network analysis |

---

## Limitations

The following limitations, assumptions, and weaknesses are declared in
accordance with the project requirements.

- **English only.** The ArXiv corpus is English-language. Research published
  in other languages is not represented.

- **No citation data.** The raw ArXiv JSON does not include citation counts.
  The co-authorship network is built from author lists only, not citation links.

- **Cluster names are auto-generated.** Names are derived from top TF-IDF
  keywords and may be imprecise for cross-disciplinary clusters that span
  multiple fields.

- **Recommender cold start.** Papers not present in the 40000-paper clustered
  corpus cannot be recommended. Only papers with pre-computed embeddings are
  searchable.

- **2015 sampling bias in clustering corpus.** The clustering pipeline samples
  5,000 papers per category using oldest-first streaming, which over-represents
  2015. The time series analysis uses the full 1.9M corpus to avoid this.
  Cluster trend analysis uses in-memory year-stratified resampling.

- **Partial 2026 data.** The dataset includes papers up to mid-2026. All time
  series metrics flag 2026 as a partial year and exclude it from statistical
  calculations including anomaly detection and CAGR computation.

- **System capabilities are not overstated.** The recommender returns papers
  similar to the query based on text similarity — it does not model user
  preferences or reading history.

---

## AI Tool Declaration

In accordance with the project requirements, the team declares the following
use of AI tools during this project:

**Claude (Anthropic)** was used for project ideation, code scaffolding,
debugging assistance, analysis interpretation.

All data mining logic, model parameter selection, evaluation interpretation,
pipeline implementation, and final results were independently implemented,
reviewed, and verified by the team members. No AI-generated content was
submitted as analysis or results without independent verification.

---

## Requirements File

Key dependencies (see `requirements.txt` for full pinned versions):

```
pandas
numpy
pyarrow
sentence-transformers
umap-learn
hdbscan
scikit-learn
networkx
mlxtend
pmdarima
matplotlib
plotly
streamlit
```

---

## Acknowledgements

- [ArXiv](https://arxiv.org) and Cornell University for the open research dataset
- The `sentence-transformers`, `umap-learn`, `hdbscan`, `networkx`, `mlxtend`,
  `pmdarima`, and `streamlit` open-source communities
- Dr.-Ing. Ahmed Abualia for project supervision and guidance
- An-Najah National University, Faculty of Information Technology and Artificial Intelligence
