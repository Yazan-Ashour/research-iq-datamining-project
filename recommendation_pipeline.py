from __future__ import annotations

import ast
import json
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, Normalizer, StandardScaler

try:
    import umap
except ImportError as exc:
    raise ImportError("Install UMAP with: pip install umap-learn") from exc

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
PAPER_IDS_PATH = DATA_DIR / "paper_ids.npy"
PAPERS_PATH = DATA_DIR / "papers_clean.parquet"

RANDOM_STATE = 42

KMEANS_PARAMS = {
    "n_clusters": 5,
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 300,
    "random_state": RANDOM_STATE,
}

PCA_N_COMPONENTS = 50

UMAP_PARAMS = {
    "n_components": 2,
    "n_neighbors": 15,
    "min_dist": 0.0,
    "metric": "euclidean",
    "random_state": RANDOM_STATE,
    "low_memory": True,
}

TFIDF_MAX_FEATURES = 80_000
KEYWORD_MAX_FEATURES = 8_000

TOP_K_SEARCH = 5
TOP_K_RECS = 10

MAX_KEYWORDS_PER_PAPER = 4
MAX_QUERY_KEYWORDS = 6
MAX_TRANSACTION_AUTHORS = 5

MIN_SUPPORT = 0.001
MIN_CONFIDENCE = 0.05
MAX_APRIORI_ITEMSET_LEN = 2

TEXT_W = 0.60
AUTHOR_W = 0.25
CLUSTER_W = 0.15

ASSOC_AUTHOR_W = 0.25
ASSOC_CLUSTER_W = 0.20
ASSOC_KEYWORD_W = 0.30
ASSOC_EMBED_W = 0.25

MIN_ASSOC_EMBED_SIM = 0.35


@dataclass
class SearchContext:
    """Search input. query_text can be topic, paper title, keywords, or title + topic."""

    query_text: str | None = None
    author_query: str | None = None
    cluster_query: int | None = None
    top_k: int = TOP_K_SEARCH

def save_json(obj: dict, path: Path) -> None:
    """Save dictionary as formatted JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_value(value: Any) -> Any:
    """Convert NumPy/Pandas values to JSON-safe Python values."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): safe_value(v) for k, v in value.items()}
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def records(df: pd.DataFrame, score_cols: list[str] | None = None) -> list[dict]:
    """Convert paper DataFrame rows to clean JSON records."""
    if df is None or df.empty:
        return []

    base_cols = [
        "paper_id",
        "title_clean",
        "authors_list",
        "paper_keywords",
        "year",
        "cluster_id",
        "doi",
        "journal_ref",
    ]
    cols = [c for c in base_cols + (score_cols or []) if c in df.columns]

    return [{c: safe_value(row[c]) for c in cols} for _, row in df[cols].iterrows()]


def rule_records(df: pd.DataFrame) -> list[dict]:
    """Convert matched rules to clean JSON records."""
    if df is None or df.empty:
        return []

    output = []
    for _, row in df.iterrows():
        output.append(
            {
                "antecedents": sorted(list(row["antecedents"])),
                "consequents": sorted(list(row["consequents"])),
                "rule_type": row.get("rule_type"),
                "support": float(row["support"]),
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"]),
                "context_rule_score": float(row.get("context_rule_score", 0.0)),
            }
        )
    return output

def parse_authors(value: Any) -> list[str]:
    """Parse authors_list values from list/array/string to a list of names."""
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, np.ndarray):
        return [str(x).strip() for x in value.tolist() if str(x).strip()]
    if isinstance(value, (tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set, np.ndarray)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    return [x.strip() for x in re.split(r";|,", text) if x.strip()]


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load and align paper metadata with embeddings."""
    print("=" * 90)
    print("Loading processed data")
    print("=" * 90)

    for path in [EMBEDDINGS_PATH, PAPER_IDS_PATH, PAPERS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False).astype("float32")
    paper_ids = np.load(PAPER_IDS_PATH, allow_pickle=True).astype(str)
    papers = pd.read_parquet(PAPERS_PATH)

    if embeddings.shape[0] != len(paper_ids):
        raise ValueError("embeddings.npy and paper_ids.npy row counts do not match.")

    if "paper_id" not in papers.columns:
        raise ValueError("papers_clean.parquet must contain paper_id.")

    order = pd.DataFrame({"paper_id": paper_ids, "embedding_index": np.arange(len(paper_ids))})
    papers["paper_id"] = papers["paper_id"].astype(str)
    papers = order.merge(papers, on="paper_id", how="left")

    for col in ["title_clean", "abstract_clean", "text_for_embedding", "doi", "journal_ref"]:
        if col not in papers.columns:
            papers[col] = ""
        papers[col] = papers[col].fillna("").astype(str)

    if "authors_list" not in papers.columns:
        papers["authors_list"] = [[] for _ in range(len(papers))]
    if "year" not in papers.columns:
        papers["year"] = None

    papers["authors_list"] = papers["authors_list"].apply(parse_authors)
    papers["authors_text"] = papers["authors_list"].apply(lambda xs: " ; ".join(xs))
    papers["search_text"] = (
        papers["title_clean"] + " " + papers["abstract_clean"] + " " + papers["authors_text"]
    ).str.lower()

    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.linalg.norm(embeddings, axis=1) > 0
    removed = int(np.sum(~valid))

    papers = papers.loc[valid].reset_index(drop=True)
    embeddings = embeddings[valid].astype("float32")

    print(f"Papers: {papers.shape}")
    print(f"Embeddings: {embeddings.shape}")
    print(f"Removed zero-vector embeddings: {removed}")

    return papers, embeddings


def preprocess_embeddings(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize embeddings and create normalized vectors for similarity."""
    print("\n" + "=" * 90)
    print("Preprocessing embeddings")
    print("=" * 90)

    x_scaled = StandardScaler().fit_transform(embeddings).astype("float32")
    x_similarity = Normalizer(norm="l2").fit_transform(x_scaled).astype("float32")

    print(f"X_scaled: {x_scaled.shape}")
    print(f"X_similarity: {x_similarity.shape}")

    return x_similarity, x_scaled

def run_clustering(papers: pd.DataFrame, x_similarity: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Apply K-Means clustering to all papers."""
    print("\n" + "=" * 90)
    print("Applying final K-Means clustering")
    print("=" * 90)

    model = KMeans(**KMEANS_PARAMS)

    start = time.time()
    labels = model.fit_predict(x_similarity)
    runtime = time.time() - start

    papers = papers.copy()
    papers["cluster_id"] = labels.astype(int)
    papers["is_noise"] = False

    sample_silhouette = None
    try:
        n = min(5000, x_similarity.shape[0])
        idx = np.random.default_rng(RANDOM_STATE).choice(x_similarity.shape[0], size=n, replace=False)
        sample_silhouette = float(silhouette_score(x_similarity[idx], labels[idx]))
    except Exception:
        pass

    result = {
        "method": "K-Means",
        "parameters": {k: v for k, v in KMEANS_PARAMS.items() if k != "random_state"},
        "runtime_seconds": runtime,
        "inertia_sse": float(model.inertia_),
        "silhouette_score_sample": sample_silhouette,
        "cluster_counts": {
            str(k): int(v) for k, v in papers["cluster_id"].value_counts().sort_index().to_dict().items()
        },
        "selection_reason": (
            "DBSCAN was not selected because it produced a very high noise ratio. "
            "K-Means assigns every paper to a usable topic cluster."
        ),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return papers, result


def run_dimensionality_reduction(
    papers: pd.DataFrame,
    x_scaled: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Run PCA_SVD for compressed vectors and UMAP for 2D visualization."""
    print("\n" + "=" * 90)
    print("Applying PCA_SVD and UMAP")
    print("=" * 90)

    pca = PCA(
        n_components=PCA_N_COMPONENTS,
        svd_solver="randomized",
        iterated_power=7,
        random_state=RANDOM_STATE,
    )

    start = time.time()
    reduced = pca.fit_transform(x_scaled).astype("float32")
    pca_time = time.time() - start

    reducer = umap.UMAP(**UMAP_PARAMS)

    start = time.time()
    projection = reducer.fit_transform(x_scaled)
    umap_time = time.time() - start

    projection_df = pd.DataFrame(
        {
            "paper_id": papers["paper_id"].astype(str).values,
            "cluster_id": papers["cluster_id"].astype(int).values,
            "umap_x": projection[:, 0],
            "umap_y": projection[:, 1],
        }
    )

    result = {
        "compression": {
            "method": "PCA_SVD",
            "n_components": PCA_N_COMPONENTS,
            "cumulative_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
            "runtime_seconds": pca_time,
        },
        "visualization": {
            "method": "UMAP",
            "parameters": {k: v for k, v in UMAP_PARAMS.items() if k != "random_state"},
            "runtime_seconds": umap_time,
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return reduced, projection_df, result

def clean_keyword(phrase: str) -> str | None:
    """Normalize TF-IDF keyword phrase."""
    phrase = phrase.lower().strip()
    phrase = re.sub(r"[^a-z0-9\s_-]+", " ", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()

    if not phrase:
        return None

    tokens = [t for t in phrase.split() if len(t) >= 3]
    if not tokens:
        return None
    if len(tokens) == 1 and tokens[0] in ENGLISH_STOP_WORDS:
        return None

    return "_".join(tokens)


def extract_keywords_from_row(feature_names: np.ndarray, row: Any, top_n: int) -> list[str]:
    """Extract top keywords from one TF-IDF sparse row."""
    if row.nnz == 0:
        return []

    coo = row.tocoo()
    top = np.argsort(coo.data)[::-1][:top_n]

    keywords = []
    for idx in top:
        cleaned = clean_keyword(feature_names[coo.col[idx]])
        if cleaned:
            keywords.append(cleaned)

    return list(dict.fromkeys(keywords))


def add_paper_keywords(papers: pd.DataFrame) -> tuple[pd.DataFrame, TfidfVectorizer]:
    """Build keyword TF-IDF model and add paper_keywords column."""
    print("\n" + "=" * 90)
    print("Building keyword extractor and extracting paper keywords")
    print("=" * 90)

    vectorizer = TfidfVectorizer(
        max_features=KEYWORD_MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.60,
    )

    matrix = vectorizer.fit_transform(papers["search_text"].fillna(""))
    feature_names = np.array(vectorizer.get_feature_names_out())

    print(f"Keyword matrix: {matrix.shape}")

    all_keywords = []
    n_rows = matrix.shape[0]

    for i in range(n_rows):
        all_keywords.append(
            extract_keywords_from_row(feature_names, matrix.getrow(i), MAX_KEYWORDS_PER_PAPER)
        )

        if (i + 1) % 2000 == 0 or (i + 1) == n_rows:
            print(f"Extracted keywords for {i + 1:,} / {n_rows:,} papers")

    papers = papers.copy()
    papers["paper_keywords"] = all_keywords

    print("Example keywords:")
    print(papers[["paper_id", "title_clean", "paper_keywords"]].head(3).to_string(index=False))

    return papers, vectorizer


def extract_query_keywords(query: str | None, keyword_vectorizer: TfidfVectorizer) -> list[str]:
    """Extract keywords from user query."""
    if not query:
        return []

    row = keyword_vectorizer.transform([query.lower()])
    feature_names = np.array(keyword_vectorizer.get_feature_names_out())

    return extract_keywords_from_row(feature_names, row, MAX_QUERY_KEYWORDS)


def build_transactions(papers: pd.DataFrame) -> list[list[str]]:
    """Build Apriori transactions from clusters, keywords, and authors."""
    transactions = []

    for _, row in papers.iterrows():
        items = [f"cluster:{int(row['cluster_id'])}"]

        for keyword in row.get("paper_keywords", []):
            if keyword:
                items.append(f"keyword:{keyword}")

        for author in row["authors_list"][:MAX_TRANSACTION_AUTHORS]:
            author = str(author).strip()
            if author:
                items.append(f"author:{author}")

        items = list(dict.fromkeys(items))

        if len(items) >= 2:
            transactions.append(items)

    return transactions


def classify_rule(row: pd.Series) -> str:
    """Classify association rule type."""
    ant = set(row["antecedents"])
    con = set(row["consequents"])

    ant_author = any(x.startswith("author:") for x in ant)
    ant_cluster = any(x.startswith("cluster:") for x in ant)
    ant_keyword = any(x.startswith("keyword:") for x in ant)

    con_author = any(x.startswith("author:") for x in con)
    con_cluster = any(x.startswith("cluster:") for x in con)
    con_keyword = any(x.startswith("keyword:") for x in con)

    if ant_keyword and con_cluster:
        return "keyword_to_topic"
    if ant_cluster and con_keyword:
        return "topic_to_keyword"
    if ant_keyword and con_author:
        return "keyword_to_author"
    if ant_author and con_keyword:
        return "author_to_keyword"
    if ant_author and con_cluster:
        return "author_to_topic"
    if ant_cluster and con_author:
        return "topic_to_author"
    if ant_author and con_author:
        return "author_to_author"
    if ant_keyword and con_keyword:
        return "keyword_to_keyword"
    if ant_cluster and con_cluster:
        return "topic_to_topic"

    return "mixed_rule"


def generate_rules(papers: pd.DataFrame) -> pd.DataFrame:
    """Generate memory-safe Association Rules."""
    print("\n" + "=" * 90)
    print("Generating Association Rules")
    print("=" * 90)

    transactions = build_transactions(papers)
    encoder = MultiLabelBinarizer()
    basket = pd.DataFrame(
        encoder.fit_transform(transactions).astype(bool),
        columns=encoder.classes_,
    )

    print(f"Transactions: {len(transactions)}")
    print(f"Basket shape: {basket.shape}")
    print(f"min_support={MIN_SUPPORT}, min_confidence={MIN_CONFIDENCE}, max_len={MAX_APRIORI_ITEMSET_LEN}")

    itemsets = apriori(
        basket,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_APRIORI_ITEMSET_LEN,
        low_memory=True,
    )

    if itemsets.empty:
        print("No frequent itemsets found.")
        return pd.DataFrame()

    rules = association_rules(itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
    if rules.empty:
        print("No association rules found.")
        return pd.DataFrame()

    rules = rules[rules["consequents"].apply(lambda x: len(x) == 1)].copy()
    if rules.empty:
        print("No rules after filtering to one consequent.")
        return pd.DataFrame()

    rules["antecedents_text"] = rules["antecedents"].apply(lambda x: sorted(list(x)))
    rules["consequents_text"] = rules["consequents"].apply(lambda x: sorted(list(x)))
    rules["rule_type"] = rules.apply(classify_rule, axis=1)

    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)

    print(f"Frequent itemsets: {len(itemsets)}")
    print(f"Generated rules: {len(rules)}")
    print(
        rules[["antecedents_text", "consequents_text", "support", "confidence", "lift", "rule_type"]]
        .head(10)
        .to_string(index=False)
    )

    return rules

def build_search_index(papers: pd.DataFrame) -> tuple[TfidfVectorizer, Any]:
    """Build TF-IDF page ranking index."""
    print("\n" + "=" * 90)
    print("Building search index")
    print("=" * 90)

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    matrix = vectorizer.fit_transform(papers["search_text"].fillna(""))
    print(f"Search matrix: {matrix.shape}")

    return vectorizer, matrix


def author_score(authors: list[str], query: str | None) -> float:
    """Score how well a paper's authors match the author query."""
    if not query:
        return 0.0

    q = query.lower().strip()
    if not q:
        return 0.0

    for author in authors:
        a = author.lower().strip()
        if q == a:
            return 1.0
        if q in a or a in q:
            return 0.8

    return 0.0


def rank_results(
    context: SearchContext,
    papers: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix: Any,
) -> pd.DataFrame:
    """Return top ranked direct search results."""
    query_vec = vectorizer.transform([(context.query_text or "").lower()])
    text_scores = cosine_similarity(query_vec, matrix).ravel()

    a_scores = papers["authors_list"].apply(lambda xs: author_score(xs, context.author_query)).astype(float).values

    if context.cluster_query is None:
        c_scores = np.zeros(len(papers), dtype=float)
    else:
        c_scores = (papers["cluster_id"].astype(int).values == int(context.cluster_query)).astype(float)

    final = TEXT_W * text_scores + AUTHOR_W * a_scores + CLUSTER_W * c_scores

    ranked = papers.copy()
    ranked["search_text_score"] = text_scores
    ranked["author_match_score"] = a_scores
    ranked["cluster_match_score"] = c_scores
    ranked["page_rank_score"] = final

    return (
        ranked.sort_values(["page_rank_score", "author_match_score", "search_text_score"], ascending=False)
        .head(context.top_k)
        .reset_index(drop=True)
    )


def mean_seed_similarity(top_results: pd.DataFrame, papers: pd.DataFrame, x_similarity: np.ndarray) -> np.ndarray:
    """Cosine similarity between all papers and the average vector of top results."""
    seed_ids = set(top_results["paper_id"].astype(str))
    seed_idx = papers.index[papers["paper_id"].astype(str).isin(seed_ids)].tolist()

    if not seed_idx:
        return np.zeros(len(papers), dtype=float)

    seed_vec = x_similarity[seed_idx].mean(axis=0, keepdims=True)
    return cosine_similarity(seed_vec, x_similarity).ravel()


def normal_recommendations(
    context: SearchContext,
    top_results: pd.DataFrame,
    papers: pd.DataFrame,
    x_similarity: np.ndarray,
) -> dict:
    """Create normal recommendations: embedding, same cluster, same author."""
    seed_ids = set(top_results["paper_id"].astype(str))
    sim = mean_seed_similarity(top_results, papers, x_similarity)

    all_papers = papers.copy()
    all_papers["embedding_similarity_score"] = sim

    paper_to_paper = (
        all_papers[~all_papers["paper_id"].astype(str).isin(seed_ids)]
        .sort_values("embedding_similarity_score", ascending=False)
        .head(TOP_K_RECS)
        .reset_index(drop=True)
    )

    seed_clusters = set(top_results["cluster_id"].astype(int).tolist())
    same_cluster = (
        all_papers[
            all_papers["cluster_id"].astype(int).isin(seed_clusters)
            & (~all_papers["paper_id"].astype(str).isin(seed_ids))
        ]
        .rename(columns={"embedding_similarity_score": "same_cluster_similarity"})
        .sort_values("same_cluster_similarity", ascending=False)
        .head(TOP_K_RECS)
        .reset_index(drop=True)
    )

    target_authors = set()
    if context.author_query:
        target_authors.add(context.author_query.lower().strip())
    for authors in top_results["authors_list"]:
        target_authors.update(a.lower().strip() for a in authors)

    same_author = pd.DataFrame()
    if target_authors:
        tmp = papers.copy()

        def score(xs: list[str]) -> float:
            return float(len({a.lower().strip() for a in xs}.intersection(target_authors)))

        tmp["same_author_score"] = tmp["authors_list"].apply(score)
        same_author = (
            tmp[
                (tmp["same_author_score"] > 0)
                & (~tmp["paper_id"].astype(str).isin(seed_ids))
            ]
            .sort_values(["same_author_score", "year"], ascending=[False, False])
            .head(TOP_K_RECS)
            .reset_index(drop=True)
        )

    return {
        "paper_to_paper_embedding_similarity": paper_to_paper,
        "same_cluster_topic_papers": same_cluster,
        "same_author_papers": same_author,
    }


def context_items(
    context: SearchContext,
    top_results: pd.DataFrame,
    keyword_vectorizer: TfidfVectorizer,
) -> set[str]:
    """Build association-rule context items."""
    items = set()

    for kw in extract_query_keywords(context.query_text, keyword_vectorizer):
        items.add(f"keyword:{kw}")

    if context.author_query:
        items.add(f"author:{context.author_query}")

    if context.cluster_query is not None:
        items.add(f"cluster:{int(context.cluster_query)}")

    for _, paper in top_results.iterrows():
        items.add(f"cluster:{int(paper['cluster_id'])}")
        for kw in paper.get("paper_keywords", []):
            items.add(f"keyword:{kw}")
        for author in paper["authors_list"]:
            items.add(f"author:{author}")

    return items


def match_rules(
    context: SearchContext,
    top_results: pd.DataFrame,
    rules: pd.DataFrame,
    keyword_vectorizer: TfidfVectorizer,
) -> pd.DataFrame:
    """Find association rules relevant to the current search."""
    if rules.empty:
        return pd.DataFrame()

    items = context_items(context, top_results, keyword_vectorizer)

    def score(row: pd.Series) -> float:
        overlap = len(set(row["antecedents"]).intersection(items))
        if overlap == 0:
            return 0.0
        return float(overlap) * float(row["lift"]) * float(row["confidence"]) * (1.0 + float(row["support"]))

    matched = rules.copy()
    matched["context_rule_score"] = matched.apply(score, axis=1)
    matched = matched[matched["context_rule_score"] > 0]

    return (
        matched.sort_values(["context_rule_score", "lift", "confidence"], ascending=False)
        .head(TOP_K_RECS)
        .reset_index(drop=True)
    )


def extract_targets(matched_rules: pd.DataFrame) -> tuple[list[str], list[int], list[str]]:
    """Extract author, cluster, and keyword targets from matched rules."""
    authors, clusters, keywords = [], [], []

    for consequents in matched_rules.get("consequents", []):
        for item in consequents:
            if item.startswith("author:"):
                authors.append(item.replace("author:", "", 1))
            elif item.startswith("cluster:"):
                try:
                    clusters.append(int(item.replace("cluster:", "", 1)))
                except ValueError:
                    pass
            elif item.startswith("keyword:"):
                keywords.append(item.replace("keyword:", "", 1))

    return list(dict.fromkeys(authors)), list(dict.fromkeys(clusters)), list(dict.fromkeys(keywords))


def association_recommendations(
    matched_rules: pd.DataFrame,
    top_results: pd.DataFrame,
    papers: pd.DataFrame,
    x_similarity: np.ndarray,
) -> tuple[pd.DataFrame, list[str], list[int], list[str]]:
    """
    Recommend papers from Association Rules.

    Quality fix:
    - A candidate is NOT accepted only because it shares a broad cluster.
    - It must have at least one keyword match, author match, or strong semantic similarity.
    """
    rec_authors, rec_clusters, rec_keywords = extract_targets(matched_rules)

    if not rec_authors and not rec_clusters and not rec_keywords:
        return pd.DataFrame(), [], [], []

    seed_ids = set(top_results["paper_id"].astype(str))
    sim = mean_seed_similarity(top_results, papers, x_similarity)

    author_set = {a.lower().strip() for a in rec_authors}
    cluster_set = set(rec_clusters)
    keyword_set = {k.lower().strip() for k in rec_keywords}

    candidates = papers.copy()
    candidates["association_embedding_similarity"] = sim

    def author_match(xs: list[str]) -> float:
        return float(len({a.lower().strip() for a in xs}.intersection(author_set)))

    def keyword_match(xs: list[str]) -> float:
        return float(len({k.lower().strip() for k in xs}.intersection(keyword_set)))

    candidates["association_author_score"] = candidates["authors_list"].apply(author_match)
    candidates["association_cluster_score"] = candidates["cluster_id"].apply(
        lambda c: 1.0 if int(c) in cluster_set else 0.0
    )
    candidates["association_keyword_score"] = candidates["paper_keywords"].apply(keyword_match)

    quality_mask = (
        (candidates["association_author_score"] > 0)
        | (candidates["association_keyword_score"] > 0)
        | (candidates["association_embedding_similarity"] >= MIN_ASSOC_EMBED_SIM)
    )

    candidates["association_recommendation_score"] = (
        ASSOC_AUTHOR_W * candidates["association_author_score"]
        + ASSOC_CLUSTER_W * candidates["association_cluster_score"]
        + ASSOC_KEYWORD_W * candidates["association_keyword_score"]
        + ASSOC_EMBED_W * candidates["association_embedding_similarity"]
    )

    candidates = candidates[
        quality_mask
        & (candidates["association_recommendation_score"] > 0)
        & (~candidates["paper_id"].astype(str).isin(seed_ids))
    ]

    if candidates.empty:
        return pd.DataFrame(), rec_authors, rec_clusters, rec_keywords

    return (
        candidates.sort_values("association_recommendation_score", ascending=False)
        .head(TOP_K_RECS)
        .reset_index(drop=True),
        rec_authors,
        rec_clusters,
        rec_keywords,
    )


def run_recommendation(
    context: SearchContext,
    papers: pd.DataFrame,
    x_similarity: np.ndarray,
    search_vectorizer: TfidfVectorizer,
    search_matrix: Any,
    rules: pd.DataFrame,
    keyword_vectorizer: TfidfVectorizer,
) -> dict:
    """Run full recommendation workflow for one query."""
    top_results = rank_results(context, papers, search_vectorizer, search_matrix)
    normal = normal_recommendations(context, top_results, papers, x_similarity)
    matched = match_rules(context, top_results, rules, keyword_vectorizer)

    assoc_papers, assoc_authors, assoc_clusters, assoc_keywords = association_recommendations(
        matched, top_results, papers, x_similarity
    )

    return {
        "search_context": {
            "query_text": context.query_text,
            "author_query": context.author_query,
            "cluster_query": context.cluster_query,
            "top_k": context.top_k,
            "query_keywords_used_for_rules": extract_query_keywords(context.query_text, keyword_vectorizer),
        },
        "page_ranking_top_5": records(
            top_results,
            ["page_rank_score", "search_text_score", "author_match_score", "cluster_match_score"],
        ),
        "normal_recommendations": {
            "paper_to_paper_embedding_similarity": records(
                normal["paper_to_paper_embedding_similarity"], ["embedding_similarity_score"]
            ),
            "same_cluster_topic_papers": records(
                normal["same_cluster_topic_papers"], ["same_cluster_similarity"]
            ),
            "same_author_papers": records(normal["same_author_papers"], ["same_author_score"]),
        },
        "association_rule_recommendations": {
            "recommended_papers": records(
                assoc_papers,
                [
                    "association_author_score",
                    "association_cluster_score",
                    "association_keyword_score",
                    "association_embedding_similarity",
                    "association_recommendation_score",
                ],
            ),
            "recommended_authors": assoc_authors,
            "recommended_clusters": assoc_clusters,
            "recommended_keywords": assoc_keywords,
            "matched_rules": rule_records(matched),
        },
    }

def save_plots(papers: pd.DataFrame, projection: pd.DataFrame) -> None:
    """Save cluster distribution and UMAP plots."""
    counts = papers["cluster_id"].value_counts().sort_index()

    plt.figure(figsize=(9, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Papers")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cluster_distribution.png", dpi=200, bbox_inches="tight")
    plt.show()

    plot_df = projection.sample(min(len(projection), 12000), random_state=RANDOM_STATE)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(plot_df["umap_x"], plot_df["umap_y"], c=plot_df["cluster_id"], s=5, alpha=0.7)
    plt.title("2D Research Map using UMAP")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "research_map_umap2d.png", dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved plots in: {RESULTS_DIR}")


def save_static_outputs(
    papers: pd.DataFrame,
    reduced: np.ndarray,
    projection: pd.DataFrame,
    rules: pd.DataFrame,
    clustering_result: dict,
    dr_result: dict,
) -> None:
    """Save pipeline outputs in results/."""
    papers.to_parquet(RESULTS_DIR / "papers_with_clusters.parquet", index=False)
    np.save(RESULTS_DIR / "reduced_embeddings_pca50.npy", reduced.astype("float32"))
    projection.to_csv(RESULTS_DIR / "papers_2d_projection.csv", index=False)

    rules_export = rules.copy()
    if not rules_export.empty:
        rules_export["antecedents"] = rules_export["antecedents"].apply(lambda x: sorted(list(x)))
        rules_export["consequents"] = rules_export["consequents"].apply(lambda x: sorted(list(x)))
    rules_export.to_csv(RESULTS_DIR / "association_rules.csv", index=False)

    save_json(clustering_result, RESULTS_DIR / "clustering_results.json")
    save_json(dr_result, RESULTS_DIR / "dimensionality_reduction_results.json")

    save_json(
        {
            "outputs_folder": str(RESULTS_DIR),
            "user_history_used": False,
            "search_cases_supported": [
                "topic only",
                "author only",
                "paper title only",
                "topic + author",
                "topic + cluster",
                "paper title + author",
                "paper title + author + topic",
            ],
            "clustering": clustering_result,
            "dimensionality_reduction": dr_result,
            "association_rules": {
                "min_support": MIN_SUPPORT,
                "min_confidence": MIN_CONFIDENCE,
                "max_itemset_length": MAX_APRIORI_ITEMSET_LEN,
                "rule_count": int(len(rules)),
                "quality_filter": (
                    "Association recommendation does not accept broad-cluster-only candidates; "
                    "candidate must match keyword/author or have enough embedding similarity."
                ),
            },
        },
        RESULTS_DIR / "pipeline_summary.json",
    )


def read_context() -> SearchContext | None:
    """Read one interactive query. Return None on exit."""
    print("\n" + "=" * 90)
    print("Interactive Search")
    print("=" * 90)
    print("Type 'exit' in any input to stop.\n")

    query = input("Topic / paper title / keywords: ").strip()
    if query.lower() == "exit":
        return None

    author = input("Author name (optional): ").strip()
    if author.lower() == "exit":
        return None

    cluster_text = input("Cluster ID / topic ID (optional): ").strip()
    if cluster_text.lower() == "exit":
        return None

    cluster = None
    if cluster_text:
        try:
            cluster = int(cluster_text)
        except ValueError:
            print("Invalid cluster ID. Ignoring cluster filter.")

    if not query and not author and cluster is None:
        print("Empty search. Please enter at least one field.")
        return read_context()

    return SearchContext(
        query_text=query or None,
        author_query=author or None,
        cluster_query=cluster,
        top_k=TOP_K_SEARCH,
    )


def result_filename(counter: int, context: SearchContext) -> str:
    """Build safe output file name for recommendation result."""
    raw = context.query_text or context.author_query or f"cluster_{context.cluster_query}" or f"search_{counter}"
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", raw.lower()).strip("_") or f"search_{counter}"
    return f"recommendation_results_{counter}_{safe[:50]}.json"


def main() -> None:
    """Run heavy pipeline once, then allow repeated searches."""
    papers, embeddings = load_data()
    x_similarity, x_scaled = preprocess_embeddings(embeddings)

    papers, clustering_result = run_clustering(papers, x_similarity)
    reduced, projection, dr_result = run_dimensionality_reduction(papers, x_scaled)

    papers, keyword_vectorizer = add_paper_keywords(papers)
    search_vectorizer, search_matrix = build_search_index(papers)
    rules = generate_rules(papers)

    save_static_outputs(papers, reduced, projection, rules, clustering_result, dr_result)
    save_plots(papers, projection)

    counter = 1
    while True:
        context = read_context()
        if context is None:
            print("Stopping interactive search.")
            break

        result = run_recommendation(
            context=context,
            papers=papers,
            x_similarity=x_similarity,
            search_vectorizer=search_vectorizer,
            search_matrix=search_matrix,
            rules=rules,
            keyword_vectorizer=keyword_vectorizer,
        )

        output_path = RESULTS_DIR / result_filename(counter, context)
        save_json(result, output_path)

        print(f"Saved recommendation result: {output_path}")
        print("\n" + "=" * 90)
        print("Recommendation Result")
        print("=" * 90)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        counter += 1

    print("\nDone.")


if __name__ == "__main__":
    main()
