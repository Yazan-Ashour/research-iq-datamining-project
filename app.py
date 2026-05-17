"""
ResearchIQ — Streamlit UI
===============================================
Run:  streamlit run app.py

"""

import ast
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchIQ",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; }
.stExpander > details > summary {
    background: #1A1A24; border-radius: 6px; padding: 8px 14px; font-weight: 500;
}
.cluster-pill {
    display: inline-block; padding: 2px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700; color: #fff;
    letter-spacing: 0.03em; margin-right: 4px;
}
.author-chip {
    display: inline-block; background: #1e2235; border: 1px solid #2e3555;
    color: #aab0cc; border-radius: 4px; padding: 1px 7px;
    font-size: 0.78rem; margin: 2px 2px 2px 0;
}
.paper-meta { color: #7a8099; font-size: 0.82rem; margin-top: 2px; }
.track-header {
    background: linear-gradient(90deg,#1a1a2e,#16213e);
    border-left: 3px solid #4C8EDA; border-radius: 4px;
    padding: 6px 14px; margin: 18px 0 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
DATA_DIR = BASE / "data" / "prosecced"
RESULTS  = BASE / "results"
TS_DIR   = RESULTS / "time_series"

PAPERS_PATH   = RESULTS / "papers_with_clusters.parquet"
PROJ_PATH     = RESULTS / "papers_2d_projection.csv"
RULES_PATH    = RESULTS / "association_rules.csv"
EMBED_PATH    = DATA_DIR / "embeddings.npy"
REDUCED_PATH  = RESULTS / "reduced_embeddings_pca50.npy"
CLUSTER_JSON  = RESULTS / "clustering_results.json"
DIM_JSON      = RESULTS / "dimensionality_reduction_results.json"
PIPELINE_JSON = RESULTS / "pipeline_summary.json"
# report may be in either location
TS_REPORT_PATHS = [
    TS_DIR / "time_series_report.txt",
    RESULTS / "time_series_report.txt",
]

# ── Cluster metadata ───────────────────────────────────────────────────────────
CLUSTER_NAMES = {
    0: "High Energy Physics",
    1: "CS / ML / Networks",
    2: "Astrophysics",
    3: "Mathematics",
    4: "Quantum / Condensed Matter",
}
CLUSTER_COLOURS = {
    0: "#E05C5C", 1: "#4C8EDA", 2: "#F0C93B", 3: "#E8733A", 4: "#6ABF69",
}
CAT_COLOURS = {
    "cs": "#4C8EDA", "math": "#E8733A", "astro-ph": "#F0C93B",
    "quant-ph": "#6ABF69", "hep": "#E05C5C", "cond-mat": "#B07FD4",
    "physics": "#4DBDBD", "stat": "#A0A0A0",
}
DARK = dict(plot_bgcolor="#0F0F14", paper_bgcolor="#0F0F14", font_color="white")

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_authors(value) -> list:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, np.ndarray):
        return [str(x).strip() for x in value.tolist() if str(x).strip()]
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in re.split(r";|,", text) if x.strip()]


def parse_kws(val):
    if isinstance(val, list): return val
    if isinstance(val, str):
        try: return ast.literal_eval(val)
        except: return [v.strip() for v in val.split(",") if v.strip()]
    return []

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading papers …")
def load_papers():
    df = pd.read_parquet(PAPERS_PATH)
    if "cluster_id" in df.columns and "cluster" not in df.columns:
        df = df.rename(columns={"cluster_id": "cluster"})
    df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES).fillna("Unknown")
    df["year"] = df["year"].astype("Int64")

    if "title_clean" not in df.columns:
        df["title_clean"] = df.get("title", pd.Series([""] * len(df)))
    df["title_clean"] = df["title_clean"].fillna("").astype(str)

    if "authors_list" not in df.columns:
        df["authors_list"] = [[] for _ in range(len(df))]
    df["authors_list"] = df["authors_list"].apply(_parse_authors)
    df["authors_text"] = df["authors_list"].apply(lambda xs: " ; ".join(xs))

    if "abstract_clean" not in df.columns:
        df["abstract_clean"] = df.get("abstract", pd.Series([""] * len(df)))
    df["abstract_clean"] = df["abstract_clean"].fillna("").astype(str)

    for col in ("doi", "journal_ref"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # Always string paper_id
    df["paper_id"] = df["paper_id"].astype(str)
    return df


@st.cache_data(show_spinner="Loading 2D map …")
def load_projection():
    df = pd.read_csv(PROJ_PATH)
    if "cluster_id" in df.columns and "cluster" not in df.columns:
        df = df.rename(columns={"cluster_id": "cluster"})
    # FIX: coerce to string so merge with papers["paper_id"] (str) works
    df["paper_id"] = df["paper_id"].astype(str)
    return df


@st.cache_data(show_spinner="Loading association rules …")
def load_rules():
    df = pd.read_csv(RULES_PATH)
    for col, base in [("antecedents_text", "antecedents"), ("consequents_text", "consequents")]:
        if col not in df.columns and base in df.columns:
            df[col] = df[base].astype(str)
    return df


@st.cache_data(show_spinner="Loading embeddings …")
def load_embeddings():
    for p in [REDUCED_PATH, EMBED_PATH]:
        if p.exists():
            return np.load(str(p)).astype("float32")
    return None


def load_json(path):
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner="Parsing time series report …")
def parse_ts_report():
    text = ""
    for p in TS_REPORT_PATHS:
        if Path(p).exists():
            text = Path(p).read_text(encoding="utf-8")
            break
    if not text:
        return {}

    out = {}

    # Annual volume
    av_rows, in_av = [], False
    for line in text.splitlines():
        if "ANNUAL VOLUME" in line:
            in_av = True; continue
        if in_av:
            if line.startswith("  20"):
                parts   = line.strip().split(":")
                year    = int(parts[0].strip()[:4])
                raw     = int(parts[1].strip().split()[0].replace(",", ""))
                partial = "[PARTIAL]" in line
                av_rows.append({"year": year, "papers": raw, "partial": partial})
            elif line.strip() and not line.startswith("  20"):
                if av_rows: in_av = False
    out["annual_volume"] = pd.DataFrame(av_rows) if av_rows else pd.DataFrame()

    # Dominant category
    dom_rows, in_dom = [], False
    for line in text.splitlines():
        if "DOMINANT CATEGORY" in line:
            in_dom = True; continue
        if in_dom:
            if "→" in line:
                yr   = int(line.strip().split("→")[0].strip())
                rest = line.split("→")[1].strip()
                cat  = rest.split("(")[0].strip()
                pct  = float(rest.split("(")[1].replace("%","").replace(")","").strip())
                dom_rows.append({"year": yr, "category": cat, "share_pct": pct})
            elif line.strip() and "→" not in line and line.startswith("---"):
                in_dom = False
    out["dominant_category"] = pd.DataFrame(dom_rows) if dom_rows else pd.DataFrame()

    # YoY growth
    yoy_rows, cur_cat, in_yoy = [], None, False
    for line in text.splitlines():
        if "YoY GROWTH PER CATEGORY" in line:
            in_yoy = True; continue
        if in_yoy:
            s = line.rstrip()
            if s.startswith("  ") and not s.startswith("    "):
                cur_cat = s.strip()
            elif s.startswith("    20") and cur_cat:
                parts = s.strip().split(":")
                yr    = int(parts[0].strip())
                val   = float(parts[1].strip().replace("%","").replace("+",""))
                yoy_rows.append({"category": cur_cat, "year": yr, "yoy_pct": val})
            elif s.startswith("TOP KEYWORDS") or s.startswith("ANOMALY"):
                in_yoy = False
    out["yoy_growth"] = pd.DataFrame(yoy_rows) if yoy_rows else pd.DataFrame()

    # Anomaly detection
    anomalies  = {"normal_lo": None, "normal_hi": None, "spikes": [], "dips": []}
    in_anom, cur_type = False, None
    for line in text.splitlines():
        if "ANOMALY DETECTION" in line:
            in_anom = True; continue
        if in_anom:
            if "Normal range" in line:
                parts = line.split(":")[1].strip().split("–")
                anomalies["normal_lo"] = float(parts[0].strip())
                anomalies["normal_hi"] = float(parts[1].strip())
            elif "Spikes" in line: cur_type = "spikes"
            elif "Dips"   in line: cur_type = "dips"
            elif line.strip().startswith("20") and cur_type:
                month = line.strip().split()[0]
                share = float(line.split("share=")[1].split()[0]) if "share=" in line else 0
                raw   = int(line.split("raw=")[1].replace(",","").strip()) if "raw=" in line else 0
                anomalies[cur_type].append({"month": month, "share": share, "raw": raw})
            elif line.startswith("ARIMA") or line.startswith("TOP"):
                in_anom = False
    out["anomalies"] = anomalies

    # ARIMA overall
    arima_rows, in_arima = [], False
    for line in text.splitlines():
        if "ARIMA — OVERALL" in line:
            in_arima = True; continue
        if in_arima:
            if line.strip().startswith("202") and "~" in line:
                month = line.strip().split()[0]
                val   = float(line.split("~")[1].split()[0])
                lo    = float(line.split("[")[1].split("–")[0].strip())
                hi    = float(line.split("–")[1].replace("]","").strip())
                arima_rows.append({"month": month, "forecast": val, "lo": lo, "hi": hi})
            elif line.startswith("ARIMA — PER"):
                in_arima = False
    out["arima_forecast"] = pd.DataFrame(arima_rows) if arima_rows else pd.DataFrame()

    # ARIMA per category
    cat_arima, cur_cat, in_cat = {}, None, False
    for line in text.splitlines():
        if "ARIMA — PER CATEGORY" in line:
            in_cat = True; continue
        if in_cat:
            s = line.strip()
            if s and "ARIMA" in s and "MAPE" in s:
                parts   = s.split()
                cur_cat = parts[0]
                ms      = [p for p in parts if "MAPE=" in p]
                mape    = float(ms[0].replace("MAPE=","").replace("%","")) if ms else None
                cat_arima[cur_cat] = {"mape": mape, "forecast_range": None}
            elif cur_cat and "Forecast:" in s and "range" in s:
                rng    = s.split("range")[1].strip().strip("[]")
                lo, hi = [float(x.strip()) for x in rng.split("–")]
                cat_arima[cur_cat]["forecast_range"] = (lo, hi)
    out["arima_per_cat"] = cat_arima

    # Top keywords
    kw_rows, in_kw = [], False
    for line in text.splitlines():
        if "TOP KEYWORDS PER YEAR" in line:
            in_kw = True; continue
        if in_kw:
            if line.strip().startswith("20"):
                parts = line.split(":")
                yr    = int(parts[0].strip()[:4])
                kws   = [k.strip() for k in parts[1].split(",") if k.strip()]
                kw_rows.append({"year": yr, "keywords": kws})
            elif line.startswith("ANOMALY") or line.startswith("ARIMA"):
                in_kw = False
    out["keywords"] = kw_rows
    return out


@st.cache_data(show_spinner="Loading time series figures …")
def load_ts_figures():
    # Exact filenames as shown in the GitHub screenshot
    figs = {}
    for fname in [
        "fig1_annual_trends.png",
        "fig2_keyword_heatmap.png",
        "fig3_keyword_trends.png",
        "fig4_category_trends.png",
        "fig5_growth_rates.png",
        "fig6_category_keyword_drift.png",
        "fig7_anomalies.png",
        "fig8_arima_overall.png",
        "fig9_arima_per_category.png",
        "fig10_arima_comparison.png",
    ]:
        p = TS_DIR / fname
        if p.exists():
            figs[fname] = str(p)
    return figs

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH & RANKING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _build_corpus(df):
    return (
        df["title_clean"].fillna("") + " " +
        df["abstract_clean"].fillna("") + " " +
        df["authors_text"].fillna("")
    ).tolist()


def search_papers(query: str, author_query: str, df: pd.DataFrame, top_k: int = 10):
    """
    TF-IDF search over title + abstract + authors, with PageRank-style re-ranking.
    Multi-word author query is handled correctly (all tokens checked).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    full_query = " ".join(filter(None, [query.strip(), author_query.strip()]))
    if not full_query:
        out = df.head(top_k).copy()
        out["search_score"] = 0.0
        out["page_rank_score"] = 0.0
        return out

    vec  = TfidfVectorizer(max_features=40_000, stop_words="english")
    mat  = vec.fit_transform(_build_corpus(df))
    text_scores = cosine_similarity(vec.transform([full_query]), mat).flatten()

    # Author boost — all tokens of author query must appear in authors_text
    author_boost = np.zeros(len(df))
    if author_query.strip():
        tokens = author_query.strip().lower().split()
        mask   = pd.Series([True] * len(df), index=df.index)
        for tok in tokens:
            mask = mask & df["authors_text"].str.lower().str.contains(
                re.escape(tok), regex=False, na=False
            )
        author_boost = mask.astype(float).values * 0.5

    raw_scores = text_scores + author_boost

    # PageRank re-ranking: TF-IDF (65%) + cluster popularity (20%) + recency (15%)
    cluster_pop = df["cluster"].map(
        df["cluster"].value_counts(normalize=True)
    ).fillna(0).values
    max_yr = float(df["year"].max()); min_yr = float(df["year"].min())
    yr_arr  = df["year"].fillna(min_yr).astype(float).values
    recency = (yr_arr - min_yr) / max(max_yr - min_yr, 1)

    page_rank_score = (
        0.65 * raw_scores
        + 0.20 * cluster_pop / (cluster_pop.max() + 1e-9)
        + 0.15 * recency
    )

    out = df.copy()
    out["search_score"]    = raw_scores
    out["page_rank_score"] = page_rank_score
    return out.nlargest(top_k, "page_rank_score")


def recommend_similar(paper_id, df, embeddings, top_k=10, all_papers=None):
    """
    FIX: embeddings rows are indexed 1-to-1 with `all_papers` (the full dataset).
    When `df` is a filtered subset, df.iloc[i] no longer corresponds to
    embeddings[i], causing out-of-bounds errors.

    Strategy:
      1. Find the query paper's position in the FULL papers frame.
      2. Compute cosine similarity against ALL embeddings.
      3. Map the top-scoring embedding indices back to paper_ids via all_papers.
      4. Filter those paper_ids to only those present in the filtered `df`.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    if embeddings is None:
        return pd.DataFrame()

    # Use the full dataset for index lookup if provided, else fall back to df
    ref = all_papers if all_papers is not None else df
    all_ids = ref["paper_id"].tolist()
    if paper_id not in all_ids:
        return pd.DataFrame()

    # Query embedding from the full index
    idx = all_ids.index(paper_id)
    scores = cosine_similarity(embeddings[idx:idx+1], embeddings).flatten()
    scores[idx] = -1  # exclude the query paper itself

    # Get top-k indices into the FULL embeddings array
    top_indices = scores.argsort()[::-1][: top_k * 5]  # oversample to allow for filter loss

    # Map indices → paper_ids → filter to the current filtered pool
    top_pids    = [all_ids[i] for i in top_indices]
    top_scores  = {all_ids[i]: scores[i] for i in top_indices}

    filtered_pool = df[df["paper_id"].isin(top_pids)].copy()
    if filtered_pool.empty:
        return pd.DataFrame()

    filtered_pool["similarity_score"] = filtered_pool["paper_id"].map(top_scores)
    filtered_pool = filtered_pool.sort_values("similarity_score", ascending=False).head(top_k)

    # Recency boost
    max_yr = float(df["year"].max()); min_yr = float(df["year"].min())
    yr      = filtered_pool["year"].fillna(min_yr).astype(float)
    recency = (yr - min_yr) / max(max_yr - min_yr, 1)
    filtered_pool["page_rank_score"] = (
        0.75 * filtered_pool["similarity_score"] + 0.25 * recency.values
    )
    return filtered_pool.sort_values("page_rank_score", ascending=False)


def apply_filters(df, filter_clusters, year_range):
    """
    FIX: Cast year to plain Python int (works with nullable Int64) before
    comparing against the slider tuple values, which are also plain ints.
    Using .fillna(-1) ensures NA years are excluded rather than causing
    silent comparison failures with pandas nullable Int64.
    """
    out = df.copy()
    if filter_clusters:
        out = out[out["cluster"].isin(filter_clusters)]
    yr = out["year"].fillna(-1).astype(int)
    out = out[(yr >= year_range[0]) & (yr <= year_range[1])]
    return out

# ══════════════════════════════════════════════════════════════════════════════
# PAPER CARD
# FIX: Replace st.button + st.rerun() abstract toggle with st.checkbox.
#      st.checkbox manages its own state natively without triggering a full
#      rerun that collapses the result list and re-executes the search.
# ══════════════════════════════════════════════════════════════════════════════

def paper_card(row, score_col=None, score_label="Score", track_id="t", idx=0):
    title    = str(row.get("title_clean", row.get("title", "No title")))
    year     = row.get("year", "?")
    cl       = int(row.get("cluster", row.get("cluster_id", -1)))
    cname    = CLUSTER_NAMES.get(cl, "Unknown")
    ccol     = CLUSTER_COLOURS.get(cl, "#888888")
    authors  = _parse_authors(row.get("authors_list", []))
    kws      = parse_kws(row.get("paper_keywords", []))[:8]
    doi      = str(row.get("doi", "")).strip()
    jref     = str(row.get("journal_ref", "")).strip()
    abstract = str(row.get("abstract_clean", "")).strip()
    pid      = str(row.get("paper_id", ""))

    score_str = (
        f"  ·  **{score_label}: {row[score_col]:.3f}**"
        if (score_col and score_col in row.index) else ""
    )
    pr_str = (
        f"  PageRank: {row['page_rank_score']:.3f}"
        if ("page_rank_score" in row.index and score_col != "page_rank_score") else ""
    )

    short = title[:82] + ("…" if len(title) > 82 else "")
    header = f"📄 {short} ({year}){score_str}"

    with st.expander(header):
        # Cluster badge + pr
        st.markdown(
            f"<span class='cluster-pill' style='background:{ccol}'>● {cl}: {cname}</span>"
            + (f"<span style='color:#7a8099;font-size:0.8rem;margin-left:8px'>{pr_str}</span>" if pr_str else ""),
            unsafe_allow_html=True,
        )

        # Authors
        if authors:
            chips = "".join(f"<span class='author-chip'>{a}</span>" for a in authors[:14])
            if len(authors) > 14:
                chips += f"<span class='author-chip'>+{len(authors)-14} more</span>"
            st.markdown(f"**Authors:** {chips}", unsafe_allow_html=True)
        else:
            st.markdown("<span class='paper-meta'>Authors: not available</span>", unsafe_allow_html=True)

        # Keywords
        if kws:
            st.markdown("**Keywords:** " + " ".join(f"`{k}`" for k in kws))

        # ArXiv / DOI / journal
        meta = []
        if pid:
            meta.append(f"[🔗 arxiv.org/abs/{pid}](https://arxiv.org/abs/{pid})")
        if doi and doi not in {"", "nan"}:
            meta.append(f"DOI: `{doi}`")
        if jref and jref not in {"", "nan"}:
            meta.append(f"*{jref}*")
        if meta:
            st.markdown("  ·  ".join(meta))

        # Score metrics
        if score_col and score_col in row.index:
            c1, c2 = st.columns([1, 1])
            c1.metric(score_label, f"{row[score_col]:.4f}")
            if "page_rank_score" in row.index and score_col != "page_rank_score":
                c2.metric("PageRank Score", f"{row['page_rank_score']:.4f}")

        if abstract:
            with st.expander("📄 Show abstract"):
                st.write(abstract[:700] + ("…" if len(abstract) > 700 else ""))

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔬 ResearchIQ")
    st.caption("ArXiv Research Intelligence Platform")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Overview",
        "🔍 Search & Recommend",
        "🗺️ Research Map",
        "🔗 Association Rules",
        "📈 Time Series",
        "⚙️ Pipeline Info",
    ], label_visibility="collapsed")
    st.divider()
    st.caption("Data Mining · An-Najah National University")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ══════════════════════════════════════════════════════════════════════════════

papers     = load_papers()
projection = load_projection()
rules      = load_rules()
embeddings = load_embeddings()
ts_data    = parse_ts_report()
ts_figs    = load_ts_figures()
cl_meta    = load_json(CLUSTER_JSON)   if CLUSTER_JSON.exists()  else {}
dim_meta   = load_json(DIM_JSON)       if DIM_JSON.exists()      else {}
pipe_meta  = load_json(PIPELINE_JSON)  if PIPELINE_JSON.exists() else {}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("🔬 ResearchIQ")
    st.markdown(
        "**An intelligent research discovery platform** built on ~40K sampled ArXiv papers "
        "backed by a full pipeline of 1.9M papers for time series analysis."
    )
    st.divider()

    sil     = cl_meta.get("silhouette_score_sample", 0.034)
    pca_var = dim_meta.get("compression", {}).get("cumulative_explained_variance", 0.444)
    n_rules = pipe_meta.get("association_rules", {}).get("rule_count", 659)

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("📄 Papers (sample)",  f"{len(papers):,}")
    k2.metric("📄 Papers (full TS)", "1.9M")
    k3.metric("🗂️ Clusters",         f"{papers['cluster'].nunique()}")
    k4.metric("📐 Silhouette",        f"{sil:.4f}")
    k5.metric("🔗 Assoc. Rules",      f"{n_rules:,}")
    k6.metric("📊 PCA Variance",      f"{pca_var*100:.1f}%")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Research Cluster Distribution")
        counts = papers.groupby(["cluster","cluster_name"]).size().reset_index(name="count")
        fig = px.bar(counts, x="cluster_name", y="count", color="cluster_name",
                     color_discrete_map={CLUSTER_NAMES[k]: v for k,v in CLUSTER_COLOURS.items()},
                     text_auto=True, labels={"cluster_name":"","count":"Papers"})
        fig.update_layout(**DARK, showlegend=False, height=340, xaxis=dict(tickangle=-15))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Full Dataset — Annual Volume (1.9M papers)")
        av = ts_data.get("annual_volume", pd.DataFrame())
        if not av.empty:
            fig2 = px.bar(av[~av["partial"]], x="year", y="papers", text_auto=True,
                          color_discrete_sequence=["#4C8EDA"],
                          labels={"year":"Year","papers":"Papers"})
            partial = av[av["partial"]]
            if not partial.empty:
                fig2.add_bar(x=partial["year"], y=partial["papers"],
                             name="Partial year", marker_color="#888888",
                             text=partial["papers"])
            fig2.update_layout(**DARK, height=340, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Run time_series.py to populate this chart.")

    dom = ts_data.get("dominant_category", pd.DataFrame())
    if not dom.empty:
        st.subheader("Dominant ArXiv Category per Year")
        fig3 = px.bar(dom, x="year", y="share_pct", color="category",
                      color_discrete_map=CAT_COLOURS, text_auto=".1f",
                      labels={"year":"Year","share_pct":"Share (%)","category":"Category"})
        fig3.update_layout(**DARK, height=320,
                           legend=dict(orientation="h",yanchor="bottom",y=1.0))
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("What can ResearchIQ do?")
    f1,f2,f3,f4,f5 = st.columns(5)
    f1.info("**🔍 Search & Recommend**\nTF-IDF + PageRank + embedding similarity + association rules. Topic, author & title search.")
    f2.info("**🗺️ Research Map**\n2D UMAP of 40K papers, filterable by cluster and year")
    f3.info("**🔗 Association Rules**\n659 rules linking keywords, authors, clusters")
    f4.info("**📈 Time Series**\nARIMA forecasts, anomalies, YoY growth on 1.9M papers")
    f5.info("**⚙️ Pipeline Info**\nClustering, PCA, UMAP, and association-rule parameters")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SEARCH & RECOMMEND
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Search & Recommend":
    st.title("🔍 Search & Recommend")
    st.markdown(
        "Search by **topic**, **keywords**, **paper title**, or **author name**. "
        "All results ranked by **PageRank score** (TF-IDF 65% + cluster popularity 20% + recency 15%). "
        "**Advanced Filters apply to every track.**"
    )

    # ── FIX: Initialise year-range session state BEFORE the slider is rendered.
    # When the form is submitted, Streamlit reruns from the top. Without this
    # initialisation the slider resets to its default values on every rerun,
    # making the year filter appear to do nothing.
    year_min = int(papers["year"].min())
    year_max = int(papers["year"].max())
    if "adv_years" not in st.session_state:
        st.session_state["adv_years"] = (year_min, year_max)

    with st.form("search_form"):
        c1, c2, c3 = st.columns([3, 2, 1])
        query        = c1.text_input("Topic / Keywords / Paper title",
                                      placeholder="e.g. quantum entanglement graph neural networks")
        author_query = c2.text_input("Author name (optional)",
                                      placeholder="e.g. Yoshua Bengio")
        top_k        = c3.selectbox("Results per track", [5, 10, 20], index=1)
        submitted    = st.form_submit_button("🔍 Search", use_container_width=True)

    # Advanced Filters — rendered OUTSIDE the form so they persist
    with st.expander("⚙️ Advanced Filters — applied to ALL tracks", expanded=True):
        fc1, fc2 = st.columns(2)
        filter_cluster = fc1.multiselect(
            "Limit to clusters",
            options=list(CLUSTER_NAMES.keys()),
            format_func=lambda x: f"{x}: {CLUSTER_NAMES[x]}",
            key="adv_clusters",
        )
        # FIX: The slider reads from and writes to st.session_state["adv_years"]
        # automatically via the key= argument. Providing explicit min_value and
        # max_value (plain ints) ensures no type mismatch with Int64 year values.
        year_range = fc2.slider(
            "Year range",
            min_value=year_min,
            max_value=year_max,
            key="adv_years",
        )

    if submitted and (query.strip() or author_query.strip()):

        # year_range is now always a reliable (int, int) tuple from session state
        filtered = apply_filters(papers, filter_cluster, year_range)

        if filtered.empty:
            st.warning("No papers match the current filters. Relax the cluster or year selection.")
            st.stop()

        label_parts = []
        if query.strip():        label_parts.append(f"*{query}*")
        if author_query.strip(): label_parts.append(f"author: *{author_query}*")
        combined = " + ".join(label_parts)

        filter_info = []
        if filter_cluster:
            filter_info.append("Clusters: " + ", ".join(CLUSTER_NAMES.get(c, str(c)) for c in filter_cluster))
        filter_info.append(f"Years: {year_range[0]}–{year_range[1]}")
        st.caption("Active filters: " + "  ·  ".join(filter_info))

        with st.spinner("Searching and ranking …"):
            results = search_papers(query, author_query, filtered, top_k=top_k)

        # ── Track 1: Main search results ──────────────────────────────────────
        st.markdown(
            f"<div class='track-header'>🏆 Top {len(results)} results for {combined} "
            f"<small style='color:#7a8099'>(PageRank ranked)</small></div>",
            unsafe_allow_html=True,
        )
        for i, (_, row) in enumerate(results.iterrows()):
            paper_card(row, score_col="page_rank_score", score_label="PageRank",
                       track_id="main", idx=i)

        # ── Track 2: Embedding similarity ─────────────────────────────────────
        if embeddings is not None and len(results) > 0:
            st.markdown(
                "<div class='track-header'>📌 Similar Papers — Embedding Similarity "
                "<small style='color:#7a8099'>(PCA-50 embedding of top result · filtered pool)</small></div>",
                unsafe_allow_html=True,
            )
            top_pid = results.iloc[0]["paper_id"]
            similar = recommend_similar(top_pid, filtered, embeddings, top_k=top_k, all_papers=papers)
            if similar.empty:
                st.info("Top result not found in filtered embedding pool.")
            else:
                for i, (_, row) in enumerate(similar.iterrows()):
                    paper_card(row, score_col="page_rank_score", score_label="PageRank",
                               track_id="embed", idx=i)

        # ── Track 3: Same cluster ──────────────────────────────────────────────
        if len(results) > 0:
            top_cl = int(results.iloc[0]["cluster"])
            st.markdown(
                f"<div class='track-header'>🗂️ More from Cluster {top_cl}: "
                f"{CLUSTER_NAMES.get(top_cl,'')} "
                f"<small style='color:#7a8099'>(recency-weighted sample · filtered pool)</small></div>",
                unsafe_allow_html=True,
            )
            same_cl_pool = filtered[
                (filtered["cluster"] == top_cl) &
                (~filtered["paper_id"].isin(results["paper_id"]))
            ].copy()

            if same_cl_pool.empty:
                st.info("No additional papers in this cluster within the current filters.")
            else:
                max_yr = float(same_cl_pool["year"].max())
                min_yr = float(same_cl_pool["year"].min())
                yr_w   = same_cl_pool["year"].fillna(min_yr).astype(float)
                weights = (yr_w - min_yr + 1) / (max_yr - min_yr + 1)
                weights = weights / weights.sum()
                n_samp  = min(top_k, len(same_cl_pool))
                sampled = same_cl_pool.sample(n_samp, weights=weights, random_state=42)
                for i, (_, row) in enumerate(sampled.iterrows()):
                    paper_card(row, track_id="cluster", idx=i)

        # ── Track 4: Association rules ─────────────────────────────────────────
        st.markdown(
            "<div class='track-header'>🔗 Association Rule Suggestions "
            "<small style='color:#7a8099'>(multi-word phrase matching)</small></div>",
            unsafe_allow_html=True,
        )

        def _rule_matches(rule_text: str) -> bool:
            rl = str(rule_text).lower()
            # match any meaningful word from topic query (len > 2)
            for w in query.lower().split():
                if len(w) > 2 and w in rl:
                    return True
            # match all tokens from author query
            if author_query.strip():
                if all(t in rl for t in author_query.lower().split() if len(t) > 2):
                    return True
            return False

        mask = rules["antecedents_text"].apply(_rule_matches)
        matched_rules = rules[mask].sort_values("confidence", ascending=False).head(8)

        if len(matched_rules) > 0:
            st.dataframe(
                matched_rules[["antecedents_text","consequents_text",
                               "confidence","lift","rule_type"]]
                .rename(columns={
                    "antecedents_text":"If topic …",
                    "consequents_text":"… also explore",
                    "confidence":"Confidence",
                    "lift":"Lift",
                    "rule_type":"Type",
                }),
                use_container_width=True, hide_index=True,
            )
            # Papers matching rule consequents within filtered pool
            rule_kws = set()
            for _, r in matched_rules.iterrows():
                rule_kws.update(str(r["consequents_text"]).lower().split())
            rule_kws = {w for w in rule_kws if len(w) > 3}

            if rule_kws:
                mask2 = (
                    filtered["title_clean"].str.lower().apply(
                        lambda t: any(kw in t for kw in rule_kws)
                    ) |
                    filtered["abstract_clean"].str.lower().apply(
                        lambda t: any(kw in t for kw in rule_kws)
                    )
                )
                rule_papers = filtered[mask2].head(top_k)
                if not rule_papers.empty:
                    st.caption(
                        f"Papers matching rule consequents ({len(rule_papers)} found, "
                        "filtered by active filters):"
                    )
                    for i, (_, row) in enumerate(rule_papers.iterrows()):
                        paper_card(row, track_id="rules", idx=i)
        else:
            st.info("No matching association rules. Try a shorter or single keyword.")

        with st.expander("ℹ️ Supported search modes"):
            for m in pipe_meta.get("search_cases_supported", [
                "topic only", "author only", "paper title only",
                "topic + author", "topic + cluster",
            ]):
                st.markdown(f"- {m}")

    elif submitted:
        st.warning("Please enter at least a topic or an author name.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESEARCH MAP
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🗺️ Research Map":
    st.title("🗺️ Research Map")
    st.markdown("Interactive **2D UMAP projection** of ~40K papers. Each dot is a paper; colour = cluster.")

    c1,c2,c3 = st.columns(3)
    sel_clusters = c1.multiselect(
        "Clusters", list(CLUSTER_NAMES.keys()),
        default=list(CLUSTER_NAMES.keys()),
        format_func=lambda x: f"{x}: {CLUSTER_NAMES[x]}"
    )
    sample_n = c2.slider("Max points", 1000, min(15000, len(projection)), 8000, 1000)
    yr_range = c3.slider(
        "Year range",
        int(papers["year"].min()), int(papers["year"].max()),
        (int(papers["year"].min()), int(papers["year"].max()))
    )

    # FIX: both sides are already str after load_papers / load_projection
    proj = projection.copy()

    merge_cols = [c for c in ["paper_id","year","title_clean","cluster","authors_text"]
                  if c in papers.columns]
    papers_m   = papers[merge_cols].copy()

    # safe merge — both paper_id columns are str
    proj = proj.merge(papers_m, on="paper_id", how="left", suffixes=("","_p"))

    if "cluster_p" in proj.columns:
        proj["cluster"] = proj["cluster"].combine_first(proj["cluster_p"])
        proj.drop(columns=["cluster_p"], inplace=True)

    proj["cluster"] = pd.to_numeric(proj["cluster"], errors="coerce")
    proj = proj[proj["cluster"].isin(sel_clusters)]
    proj = proj[(proj["year"] >= yr_range[0]) & (proj["year"] <= yr_range[1])]
    if len(proj) > sample_n:
        proj = proj.sample(sample_n, random_state=42)
    proj["cluster_name"] = proj["cluster"].map(CLUSTER_NAMES)

    hover = {"title_clean": True, "year": True, "cluster_name": True,
             "umap_x": False, "umap_y": False}
    if "authors_text" in proj.columns:
        hover["authors_text"] = True

    fig = px.scatter(
        proj, x="umap_x", y="umap_y", color="cluster_name",
        color_discrete_map={CLUSTER_NAMES[k]: v for k,v in CLUSTER_COLOURS.items()},
        hover_data=hover, opacity=0.55, size_max=4,
        labels={"umap_x":"UMAP-1","umap_y":"UMAP-2","cluster_name":"Cluster",
                "title_clean":"Title","authors_text":"Authors"},
        title=f"Research Map — {len(proj):,} papers",
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(**DARK, height=600, legend=dict(orientation="h",yanchor="bottom",y=1.01))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Summary")
    smry = (
        papers[papers["cluster"].isin(sel_clusters)]
        .groupby(["cluster","cluster_name"])
        .agg(papers=("paper_id","count"), first_year=("year","min"), last_year=("year","max"))
        .reset_index().sort_values("cluster")
    )
    smry.columns = ["ID","Cluster","Papers","First Year","Last Year"]
    st.dataframe(smry, use_container_width=True, hide_index=True)

    with st.expander("📐 Dimensionality Reduction Parameters"):
        cc1,cc2 = st.columns(2)
        cmp = dim_meta.get("compression",{}); viz2 = dim_meta.get("visualization",{})
        with cc1:
            st.markdown("**PCA**")
            st.json({"n_components": cmp.get("n_components",50),
                     "cumulative_variance": f"{cmp.get('cumulative_explained_variance',0.444)*100:.1f}%",
                     "runtime_s": cmp.get("runtime_seconds",0.78)})
        with cc2:
            st.markdown("**UMAP**")
            st.json({**viz2.get("parameters",{}), "runtime_s": viz2.get("runtime_seconds",59.5)})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔗 Association Rules":
    st.title("🔗 Association Rules")
    st.markdown(
        "**659 rules** mined with Apriori from keywords, authors, and clusters. "
        "min_support = 0.001 · min_confidence = 0.05 · max_itemset_length = 2"
    )

    c1,c2,c3,c4 = st.columns(4)
    min_conf   = c1.slider("Min Confidence", 0.0, 1.0, 0.3, 0.05)
    min_lift   = c2.slider("Min Lift",       0.0, 200.0, 1.0, 1.0)
    rule_types = c3.multiselect("Rule Type", rules["rule_type"].unique().tolist(),
                                 default=rules["rule_type"].unique().tolist())
    kw_search  = c4.text_input("Search keyword(s)", placeholder="e.g. neural network")

    fr = rules[
        (rules["confidence"] >= min_conf) &
        (rules["lift"]       >= min_lift) &
        (rules["rule_type"].isin(rule_types))
    ]
    if kw_search.strip():
        # Multi-word: ANY token matches
        tokens = kw_search.strip().lower().split()
        mask = (
            fr["antecedents_text"].str.lower().apply(lambda x: any(t in x for t in tokens)) |
            fr["consequents_text"].str.lower().apply(lambda x: any(t in x for t in tokens))
        )
        fr = fr[mask]

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Rules shown",    len(fr))
    m2.metric("Avg Confidence", f"{fr['confidence'].mean():.3f}" if len(fr) else "—")
    m3.metric("Avg Lift",       f"{fr['lift'].mean():.1f}"       if len(fr) else "—")
    m4.metric("Max Lift",       f"{fr['lift'].max():.1f}"        if len(fr) else "—")

    st.divider()
    ca,cb = st.columns([2,1])
    with ca:
        st.subheader("Lift vs Confidence")
        if len(fr):
            figr = px.scatter(fr, x="confidence", y="lift", color="rule_type",
                              size="support", opacity=0.7,
                              hover_data=["antecedents_text","consequents_text"],
                              labels={"confidence":"Confidence","lift":"Lift","rule_type":"Type"})
            figr.update_layout(**DARK, height=370)
            st.plotly_chart(figr, use_container_width=True)
    with cb:
        st.subheader("By Type")
        tc = fr["rule_type"].value_counts().reset_index(); tc.columns = ["Type","Count"]
        figp = px.pie(tc, names="Type", values="Count", hole=0.45)
        figp.update_layout(**DARK, height=370)
        st.plotly_chart(figp, use_container_width=True)

    st.subheader(f"Rules Table ({len(fr):,})")
    st.dataframe(
        fr[["antecedents_text","consequents_text","confidence","lift","support","rule_type"]]
        .rename(columns={"antecedents_text":"If …","consequents_text":"… Then",
                         "confidence":"Conf.","lift":"Lift","support":"Support","rule_type":"Type"})
        .sort_values("Lift", ascending=False).reset_index(drop=True),
        use_container_width=True, height=380,
    )

    st.divider()
    st.subheader("🏆 Top 10 Rules by Lift")
    top10 = fr.nlargest(10,"lift")[
        ["antecedents_text","consequents_text","confidence","lift","rule_type"]
    ].reset_index(drop=True)
    st.dataframe(top10.rename(columns={
        "antecedents_text":"If …","consequents_text":"… Then",
        "confidence":"Conf.","lift":"Lift","rule_type":"Type",
    }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Time Series":
    st.title("📈 Time Series Analysis")
    st.markdown(
        "Based on the **full 1.9M-paper ArXiv dataset** (2015–2026). "
        "All share metrics normalised to correct for yearly volume differences."
    )

    tabs = st.tabs([
        "📊 Trends", "📉 YoY Growth", "🔮 ARIMA Forecast",
        "⚠️ Anomalies", "🔑 Keywords", "🖼️ Pipeline Figures",
    ])

    # ── Tab 1: Trends ─────────────────────────────────────────────────────────
    with tabs[0]:
        av = ts_data.get("annual_volume", pd.DataFrame())
        if not av.empty:
            st.subheader("Annual Publication Volume — Full Dataset (1.9M papers)")
            figt = go.Figure()
            complete = av[~av["partial"]]; partial = av[av["partial"]]
            figt.add_bar(x=complete["year"], y=complete["papers"], name="Complete year",
                         marker_color="#4C8EDA",
                         text=complete["papers"].apply(lambda x: f"{x:,}"), textposition="outside")
            if not partial.empty:
                figt.add_bar(x=partial["year"], y=partial["papers"], name="Partial year ⚠",
                             marker_color="#888888",
                             text=partial["papers"].apply(lambda x: f"{x:,}"), textposition="outside")
            figt.update_layout(**DARK, height=380, barmode="group",
                               yaxis_title="Papers Published", xaxis_title="Year")
            st.plotly_chart(figt, use_container_width=True)
            st.caption("Note: 2026 is a partial year.")
        else:
            st.info("Annual volume: run time_series.py and ensure time_series_report.txt is in results/time_series/")

        dom = ts_data.get("dominant_category", pd.DataFrame())
        if not dom.empty:
            st.subheader("Dominant ArXiv Category per Year (normalised share)")
            figd = px.bar(dom, x="year", y="share_pct", color="category",
                          color_discrete_map=CAT_COLOURS, text_auto=".1f",
                          labels={"year":"Year","share_pct":"Share (%)","category":"Category"})
            figd.update_layout(**DARK, height=360,
                               legend=dict(orientation="h",yanchor="bottom",y=1.0))
            st.plotly_chart(figd, use_container_width=True)

        st.subheader("Research Cluster Share per Year — Sampled Dataset (normalised)")
        norm = papers.groupby(["year","cluster","cluster_name"]).size().reset_index(name="count")
        norm["share"] = norm["count"] / norm.groupby("year")["count"].transform("sum")
        fign = px.bar(norm, x="year", y="share", color="cluster_name",
                      color_discrete_map={CLUSTER_NAMES[k]:v for k,v in CLUSTER_COLOURS.items()},
                      barmode="stack", text_auto=".0%",
                      labels={"share":"Share","year":"Year","cluster_name":"Cluster"})
        fign.update_layout(**DARK, height=360, yaxis_tickformat=".0%",
                           legend=dict(orientation="h",yanchor="bottom",y=1.0))
        st.plotly_chart(fign, use_container_width=True)

    # ── Tab 2: YoY Growth ────────────────────────────────────────────────────
    with tabs[1]:
        yoy = ts_data.get("yoy_growth", pd.DataFrame())
        if not yoy.empty:
            st.subheader("YoY Growth per ArXiv Category (full 1.9M dataset, normalised share)")
            sel_cats = st.multiselect("Select categories", yoy["category"].unique().tolist(),
                                       default=yoy["category"].unique().tolist())
            yoy_f = yoy[yoy["category"].isin(sel_cats)]
            figy = px.line(yoy_f, x="year", y="yoy_pct", color="category",
                           color_discrete_map=CAT_COLOURS, markers=True,
                           labels={"yoy_pct":"YoY Growth (%)","year":"Year","category":"Category"})
            figy.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
            figy.update_layout(**DARK, height=420,
                               legend=dict(orientation="h",yanchor="bottom",y=1.0))
            st.plotly_chart(figy, use_container_width=True)

            pivot = yoy_f.pivot_table(index="year", columns="category", values="yoy_pct").round(1)
            pivot.columns.name = None
            st.dataframe(pivot.style.background_gradient(cmap="RdYlGn", axis=None),
                         use_container_width=True)

            st.subheader("YoY Growth per Cluster — Sampled Dataset")
            norm2 = papers.groupby(["year","cluster","cluster_name"]).size().reset_index(name="count")
            norm2["share"] = norm2["count"] / norm2.groupby("year")["count"].transform("sum")
            yoy2_rows = []
            for cl, grp in norm2.groupby("cluster"):
                g = grp.groupby("year")["share"].sum().reset_index()
                g["yoy"] = g["share"].pct_change() * 100
                g["cluster_name"] = CLUSTER_NAMES.get(cl, str(cl))
                yoy2_rows.append(g)
            yoy2 = pd.concat(yoy2_rows).dropna(subset=["yoy"])
            figy2 = px.line(yoy2, x="year", y="yoy", color="cluster_name",
                            color_discrete_map={CLUSTER_NAMES[k]:v for k,v in CLUSTER_COLOURS.items()},
                            markers=True,
                            labels={"yoy":"YoY Growth (%)","year":"Year","cluster_name":"Cluster"})
            figy2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
            figy2.update_layout(**DARK, height=380,
                                legend=dict(orientation="h",yanchor="bottom",y=1.0))
            st.plotly_chart(figy2, use_container_width=True)
        else:
            st.info("YoY data not found. Ensure time_series_report.txt is in results/time_series/")

    # ── Tab 3: ARIMA Forecast ─────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("ARIMA Forecast — Overall Monthly Share (2026)")
        st.markdown("In-sample **MAPE: 5.86%** · Horizon: 12 months")
        fc = ts_data.get("arima_forecast", pd.DataFrame())
        if not fc.empty:
            figa = go.Figure()
            figa.add_scatter(x=fc["month"], y=fc["forecast"],
                             mode="lines+markers", name="Forecast",
                             line=dict(color="#4C8EDA",width=2.5), marker=dict(size=7))
            figa.add_scatter(x=fc["month"], y=fc["hi"], mode="lines", showlegend=False,
                             line=dict(color="#4C8EDA",width=0))
            figa.add_scatter(x=fc["month"], y=fc["lo"], mode="lines", name="95% CI",
                             fill="tonexty", fillcolor="rgba(76,142,218,0.2)",
                             line=dict(color="#4C8EDA",width=0))
            figa.update_layout(**DARK, height=380,
                               xaxis_title="Month", yaxis_title="Normalised Monthly Share")
            st.plotly_chart(figa, use_container_width=True)
            st.dataframe(fc.rename(columns={"month":"Month","forecast":"Forecast",
                                            "lo":"Lower 95% CI","hi":"Upper 95% CI"}),
                         use_container_width=True, hide_index=True)
        else:
            st.info("ARIMA data not found in report.")

        cat_arima = ts_data.get("arima_per_cat", {})
        if cat_arima:
            st.subheader("ARIMA per ArXiv Category")
            rows = []
            for cat, info in cat_arima.items():
                lo2, hi2 = info.get("forecast_range") or (None, None)
                rows.append({"Category":cat,"MAPE (%)":info.get("mape"),
                             "Forecast Lo":lo2,"Forecast Hi":hi2})
            cat_df = pd.DataFrame(rows).sort_values("MAPE (%)")
            figc = px.bar(cat_df, x="Category", y="MAPE (%)",
                          color="Category", color_discrete_map=CAT_COLOURS, text_auto=".1f")
            figc.update_layout(**DARK, height=320, showlegend=False)
            st.plotly_chart(figc, use_container_width=True)
            st.dataframe(cat_df.style.format({
                "MAPE (%)":"{:.2f}","Forecast Lo":"{:.5f}","Forecast Hi":"{:.5f}",
            }), use_container_width=True, hide_index=True)

    # ── Tab 4: Anomalies ──────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Anomaly Detection — Normalised Monthly Share (IQR × 1.5)")
        anom   = ts_data.get("anomalies", {})
        lo     = anom.get("normal_lo")
        hi     = anom.get("normal_hi")
        spikes = anom.get("spikes", [])
        dips   = anom.get("dips",   [])

        if lo is not None:
            a1,a2,a3,a4 = st.columns(4)
            a1.metric("Normal Low",  f"{lo:.5f}")
            a2.metric("Normal High", f"{hi:.5f}")
            a3.metric("🔴 Spikes",   len(spikes))
            a4.metric("🟡 Dips",     len(dips))

            all_anom = (
                [{"month":s["month"],"share":s["share"],"raw":s["raw"],"type":"Spike"} for s in spikes]+
                [{"month":d["month"],"share":d["share"],"raw":d["raw"],"type":"Dip"}   for d in dips]
            )
            if all_anom:
                anom_df = pd.DataFrame(all_anom)
                figan = px.scatter(anom_df, x="month", y="share", color="type",
                                   color_discrete_map={"Spike":"#E05C5C","Dip":"#F0C93B"},
                                   size="raw", hover_data=["month","share","raw","type"],
                                   labels={"month":"Month","share":"Normalised Share","type":"Type"})
                figan.add_hline(y=hi, line_dash="dash", line_color="#E05C5C",
                                annotation_text="Upper IQR bound")
                figan.add_hline(y=lo, line_dash="dash", line_color="#F0C93B",
                                annotation_text="Lower IQR bound")
                figan.update_layout(**DARK, height=360)
                st.plotly_chart(figan, use_container_width=True)
                st.dataframe(anom_df.rename(columns={
                    "month":"Month","share":"Share","raw":"Raw Papers","type":"Type",
                }), use_container_width=True, hide_index=True)
            with st.expander("📖 Anomaly Interpretation"):
                st.markdown("""
- **Oct 2023 spike**: surge in CS/ML/LLM submissions post-ChatGPT.
- **Oct 2025 spike**: continued AI/ML growth.
- **Jan dips**: seasonal academic slow-down (post-holiday, pre-deadline lull).
""")
        else:
            st.info("Anomaly data not found. Ensure time_series_report.txt is in results/time_series/")

    # ── Tab 5: Keywords ───────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Top Keywords per Year (TF-IDF, full dataset)")
        kw_data = ts_data.get("keywords", [])
        if kw_data:
            kw_df = pd.DataFrame([
                {"year": r["year"], "rank": i+1, "keyword": kw}
                for r in kw_data for i, kw in enumerate(r["keywords"])
            ])
            pivot_kw = kw_df.pivot_table(index="keyword", columns="year",
                                          values="rank", aggfunc="min")
            pivot_kw.columns = [str(c) for c in pivot_kw.columns]
            st.dataframe(pivot_kw.style.background_gradient(cmap="YlOrRd_r", axis=None),
                         use_container_width=True)
            st.caption("Lower rank = higher TF-IDF score. Empty = not in top 5 that year.")
            for r in kw_data:
                st.markdown(f"**{r['year']}:** " + "  ".join(f"`{k}`" for k in r["keywords"]))
        else:
            st.info("Keyword data not found in report.")

    # ── Tab 6: Pipeline Figures — exact filenames from the screenshot ─────────
    with tabs[5]:
        fig_display = [
            ("fig1_annual_trends.png",         "Annual Trends"),
            ("fig2_keyword_heatmap.png",        "Keyword Heatmap"),
            ("fig3_keyword_trends.png",         "Keyword Trends"),
            ("fig4_category_trends.png",        "Category Trends"),
            ("fig5_growth_rates.png",           "Growth Rates"),
            ("fig6_category_keyword_drift.png", "Category Keyword Drift"),
            ("fig7_anomalies.png",              "Anomaly Detection"),
            ("fig8_arima_overall.png",          "ARIMA Overall"),
            ("fig9_arima_per_category.png",     "ARIMA per Category"),
            ("fig10_arima_comparison.png",      "ARIMA Comparison"),
        ]
        found_any = False
        for fname, label in fig_display:
            if fname in ts_figs:
                found_any = True
                with st.expander(f"📊 {label}", expanded=(fname=="fig1_annual_trends.png")):
                    st.image(ts_figs[fname], use_column_width=True)
            else:
                st.caption(f"*{label} ({fname}) — not found in results/time_series/*")
        if not found_any:
            st.warning("No figures found. Run time_series.py to generate them.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PIPELINE INFO
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Pipeline Info":
    st.title("⚙️ Pipeline Information")
    st.markdown("Technical details of every stage in the ResearchIQ pipeline.")

    st.subheader("🗂️ Clustering — K-Means")
    cc = cl_meta
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Method",           cc.get("method","K-Means"))
    c2.metric("Clusters (k)",     cc.get("parameters",{}).get("n_clusters",5))
    c3.metric("Silhouette Score", f"{cc.get('silhouette_score_sample',0.034):.4f}")
    c4.metric("SSE / Inertia",    f"{cc.get('inertia_sse',36815):.0f}")
    if cc.get("selection_reason"):
        st.info(cc["selection_reason"], icon="ℹ️")

    counts = cc.get("cluster_counts",{})
    if counts:
        df_c = pd.DataFrame([
            {"Cluster":int(k),"Name":CLUSTER_NAMES.get(int(k),"?"),"Papers":v}
            for k,v in counts.items()
        ])
        figcc = px.bar(df_c, x="Name", y="Papers", color="Name",
                       color_discrete_map={CLUSTER_NAMES[k]:v for k,v in CLUSTER_COLOURS.items()},
                       text_auto=True)
        figcc.update_layout(**DARK, height=300, showlegend=False)
        st.plotly_chart(figcc, use_container_width=True)
    st.caption(f"Runtime: {cc.get('runtime_seconds',19.99):.1f} s")

    st.divider()
    st.subheader("📐 Dimensionality Reduction")
    d1,d2 = st.columns(2)
    cmp = dim_meta.get("compression",{}); viz3 = dim_meta.get("visualization",{})
    with d1:
        st.markdown("#### PCA — Compression")
        p1,p2,p3 = st.columns(3)
        p1.metric("Components",     cmp.get("n_components",50))
        p2.metric("Var. Explained", f"{cmp.get('cumulative_explained_variance',0.444)*100:.1f}%")
        p3.metric("Runtime",        f"{cmp.get('runtime_seconds',0.78):.2f}s")
        st.progress(float(cmp.get("cumulative_explained_variance",0.444)),
                    text="Cumulative variance captured")
    with d2:
        st.markdown("#### UMAP — Visualisation")
        params = viz3.get("parameters",{})
        u1,u2,u3 = st.columns(3)
        u1.metric("n_neighbors", params.get("n_neighbors",15))
        u2.metric("min_dist",    params.get("min_dist",0.0))
        u3.metric("Runtime",     f"{viz3.get('runtime_seconds',59.5):.1f}s")
        st.json(params)

    st.divider()
    st.subheader("🔗 Association Rule Mining — Apriori")
    ar = pipe_meta.get("association_rules",{})
    r1,r2,r3,r4 = st.columns(4)
    r1.metric("Total Rules",     ar.get("rule_count",659))
    r2.metric("Min Support",     ar.get("min_support",0.001))
    r3.metric("Min Confidence",  ar.get("min_confidence",0.05))
    r4.metric("Max Itemset Len", ar.get("max_itemset_length",2))
    if ar.get("quality_filter"):
        st.caption(f"Quality filter: {ar['quality_filter']}")

    rtc = rules["rule_type"].value_counts().reset_index(); rtc.columns = ["Type","Count"]
    figrp = px.pie(rtc, names="Type", values="Count", hole=0.4, title="Rules by Type")
    figrp.update_layout(**DARK, height=300)
    st.plotly_chart(figrp, use_container_width=True)

    st.divider()
    st.subheader("🔍 Supported Search Modes")
    for m in pipe_meta.get("search_cases_supported",[]):
        st.markdown(f"✅ {m}")

    st.divider()
    st.subheader("Full Pipeline Summary JSON")
    with st.expander("View raw JSON"):
        st.json(pipe_meta)