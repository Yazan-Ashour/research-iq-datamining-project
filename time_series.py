import warnings
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")


PAPERS_PATH          = Path("data/prosecced/papers_timeseries.parquet")
OUTPUT_DIR           = Path("results/time_series")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_START           = 2015
YEAR_END             = 2026
PARTIAL_YEARS        = {2026}
MIN_MONTH_PAPERS     = 30
TOP_N_KEYWORDS_YEAR  = 10
TOP_N_KEYWORDS_CAT   = 6
TOP_N_HEATMAP        = 20
FORECAST_PERIODS     = 12
IQR_FACTOR           = 1.5
TFIDF_MAX_FEATURES   = 3000
TFIDF_NGRAM_RANGE    = (1, 2)
TFIDF_MIN_DF         = 5
TFIDF_MAX_DF         = 0.85

ARIMA_MAX_P          = 3
ARIMA_MAX_Q          = 3
ARIMA_MAX_D          = 2
ARIMA_SEASONAL       = False
ARIMA_IC             = "aic"

DARK_BG      = "#0F0F14"
PANEL_BG     = "#1A1A24"
SPINE_COL    = "#444444"
TICK_COL     = "#AAAAAA"
TEXT_COL     = "#CCCCCC"
WHITE        = "#FFFFFF"
PARTIAL_COL  = "#F0C93B"

CATEGORY_PALETTE = [
    "#4C8EDA", "#E05C5C", "#6ABF69", "#F0C93B",
    "#E8733A", "#B07FD4", "#4DBDBD", "#D4537E",
]


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TICK_COL)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(SPINE_COL)


def _savefig(fig, name):
    out = OUTPUT_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def _extract_month(paper_id):
    pid = str(paper_id).strip()
    m = re.match(r"^(\d{2})(\d{2})\.\d+", pid)
    if m:
        mm = int(m.group(2))
        if 1 <= mm <= 12:
            return mm
    m2 = re.search(r"/(\d{2})(\d{2})\d{3}$", pid)
    if m2:
        mm = int(m2.group(2))
        if 1 <= mm <= 12:
            return mm
    return None


def _partial_tag(year):
    return "  [PARTIAL]" if int(year) in PARTIAL_YEARS else ""


def load_data():
    if not PAPERS_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {PAPERS_PATH}")

    print("Loading data...")
    df = pd.read_parquet(PAPERS_PATH)
    df["year"] = df["year"].astype("Int64")
    df = df[
        (df["year"] >= YEAR_START) &
        (df["year"] <= YEAR_END)
    ].reset_index(drop=True)

    df["pub_month"] = df["paper_id"].apply(_extract_month).astype("Int64")
    valid = df["pub_month"].notna()
    df.loc[valid, "year_month"] = pd.to_datetime(
        df.loc[valid, "year"].astype(int).astype(str)
        + "-"
        + df.loc[valid, "pub_month"].astype(int).astype(str).str.zfill(2),
        format="%Y-%m",
    ).dt.to_period("M")

    year_totals        = df.groupby("year")["paper_id"].transform("count")
    df["norm_weight"]  = 1.0 / year_totals
    df["is_partial_year"] = df["year"].isin(PARTIAL_YEARS)

    categories = sorted(df["category_group"].unique())
    colour_map = {
        cat: CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)]
        for i, cat in enumerate(categories)
    }

    print(f"  Papers      : {len(df):,}")
    print(f"  Year range  : {int(df['year'].min())} – {int(df['year'].max())}")
    print(f"  Categories  : {categories}")
    print(f"  Partial yrs : {sorted(df[df['is_partial_year']]['year'].unique().tolist())}")

    df.attrs["colour_map"] = colour_map
    df.attrs["categories"] = categories
    return df


def compute_annual_aggregates(df):
    raw_annual = (
        df.groupby(["year", "is_partial_year"])
        .size()
        .reset_index(name="count")
        .sort_values("year")
    )
    norm_by_cat = (
        df.groupby(["year", "category_group", "is_partial_year"])["norm_weight"]
        .sum()
        .reset_index(name="share")
        .sort_values(["year", "category_group"])
    )
    return raw_annual, norm_by_cat


def compute_monthly_aggregates(df):
    monthly = (
        df.dropna(subset=["year_month"])
        .groupby(["year_month", "is_partial_year"])
        .agg(
            count=("paper_id", "count"),
            norm_share=("norm_weight", "sum"),
        )
        .reset_index()
        .sort_values("year_month")
    )
    monthly["year_month_dt"] = monthly["year_month"].dt.to_timestamp()
    monthly = monthly[monthly["count"] >= MIN_MONTH_PAPERS].reset_index(drop=True)
    return monthly


def compute_monthly_by_category(df):
    monthly_cat = (
        df.dropna(subset=["year_month"])
        .groupby(["year_month", "category_group", "is_partial_year"])
        .agg(
            count=("paper_id", "count"),
            norm_share=("norm_weight", "sum"),
        )
        .reset_index()
        .sort_values(["category_group", "year_month"])
    )
    monthly_cat["year_month_dt"] = monthly_cat["year_month"].dt.to_timestamp()
    monthly_cat = monthly_cat[
        monthly_cat["count"] >= max(MIN_MONTH_PAPERS // 4, 5)
    ].reset_index(drop=True)
    return monthly_cat


def extract_keywords_per_year(df):
    result = {}
    for year, grp in df.groupby("year"):
        if not (YEAR_START <= int(year) <= YEAR_END):
            continue
        texts = grp["abstract_clean"].fillna("").tolist()
        if len(texts) < 10:
            result[int(year)] = []
            continue
        try:
            vec = TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                stop_words="english",
                ngram_range=TFIDF_NGRAM_RANGE,
                min_df=TFIDF_MIN_DF,
                max_df=TFIDF_MAX_DF,
            )
            matrix      = vec.fit_transform(texts)
            mean_scores = matrix.mean(axis=0).A1
            features    = vec.get_feature_names_out()
            top_idx     = mean_scores.argsort()[::-1][:TOP_N_KEYWORDS_YEAR]
            result[int(year)] = [
                (features[i], round(float(mean_scores[i]), 6))
                for i in top_idx
            ]
        except Exception:
            result[int(year)] = []
    return result


def derive_trending_keywords(kw_by_year, top_n=8):
    totals = Counter()
    for kws in kw_by_year.values():
        for kw, score in kws:
            totals[kw] += score
    return [kw for kw, _ in totals.most_common(top_n)]


def track_keywords_over_time(df, keywords):
    rows = []
    for year, grp in df.groupby("year"):
        if not (YEAR_START <= int(year) <= YEAR_END):
            continue
        total  = len(grp)
        corpus = " ".join(grp["abstract_clean"].fillna("").str.lower())
        for kw in keywords:
            cnt = corpus.count(kw.lower())
            rows.append({
                "year"            : int(year),
                "keyword"         : kw,
                "count"           : cnt,
                "rate"            : cnt / max(total, 1),
                "is_partial_year" : int(year) in PARTIAL_YEARS,
            })
    return pd.DataFrame(rows).sort_values(["keyword", "year"])


def compute_yoy_growth(norm_by_cat):
    rows = []
    for cat, grp in norm_by_cat.groupby("category_group"):
        grp = grp.sort_values("year").copy()
        for i in range(1, len(grp)):
            prev = grp.iloc[i - 1]["share"]
            curr = grp.iloc[i]["share"]
            year = int(grp.iloc[i]["year"])
            yoy  = (curr - prev) / prev * 100 if prev > 0 else None
            rows.append({
                "year"            : year,
                "category_group"  : cat,
                "share"           : round(curr, 6),
                "yoy_pct"         : round(yoy, 2) if yoy is not None else None,
                "is_partial_year" : year in PARTIAL_YEARS,
            })
    return pd.DataFrame(rows)


def detect_anomalies(monthly):
    monthly  = monthly.copy()
    complete = monthly[~monthly["is_partial_year"]]
    q1  = complete["norm_share"].quantile(0.25)
    q3  = complete["norm_share"].quantile(0.75)
    iqr = q3 - q1
    lo  = q1 - IQR_FACTOR * iqr
    hi  = q3 + IQR_FACTOR * iqr

    monthly["anomaly"]      = False
    monthly["anomaly_type"] = ""
    mask_hi = (~monthly["is_partial_year"]) & (monthly["norm_share"] > hi)
    mask_lo = (~monthly["is_partial_year"]) & (monthly["norm_share"] < lo)
    monthly.loc[mask_hi, ["anomaly", "anomaly_type"]] = [True, "spike"]
    monthly.loc[mask_lo, ["anomaly", "anomaly_type"]] = [True, "dip"]
    return monthly, (lo, hi)


def _fit_arima(series, label="series"):
    """
    Fit an ARIMA model using auto_arima on a pandas Series with a DatetimeIndex.
    Returns (fitted_model, order_tuple) or (None, None) if pmdarima is missing.
    """
    try:
        from pmdarima import auto_arima
    except ImportError:
        print("  pmdarima not installed — pip install pmdarima")
        return None, None

    try:
        model = auto_arima(
            series,
            max_p=ARIMA_MAX_P,
            max_q=ARIMA_MAX_Q,
            max_d=ARIMA_MAX_D,
            seasonal=ARIMA_SEASONAL,
            information_criterion=ARIMA_IC,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        order = model.order
        print(f"    {label:<25} → ARIMA{order}  AIC={model.aic():.2f}")
        return model, order
    except Exception as e:
        print(f"    {label:<25} → fit failed: {e}")
        return None, None


def _arima_forecast_df(model, series, n_periods, label):
    """
    Generate forecast from a fitted auto_arima model.
    Returns a DataFrame with ds, yhat, yhat_lower, yhat_upper.
    """
    try:
        preds, conf = model.predict(n_periods=n_periods, return_conf_int=True, alpha=0.05)
        last_date   = series.index[-1]
        future_idx  = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=n_periods,
            freq="MS",
        )
        return pd.DataFrame({
            "ds"          : future_idx,
            "yhat"        : preds,
            "yhat_lower"  : conf[:, 0],
            "yhat_upper"  : conf[:, 1],
            "label"       : label,
        })
    except Exception as e:
        print(f"    Forecast failed for {label}: {e}")
        return pd.DataFrame()


def _arima_fitted_values(model, series):
    """Return in-sample fitted values aligned to the training series index."""
    try:
        fitted = model.predict_in_sample()
        return pd.Series(fitted, index=series.index)
    except Exception:
        return pd.Series(dtype=float)


def _mape(actual, predicted):
    mask = actual != 0
    if mask.sum() == 0:
        return None
    return round(float((abs(actual[mask] - predicted[mask]) / actual[mask]).mean() * 100), 2)


def run_arima_overall(monthly):
    """
    Fit one ARIMA model on the overall normalised monthly share
    (complete years only) and forecast FORECAST_PERIODS months ahead.

    Returns (series, fitted_values, forecast_df, model, mape).
    """
    print("\n── ARIMA — overall series...")
    complete = monthly[~monthly["is_partial_year"]].copy()
    complete = complete.set_index("year_month_dt").sort_index()
    series   = complete["norm_share"].asfreq("MS")

    model, order = _fit_arima(series, label="overall")
    if model is None:
        return series, None, pd.DataFrame(), None, None

    fitted   = _arima_fitted_values(model, series)
    forecast = _arima_forecast_df(model, series, FORECAST_PERIODS, "overall")
    mape_val = _mape(series, fitted) if len(fitted) > 0 else None

    if mape_val is not None:
        print(f"    Overall MAPE (in-sample): {mape_val:.2f}%")
    return series, fitted, forecast, model, mape_val


def run_arima_per_category(monthly_cat, categories):
    """
    Fit one ARIMA model per category on normalised monthly share
    (complete years only). Returns a dict keyed by category with
    (series, fitted, forecast_df, model, mape, order).
    """
    print("\n── ARIMA — per category...")
    results = {}
    for cat in categories:
        sub      = monthly_cat[
            (monthly_cat["category_group"] == cat) &
            (~monthly_cat["is_partial_year"])
        ].copy()
        sub = sub.set_index("year_month_dt").sort_index()
        if len(sub) < 12:
            print(f"    {cat:<25} → not enough data, skipping")
            continue

        series = sub["norm_share"].asfreq("MS")
        series = series.interpolate(method="time")

        model, order = _fit_arima(series, label=cat)
        if model is None:
            continue

        fitted   = _arima_fitted_values(model, series)
        forecast = _arima_forecast_df(model, series, FORECAST_PERIODS, cat)
        mape_val = _mape(series, fitted) if len(fitted) > 0 else None
        results[cat] = (series, fitted, forecast, model, mape_val, order)

    return results


def plot_annual_trends(raw_annual, norm_by_cat, colour_map, categories):
    fig, (ax_raw, ax_norm) = plt.subplots(2, 1, figsize=(14, 12), facecolor=DARK_BG)
    fig.suptitle("Annual Publication Trends  |  ResearchIQ",
                 fontsize=14, color=WHITE, fontweight="bold", y=1.01)

    for ax in (ax_raw, ax_norm):
        _style_ax(ax)

    complete = raw_annual[~raw_annual["is_partial_year"]]
    partial  = raw_annual[raw_annual["is_partial_year"]]

    ax_raw.bar(complete["year"].astype(int), complete["count"],
               color="#4C8EDA", alpha=0.75, width=0.7)
    ax_raw.bar(partial["year"].astype(int), partial["count"],
               color=PARTIAL_COL, alpha=0.55, width=0.7, label="Partial year (2026)")
    ax_raw.plot(raw_annual["year"].astype(int), raw_annual["count"],
                color=WHITE, lw=1.5, marker="o", ms=4)
    for _, row in raw_annual.iterrows():
        ax_raw.text(
            int(row["year"]),
            row["count"] + raw_annual["count"].max() * 0.01,
            f'{int(row["count"]):,}',
            ha="center", color=TEXT_COL, fontsize=6.5,
        )
    ax_raw.set_title("Raw Paper Count per Year", color=TEXT_COL, fontsize=11)
    ax_raw.set_ylabel("Papers", color=TICK_COL)
    ax_raw.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_raw.tick_params(axis="x", rotation=45)
    ax_raw.legend(fontsize=8, framealpha=0.3, labelcolor=WHITE, facecolor=PANEL_BG)

    pivot       = norm_by_cat.pivot_table(
        index="year", columns="category_group", values="share", fill_value=0
    )
    pivot.index = pivot.index.astype(int)
    bottoms     = np.zeros(len(pivot))
    for cat in categories:
        if cat not in pivot.columns:
            continue
        ax_norm.bar(pivot.index, pivot[cat], bottom=bottoms,
                    color=colour_map.get(cat, "#888"), alpha=0.88, width=0.7, label=cat)
        bottoms += pivot[cat].values

    for y in PARTIAL_YEARS:
        if y in pivot.index:
            ax_norm.bar(y, 1.0, color="none", edgecolor=PARTIAL_COL,
                        linewidth=1.5, width=0.7, zorder=5)

    ax_norm.set_title("Research Category Share per Year  (normalised)",
                      color=TEXT_COL, fontsize=11)
    ax_norm.set_ylabel("Share of Year's Papers", color=TICK_COL)
    ax_norm.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax_norm.tick_params(axis="x", rotation=45)
    ax_norm.legend(loc="upper right", fontsize=8, framealpha=0.3,
                   labelcolor=WHITE, facecolor=PANEL_BG, ncol=2)

    plt.tight_layout()
    _savefig(fig, "fig1_annual_trends.png")


def plot_keyword_heatmap(kw_by_year):
    score_map = defaultdict(lambda: defaultdict(float))
    all_kws   = set()
    for yr, kws in kw_by_year.items():
        for kw, sc in kws:
            score_map[kw][yr] = sc
            all_kws.add(kw)

    years   = sorted(kw_by_year.keys())
    totals  = {kw: sum(score_map[kw].values()) for kw in all_kws}
    top_kws = sorted(totals, key=lambda k: -totals[k])[:TOP_N_HEATMAP]
    matrix  = np.array([
        [score_map[kw].get(yr, 0.0) for yr in years]
        for kw in top_kws
    ])

    fig, ax = plt.subplots(figsize=(max(9, len(years)), 7), facecolor=DARK_BG)
    _style_ax(ax)
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    xlabels = [f"{y}*" if y in PARTIAL_YEARS else str(y) for y in years]
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", color=TEXT_COL, fontsize=9)
    ax.set_yticks(range(len(top_kws)))
    ax.set_yticklabels(top_kws, color=TEXT_COL, fontsize=9)
    ax.set_title("Top Keywords by Year — Mean TF-IDF Score  (* = partial year)",
                 color=WHITE, fontsize=12, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, label="Mean TF-IDF Score")
    cbar.ax.yaxis.label.set_color(TICK_COL)
    cbar.ax.tick_params(colors=TICK_COL)
    plt.tight_layout()
    _savefig(fig, "fig2_keyword_heatmap.png")


def plot_keyword_trends(kw_df):
    keywords = kw_df["keyword"].unique().tolist()
    palette  = plt.cm.get_cmap("tab10", len(keywords))

    fig, ax = plt.subplots(figsize=(13, 5), facecolor=DARK_BG)
    _style_ax(ax)
    for i, kw in enumerate(keywords):
        sub      = kw_df[kw_df["keyword"] == kw]
        complete = sub[~sub["is_partial_year"]]
        partial  = sub[sub["is_partial_year"]]
        color    = palette(i)
        ax.plot(complete["year"], complete["rate"],
                label=kw, color=color, lw=2, marker="o", ms=5)
        if not partial.empty:
            ax.plot(partial["year"], partial["rate"],
                    color=color, lw=2, linestyle="--", marker="o", ms=5, alpha=0.6)

    ax.set_title("Keyword Frequency Over Time  (dashed = partial year)",
                 color=WHITE, fontsize=12, fontweight="bold")
    ax.set_xlabel("Year", color=TICK_COL)
    ax.set_ylabel("Occurrences / Papers in Year", color=TICK_COL)
    ax.legend(fontsize=8, framealpha=0.3, labelcolor=WHITE, facecolor=PANEL_BG)
    plt.tight_layout()
    _savefig(fig, "fig3_keyword_trends.png")


def plot_category_trends(norm_by_cat, colour_map, categories):
    fig, (ax_line, ax_heat) = plt.subplots(2, 1, figsize=(14, 12), facecolor=DARK_BG)
    fig.suptitle("Research Category Trends Over Time  |  Normalised Share",
                 fontsize=14, color=WHITE, fontweight="bold")

    for ax in (ax_line, ax_heat):
        _style_ax(ax)

    for cat in categories:
        sub      = norm_by_cat[norm_by_cat["category_group"] == cat]
        complete = sub[~sub["is_partial_year"]]
        partial  = sub[sub["is_partial_year"]]
        col      = colour_map.get(cat, "#888")
        ax_line.plot(complete["year"].astype(int), complete["share"],
                     label=cat, color=col, lw=2.5, marker="o", ms=6)
        if not partial.empty:
            ax_line.plot(partial["year"].astype(int), partial["share"],
                         color=col, lw=2.5, linestyle="--", marker="o", ms=6, alpha=0.6)

    ax_line.set_ylabel("Share of Year's Papers", color=TICK_COL)
    ax_line.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax_line.tick_params(axis="x", rotation=45)
    ax_line.legend(fontsize=9, framealpha=0.3, labelcolor=WHITE,
                   facecolor=PANEL_BG, ncol=2)
    ax_line.set_title("Normalised Share per Category per Year  (dashed = partial)",
                      color=TEXT_COL, fontsize=11)

    pivot = norm_by_cat.pivot_table(
        index="category_group", columns="year", values="share", fill_value=0
    )
    im = ax_heat.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    xlabels = [f"{int(c)}*" if int(c) in PARTIAL_YEARS else str(int(c)) for c in pivot.columns]
    ax_heat.set_xticks(range(len(pivot.columns)))
    ax_heat.set_xticklabels(xlabels, rotation=45, ha="right", color=TEXT_COL, fontsize=9)
    ax_heat.set_yticks(range(len(pivot.index)))
    ax_heat.set_yticklabels(pivot.index, color=TEXT_COL, fontsize=9)
    ax_heat.set_title("Category Share Heatmap  (* = partial year)",
                      color=TEXT_COL, fontsize=11)
    cbar = plt.colorbar(im, ax=ax_heat, label="Normalised Share")
    cbar.ax.yaxis.label.set_color(TICK_COL)
    cbar.ax.tick_params(colors=TICK_COL)
    plt.tight_layout()
    _savefig(fig, "fig4_category_trends.png")


def plot_growth_rates(yoy_df, norm_by_cat, colour_map, categories):
    fig, (ax_yoy, ax_cagr) = plt.subplots(1, 2, figsize=(16, 6), facecolor=DARK_BG)
    fig.suptitle("Growth Analysis  |  Normalised Share — ResearchIQ",
                 color=WHITE, fontsize=14, fontweight="bold")

    for ax in (ax_yoy, ax_cagr):
        _style_ax(ax)

    for cat in categories:
        sub      = yoy_df[yoy_df["category_group"] == cat].dropna(subset=["yoy_pct"])
        complete = sub[~sub["is_partial_year"]]
        partial  = sub[sub["is_partial_year"]]
        col      = colour_map.get(cat, "#888")
        ax_yoy.plot(complete["year"].astype(int), complete["yoy_pct"],
                    label=cat, color=col, lw=2, marker="o", ms=5)
        if not partial.empty:
            ax_yoy.plot(partial["year"].astype(int), partial["yoy_pct"],
                        color=col, lw=2, linestyle="--", marker="o", ms=5, alpha=0.6)

    ax_yoy.axhline(0, color=WHITE, lw=0.8, linestyle="--", alpha=0.5)
    ax_yoy.set_title("YoY Growth on Normalised Share (%)", color=TEXT_COL, fontsize=11)
    ax_yoy.set_ylabel("YoY Growth %", color=TICK_COL)
    ax_yoy.tick_params(axis="x", rotation=45)
    ax_yoy.legend(fontsize=8, framealpha=0.3, labelcolor=WHITE, facecolor=PANEL_BG)

    cagr_rows    = []
    complete_norm = norm_by_cat[~norm_by_cat["is_partial_year"]]
    for cat, grp in complete_norm.groupby("category_group"):
        grp  = grp.sort_values("year")
        n    = grp["year"].max() - grp["year"].min()
        if n > 0:
            s    = grp.iloc[0]["share"]
            e    = grp.iloc[-1]["share"]
            cagr = ((e / max(s, 1e-9)) ** (1 / n) - 1) * 100
            cagr_rows.append({"label": cat, "cagr": cagr, "cat": cat})

    cagr_df = pd.DataFrame(cagr_rows).sort_values("cagr", ascending=True)
    cols    = [colour_map.get(r["cat"], "#888") for _, r in cagr_df.iterrows()]
    ax_cagr.barh(cagr_df["label"], cagr_df["cagr"], color=cols, alpha=0.88)
    ax_cagr.axvline(0, color=WHITE, lw=0.8, linestyle="--", alpha=0.5)
    ax_cagr.set_title("CAGR by Category  (complete years only, %)",
                      color=TEXT_COL, fontsize=11)
    ax_cagr.set_xlabel("CAGR (%)", color=TICK_COL)
    ax_cagr.tick_params(axis="y", labelcolor=TEXT_COL)
    plt.tight_layout()
    _savefig(fig, "fig5_growth_rates.png")


def plot_category_keyword_drift(df, colour_map, categories):
    n_cat = len(categories)
    fig, axes = plt.subplots(n_cat, 1, figsize=(14, 5 * n_cat), facecolor=DARK_BG)
    fig.suptitle("Keyword Drift per Category  |  Normalised Rate",
                 fontsize=14, color=WHITE, fontweight="bold", y=1.0)
    if n_cat == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories):
        _style_ax(ax)
        sub = df[df["category_group"] == cat]
        try:
            vec = TfidfVectorizer(
                max_features=500, stop_words="english",
                ngram_range=(1, 1), min_df=3,
            )
            matrix   = vec.fit_transform(sub["abstract_clean"].fillna(""))
            scores   = matrix.mean(axis=0).A1
            features = vec.get_feature_names_out()
            top_kws  = [features[i] for i in scores.argsort()[::-1][:TOP_N_KEYWORDS_CAT]]
        except Exception:
            top_kws = []

        palette = plt.cm.get_cmap("tab10", max(len(top_kws), 1))
        for i, kw in enumerate(top_kws):
            rows = []
            for year, grp in sub.groupby("year"):
                if not (YEAR_START <= int(year) <= YEAR_END):
                    continue
                total = len(grp)
                cnt   = grp["abstract_clean"].fillna("").str.lower().str.count(
                    rf"\b{re.escape(kw)}\b"
                ).sum()
                rows.append({
                    "year"            : int(year),
                    "rate"            : cnt / max(total, 1),
                    "is_partial_year" : int(year) in PARTIAL_YEARS,
                })
            if rows:
                kdf      = pd.DataFrame(rows).sort_values("year")
                complete = kdf[~kdf["is_partial_year"]]
                partial  = kdf[kdf["is_partial_year"]]
                color    = palette(i)
                ax.plot(complete["year"], complete["rate"],
                        label=kw, color=color, lw=2, marker="o", ms=4)
                if not partial.empty:
                    ax.plot(partial["year"], partial["rate"],
                            color=color, lw=2, linestyle="--", marker="o", ms=4, alpha=0.6)

        ax.set_title(f"{cat}  —  Keyword Drift",
                     color=colour_map.get(cat, WHITE), fontsize=10)
        ax.set_ylabel("Keyword rate", color=TICK_COL)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2%}"))
        ax.legend(fontsize=7.5, framealpha=0.3, labelcolor=WHITE,
                  facecolor=PANEL_BG, ncol=2)

    plt.tight_layout()
    _savefig(fig, "fig6_category_keyword_drift.png")


def plot_anomalies(monthly_anom, bounds):
    lo, hi  = bounds
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=DARK_BG)
    _style_ax(ax)

    complete = monthly_anom[~monthly_anom["is_partial_year"]]
    partial  = monthly_anom[monthly_anom["is_partial_year"]]
    spikes   = complete[complete["anomaly_type"] == "spike"]
    dips     = complete[complete["anomaly_type"] == "dip"]

    ax.plot(complete["year_month_dt"], complete["norm_share"],
            color="#4C8EDA", lw=1.5, alpha=0.85)
    ax.plot(partial["year_month_dt"], partial["norm_share"],
            color=PARTIAL_COL, lw=1.5, linestyle="--", alpha=0.7, label="Partial year (2026)")
    ax.scatter(spikes["year_month_dt"], spikes["norm_share"],
               color="#E05C5C", s=65, zorder=5,
               label=f"Spike (>{hi:.4f})  n={len(spikes)}")
    ax.scatter(dips["year_month_dt"], dips["norm_share"],
               color="#F0C93B", s=65, zorder=5,
               label=f"Dip (<{lo:.4f})  n={len(dips)}")
    ax.axhline(hi, color="#E05C5C", lw=1, linestyle="--", alpha=0.5)
    ax.axhline(lo, color="#F0C93B", lw=1, linestyle="--", alpha=0.5)
    ax.set_title(
        f"Anomaly Detection — Normalised Monthly Share  "
        f"(IQR × {IQR_FACTOR}, min {MIN_MONTH_PAPERS} papers/month)",
        color=WHITE, fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Month", color=TICK_COL)
    ax.set_ylabel("Normalised Share", color=TICK_COL)
    ax.legend(fontsize=9, framealpha=0.3, labelcolor=WHITE, facecolor=PANEL_BG)
    plt.tight_layout()
    _savefig(fig, "fig7_anomalies.png")


def plot_arima_overall(series, fitted, forecast_df, mape):
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=DARK_BG)
    _style_ax(ax)

    ax.bar(series.index, series.values, width=20,
           color="#4C8EDA", alpha=0.5, label="Actual (normalised share)")
    if fitted is not None and len(fitted):
        ax.plot(fitted.index, fitted.values,
                color="#E8733A", lw=1.5, label="ARIMA fitted")
    if not forecast_df.empty:
        ax.plot(forecast_df["ds"], forecast_df["yhat"],
                color="#F0C93B", lw=2, linestyle="--", label="ARIMA forecast")
        ax.fill_between(forecast_df["ds"],
                        forecast_df["yhat_lower"], forecast_df["yhat_upper"],
                        color="#F0C93B", alpha=0.2, label="95% CI")
        cutoff = series.index[-1]
        ax.axvline(cutoff, color=WHITE, lw=1, linestyle=":", alpha=0.6)
        ax.text(cutoff, ax.get_ylim()[1] * 0.92, "  Forecast →",
                color=WHITE, fontsize=9)

    title = "Overall Monthly Volume — ARIMA Forecast"
    if mape is not None:
        title += f"  |  In-sample MAPE: {mape:.2f}%"
    ax.set_title(title, color=WHITE, fontsize=13, fontweight="bold")
    ax.set_xlabel("Month", color=TICK_COL)
    ax.set_ylabel("Normalised Share", color=TICK_COL)
    ax.legend(fontsize=9, framealpha=0.3, labelcolor=WHITE, facecolor=PANEL_BG)
    plt.tight_layout()
    _savefig(fig, "fig8_arima_overall.png")


def plot_arima_per_category(arima_results, colour_map):
    n_cat = len(arima_results)
    if n_cat == 0:
        return

    fig, axes = plt.subplots(n_cat, 1, figsize=(14, 5 * n_cat), facecolor=DARK_BG)
    fig.suptitle("Per-Category ARIMA Forecasts  |  Normalised Monthly Share",
                 fontsize=14, color=WHITE, fontweight="bold", y=1.0)
    if n_cat == 1:
        axes = [axes]

    for ax, (cat, (series, fitted, forecast_df, model, mape, order)) in zip(axes, arima_results.items()):
        _style_ax(ax)
        col = colour_map.get(cat, "#4C8EDA")

        ax.bar(series.index, series.values, width=20, color=col, alpha=0.4, label="Actual")
        if fitted is not None and len(fitted):
            ax.plot(fitted.index, fitted.values, color=WHITE, lw=1.2, alpha=0.8, label="Fitted")
        if not forecast_df.empty:
            ax.plot(forecast_df["ds"], forecast_df["yhat"],
                    color=col, lw=2, linestyle="--", label="Forecast")
            ax.fill_between(forecast_df["ds"],
                            forecast_df["yhat_lower"], forecast_df["yhat_upper"],
                            color=col, alpha=0.15, label="95% CI")
            cutoff = series.index[-1]
            ax.axvline(cutoff, color=WHITE, lw=0.8, linestyle=":", alpha=0.5)

        title = f"{cat}  —  ARIMA{order}"
        if mape is not None:
            title += f"  |  MAPE: {mape:.2f}%"
        ax.set_title(title, color=colour_map.get(cat, WHITE), fontsize=10)
        ax.set_ylabel("Normalised Share", color=TICK_COL)
        ax.legend(fontsize=8, framealpha=0.3, labelcolor=WHITE, facecolor=PANEL_BG)

    plt.tight_layout()
    _savefig(fig, "fig9_arima_per_category.png")


def plot_arima_comparison(arima_results, colour_map):
    """All category forecasts on a single axes for direct comparison."""
    if not arima_results:
        return

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=DARK_BG)
    _style_ax(ax)

    for cat, (series, fitted, forecast_df, model, mape, order) in arima_results.items():
        col = colour_map.get(cat, "#888")
        ax.plot(series.index, series.values, color=col, lw=1, alpha=0.4)
        if not forecast_df.empty:
            full_x = list(series.index) + list(forecast_df["ds"])
            full_y = list(series.values) + list(forecast_df["yhat"])
            ax.plot(full_x[-FORECAST_PERIODS - 3:], full_y[-FORECAST_PERIODS - 3:],
                    color=col, lw=2.2, linestyle="--", label=f"{cat} ARIMA{order}")

    cutoff = max(s.index[-1] for s, *_ in arima_results.values())
    ax.axvline(cutoff, color=WHITE, lw=1, linestyle=":", alpha=0.6)
    ax.text(cutoff, ax.get_ylim()[1] * 0.95, "  Forecast →", color=WHITE, fontsize=9)
    ax.set_title("Category Forecast Comparison  (dashed = forecast horizon)",
                 color=WHITE, fontsize=13, fontweight="bold")
    ax.set_xlabel("Month", color=TICK_COL)
    ax.set_ylabel("Normalised Share", color=TICK_COL)
    ax.legend(fontsize=8, framealpha=0.3, labelcolor=WHITE,
              facecolor=PANEL_BG, ncol=2)
    plt.tight_layout()
    _savefig(fig, "fig10_arima_comparison.png")


def write_report(df, raw_annual, norm_by_cat, yoy_df,
                 kw_by_year, monthly_anom, bounds,
                 series_overall, forecast_overall, mape_overall,
                 arima_results, categories):
    sep   = "=" * 68
    lines = [sep, "ResearchIQ — Time Series Analysis Report", sep, ""]

    lines += [
        "DATASET OVERVIEW", "-" * 50,
        f"  Total papers  : {len(df):,}",
        f"  Year range    : {int(df['year'].min())} – {int(df['year'].max())}",
        f"  Partial years : {sorted(PARTIAL_YEARS)}  (flagged, not excluded from charts)",
        f"  Categories    : {categories}",
        "",
    ]

    lines += ["ANNUAL VOLUME", "-" * 50]
    for _, row in raw_annual.iterrows():
        lines.append(
            f"  {int(row['year'])} : {int(row['count']):>8,} papers{_partial_tag(row['year'])}"
        )
    lines.append("")

    lines += ["DOMINANT CATEGORY PER YEAR  (normalised share)", "-" * 50]
    for year, grp in norm_by_cat.groupby("year"):
        top = grp.sort_values("share", ascending=False).iloc[0]
        lines.append(
            f"  {int(year)}  →  {top['category_group']} "
            f"({top['share']:.1%}){_partial_tag(year)}"
        )
    lines.append("")

    lines += ["YoY GROWTH PER CATEGORY  (complete years only)", "-" * 50]
    complete_yoy = yoy_df[~yoy_df["is_partial_year"]]
    for cat in categories:
        sub = complete_yoy[complete_yoy["category_group"] == cat].dropna(subset=["yoy_pct"])
        lines.append(f"  {cat}")
        for _, row in sub.iterrows():
            sign = "+" if row["yoy_pct"] >= 0 else ""
            lines.append(f"    {int(row['year'])} : {sign}{row['yoy_pct']:.1f}%")
    lines.append("")

    lines += ["TOP KEYWORDS PER YEAR  (mean TF-IDF)", "-" * 50]
    for yr in sorted(kw_by_year):
        kws = ", ".join(k for k, _ in kw_by_year[yr][:5])
        lines.append(f"  {yr}{_partial_tag(yr)} : {kws}")
    lines.append("")

    lo, hi = bounds
    spikes = monthly_anom[monthly_anom["anomaly_type"] == "spike"]
    dips   = monthly_anom[monthly_anom["anomaly_type"] == "dip"]
    lines += [
        f"ANOMALY DETECTION  (IQR × {IQR_FACTOR}, "
        f"min {MIN_MONTH_PAPERS} papers/month, complete years only)",
        "-" * 50,
        f"  Normal range : {lo:.5f} – {hi:.5f}",
        f"  Spikes       : {len(spikes)}",
    ]
    for _, row in spikes.iterrows():
        lines.append(
            f"    {row['year_month_dt'].strftime('%Y-%m')}  "
            f"share={row['norm_share']:.5f}  raw={int(row['count']):,}"
        )
    lines.append(f"  Dips         : {len(dips)}")
    for _, row in dips.iterrows():
        lines.append(
            f"    {row['year_month_dt'].strftime('%Y-%m')}  "
            f"share={row['norm_share']:.5f}  raw={int(row['count']):,}"
        )
    lines.append("")

    lines += ["ARIMA — OVERALL SERIES", "-" * 50]
    if mape_overall is not None:
        lines.append(f"  In-sample MAPE : {mape_overall:.2f}%")
    if forecast_overall is not None and not forecast_overall.empty:
        lines.append(f"  Forecast horizon : {FORECAST_PERIODS} months")
        for _, row in forecast_overall.iterrows():
            lines.append(
                f"  {row['ds'].strftime('%Y-%m')}  "
                f"~{row['yhat']:.5f}  "
                f"[{row['yhat_lower']:.5f} – {row['yhat_upper']:.5f}]"
            )
    lines.append("")

    lines += ["ARIMA — PER CATEGORY", "-" * 50]
    for cat, (series, fitted, forecast_df, model, mape, order) in arima_results.items():
        mape_str = f"{mape:.2f}%" if mape is not None else "N/A"
        lines.append(f"  {cat:<22} ARIMA{order}  MAPE={mape_str}")
        if not forecast_df.empty:
            first = forecast_df.iloc[0]
            last  = forecast_df.iloc[-1]
            lines.append(
                f"    Forecast: {first['ds'].strftime('%Y-%m')} → "
                f"{last['ds'].strftime('%Y-%m')}  "
                f"range [{forecast_df['yhat'].min():.5f} – {forecast_df['yhat'].max():.5f}]"
            )
    lines.append("")

    lines += [sep, "End of Report", sep]
    out = OUTPUT_DIR / "time_series_report.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {out}")


def run_pipeline():
    print("\n" + "=" * 60)
    print("ResearchIQ — Time Series Pipeline  (ARIMA)")
    print("=" * 60 + "\n")

    df         = load_data()
    colour_map = df.attrs["colour_map"]
    categories = df.attrs["categories"]

    print("\n── Annual aggregates...")
    raw_annual, norm_by_cat = compute_annual_aggregates(df)

    print("── Monthly aggregates (overall)...")
    monthly = compute_monthly_aggregates(df)
    print(f"  Months retained: {len(monthly)}")

    print("── Monthly aggregates (per category)...")
    monthly_cat = compute_monthly_by_category(df)

    print("── Keyword extraction...")
    kw_by_year   = extract_keywords_per_year(df)
    trending_kws = derive_trending_keywords(kw_by_year)
    print(f"  Trending keywords: {trending_kws}")
    kw_df = track_keywords_over_time(df, trending_kws)

    print("── YoY growth...")
    yoy_df = compute_yoy_growth(norm_by_cat)

    print("── Anomaly detection...")
    monthly_anom, bounds = detect_anomalies(monthly)

    series_overall, fitted_overall, forecast_overall, model_overall, mape_overall = \
        run_arima_overall(monthly)

    arima_results = run_arima_per_category(monthly_cat, categories)

    print("\n── Generating figures...")
    plot_annual_trends(raw_annual, norm_by_cat, colour_map, categories)
    plot_keyword_heatmap(kw_by_year)
    plot_keyword_trends(kw_df)
    plot_category_trends(norm_by_cat, colour_map, categories)
    plot_growth_rates(yoy_df, norm_by_cat, colour_map, categories)
    plot_category_keyword_drift(df, colour_map, categories)
    plot_anomalies(monthly_anom, bounds)
    plot_arima_overall(series_overall, fitted_overall, forecast_overall, mape_overall)
    plot_arima_per_category(arima_results, colour_map)
    plot_arima_comparison(arima_results, colour_map)

    print("── Writing report...")
    write_report(
        df, raw_annual, norm_by_cat, yoy_df,
        kw_by_year, monthly_anom, bounds,
        series_overall, forecast_overall, mape_overall,
        arima_results, categories,
    )

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete — outputs in: {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()