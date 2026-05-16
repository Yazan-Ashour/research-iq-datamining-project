"""
ResearchIQ — Time Series Data Extractor
========================================
Extracts a SEPARATE dataset from the raw ArXiv JSONL specifically
for the time series pipeline.

Filters applied:
  - update_date year between 2015 and 2026 (inclusive)
  - primary category matches one of the 8 CATEGORY_GROUPS
  - abstract length >= MIN_ABSTRACT_LEN (removes very short/empty abstracts)

2026 is flagged with is_partial_year=True since the year is not yet
complete at time of extraction.

Output:
  data/timeseries/papers_timeseries.parquet
"""

import json
import re
from pathlib import Path

import pandas as pd


INPUT_FILE = Path(r"C:\Users\HP\Desktop\DATA MINING\DM PROJECT\arxiv-metadata-oai-snapshot.json")
OUTPUT_DIR = Path("data/prosecced")

YEAR_START       = 2015
YEAR_END         = 2026
MIN_ABSTRACT_LEN = 400  
PARTIAL_YEARS    = {2026} 

CATEGORY_GROUPS = {
    "cs":       "Computer Science",
    "math":     "Mathematics",
    "physics":  "Physics",
    "cond-mat": "Condensed Matter",
    "astro-ph": "Astrophysics",
    "hep":      "High Energy Physics",
    "quant-ph": "Quantum Physics",
    "stat":     "Statistics",
}



def get_primary_group(categories_str):
    """Return the top-level category group or None if not in our list."""
    if not isinstance(categories_str, str) or not categories_str.strip():
        return None
    primary = categories_str.strip().split()[0]
    for prefix in CATEGORY_GROUPS:
        if primary.startswith(prefix):
            return prefix
    return None


def extract_year(update_date):
    """Parse year from update_date string (format: YYYY-MM-DD)."""
    if isinstance(update_date, str) and len(update_date) >= 4:
        try:
            return int(update_date[:4])
        except ValueError:
            return None
    return None


def clean_text(text):
    """
    Clean and normalize abstract text.
    Identical to original load_and_clean.py for consistency.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\$\$.*?\$\$",            " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]",            " ", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]{0,200}\$",        " ", text)
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+",            " ", text)
    text = re.sub(r"https?://\S+",           " ", text)
    text = re.sub(r"\barXiv:\S+",            " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\S+@\S+\.\S+",          " ", text)
    text = re.sub(r"\{[^}]*\}",             " ", text)
    text = re.sub(r"\[[^\]]*\]",            " ", text)
    text = re.sub(r"\([^)]{0,6}\)",         " ", text)
    text = re.sub(r"\b\d+(\.\d+)?\b",       " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,:;\-]",  " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_authors(authors_parsed):
    """Convert ArXiv author structure into a list of full name strings."""
    if not isinstance(authors_parsed, list):
        return []
    names = []
    for author in authors_parsed:
        if isinstance(author, list) and len(author) >= 2:
            last  = str(author[0]).strip()
            first = str(author[1]).strip()
            names.append(f"{first} {last}".strip())
        elif isinstance(author, str):
            names.append(author.strip())
    return names



def extract_all(filepath):
    """
    Stream the full ArXiv JSONL and collect every paper that passes
    the year, category, and abstract-length filters.

    No per-year or per-category cap — we take everything that qualifies.
    Year imbalance is handled by normalization in time_series.py.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"ArXiv dataset not found: {filepath}")

    rows = []
    n_read = n_skip_cat = n_skip_year = n_skip_len = 0
    year_counts = {}

    print(f"Streaming: {path.name}")
    print(f"Filters  : years {YEAR_START}-{YEAR_END} | "
          f"abstract >= {MIN_ABSTRACT_LEN} chars | "
          f"{len(CATEGORY_GROUPS)} category groups\n")

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            n_read += 1
            if n_read % 300_000 == 0:
                total_kept = sum(year_counts.values())
                print(f"  Scanned {n_read:>9,} | Kept so far: {total_kept:,}")

            group = get_primary_group(record.get("categories", ""))
            if group is None:
                n_skip_cat += 1
                continue

            year = extract_year(record.get("update_date", ""))
            if year is None or not (YEAR_START <= year <= YEAR_END):
                n_skip_year += 1
                continue

            abstract = (record.get("abstract", "") or "").strip()
            if len(abstract) < MIN_ABSTRACT_LEN:
                n_skip_len += 1
                continue

            rows.append({
                "paper_id"            : record["id"],
                "title"               : record.get("title", ""),
                "abstract_clean"      : clean_text(abstract),
                "authors_list"        : parse_authors(record.get("authors_parsed", [])),
                "year"                : year,
                "update_date"         : record.get("update_date", ""),
                "categories"          : record.get("categories", ""),
                "category_group"      : group,
                "category_group_name" : CATEGORY_GROUPS[group],
                "doi"                 : record.get("doi"),
                "journal_ref"         : record.get("journal-ref"),
                "is_partial_year"     : year in PARTIAL_YEARS,
            })
            year_counts[year] = year_counts.get(year, 0) + 1

    print(f"\nDone scanning {n_read:,} total records")
    print(f"  Skipped — category mismatch  : {n_skip_cat:,}")
    print(f"  Skipped — outside year range : {n_skip_year:,}")
    print(f"  Skipped — short abstract     : {n_skip_len:,}")
    print(f"  Kept                         : {sum(year_counts.values()):,}")
    return rows, year_counts


def print_summary(df, year_counts):
    print("\n" + "=" * 60)
    print("YEAR DISTRIBUTION  (all qualifying papers — no cap)")
    print("=" * 60)
    max_count = max(year_counts.values()) if year_counts else 1
    for year in sorted(year_counts):
        count     = year_counts[year]
        bar       = "█" * int(count / max_count * 40)
        partial   = "  <- partial year (2026, flagged)" if year in PARTIAL_YEARS else ""
        print(f"  {year} : {count:>7,}  {bar}{partial}")

    print(f"\n  Total papers      : {len(df):,}")
    print(f"  Year range        : {YEAR_START} - {YEAR_END}")
    print(f"  Category groups   : {sorted(df['category_group'].unique())}")
    print(f"  Partial years     : {sorted(PARTIAL_YEARS)} (is_partial_year=True)")
    print()
    print("  Category breakdown:")
    for grp, name in CATEGORY_GROUPS.items():
        n = (df['category_group'] == grp).sum()
        print(f"    {name:<22} : {n:>7,}")

    print()
    print("  NOTE: Year imbalance is expected and intentional.")
    print("  time_series.py will normalize counts by year total")
    print("  so trends reflect share of activity, not raw volume.")
    print("=" * 60)


def main():
    print("=" * 60)
    print("ResearchIQ — Time Series Data Extractor")
    print("=" * 60)

    rows, year_counts = extract_all(INPUT_FILE)

    print("\nBuilding DataFrame and cleaning abstracts...")
    df     = pd.DataFrame(rows)
    before = len(df)
    df     = df[df["abstract_clean"].str.len() > 100].reset_index(drop=True)
    print(f"Dropped {before - len(df):,} rows with empty abstracts after cleaning")

    print_summary(df, year_counts)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "papers_timeseries.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\n  Saved -> {out_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()