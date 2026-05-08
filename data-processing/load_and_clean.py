import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("arxiv-metadata-oai-snapshot.json")
OUTPUT_DIR = Path("data")

CATEGORY_GROUPS = {
    "cs": "Computer Science",
    "math": "Mathematics",
    "physics": "Physics",
    "cond-mat": "Condensed Matter",
    "astro-ph": "Astrophysics",
    "hep": "High Energy Physics",
    "quant-ph": "Quantum Physics",
    "stat": "Statistics",
}

MIN_ABSTRACT_LEN = 400
YEAR_START = 2015
SAMPLES_PER_CAT = 5000
RANDOM_SEED = 42


def get_primary_group(categories_str):
    """Extract the primary top-level ArXiv category group."""
    if not isinstance(categories_str, str) or not categories_str.strip():
        return None
    primary = categories_str.strip().split()[0]
    for prefix in CATEGORY_GROUPS:
        if primary.startswith(prefix):
            return prefix
    return None


def clean_text(text):
    """
    Clean and normalize text for embedding quality.

    Removes LaTeX markup, URLs, arXiv IDs, email addresses, standalone
    numbers, and non-standard characters. Retains only meaningful prose
    with basic punctuation to maximize clustering signal.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]{0,200}\$", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)

    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\barXiv:\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]{0,6}\)", " ", text)

    text = re.sub(r"\b\d+(\.\d+)?\b", " ", text)

    text = re.sub(r"[^a-zA-Z0-9\s.,:;\-]", " ", text)

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def parse_authors(authors_parsed):
    """Convert ArXiv author structure into a clean list of full name strings."""
    if not isinstance(authors_parsed, list):
        return []
    names = []
    for author in authors_parsed:
        if isinstance(author, list) and len(author) >= 2:
            last = str(author[0]).strip()
            first = str(author[1]).strip()
            names.append(f"{first} {last}".strip())
        elif isinstance(author, str):
            names.append(author.strip())
    return names


def extract_year(update_date):
    """Parse a four-digit year from an update_date string."""
    if isinstance(update_date, str) and len(update_date) >= 4:
        try:
            return int(update_date[:4])
        except ValueError:
            return None
    return None


def load_arxiv_json(filepath, samples_per_cat, min_abstract_len, year_start):
    """
    Stream ArXiv metadata from a JSONL file and bucket records by category group.

    Stops early once all category buckets have reached capacity.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"ArXiv dataset not found:\n{filepath}")

    print(f"\nReading dataset: {path.name}\n")

    buckets = defaultdict(list)
    full_groups = set()
    n_read = n_skipped_cat = n_skipped_len = n_skipped_year = 0

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if len(full_groups) == len(CATEGORY_GROUPS):
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            n_read += 1
            if n_read % 100_000 == 0:
                print(
                    f"Scanned {n_read:,} records "
                    f"| Bucket sizes: { {g: len(v) for g, v in buckets.items()} }"
                )

            group = get_primary_group(record.get("categories", ""))
            if group is None or group in full_groups:
                n_skipped_cat += 1
                continue

            abstract = (record.get("abstract", "") or "").strip()
            if len(abstract) < min_abstract_len:
                n_skipped_len += 1
                continue

            year = extract_year(record.get("update_date", ""))
            if year is None or year < year_start:
                n_skipped_year += 1
                continue

            buckets[group].append(record)
            if len(buckets[group]) >= samples_per_cat:
                full_groups.add(group)
                print(f"{CATEGORY_GROUPS[group]} bucket full ({samples_per_cat:,} papers)")

    print(f"\nFinished scanning {n_read:,} records.")
    print(f"Skipped category mismatch : {n_skipped_cat:,}")
    print(f"Skipped short abstracts   : {n_skipped_len:,}")
    print(f"Skipped old papers        : {n_skipped_year:,}\n")

    rows = []
    for group, records in buckets.items():
        random.shuffle(records)
        sample = records[:samples_per_cat]
        for record in sample:
            record["_group"] = group
            record["_group_name"] = CATEGORY_GROUPS[group]
        rows.extend(sample)
        print(f"Collected {len(sample):,} papers from {CATEGORY_GROUPS[group]}")

    df = pd.DataFrame(rows)
    print(f"\nTotal collected papers: {len(df):,}")
    return df


def build_clean_dataset(df):
    """
    Produce a cleaned feature DataFrame and a separate ground-truth labels DataFrame.

    Applies text cleaning, author parsing, title-weighted embedding construction,
    and drops rows whose abstract is too thin after normalization.
    """
    df = df.copy()
    df["year"] = pd.to_numeric(df["update_date"].str[:4], errors="coerce").astype("Int64")

    print("\nCleaning titles and abstracts...\n")

    df["title_clean"] = df["title"].fillna("").apply(clean_text)
    df["abstract_clean"] = df["abstract"].fillna("").apply(clean_text)

    df["text_for_embedding"] = (
        df["title_clean"]
        + " . "
        + df["title_clean"]
        + " . "
        + df["abstract_clean"]
    )

    df["authors_list"] = df["authors_parsed"].apply(parse_authors)

    ground_truth_df = (
        df[["id", "categories", "_group", "_group_name"]]
        .copy()
        .rename(columns={
            "id": "paper_id",
            "_group": "category_group",
            "_group_name": "category_group_name",
        })
    )

    clean_df = (
        df[[
            "id", "title", "title_clean", "abstract_clean",
            "text_for_embedding", "authors_list", "year", "doi", "journal-ref",
        ]]
        .copy()
        .rename(columns={"id": "paper_id", "journal-ref": "journal_ref"})
    )

    before = len(clean_df)
    clean_df = clean_df[clean_df["abstract_clean"].str.len() > 100].reset_index(drop=True)
    ground_truth_df = ground_truth_df[
        ground_truth_df["paper_id"].isin(clean_df["paper_id"])
    ].reset_index(drop=True)

    print(f"Dropped {before - len(clean_df):,} rows after final cleaning.")
    return clean_df, ground_truth_df


def save_datasets(clean_df, ground_truth_df, output_dir):
    """Persist both DataFrames to Parquet files under the given output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clean_df.to_parquet(output_path / "papers_clean.parquet", index=False)
    ground_truth_df.to_parquet(output_path / "ground_truth_labels.parquet", index=False)

    print("\n" + "=" * 60)
    print(f"papers_clean.parquet        : {len(clean_df):,} papers")
    print(f"ground_truth_labels.parquet : {len(ground_truth_df):,} labels")
    print(f"Saved to: {output_path.resolve()}")
    print("=" * 60)
    print("\npapers_clean.parquet columns:")
    print(list(clean_df.columns))
    print("\nground_truth_labels.parquet columns:")
    print(list(ground_truth_df.columns))


def print_sample(clean_df):
    """Print the first record from the cleaned DataFrame for visual validation."""
    if clean_df.empty:
        return
    row = clean_df.iloc[0]
    print("\nSample Record\n")
    print(f"paper_id  : {row['paper_id']}")
    print(f"title     : {row['title'][:80]}...")
    print(f"year      : {row['year']}")
    print(f"abstract  : {row['abstract_clean'][:200]}...")
    print(f"embedding : {row['text_for_embedding'][:200]}...")
    print(f"authors   : {row['authors_list'][:3]}")


def main():
    """Run the full ArXiv preprocessing pipeline from CLI arguments."""
    parser = argparse.ArgumentParser(description="ResearchIQ - Load and clean ArXiv metadata")
    parser.add_argument("--input", default=INPUT_FILE, help="Path to ArXiv metadata JSONL file")
    parser.add_argument("--samples_per_cat", type=int, default=SAMPLES_PER_CAT)
    parser.add_argument("--min_abstract_len", type=int, default=MIN_ABSTRACT_LEN)
    parser.add_argument("--year_start", type=int, default=YEAR_START)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)

    raw_df = load_arxiv_json(
        filepath=args.input,
        samples_per_cat=args.samples_per_cat,
        min_abstract_len=args.min_abstract_len,
        year_start=args.year_start,
    )
    clean_df, ground_truth_df = build_clean_dataset(raw_df)
    save_datasets(clean_df, ground_truth_df, args.output_dir)
    print_sample(clean_df)
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()