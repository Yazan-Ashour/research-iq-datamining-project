import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


INPUT_FILE = Path("arxiv-metadata-oai-snapshot.json")
RESULTS_DIR = Path("results")
REPORT_FILE = RESULTS_DIR / "analysis_report.txt"


def stream_records(filepath):
    """Stream all valid JSON records from a newline-delimited JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def analyse_field_presence(filepath):
    """
    Report every field present in the dataset and its completeness rate.

    Streams the full file to count how often each key holds a non-null,
    non-empty value. Useful for deciding which fields are reliable enough
    to use in downstream steps.
    """
    key_total = defaultdict(int)
    key_present = defaultdict(int)
    total = 0

    for r in stream_records(filepath):
        total += 1
        for k, v in r.items():
            key_total[k] += 1
            if v not in (None, "", [], {}, "None"):
                key_present[k] += 1

    lines = []
    lines.append(f"\n{'─'*65}")
    lines.append("FIELD PRESENCE RATES  (full dataset)")
    lines.append(f"{'─'*65}")
    lines.append(f"  {'Field':<25}  {'Present':>10}  {'Total':>10}  {'Rate':>7}")
    lines.append(f"  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*7}")

    for key in sorted(key_total):
        present = key_present[key]
        tot = key_total[key]
        pct = 100 * present / tot if tot else 0
        note = "  ← sparse" if pct < 30 else ("  ← moderate" if pct < 70 else "")
        lines.append(f"  {key:<25}  {present:>10,}  {tot:>10,}  {pct:>6.1f}%{note}")

    lines.append(f"\n  Total records scanned : {total:,}")
    return "\n".join(lines), total


def analyse_sample_records(filepath, n=3):
    """Display a small number of raw records for structural inspection."""
    lines = []
    lines.append(f"\n{'─'*65}")
    lines.append(f"SAMPLE RECORDS (first {n})")
    lines.append(f"{'─'*65}")

    import pprint
    count = 0
    for r in stream_records(filepath):
        if count >= n:
            break
        lines.append(f"\n  --- Record {count} ---")
        lines.append(pprint.pformat(r))
        count += 1

    return "\n".join(lines)


def analyse_abstracts(filepath):
    """
    Analyse abstract character length, word count, and LaTeX prevalence.

    Streams the full dataset and reports the full distribution so that
    an appropriate minimum-length threshold can be chosen based on evidence.
    """
    lengths = []
    word_counts = []
    latex_count = 0
    empty_count = 0

    for r in stream_records(filepath):
        abstract = (r.get("abstract") or "").strip()
        length = len(abstract)
        lengths.append(length)
        word_counts.append(len(abstract.split()) if abstract else 0)
        if length == 0:
            empty_count += 1
        if re.search(r'\$|\\\w+|\\\[', abstract):
            latex_count += 1

    n = len(lengths)
    lines = []
    lines.append(f"\n{'─'*65}")
    lines.append("ABSTRACTS — LENGTH AND QUALITY  (full dataset)")
    lines.append(f"{'─'*65}")

    lines.append(f"  Total records           : {n:,}")
    lines.append(f"  Empty abstracts         : {empty_count:,}  ({100*empty_count/n:.2f}%)")
    lines.append(f"  Abstracts with LaTeX    : {latex_count:,}  ({100*latex_count/n:.2f}%)")

    lines.append(f"\n  Character length:")
    lines.append(f"    Min    : {min(lengths)}")
    lines.append(f"    Max    : {max(lengths)}")
    lines.append(f"    Mean   : {statistics.mean(lengths):.1f}")
    lines.append(f"    Median : {statistics.median(lengths):.1f}")
    lines.append(f"    Stdev  : {statistics.stdev(lengths):.1f}")
    lines.append(f"    P10    : {sorted(lengths)[n//10]}")
    lines.append(f"    P25    : {sorted(lengths)[n//4]}")
    lines.append(f"    P75    : {sorted(lengths)[3*n//4]}")
    lines.append(f"    P90    : {sorted(lengths)[9*n//10]}")

    lines.append(f"\n  Word count:")
    lines.append(f"    Mean   : {statistics.mean(word_counts):.1f}")
    lines.append(f"    Median : {statistics.median(word_counts):.1f}")
    lines.append(f"    Max    : {max(word_counts)}")

    lines.append(f"\n  Character length distribution:")
    bins = [0, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 99999]
    labels = ["0–50", "50–100", "100–150", "150–200", "200–300",
              "300–400", "400–500", "500–700", "700–1000", "1000–1500", "1500+"]
    cumulative = 0
    lines.append(f"  {'Range':>12}  {'Count':>10}  {'%':>6}  {'Cumulative%':>12}")
    for lo, hi, label in zip(bins, bins[1:], labels):
        count = sum(1 for l in lengths if lo <= l < hi)
        pct = 100 * count / n
        cumulative += pct
        lines.append(f"  {label:>12}  {count:>10,}  {pct:>5.1f}%  {cumulative:>11.1f}%")

    lines.append(f"\n  Retention at candidate MIN_ABSTRACT_LENGTH thresholds:")
    lines.append(f"  {'Threshold':>10}  {'Kept':>10}  {'Dropped':>10}  {'Retention%':>11}")
    for t in [50, 100, 150, 200, 250, 300, 400, 500]:
        kept = sum(1 for l in lengths if l >= t)
        dropped = n - kept
        lines.append(f"  {t:>10}  {kept:>10,}  {dropped:>10,}  {100*kept/n:>10.1f}%")

    return "\n".join(lines)


def analyse_dates(filepath):
    """
    Analyse the temporal distribution of records using update_date and
    the original submission date embedded in versions[0].created.

    Reports full year-by-year breakdown so an appropriate YEAR_START
    cutoff can be chosen based on actual data density.
    """
    years_update = defaultdict(int)
    years_v1 = defaultdict(int)
    malformed = 0

    for r in stream_records(filepath):
        d = r.get("update_date", "")
        if isinstance(d, str) and len(d) >= 4:
            try:
                years_update[int(d[:4])] += 1
            except ValueError:
                malformed += 1

        versions = r.get("versions", [])
        if isinstance(versions, list) and versions:
            m = re.search(r'\d{4}', versions[0].get("created", ""))
            if m:
                years_v1[int(m.group())] += 1

    total = sum(years_update.values())
    lines = []
    lines.append(f"\n{'─'*65}")
    lines.append("DATES — TEMPORAL COVERAGE  (full dataset)")
    lines.append(f"{'─'*65}")
    lines.append(f"  update_date format      : YYYY-MM-DD")
    lines.append(f"  versions[0].created     : RFC-2822 (year via regex)")
    lines.append(f"  Malformed update_date   : {malformed:,}")

    lines.append(f"\n  Year distribution — update_date:")
    lines.append(f"  {'Year':<6}  {'Count':>10}  {'%':>6}  {'Cumulative%':>12}")
    cumulative = 0
    for yr in sorted(years_update):
        count = years_update[yr]
        pct = 100 * count / total
        cumulative += pct
        lines.append(f"  {yr:<6}  {count:>10,}  {pct:>5.1f}%  {cumulative:>11.1f}%")

    lines.append(f"\n  Retention at candidate YEAR_START thresholds:")
    lines.append(f"  {'Year':>6}  {'Kept':>10}  {'Dropped':>10}  {'Retention%':>11}")
    for y in [2010, 2012, 2014, 2015, 2016, 2018, 2020]:
        kept = sum(v for k, v in years_update.items() if k >= y)
        lines.append(f"  {y:>6}  {kept:>10,}  {total-kept:>10,}  {100*kept/total:>10.1f}%")

    return "\n".join(lines)


def analyse_categories(filepath):
    """
    Report category field format, multi-label frequency, and the full
    top-level subject group distribution across the dataset.
    """
    multi = 0
    single = 0
    top_level = defaultdict(int)
    all_categories = defaultdict(int)

    prefixes = [
        "cond-mat", "astro-ph", "hep-ph", "hep-th", "hep-ex", "hep-lat",
        "quant-ph", "gr-qc", "math-ph", "nlin", "nucl-th", "nucl-ex",
        "cs", "math", "physics", "stat", "q-bio", "econ"
    ]

    for r in stream_records(filepath):
        cats = (r.get("categories") or "").strip().split()
        if len(cats) > 1:
            multi += 1
        elif len(cats) == 1:
            single += 1

        if cats:
            primary = cats[0]
            for prefix in prefixes:
                if primary.startswith(prefix):
                    top_level[prefix] += 1
                    break
            else:
                top_level[primary] += 1

            for cat in cats:
                all_categories[cat] += 1

    total = multi + single
    lines = []
    lines.append(f"\n{'─'*65}")
    lines.append("CATEGORIES — DISTRIBUTION  (full dataset)")
    lines.append(f"{'─'*65}")
    lines.append(f"  Multi-category papers   : {multi:,}  ({100*multi/total:.1f}%)")
    lines.append(f"  Single-category papers  : {single:,}  ({100*single/total:.1f}%)")

    lines.append(f"\n  Top-level group distribution (by primary category):")
    lines.append(f"  {'Group':<15}  {'Count':>10}  {'%':>6}")
    for cat, count in sorted(top_level.items(), key=lambda x: -x[1]):
        lines.append(f"  {cat:<15}  {count:>10,}  {100*count/total:>5.1f}%")

    lines.append(f"\n  Most frequent individual category labels (top 30):")
    lines.append(f"  {'Category':<20}  {'Count':>10}  {'%':>6}")
    for cat, count in sorted(all_categories.items(), key=lambda x: -x[1])[:30]:
        lines.append(f"  {cat:<20}  {count:>10,}  {100*count/total:>5.1f}%")

    return "\n".join(lines)


def analyse_authors(filepath):
    """
    Report co-authorship statistics and confirm the authors_parsed field format.

    Streams the full dataset to compute the true author-count distribution,
    which informs whether to cap or filter extreme multi-author papers.
    """
    author_counts = []
    format_samples = []

    for r in stream_records(filepath):
        ap = r.get("authors_parsed")
        if isinstance(ap, list):
            author_counts.append(len(ap))
            if len(format_samples) < 10 and ap:
                format_samples.append((len(ap), ap[0]))

    n = len(author_counts)
    lines = []
    lines.append(f"\n{'─'*65}")
    lines.append("AUTHORS — STATISTICS  (full dataset)")
    lines.append(f"{'─'*65}")
    lines.append(f"  Records with authors_parsed : {n:,}")
    lines.append(f"  Mean authors per paper      : {statistics.mean(author_counts):.2f}")
    lines.append(f"  Median                      : {statistics.median(author_counts):.1f}")
    lines.append(f"  Max                         : {max(author_counts)}")

    sorted_counts = sorted(author_counts)
    lines.append(f"  P90                         : {sorted_counts[9*n//10]}")
    lines.append(f"  P99                         : {sorted_counts[99*n//100]}")

    lines.append(f"\n  Author count distribution:")
    lines.append(f"  {'Range':<12}  {'Count':>10}  {'%':>6}")
    buckets = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 10000)]
    labels = ["1", "2", "3", "4–5", "6–10", "11–20", "21–50", "51+"]
    for (lo, hi), label in zip(buckets, labels):
        count = sum(1 for c in author_counts if lo <= c <= hi)
        lines.append(f"  {label:<12}  {count:>10,}  {100*count/n:>5.1f}%")

    lines.append(f"\n  Entry format: [last, first, suffix]")
    lines.append(f"  Name construction: f\"{{first}} {{last}}\"")
    lines.append(f"\n  Format samples:")
    for length, first_author in format_samples:
        lines.append(f"    len={length:<3}  first_entry={first_author}")

    return "\n".join(lines)


def analyse_ids(filepath):
    """
    Confirm paper ID uniqueness and format distribution across the full dataset.
    """
    ids = []
    for r in stream_records(filepath):
        pid = r.get("id")
        if pid:
            ids.append(pid)

    total = len(ids)
    unique = len(set(ids))
    old_format = sum(1 for i in ids if "/" in i)
    new_format = total - old_format

    lines = []
    lines.append(f"\n{'─'*65}")
    lines.append("PAPER IDs — FORMAT  (full dataset)")
    lines.append(f"{'─'*65}")
    lines.append(f"  Total IDs               : {total:,}")
    lines.append(f"  Unique IDs              : {unique:,}")
    lines.append(f"  Duplicates              : {total - unique:,}")
    lines.append(f"  New format (YYMM.NNNNN) : {new_format:,}  ({100*new_format/total:.1f}%)")
    lines.append(f"  Old format (area/...)   : {old_format:,}  ({100*old_format/total:.1f}%)")
    lines.append(f"\n  Samples (first 10):")
    for pid in ids[:10]:
        lines.append(f"    '{pid}'")

    return "\n".join(lines)


def analyse_versions(filepath):
    """Analyse revision depth distribution across the full dataset."""
    version_counts = []

    for r in stream_records(filepath):
        v = r.get("versions")
        if isinstance(v, list):
            version_counts.append(len(v))

    n = len(version_counts)
    lines = []
    lines.append(f"\n{'─'*65}")
    lines.append("VERSIONS — REVISION HISTORY  (full dataset)")
    lines.append(f"{'─'*65}")
    lines.append(f"  Mean revisions per paper : {statistics.mean(version_counts):.2f}")
    lines.append(f"  Median                   : {statistics.median(version_counts):.1f}")
    lines.append(f"  Max                      : {max(version_counts)}")

    lines.append(f"\n  {'Versions':<10}  {'Count':>10}  {'%':>6}")
    for v in range(1, 8):
        count = sum(1 for c in version_counts if c == v)
        lines.append(f"  {v:<10}  {count:>10,}  {100*count/n:>5.1f}%")
    count_plus = sum(1 for c in version_counts if c >= 8)
    lines.append(f"  {'8+':<10}  {count_plus:>10,}  {100*count_plus/n:>5.1f}%")

    return "\n".join(lines)


def analyse_journal_ref(filepath):
    """
    Analyse journal-ref completeness and the string-'None' contamination issue.
    """
    real_none = 0
    str_none = 0
    empty = 0
    has_value = 0
    samples = []

    for r in stream_records(filepath):
        v = r.get("journal-ref")
        if v is None:
            real_none += 1
        elif v == "None":
            str_none += 1
        elif v == "":
            empty += 1
        else:
            has_value += 1
            if len(samples) < 8:
                samples.append(v)

    total = real_none + str_none + empty + has_value
    lines = []
    lines.append(f"\n{'─'*65}")
    lines.append("JOURNAL-REF — FIELD QUALITY  (full dataset)")
    lines.append(f"{'─'*65}")
    lines.append(f"  Python None             : {real_none:,}  ({100*real_none/total:.1f}%)")
    lines.append(f"  Empty string            : {empty:,}  ({100*empty/total:.1f}%)")
    lines.append(f"  String 'None'           : {str_none:,}  ({100*str_none/total:.1f}%)  ← requires explicit filter")
    lines.append(f"  Real values             : {has_value:,}  ({100*has_value/total:.1f}%)")
    lines.append(f"\n  Sample values:")
    for v in samples:
        lines.append(f"    '{v}'")

    return "\n".join(lines)


def analyse_titles(filepath):
    """
    Analyse title completeness and length distribution across the full dataset.
    """
    lengths = []
    empty_count = 0
    samples = []

    for r in stream_records(filepath):
        title = (r.get("title") or "").strip()
        length = len(title)
        lengths.append(length)
        if length == 0:
            empty_count += 1
        if len(samples) < 5:
            samples.append(title)

    n = len(lengths)
    lines = []
    lines.append(f"\n{'─'*65}")
    lines.append("TITLES — COMPLETENESS AND LENGTH  (full dataset)")
    lines.append(f"{'─'*65}")
    lines.append(f"  Total records           : {n:,}")
    lines.append(f"  Missing / empty titles  : {empty_count:,}  ({100*empty_count/n:.2f}%)")
    lines.append(f"  Mean length (chars)     : {statistics.mean(lengths):.1f}")
    lines.append(f"  Median length (chars)   : {statistics.median(lengths):.1f}")
    lines.append(f"  Min / Max               : {min(lengths)} / {max(lengths)}")
    lines.append(f"\n  Samples:")
    for s in samples:
        lines.append(f"    '{s}'")

    return "\n".join(lines)


def main():
    """
    Run a full structural analysis of a newline-delimited JSON dataset.

    Each probe streams the file independently so memory usage stays flat
    regardless of dataset size. Results are printed to stdout and saved
    to results/analysis_report.txt.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    RESULTS_DIR.mkdir(exist_ok=True)

    print("Dataset Analysis")
    print("=" * 65)
    print(f"File : {INPUT_FILE}")
    print()

    sections = [
        "Dataset Analysis",
        "=" * 65,
        f"File : {INPUT_FILE}",
    ]

    probes = [
        ("Field presence",   lambda: analyse_field_presence(INPUT_FILE)[0]),
        ("Sample records",   lambda: analyse_sample_records(INPUT_FILE)),
        ("Titles",           lambda: analyse_titles(INPUT_FILE)),
        ("Abstracts",        lambda: analyse_abstracts(INPUT_FILE)),
        ("Dates",            lambda: analyse_dates(INPUT_FILE)),
        ("Categories",       lambda: analyse_categories(INPUT_FILE)),
        ("Authors",          lambda: analyse_authors(INPUT_FILE)),
        ("IDs",              lambda: analyse_ids(INPUT_FILE)),
        ("Versions",         lambda: analyse_versions(INPUT_FILE)),
        ("Journal ref",      lambda: analyse_journal_ref(INPUT_FILE)),
    ]

    for label, probe in probes:
        print(f"  Analysing {label}...")
        result = probe()
        print(result)
        sections.append(result)

    full_report = "\n\n".join(sections)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"\nReport saved → {REPORT_FILE}")
    print("Analysis complete.")


if __name__ == "__main__":
    main()