import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def load_model(model_name):
    """
    Load sentence-transformer model.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed.\n"
            "Run: pip install sentence-transformers"
        )

    print(f"\nLoading model: {model_name}\n")

    return SentenceTransformer(model_name)


def validate_input_file(input_path):
    """
    Validate input parquet file existence.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{path}"
        )

    return path


def load_dataset(input_path):
    """
    Load cleaned papers dataset.
    """
    print(f"Loading dataset: {input_path.name}\n")

    df = pd.read_parquet(input_path)

    print(f"Papers loaded: {len(df):,}")

    return df


def validate_text_column(df, text_column):
    """
    Validate embedding text column existence.
    """
    if text_column not in df.columns:

        available = [
            column
            for column in df.columns
            if "text" in column.lower()
            or "abstract" in column.lower()
        ]

        raise ValueError(
            f"Column '{text_column}' not found.\n"
            f"Available text columns: {available}"
        )


def prepare_texts(df, text_column):
    """
    Prepare texts for embedding generation.
    """
    texts = (
        df[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    texts = [
        text.strip()
        for text in texts
        if text.strip()
    ]

    return texts


def generate_embeddings(
    texts,
    model,
    batch_size=64,
    show_progress=True,
):
    """
    Generate normalized semantic embeddings.
    """
    print(
        f"\nEncoding {len(texts):,} documents"
    )

    print(
        f"Batch size: {batch_size}"
    )

    print(
        "\nEstimated runtime:"
    )

    print(
        f"CPU : ~{len(texts) // 500 + 1} minutes"
    )

    print(
        f"GPU : ~{len(texts) // 5000 + 1} minutes"
    )

    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    elapsed = time.time() - start

    print(
        f"\nEmbedding completed in "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"Shape : {embeddings.shape}"
    )

    print(
        f"Dtype : {embeddings.dtype}"
    )

    print(
        f"Memory: {embeddings.nbytes / 1e6:.2f} MB"
    )

    return embeddings


def save_embeddings(
    embeddings,
    paper_ids,
    output_path,
):
    """
    Save embeddings and aligned paper IDs.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        embeddings,
    )

    paper_ids_path = (
        output_path.parent
        / "paper_ids.npy"
    )

    np.save(
        paper_ids_path,
        paper_ids,
    )

    print(
        f"\nEmbeddings saved to:\n{output_path}"
    )

    print(
        f"\nPaper IDs saved to:\n{paper_ids_path}"
    )


def print_summary(
    embeddings,
    model_name,
):
    """
    Print embedding summary statistics.
    """
    print("\n" + "=" * 60)

    print(
        f"Embedding model : {model_name}"
    )

    print(
        f"Papers embedded : "
        f"{embeddings.shape[0]:,}"
    )

    print(
        f"Vector dimension: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Embedding dtype : "
        f"{embeddings.dtype}"
    )

    print("=" * 60)

    print(
        "\nFirst embedding sample:\n"
    )

    print(
        embeddings[0][:10].round(4)
    )


def main():
    """
    Execute embedding generation pipeline.
    """
    parser = argparse.ArgumentParser(
        description=(
            "ResearchIQ - "
            "Generate sentence embeddings"
        )
    )

    parser.add_argument(
        "--input",
        default="data/papers_clean.parquet",
        help="Input parquet dataset",
    )

    parser.add_argument(
        "--output",
        default="data/embeddings.npy",
        help="Output embeddings file",
    )

    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Embedding batch size",
    )

    parser.add_argument(
        "--text_column",
        default="text_for_embedding",
        help="Column used for embeddings",
    )

    args = parser.parse_args()

    input_path = validate_input_file(
        args.input
    )

    df = load_dataset(input_path)

    validate_text_column(
        df,
        args.text_column,
    )

    texts = prepare_texts(
        df,
        args.text_column,
    )

    model = load_model(
        args.model
    )

    embeddings = generate_embeddings(
        texts=texts,
        model=model,
        batch_size=args.batch_size,
    )

    paper_ids = (
        df["paper_id"]
        .to_numpy()
    )

    save_embeddings(
        embeddings=embeddings,
        paper_ids=paper_ids,
        output_path=args.output,
    )

    print_summary(
        embeddings=embeddings,
        model_name=args.model,
    )

    print(
        "\nPipeline completed successfully."
    )

    print(
        "\nReady for Step 3:"
    )

    print(
        "python cluster.py"
    )


if __name__ == "__main__":
    main()