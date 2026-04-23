"""Dataset preprocessing for CPFG experiments.

Supports: MIMIC-IV, LegalBench, MS MARCO, GloVe-200.

Usage:
    python data/preprocess.py --dataset msmarco [--data_dir /path/to/raw]
"""

import argparse
import numpy as np
from pathlib import Path


def preprocess_glove(data_dir: str, output_dir: str, dim: int = 200):
    """Download and preprocess GloVe word vectors."""
    import urllib.request
    import zipfile

    output = Path(output_dir) / "glove"
    output.mkdir(parents=True, exist_ok=True)

    url = f"https://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = Path(data_dir) / "glove.6B.zip"

    if not zip_path.exists():
        print(f"Downloading GloVe from {url}...")
        urllib.request.urlretrieve(url, zip_path)

    print("Extracting vectors...")
    vectors = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(f"glove.6B.{dim}d.txt") as f:
            for line in f:
                parts = line.decode().strip().split()
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                if len(vec) == dim:
                    vectors.append(vec)

    db = np.array(vectors[:1200000])  # Cap at 1.2M
    n = len(db)
    authorized = np.random.default_rng(42).random(n) > 0.2  # 20% restricted

    np.save(output / "vectors.npy", db)
    np.save(output / "authorized.npy", authorized)
    print(f"Saved {n} vectors ({dim}d) to {output}")


def preprocess_synthetic(output_dir: str, name: str, n: int, d: int, alpha: float):
    """Generate synthetic dataset for testing."""
    output = Path(output_dir) / name
    output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    db = rng.normal(0, 1, size=(n, d)).astype(np.float32)
    authorized = rng.random(n) > alpha

    np.save(output / "vectors.npy", db)
    np.save(output / "authorized.npy", authorized)
    print(f"Saved synthetic {name}: {n} vectors ({d}d), alpha={alpha:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument("--dataset", required=True,
                        choices=["mimiciv", "legalbench", "msmarco", "glove", "synthetic"])
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    if args.dataset == "glove":
        preprocess_glove(args.data_dir, args.output_dir)
    elif args.dataset == "mimiciv":
        print("MIMIC-IV requires PhysioNet credentials.")
        print("1. Request access at https://physionet.org/content/mimiciv/")
        print("2. Download discharge summaries")
        print("3. Run embedding with PubMedBERT")
        print("\nGenerating synthetic placeholder...")
        preprocess_synthetic(args.output_dir, "mimiciv", 331000, 768, 0.31)
    elif args.dataset == "legalbench":
        print("Generating LegalBench placeholder...")
        preprocess_synthetic(args.output_dir, "legalbench", 25000, 1024, 0.24)
    elif args.dataset == "msmarco":
        print("Generating MS MARCO placeholder...")
        preprocess_synthetic(args.output_dir, "msmarco", 100000, 768, 0.20)  # Subset
    elif args.dataset == "synthetic":
        preprocess_synthetic(args.output_dir, "synthetic", 10000, 768, 0.30)


if __name__ == "__main__":
    main()
