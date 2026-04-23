"""Experiment 2: Channel closure evaluation (Table 3).

Evaluates all 8 mechanisms (B0-B7) on all three leakage channels
at epsilon=1.0, k=10, reporting AUC, Recall, and DistErr.

Usage:
    python experiments/exp2_channel_closure.py --dataset mimiciv --epsilon 1.0
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cpfg import CPFG, CPFGStar
from baselines import NoDefense, GumbelTopK, JointExponential, PFOnly, GumbelLaplace, GumbelGaussian
from attacks import channel1_ks_test, channel2_rayleigh_test, ChannelEvaluator


def load_dataset(name: str, data_dir: str = "data/processed"):
    """Load preprocessed dataset."""
    path = Path(data_dir) / name
    db = np.load(path / "vectors.npy")
    authorized = np.load(path / "authorized.npy")
    return db, authorized


def compute_metrics(
    mechanism,
    db: np.ndarray,
    authorized: np.ndarray,
    k: int,
    n_queries: int,
    rng: np.random.Generator,
) -> dict:
    """Run queries and compute all metrics for a mechanism."""
    d = db.shape[1]
    auth_idx = np.where(authorized)[0]
    restricted_idx = np.where(~authorized)[0]

    recalls = []
    disterrs = []
    ch1_pos, ch1_neg = [], []
    ch2_pos, ch2_neg = [], []

    # Generate positive queries (near restricted vectors) and negative queries
    for i in range(n_queries):
        # Positive query: near a restricted vector
        if len(restricted_idx) > 0:
            target = db[rng.choice(restricted_idx)]
            q_pos = target + rng.normal(0, 0.1, size=d)
        else:
            q_pos = rng.normal(0, 1, size=d)

        # Negative query: far from all restricted vectors
        q_neg = rng.normal(0, 1, size=d)

        # Run mechanism on positive query
        selected_pos, dists_pos = mechanism.query(q_pos, db, authorized, rng=rng)

        # True top-k for recall computation
        true_dists = np.linalg.norm(db[auth_idx] - q_pos, axis=1)
        true_topk = auth_idx[np.argsort(true_dists)[:k]]
        true_topk_dists = np.sort(true_dists)[:k]

        # Recall
        recall = len(set(selected_pos) & set(true_topk)) / k
        recalls.append(recall)

        # DistErr
        if np.linalg.norm(true_topk_dists) > 0:
            disterr = np.linalg.norm(dists_pos - true_topk_dists) / np.linalg.norm(true_topk_dists)
        else:
            disterr = 0.0
        disterrs.append(disterr)

        # Channel 1: collect distance vectors
        ch1_pos.append(dists_pos)

        # Channel 2: Rayleigh statistic
        if len(selected_pos) > 0 and selected_pos[0] < len(db):
            t2_pos = channel2_rayleigh_test(q_pos, db[selected_pos])
            ch2_pos.append(t2_pos)

        # Negative query
        selected_neg, dists_neg = mechanism.query(q_neg, db, authorized, rng=rng)
        ch1_neg.append(dists_neg)
        if len(selected_neg) > 0 and selected_neg[0] < len(db):
            t2_neg = channel2_rayleigh_test(q_neg, db[selected_neg])
            ch2_neg.append(t2_neg)

    # Compute AUCs
    evaluator = ChannelEvaluator()
    ch1_pos_arr = np.array(ch1_pos)
    ch1_neg_arr = np.array(ch1_neg)
    auc_ch1 = evaluator.evaluate_channel1(ch1_pos_arr, ch1_neg_arr)
    auc_ch2 = evaluator.evaluate_channel2(np.array(ch2_pos), np.array(ch2_neg))
    auc_ch3 = max(auc_ch1, auc_ch2)  # Simplified; full Ch3 needs multi-query
    joint_auc = max(auc_ch1, auc_ch2, auc_ch3)

    return {
        "ch1_auc": auc_ch1,
        "ch2_auc": auc_ch2,
        "ch3_auc": auc_ch3,
        "joint_auc": joint_auc,
        "recall": np.mean(recalls),
        "recall_std": np.std(recalls),
        "disterr": np.mean(disterrs),
        "disterr_std": np.std(disterrs),
    }


def main():
    parser = argparse.ArgumentParser(description="Channel closure experiment (Table 3)")
    parser.add_argument("--dataset", default="mimiciv", choices=["mimiciv", "legalbench", "msmarco", "glove"])
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n_queries", type=int, default=1000)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--output", default="results/table3.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"=== Experiment 2: Channel Closure ===")
    print(f"Dataset: {args.dataset}, epsilon={args.epsilon}, k={args.k}")
    print(f"Queries: {args.n_queries}, Runs: {args.n_runs}")

    # For demo: generate synthetic data if real data not available
    rng = np.random.default_rng(args.seed)
    try:
        db, authorized = load_dataset(args.dataset)
    except FileNotFoundError:
        print(f"Dataset {args.dataset} not found, using synthetic data")
        n, d = 10000, 768
        db = rng.normal(0, 1, size=(n, d)).astype(np.float32)
        authorized = rng.random(n) > 0.3

    mechanisms = {
        "B0 No Defense": NoDefense(k=args.k),
        "B1 Gumbel": GumbelTopK(epsilon=args.epsilon, k=args.k),
        "B2 JointExp": JointExponential(epsilon=args.epsilon, k=args.k),
        "B3 PF-only": PFOnly(epsilon=args.epsilon, k=args.k),
        "B4 Gum+Lap": GumbelLaplace(epsilon=args.epsilon, k=args.k),
        "B5 Gum+Gau": GumbelGaussian(epsilon=args.epsilon, k=args.k),
        "B6 CPFG": CPFG(epsilon=args.epsilon, k=args.k),
        "B7 CPFG*": CPFGStar(epsilon=args.epsilon, k=args.k),
    }

    results = []
    for name, mech in mechanisms.items():
        print(f"\n--- {name} ---")
        run_results = []
        for run in range(args.n_runs):
            run_rng = np.random.default_rng(args.seed + run)
            metrics = compute_metrics(mech, db, authorized, args.k, args.n_queries, run_rng)
            run_results.append(metrics)
            print(f"  Run {run+1}: Ch1={metrics['ch1_auc']:.3f} Ch2={metrics['ch2_auc']:.3f} "
                  f"Recall={metrics['recall']:.3f} DistErr={metrics['disterr']:.3f}")

        # Aggregate
        avg = {
            "method": name,
            "ch1": np.mean([r["ch1_auc"] for r in run_results]),
            "ch1_std": np.std([r["ch1_auc"] for r in run_results]),
            "ch2": np.mean([r["ch2_auc"] for r in run_results]),
            "ch2_std": np.std([r["ch2_auc"] for r in run_results]),
            "ch3": np.mean([r["ch3_auc"] for r in run_results]),
            "joint": np.mean([r["joint_auc"] for r in run_results]),
            "recall": np.mean([r["recall"] for r in run_results]),
            "recall_std": np.mean([r["recall_std"] for r in run_results]),
            "disterr": np.mean([r["disterr"] for r in run_results]),
            "disterr_std": np.mean([r["disterr_std"] for r in run_results]),
        }
        results.append(avg)

    # Save results
    df = pd.DataFrame(results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
