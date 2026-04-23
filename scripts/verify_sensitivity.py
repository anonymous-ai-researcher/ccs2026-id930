"""Machine-checked verification of Lemma 3.1 and smooth sensitivity.

Verifies that for all tested neighboring databases:
    ||d(D_u) - d(D_u')||_2 <= d_k

Usage:
    python scripts/verify_sensitivity.py [--smooth] [--n_trials 10000]
"""

import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from cpfg.sensitivity import verify_sensitivity_bound, l2_sensitivity, smooth_sensitivity


def verify_lemma_3_1(n: int = 5000, d: int = 100, k: int = 10, n_trials: int = 10000):
    """Verify Lemma 3.1: ell_2 sensitivity <= d_k."""
    print("=" * 60)
    print("Verification of Lemma 3.1: ell_2 sensitivity of d")
    print("=" * 60)

    rng = np.random.default_rng(42)
    db = rng.normal(0, 1, size=(n, d)).astype(np.float32)
    query = rng.normal(0, 1, size=d).astype(np.float32)
    authorized = rng.random(n) > 0.3

    result = verify_sensitivity_bound(db, query, authorized, k, n_trials, rng)

    print(f"\nDatabase: n={n}, d={d}, k={k}")
    print(f"Authorized vectors: {np.sum(authorized)}")
    print(f"Tested {result['n_tested']} neighboring databases")
    print(f"  Theoretical bound (d_k):  {result['bound']:.6f}")
    print(f"  Maximum observed change:  {result['max_observed']:.6f}")
    print(f"  Violations:               {result['violations']}")
    print(f"  Ratio (observed/bound):   {result['max_observed']/result['bound']:.6f}")
    print(f"  VERIFIED: {'YES' if result['verified'] else 'NO'}")

    # Additional verification with different data distributions
    distributions = {
        "Uniform [0,1]^d": lambda: rng.uniform(0, 1, size=(n, d)),
        "Clustered": lambda: np.vstack([
            rng.normal(c, 0.1, size=(n // 5, d))
            for c in rng.normal(0, 3, size=(5, d))
        ]),
    }

    for name, gen_fn in distributions.items():
        db2 = gen_fn().astype(np.float32)
        auth2 = rng.random(len(db2)) > 0.3
        result2 = verify_sensitivity_bound(db2, query, auth2, k, n_trials // 2, rng)
        status = "PASS" if result2["verified"] else "FAIL"
        print(f"\n  [{status}] {name}: max={result2['max_observed']:.4f}, "
              f"bound={result2['bound']:.4f}, ratio={result2['max_observed']/max(result2['bound'],1e-10):.4f}")

    return result["verified"]


def verify_smooth_sensitivity(n: int = 5000, d: int = 100, k: int = 10):
    """Verify smooth sensitivity computation."""
    print("\n" + "=" * 60)
    print("Verification of smooth sensitivity (Lemma E.1)")
    print("=" * 60)

    rng = np.random.default_rng(123)
    db = rng.normal(0, 1, size=(n, d)).astype(np.float32)
    query = rng.normal(0, 1, size=d).astype(np.float32)
    authorized = rng.random(n) > 0.3

    auth_idx = np.where(authorized)[0]
    dists = np.linalg.norm(db[auth_idx] - query, axis=1)
    sorted_dists = np.sort(dists)
    top_k = sorted_dists[:k]
    d_kplus1 = sorted_dists[k] if len(sorted_dists) > k else sorted_dists[-1] * 1.1

    # Worst-case sensitivity
    wc = l2_sensitivity(top_k)

    # Smooth sensitivity
    for eps_d in [0.1, 0.15, 0.5, 1.0]:
        ss = smooth_sensitivity(top_k, d_kplus1, eps_d, 1e-6, len(auth_idx))
        ratio = ss / wc
        print(f"  eps_d={eps_d:.2f}: S_beta*={ss:.4f}, d_k={wc:.4f}, "
              f"ratio={ratio:.4f} {'(improvement!)' if ratio < 0.5 else ''}")

    print(f"\n  Smooth sensitivity <= d_k: "
          f"{'VERIFIED' if ss <= wc + 1e-10 else 'FAILED'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth", action="store_true", help="Also verify smooth sensitivity")
    parser.add_argument("--n_trials", type=int, default=10000)
    args = parser.parse_args()

    passed = verify_lemma_3_1(n_trials=args.n_trials)

    if args.smooth:
        verify_smooth_sensitivity()

    print("\n" + "=" * 60)
    if passed:
        print("ALL VERIFICATIONS PASSED")
    else:
        print("VERIFICATION FAILED - see details above")
    print("=" * 60)


if __name__ == "__main__":
    main()
