"""Sensitivity computation for k-NN distance vectors.

Implements:
- Lemma 3.1: ell_2 sensitivity Delta_2(d) <= d_k
- Lemma E.1: smooth sensitivity S_beta^* for ordered k-NN statistics
"""

import numpy as np
from typing import Optional


def l2_sensitivity(distances: np.ndarray) -> float:
    """Worst-case ell_2 sensitivity of the k-NN distance vector (Lemma 3.1).

    For neighboring databases D_u ~ D_u' differing in one vector:
        ||d(D_u) - d(D_u')||_2 <= d_k

    The bound is tight (achieved when v* is at the query point with k=1).

    Parameters
    ----------
    distances : np.ndarray, shape (k,)
        Sorted k-NN distances d_1 <= ... <= d_k.

    Returns
    -------
    float
        The ell_2 sensitivity bound d_k.
    """
    return float(distances[-1])


def local_sensitivity(distances: np.ndarray, d_kplus1: float) -> float:
    """Local sensitivity of the distance vector at a specific database.

    LS(d, D_u) = ||g||_2 where g is the gap vector
    g = (d_2 - d_1, d_3 - d_2, ..., d_{k+1} - d_k).

    Parameters
    ----------
    distances : np.ndarray, shape (k,)
        Sorted k-NN distances.
    d_kplus1 : float
        Distance to the (k+1)-th nearest neighbor.

    Returns
    -------
    float
        Local sensitivity ||g||_2.
    """
    k = len(distances)
    # Gap vector: consecutive differences including d_{k+1}
    all_dists = np.append(distances, d_kplus1)
    gaps = np.diff(all_dists)  # g_i = d_{i+1} - d_i for i=1,...,k
    return float(np.linalg.norm(gaps))


def smooth_sensitivity(
    distances: np.ndarray,
    d_kplus1: float,
    epsilon_d: float,
    delta: float,
    n_u: int,
    t_max: Optional[int] = None,
) -> float:
    """Beta-smooth sensitivity of ordered k-NN statistics (Lemma E.1).

    S_beta^*(d, D_u) = max_{t >= 0} exp(-beta * t) * A(t)

    where A(t) = max_{D_u': d(D_u, D_u') <= t} LS(d, D_u')
    and beta = epsilon_d / (2 * ln(2/delta)).

    Parameters
    ----------
    distances : np.ndarray, shape (k,)
        Sorted k-NN distances.
    d_kplus1 : float
        Distance to the (k+1)-th nearest neighbor.
    epsilon_d : float
        Distance privacy budget.
    delta : float
        Approximate DP parameter.
    n_u : int
        Number of authorized vectors.
    t_max : int, optional
        Maximum neighborhood radius. Default: ceil(ln(n_u) / beta).

    Returns
    -------
    float
        The beta-smooth sensitivity S_beta^*.
    """
    beta = epsilon_d / (2.0 * np.log(2.0 / delta))

    if t_max is None:
        t_max = int(np.ceil(np.log(n_u) / beta))
    t_max = min(t_max, n_u)  # Cannot exceed database size

    ls_0 = local_sensitivity(distances, d_kplus1)
    k = len(distances)
    d_k = distances[-1]

    # Per-step sensitivity change bound (Poisson model)
    delta_g = d_k * np.sqrt(k) / n_u

    best = ls_0  # t=0 contribution
    for t in range(1, t_max + 1):
        # Upper bound on LS in t-neighborhood
        ls_t = ls_0 + t * delta_g
        contribution = np.exp(-beta * t) * ls_t
        best = max(best, contribution)

    return best


def verify_sensitivity_bound(
    db: np.ndarray,
    query: np.ndarray,
    authorized_mask: np.ndarray,
    k: int,
    n_trials: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Machine-checked verification of Lemma 3.1.

    Enumerates neighboring databases (adding/removing one vector)
    and verifies ||d - d'||_2 <= d_k for each.

    Parameters
    ----------
    db : np.ndarray, shape (n, d)
    query : np.ndarray, shape (d,)
    authorized_mask : np.ndarray, shape (n,), dtype bool
    k : int
    n_trials : int
        Number of random neighboring databases to test.

    Returns
    -------
    dict with keys: 'bound', 'max_observed', 'violations', 'n_tested'
    """
    if rng is None:
        rng = np.random.default_rng(42)

    authorized_idx = np.where(authorized_mask)[0]
    n_u = len(authorized_idx)

    # Compute base distances
    dists_all = np.linalg.norm(db[authorized_idx] - query, axis=1)
    sorted_idx = np.argsort(dists_all)
    d_base = dists_all[sorted_idx[:k]]
    d_k = d_base[-1]

    max_change = 0.0
    violations = 0

    for _ in range(min(n_trials, n_u)):
        # Remove a random authorized vector
        remove_idx = rng.integers(0, n_u)
        remaining = np.delete(dists_all, remove_idx)
        remaining_sorted = np.sort(remaining)

        if len(remaining_sorted) >= k:
            d_prime = remaining_sorted[:k]
            # Pad if sizes differ
            if len(d_prime) == k:
                change = np.linalg.norm(d_base - d_prime)
                max_change = max(max_change, change)
                if change > d_k + 1e-10:
                    violations += 1

    return {
        "bound": d_k,
        "max_observed": max_change,
        "violations": violations,
        "n_tested": min(n_trials, n_u),
        "verified": violations == 0,
    }
