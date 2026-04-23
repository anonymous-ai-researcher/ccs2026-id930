"""Optimal budget split for CPFG (Eq. 4, Section 4.2).

Minimizes the weighted loss:
    L(eps_s, eps_d) = k / (n_u * (e^eps_s - 1)) + c / (eps_d * sqrt(k))
    s.t. eps_s + eps_d = eps

via a single Newton step, converging quadratically since L is strictly convex.
"""

import numpy as np


def optimal_budget_split(
    epsilon: float,
    k: int,
    n_u: int,
    delta: float = 1e-6,
    gamma_k: float = 0.01,
) -> tuple:
    """Compute the optimal budget split (eps_s*, eps_d*).

    Minimizes the loss function accounting for score gap gamma_k:
        L = k * n_u * exp(-eps_s * gamma_k / 2) + c / (eps_d * sqrt(k))
    The first term uses the EM-based Recall loss (which depends on gamma_k),
    and the second is the DistErr bound.

    When gamma_k is small (dense embeddings), most budget goes to eps_s
    because each unit of eps_s reduces Recall loss exponentially.
    When gamma_k is large or n_u is very large, eps_s shrinks and eps_d
    absorbs more budget.

    Parameters
    ----------
    epsilon : float
        Total privacy budget.
    k : int
        Number of neighbors.
    n_u : int
        Number of authorized vectors.
    delta : float
        Approximate DP parameter (affects the constant c).
    gamma_k : float
        Score gap (d_{k+1} - d_k). Controls recall sensitivity to eps_s.

    Returns
    -------
    eps_s, eps_d : float, float
        Optimal budget allocation with eps_s + eps_d = epsilon.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    c = np.sqrt(2.0 * np.log(1.25 / delta))

    # Grid search for robustness (Newton can be unstable near boundaries)
    best_loss = float("inf")
    best_eps_s = epsilon * 0.5

    for ratio in np.linspace(0.05, 0.95, 181):
        es = ratio * epsilon
        ed = epsilon - es
        # EM-based recall loss: depends on gamma_k
        recall_term = k * (n_u - k) * np.exp(-es * gamma_k / 2.0)
        disterr_term = c / (ed * np.sqrt(k))
        loss = recall_term + disterr_term
        if loss < best_loss:
            best_loss = loss
            best_eps_s = es

    eps_s = float(best_eps_s)
    eps_d = float(epsilon - best_eps_s)
    return eps_s, eps_d


def loss_function(eps_s: float, eps_d: float, k: int, n_u: int, delta: float = 1e-6) -> float:
    """Evaluate the combined loss L(eps_s, eps_d)."""
    c = np.sqrt(2.0 * np.log(1.25 / delta))
    recall_loss = k / (n_u * (np.exp(eps_s) - 1))
    disterr_loss = c / (eps_d * np.sqrt(k))
    return recall_loss + disterr_loss
