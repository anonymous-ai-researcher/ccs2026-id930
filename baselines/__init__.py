"""Baseline mechanisms (B0-B5) for comparison.

B0: No defense (exact k-NN)
B1: Gumbel top-k (set only)
B2: Joint exponential mechanism (set only)
B3: PF-only (set only, no distance noise)
B4: Gumbel + Laplace (naive joint)
B5: Gumbel + Gaussian (naive joint)
"""

import numpy as np
from typing import Optional, Tuple


class NoDefense:
    """B0: No defense. Returns exact k-NN results."""

    def __init__(self, k: int = 10):
        self.k = k

    def query(self, q, db, authorized_mask, rng=None):
        auth_idx = np.where(authorized_mask)[0]
        dists = np.linalg.norm(db[auth_idx] - q, axis=1)
        topk = np.argsort(dists)[: self.k]
        return auth_idx[topk], dists[topk]


class GumbelTopK:
    """B1: Gumbel top-k mechanism (set only, exact distances).

    Adds Gumbel(0, 2*Delta_s/epsilon) noise to each score and returns
    the top-k noisy scores.
    """

    def __init__(self, epsilon: float = 1.0, k: int = 10, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.k = k
        self.sensitivity = sensitivity

    def query(self, q, db, authorized_mask, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        auth_idx = np.where(authorized_mask)[0]
        dists = np.linalg.norm(db[auth_idx] - q, axis=1)
        scores = -dists

        scale = 2.0 * self.sensitivity / self.epsilon
        noisy_scores = scores + rng.gumbel(0, scale, size=len(scores))
        topk = np.argsort(-noisy_scores)[: self.k]

        selected = auth_idx[topk]
        true_dists = dists[topk]
        order = np.argsort(true_dists)
        return selected[order], true_dists[order]


class JointExponential:
    """B2: Joint exponential mechanism (set only, exact distances)."""

    def __init__(self, epsilon: float = 1.0, k: int = 10, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.k = k
        self.sensitivity = sensitivity

    def query(self, q, db, authorized_mask, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        auth_idx = np.where(authorized_mask)[0]
        dists = np.linalg.norm(db[auth_idx] - q, axis=1)
        scores = -dists

        # Sample proportional to exp(epsilon * score / (2k * Delta_s))
        log_weights = (self.epsilon / (2.0 * self.k * self.sensitivity)) * scores
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= weights.sum()

        selected_local = rng.choice(len(auth_idx), size=self.k, replace=False, p=weights)
        selected = auth_idx[selected_local]
        true_dists = dists[selected_local]
        order = np.argsort(true_dists)
        return selected[order], true_dists[order]


class PFOnly:
    """B3: PF-only (set privatized, exact distances)."""

    def __init__(self, epsilon: float = 1.0, k: int = 10):
        self.epsilon = epsilon
        self.k = k

    def query(self, q, db, authorized_mask, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        from cpfg.pf import PermuteAndFlip

        auth_idx = np.where(authorized_mask)[0]
        dists = np.linalg.norm(db[auth_idx] - q, axis=1)
        scores = -dists

        pf = PermuteAndFlip(epsilon=self.epsilon, k=self.k)
        selected_local = pf.select(scores, rng=rng)

        selected = auth_idx[selected_local]
        true_dists = dists[selected_local]
        order = np.argsort(true_dists)
        return selected[order], true_dists[order]


class GumbelLaplace:
    """B4: Gumbel top-k + per-coordinate Laplace noise."""

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-6, k: int = 10):
        self.k = k
        # Split budget equally
        self.eps_s = epsilon / 2
        self.eps_d = epsilon / 2

    def query(self, q, db, authorized_mask, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        auth_idx = np.where(authorized_mask)[0]
        dists = np.linalg.norm(db[auth_idx] - q, axis=1)
        scores = -dists

        # Gumbel set selection
        scale = 2.0 / self.eps_s
        noisy_scores = scores + rng.gumbel(0, scale, size=len(scores))
        topk = np.argsort(-noisy_scores)[: self.k]

        selected = auth_idx[topk]
        true_dists = dists[topk]
        order = np.argsort(true_dists)
        true_dists = true_dists[order]
        selected = selected[order]

        # Laplace noise on distances
        d_k = np.sort(dists)[min(self.k - 1, len(dists) - 1)]
        lap_scale = d_k / self.eps_d
        noise = rng.laplace(0, lap_scale, size=self.k)
        noisy_dists = true_dists + noise

        return selected, noisy_dists


class GumbelGaussian:
    """B5: Gumbel top-k + global Gaussian noise."""

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-6, k: int = 10):
        self.k = k
        self.delta = delta
        self.eps_s = epsilon / 2
        self.eps_d = epsilon / 2

    def query(self, q, db, authorized_mask, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        auth_idx = np.where(authorized_mask)[0]
        dists = np.linalg.norm(db[auth_idx] - q, axis=1)
        scores = -dists

        # Gumbel set selection
        scale = 2.0 / self.eps_s
        noisy_scores = scores + rng.gumbel(0, scale, size=len(scores))
        topk = np.argsort(-noisy_scores)[: self.k]

        selected = auth_idx[topk]
        true_dists = dists[topk]
        order = np.argsort(true_dists)
        true_dists = true_dists[order]
        selected = selected[order]

        # Gaussian noise on distances
        d_k = np.sort(dists)[min(self.k - 1, len(dists) - 1)]
        sigma = d_k * np.sqrt(2.0 * np.log(1.25 / self.delta)) / self.eps_d
        noise = rng.normal(0, sigma, size=self.k)
        noisy_dists = true_dists + noise

        return selected, noisy_dists


BASELINES = {
    "B0": NoDefense,
    "B1": GumbelTopK,
    "B2": JointExponential,
    "B3": PFOnly,
    "B4": GumbelLaplace,
    "B5": GumbelGaussian,
}
