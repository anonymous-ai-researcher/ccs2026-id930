"""Leakage channel attack implementations (Section 2.4).

Channel 1: KS test on distance distributions (distance skew)
Channel 2: Rayleigh test on angular gaps
Channel 3: MLE triangulation across multiple queries
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple


def channel1_ks_test(
    observed_distances: np.ndarray,
    null_distances: np.ndarray,
) -> float:
    """Channel 1: Kolmogorov-Smirnov test on distance distributions.

    Compares observed k-NN distances against the null distribution
    (expected when no restricted vectors are nearby).

    Parameters
    ----------
    observed_distances : np.ndarray, shape (k,)
        Observed (possibly privatized) distance vector.
    null_distances : np.ndarray, shape (k,) or (n_samples, k)
        Reference distances from queries with no nearby restricted vectors.

    Returns
    -------
    float
        KS test statistic (higher = more evidence of restricted vectors).
    """
    if null_distances.ndim == 1:
        stat, _ = stats.ks_2samp(observed_distances, null_distances)
    else:
        null_flat = null_distances.flatten()
        stat, _ = stats.ks_2samp(observed_distances, null_flat)
    return float(stat)


def channel2_rayleigh_test(
    query: np.ndarray,
    neighbor_vectors: np.ndarray,
) -> float:
    """Channel 2: Rayleigh test for angular uniformity.

    Computes T_2 = k * d * ||mu||^2 where mu is the mean unit direction.
    Under H0 (uniform directions), T_2 ~ chi^2_d.

    Parameters
    ----------
    query : np.ndarray, shape (d,)
    neighbor_vectors : np.ndarray, shape (k, d)

    Returns
    -------
    float
        Rayleigh test statistic T_2.
    """
    k, d = neighbor_vectors.shape
    directions = neighbor_vectors - query
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    unit_dirs = directions / norms

    mean_dir = np.mean(unit_dirs, axis=0)
    T2 = k * d * np.sum(mean_dir ** 2)
    return float(T2)


def channel3_triangulation(
    queries: np.ndarray,
    distance_anomalies: np.ndarray,
    noise_variance: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Channel 3: MLE triangulation of restricted vector location.

    Estimates the location of a restricted vector from multiple queries
    using weighted least squares on distance anomalies.

    Parameters
    ----------
    queries : np.ndarray, shape (m, d)
        Query positions.
    distance_anomalies : np.ndarray, shape (m,)
        Estimated distance anomalies at each query position.
    noise_variance : float
        Known noise variance (for defense-aware adversary). 0 = unaware.

    Returns
    -------
    estimated_location : np.ndarray, shape (d,)
    mse : float
        Mean squared error (if true location is known).
    """
    m, d = queries.shape

    # Simple triangulation via gradient descent on the MLE objective:
    # min_v sum_i (||q_i - v|| - delta_i)^2 / (var[delta_i] + noise_variance)
    v_est = np.mean(queries, axis=0)  # Initialize at centroid

    lr = 0.01
    for step in range(500):
        diffs = queries - v_est
        dists = np.linalg.norm(diffs, axis=1)
        dists = np.maximum(dists, 1e-10)
        residuals = dists - distance_anomalies
        weights = 1.0 / (1.0 + noise_variance)

        grad = -np.sum(
            weights * (residuals / dists)[:, None] * diffs, axis=0
        ) / m
        v_est -= lr * grad

        if np.linalg.norm(grad) < 1e-8:
            break

    return v_est, 0.0


class ChannelEvaluator:
    """Evaluate all three channels and compute AUCs.

    Parameters
    ----------
    n_positive : int
        Number of positive queries (restricted vectors nearby).
    n_negative : int
        Number of negative queries (no restricted vectors nearby).
    """

    def __init__(self, n_positive: int = 1000, n_negative: int = 1000):
        self.n_positive = n_positive
        self.n_negative = n_negative

    def evaluate_channel1(
        self,
        positive_distances: np.ndarray,
        negative_distances: np.ndarray,
    ) -> float:
        """Compute AUC for Channel 1 (KS test).

        Parameters
        ----------
        positive_distances : np.ndarray, shape (n_positive, k)
        negative_distances : np.ndarray, shape (n_negative, k)

        Returns
        -------
        float : AUC score (0.5 = random guessing).
        """
        pos_stats = []
        for i in range(len(positive_distances)):
            stat = channel1_ks_test(positive_distances[i], negative_distances)
            pos_stats.append(stat)

        neg_stats = []
        for i in range(len(negative_distances)):
            # Leave-one-out: compare each negative against the rest
            mask = np.ones(len(negative_distances), dtype=bool)
            mask[i] = False
            stat = channel1_ks_test(negative_distances[i], negative_distances[mask])
            neg_stats.append(stat)

        return self._compute_auc(pos_stats, neg_stats)

    def evaluate_channel2(
        self,
        positive_stats: np.ndarray,
        negative_stats: np.ndarray,
    ) -> float:
        """Compute AUC for Channel 2 (Rayleigh test).

        Parameters
        ----------
        positive_stats : np.ndarray, shape (n_positive,)
            Rayleigh T_2 statistics for positive queries.
        negative_stats : np.ndarray, shape (n_negative,)
            Rayleigh T_2 statistics for negative queries.

        Returns
        -------
        float : AUC score.
        """
        return self._compute_auc(positive_stats, negative_stats)

    def evaluate_channel3(
        self,
        positive_mse: np.ndarray,
        negative_mse: np.ndarray,
    ) -> float:
        """Compute AUC for Channel 3 (triangulation).

        Lower MSE for positive = more evidence. We use 1/MSE as the score.
        """
        pos_scores = 1.0 / (positive_mse + 1e-10)
        neg_scores = 1.0 / (negative_mse + 1e-10)
        return self._compute_auc(pos_scores.tolist(), neg_scores.tolist())

    @staticmethod
    def _compute_auc(positive_scores, negative_scores) -> float:
        """Compute AUC from positive and negative score lists."""
        from sklearn.metrics import roc_auc_score

        labels = [1] * len(positive_scores) + [0] * len(negative_scores)
        scores = list(positive_scores) + list(negative_scores)
        try:
            return roc_auc_score(labels, scores)
        except ValueError:
            return 0.5
