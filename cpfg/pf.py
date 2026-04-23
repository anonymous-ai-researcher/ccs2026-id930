"""Permute-and-Flip mechanism for private set selection (Definition 3).

Reference: McKenna & Sheldon, "Permute-and-Flip: A new mechanism for
differentially private selection," NeurIPS 2020.
"""

import numpy as np
from typing import Optional


class PermuteAndFlip:
    """Permute-and-Flip (PF) mechanism for differentially private top-k selection.

    PF is epsilon-DP and stochastically dominates the exponential mechanism:
    for all score vectors and thresholds t,
        Pr[error(PF) >= t] <= Pr[error(EM) >= t].

    Parameters
    ----------
    epsilon : float
        Privacy parameter for set selection (epsilon_s).
    k : int
        Number of elements to select.
    sensitivity : float, optional
        Score sensitivity Delta_s. Default 1.0 (for negated distance scores
        under neighboring databases differing in one vector).
    """

    def __init__(self, epsilon: float, k: int, sensitivity: float = 1.0):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.epsilon = epsilon
        self.k = k
        self.sensitivity = sensitivity

    def select(
        self,
        scores: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Select k elements via permute-and-flip.

        Parameters
        ----------
        scores : np.ndarray, shape (n,)
            Score vector. Higher scores are preferred (e.g., s(q,v) = -dist(q,v)).
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        selected : np.ndarray, shape (k,), dtype int
            Indices of the selected elements in the original score array.
        """
        if rng is None:
            rng = np.random.default_rng()

        n = len(scores)
        if n < self.k:
            raise ValueError(f"Cannot select k={self.k} from n={n} candidates")

        s_max = np.max(scores)
        # Coin-flip probabilities: exp(epsilon / (2 * Delta_s) * (s_i - s_max))
        log_probs = (self.epsilon / (2.0 * self.sensitivity)) * (scores - s_max)
        # Clip for numerical stability (probabilities in [exp(-epsilon), 1])
        log_probs = np.clip(log_probs, -self.epsilon, 0.0)
        probs = np.exp(log_probs)

        # Random permutation
        perm = rng.permutation(n)
        selected = []

        for idx in perm:
            if len(selected) >= self.k:
                break
            # Flip biased coin
            if rng.random() < probs[idx]:
                selected.append(idx)

        # If we exhausted the permutation without k selections (very rare),
        # fill remaining slots uniformly from unselected candidates
        if len(selected) < self.k:
            remaining = [i for i in range(n) if i not in set(selected)]
            rng.shuffle(remaining)
            selected.extend(remaining[: self.k - len(selected)])

        return np.array(selected, dtype=np.int64)

    def select_with_permutation(
        self,
        scores: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple:
        """Select k elements and return the permutation order (for analysis).

        Returns
        -------
        selected : np.ndarray, shape (k,)
        permutation : np.ndarray, shape (n,)
        coin_flips : list of bool
        """
        if rng is None:
            rng = np.random.default_rng()

        n = len(scores)
        s_max = np.max(scores)
        log_probs = (self.epsilon / (2.0 * self.sensitivity)) * (scores - s_max)
        log_probs = np.clip(log_probs, -self.epsilon, 0.0)
        probs = np.exp(log_probs)

        perm = rng.permutation(n)
        selected = []
        flips = []

        for idx in perm:
            if len(selected) >= self.k:
                break
            flip = rng.random() < probs[idx]
            flips.append(flip)
            if flip:
                selected.append(idx)

        if len(selected) < self.k:
            remaining = [i for i in range(n) if i not in set(selected)]
            rng.shuffle(remaining)
            selected.extend(remaining[: self.k - len(selected)])

        return np.array(selected, dtype=np.int64), perm, flips
