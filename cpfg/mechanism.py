"""CPFG and CPFG* mechanisms (Algorithm 1, Theorems 4.1-4.2 and 6.1).

CPFG couples permute-and-flip (set selection) with calibrated Gaussian noise
(distance perturbation) through an optimal budget split. CPFG* adapts noise
to the smooth sensitivity of ordered k-NN statistics.
"""

import numpy as np
from typing import Optional, Tuple

from cpfg.pf import PermuteAndFlip
from cpfg.gaussian import GaussianMechanism
from cpfg.budget_split import optimal_budget_split
from cpfg.sensitivity import l2_sensitivity, smooth_sensitivity


class CPFG:
    """Coupled Permute-and-Flip-Gaussian mechanism (Algorithm 1).

    Satisfies joint (epsilon, delta)-DP for k-NN queries (Theorem 4.1).
    Matches the minimax lower bounds to constant factors (Theorem 4.2).

    Parameters
    ----------
    epsilon : float
        Total privacy budget.
    delta : float
        Approximate DP parameter.
    k : int
        Number of neighbors to return.
    budget_split : str or tuple
        'optimal' for automatic split via Eq. 4, or (eps_s, eps_d) tuple.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-6,
        k: int = 10,
        budget_split: str = "optimal",
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.k = k

        if isinstance(budget_split, str) and budget_split == "optimal":
            self._auto_split = True
            self.eps_s = None
            self.eps_d = None
        else:
            self._auto_split = False
            self.eps_s, self.eps_d = budget_split

    def query(
        self,
        q: np.ndarray,
        db: np.ndarray,
        authorized_mask: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute a private k-NN query.

        Parameters
        ----------
        q : np.ndarray, shape (d,)
            Query vector.
        db : np.ndarray, shape (n, d)
            Full vector database.
        authorized_mask : np.ndarray, shape (n,), dtype bool
            True for authorized vectors, False for restricted.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        selected_indices : np.ndarray, shape (k,), dtype int
            Indices into db of the selected (privatized) neighbors.
        noisy_distances : np.ndarray, shape (k,)
            Privatized distance vector.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Extract authorized vectors
        auth_idx = np.where(authorized_mask)[0]
        n_u = len(auth_idx)
        if n_u < self.k:
            raise ValueError(f"Only {n_u} authorized vectors, need k={self.k}")

        # Compute scores: s(q, v) = -dist(q, v)
        auth_vectors = db[auth_idx]
        distances_all = np.linalg.norm(auth_vectors - q, axis=1)
        scores = -distances_all

        # Step 1: Budget split
        if self._auto_split:
            # Compute score gap gamma_k for budget split
            sorted_all = np.sort(distances_all)
            if len(sorted_all) > self.k:
                gamma_k = sorted_all[self.k] - sorted_all[self.k - 1]
            else:
                gamma_k = 0.01
            self.eps_s, self.eps_d = optimal_budget_split(
                self.epsilon, self.k, n_u, self.delta, gamma_k=gamma_k
            )

        # Step 2: Set selection via PF
        pf = PermuteAndFlip(epsilon=self.eps_s, k=self.k)
        selected_local = pf.select(scores, rng=rng)

        # Map back to global indices
        selected_global = auth_idx[selected_local]

        # Step 3: Distance computation and perturbation
        true_distances = distances_all[selected_local]
        # Sort by distance (maintain ordering)
        sort_order = np.argsort(true_distances)
        true_distances = true_distances[sort_order]
        selected_global = selected_global[sort_order]

        # Sensitivity = d_k (Lemma 3.1)
        d_k = self._get_sensitivity(distances_all, n_u)

        gaussian = GaussianMechanism(
            epsilon_d=self.eps_d, delta=self.delta, sensitivity=d_k
        )
        noisy_distances = gaussian.perturb(true_distances, rng=rng)

        return selected_global, noisy_distances

    def _get_sensitivity(self, distances_all: np.ndarray, n_u: int) -> float:
        """Compute worst-case ell_2 sensitivity (Lemma 3.1)."""
        sorted_dists = np.sort(distances_all)
        d_k = sorted_dists[min(self.k - 1, len(sorted_dists) - 1)]
        return float(d_k)


class CPFGStar(CPFG):
    """Instance-optimal CPFG* using smooth sensitivity (Theorem 6.1).

    Replaces worst-case Delta_2 = d_k with the beta-smooth sensitivity
    S_beta^*(d, D_u), reducing DistErr by sqrt(n_u) on well-separated data.
    """

    def _get_sensitivity(self, distances_all: np.ndarray, n_u: int) -> float:
        """Compute smooth sensitivity of ordered k-NN statistics."""
        sorted_dists = np.sort(distances_all)
        d_k = sorted_dists[min(self.k - 1, len(sorted_dists) - 1)]

        if len(sorted_dists) > self.k:
            d_kplus1 = sorted_dists[self.k]
        else:
            d_kplus1 = d_k * 1.1  # Fallback

        top_k_dists = sorted_dists[: self.k]

        ss = smooth_sensitivity(
            distances=top_k_dists,
            d_kplus1=d_kplus1,
            epsilon_d=self.eps_d,
            delta=self.delta,
            n_u=n_u,
        )
        return float(ss)
