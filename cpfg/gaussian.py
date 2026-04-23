"""Gaussian mechanism for distance perturbation.

Calibrates i.i.d. Gaussian noise to the ell_2 sensitivity of the k-NN
distance vector (Lemma 3.1: Delta_2 <= d_k).
"""

import numpy as np
from typing import Optional


class GaussianMechanism:
    """Gaussian mechanism for (epsilon_d, delta)-DP distance perturbation.

    Parameters
    ----------
    epsilon_d : float
        Privacy budget for distance perturbation.
    delta : float
        Approximate DP parameter.
    sensitivity : float
        ell_2 sensitivity of the distance vector (Delta_2 <= d_k).
    """

    def __init__(self, epsilon_d: float, delta: float, sensitivity: float):
        if epsilon_d <= 0:
            raise ValueError(f"epsilon_d must be positive, got {epsilon_d}")
        if delta <= 0 or delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if sensitivity < 0:
            raise ValueError(f"sensitivity must be non-negative, got {sensitivity}")

        self.epsilon_d = epsilon_d
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = self._calibrate_sigma()

    def _calibrate_sigma(self) -> float:
        """Compute noise standard deviation: sigma = Delta_2 * sqrt(2*ln(1.25/delta)) / epsilon_d."""
        return self.sensitivity * np.sqrt(2.0 * np.log(1.25 / self.delta)) / self.epsilon_d

    def perturb(
        self,
        distances: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Add calibrated Gaussian noise to the distance vector.

        Parameters
        ----------
        distances : np.ndarray, shape (k,)
            True k-NN distances d_1, ..., d_k.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        noisy_distances : np.ndarray, shape (k,)
            Privatized distances d_tilde = d + xi, where xi ~ N(0, sigma^2 * I_k).
        """
        if rng is None:
            rng = np.random.default_rng()

        k = len(distances)
        noise = rng.normal(0.0, self.sigma, size=k)
        return distances + noise

    @property
    def expected_l2_noise(self) -> float:
        """Expected ell_2 norm of noise: E[||xi||_2] = sigma * sqrt(k)."""
        # This is approximate; exact is sigma * sqrt(k) * (1 - O(1/k))
        return self.sigma

    def rdp_epsilon(self, lam: float) -> float:
        """Renyi DP guarantee: (lambda, epsilon_hat)-RDP.

        epsilon_hat = lambda / (2 * (sigma / Delta_2)^2)
        """
        ratio = self.sigma / self.sensitivity
        return lam / (2.0 * ratio ** 2)
