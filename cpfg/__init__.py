"""CPFG: Coupled Permute-and-Flip-Gaussian mechanism for private k-NN search."""

from cpfg.mechanism import CPFG, CPFGStar
from cpfg.pf import PermuteAndFlip
from cpfg.gaussian import GaussianMechanism
from cpfg.budget_split import optimal_budget_split
from cpfg.sensitivity import l2_sensitivity, smooth_sensitivity

__version__ = "1.0.0"
__all__ = [
    "CPFG",
    "CPFGStar",
    "PermuteAndFlip",
    "GaussianMechanism",
    "optimal_budget_split",
    "l2_sensitivity",
    "smooth_sensitivity",
]
