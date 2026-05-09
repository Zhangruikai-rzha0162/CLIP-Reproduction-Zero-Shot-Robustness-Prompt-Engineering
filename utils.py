"""
Utility functions for reproducibility and statistical confidence intervals.
"""

import os
import random
import numpy as np
import torch
from scipy import stats
from typing import Tuple


def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds to ensure full reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def wilson_interval(p: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Wilson Score Interval for binomial proportion confidence interval.
    Suitable for accuracy and other proportion-based metrics.

    Args:
        p: Observed accuracy (0-1).
        n: Sample size.
        confidence: Confidence level (default 0.95).

    Returns:
        (lower_bound, upper_bound)
    """
    if n == 0:
        return 0.0, 1.0

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denominator

    return max(0.0, centre - margin), min(1.0, centre + margin)


def compute_accuracy_with_ci(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float, int, int]:
    """
    Compute top-1 accuracy and its Wilson confidence interval.

    Returns:
        (accuracy, lower_bound, upper_bound, correct_count, total_count)
    """
    correct = (predictions == labels).sum()
    n = len(labels)
    acc = correct / n
    lower, upper = wilson_interval(acc, n, confidence)
    return acc, lower, upper, correct, n