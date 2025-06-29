import numpy as np
from scipy.stats import entropy

def entropy_score(price_window: list[float]) -> float:
    """Calculate entropy score for a window of prices."""
    counts, _ = np.histogram(price_window, bins=10)
    return entropy(counts + 1e-9)  # Avoid log(0) 