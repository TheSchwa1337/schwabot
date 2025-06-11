"""
Schwafit Core Implementation
==========================

Implements the Schwafitting antifragile validation and partitioning system.
Now supports meta_tag tracking, validation history, and tag-based filtering.
"""

import numpy as np
import random
from typing import List, Dict, Any, Callable, Tuple, Optional
from .config import load_yaml_config, STRATEGIES_CONFIG_SCHEMA

class SchwafitManager:
    """
    Implements the Schwafitting antifragile validation and partitioning system.
    Now supports meta_tag tracking, validation history, and tag-based filtering.
    """
    def __init__(self, min_ratio: float = 0.01, max_ratio: float = 0.9, 
                 cycle_period: int = 1000, noise_scale: float = 0.05):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.cycle_period = cycle_period
        self.noise_scale = noise_scale
        self.t = 0
        self.variance_pool = 0.0
        self.memory_keys = {}  # shell class -> memory vector
        self.profit_tiers = {}  # tier -> calibration
        self.last_ratio = self.dynamic_holdout_ratio()
        self.validation_history = []  # List of dicts: {strategy, meta_tag, score, timestamp, ...}
        self.strategy_tags = {}  # strategy_id -> meta_tag
        
        # Load and validate strategy configuration using central config loader
        self.config = load_yaml_config(
            'strategies.yaml', 
            schema=STRATEGIES_CONFIG_SCHEMA
        )

    def register_strategy(self, strategy_id: str, meta_tag: str):
        """Register a strategy with its meta tag"""
        self.strategy_tags[strategy_id] = meta_tag

    def dynamic_holdout_ratio(self) -> float:
        """Dynamic r(t) with sinusoidal and stochastic components."""
        alpha = (self.max_ratio + self.min_ratio) / 2
        beta = (self.max_ratio - self.min_ratio) / 2
        T = self.cycle_period
        xi = np.random.normal(0, 1)
        gamma = self.noise_scale
        r = alpha + beta * np.sin(2 * np.pi * self.t / T) + gamma * xi
        r = max(self.min_ratio, min(self.max_ratio, r))
        self.last_ratio = r
        self.t += 1
        return r

    def split_data(self, data: List[Any], shell_states: List[Any] = None) -> Tuple[List[Any], List[Any]]:
        """Partition data into visible and holdout sets using current r(t)."""
        r = self.dynamic_holdout_ratio()
        n = len(data)
        holdout_size = int(n * r)
        indices = list(range(n))
        random.shuffle(indices)
        holdout_idx = indices[:holdout_size]
        visible_idx = indices[holdout_size:]
        
        if shell_states:
            return ([data[i] for i in visible_idx], [shell_states[i] for i in visible_idx]), \
                   ([data[i] for i in holdout_idx], [shell_states[i] for i in holdout_idx])
        else:
            return [data[i] for i in visible_idx], [data[i] for i in holdout_idx] 