"""
Schwabot AI Oracle - Profit Module
Analyzes historical and projected profit cycles for recursive signal generation.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import yaml
import logging
from pathlib import Path

from schwabot.schemas.trade_models import TradeSnapshot

logger = logging.getLogger(__name__)

# ------------------------
# Data Models
# ------------------------

@dataclass
class ProfitSignal:
    signal_id: str
    timestamp: datetime
    projected_gain: float
    duration_estimate: timedelta
    confidence: float
    source: str  # e.g., "long_term", "fallback", "matrix"
    tick_reference: Optional[int] = None

# ------------------------
# Oracle Logic
# ------------------------

class ProfitOracle:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.signal_history: List[ProfitSignal] = []
        self.tick_window: int = self.config.get("tick_window", 256)
        self.min_confidence: float = self.config.get("min_confidence", 0.65)

    def _load_config(self, path: Optional[str]) -> Dict:
        if not path:
            return {}
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Unable to load config from {path}: {e}")
            return {}

    def detect_profit_signal(self, snapshots: List[TradeSnapshot]) -> Optional[ProfitSignal]:
        if len(snapshots) < 2:
            return None

        deltas = self._calculate_returns(snapshots)
        momentum = self._calculate_momentum(deltas)
        confidence = self._calculate_signal_confidence(momentum, len(deltas))

        if confidence < self.min_confidence:
            return None

        projected_gain = float(np.mean(deltas)) * self.config.get("projection_multiplier", 1.2)
        duration = timedelta(seconds=self.tick_window * self.config.get("tick_interval_sec", 5))

        signal = ProfitSignal(
            signal_id=f"sig_{len(self.signal_history)}",
            timestamp=datetime.now(),
            projected_gain=projected_gain,
            duration_estimate=duration,
            confidence=confidence,
            source="long_term"
        )

        self.signal_history.append(signal)
        return signal

    # ------------------------
    # Metric Calculators
    # ------------------------

    def _calculate_returns(self, snapshots: List[TradeSnapshot]) -> List[float]:
        return [
            (snapshots[i].price - snapshots[i - 1].price) / snapshots[i - 1].price
            for i in range(1, len(snapshots))
            if snapshots[i - 1].price > 0
        ]

    def _calculate_momentum(self, returns: List[float]) -> float:
        return float(np.sum(returns[-self.config.get("momentum_window", 10):]))

    def _calculate_signal_confidence(self, momentum: float, sample_size: int) -> float:
        decay = np.exp(-sample_size / self.tick_window)
        return 1.0 - decay * np.abs(momentum)

    def get_recent_signals(self, window: int = 10) -> List[ProfitSignal]:
        return self.signal_history[-window:]

    def clear_old_signals(self, age_seconds: int = 3600):
        cutoff = datetime.now() - timedelta(seconds=age_seconds)
        self.signal_history = [s for s in self.signal_history if s.timestamp > cutoff]
