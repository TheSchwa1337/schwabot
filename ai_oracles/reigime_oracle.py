"""
Schwabot AI Oracle - Regime Detector
Monitors macro conditions to identify transitions between volatility/trend/volume regimes.
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

# ------------------------
# Data Model
# ------------------------

@dataclass
class MarketRegime:
    regime_id: str
    timestamp: datetime
    volatility: float
    trend: float
    volume: float
    confidence: float

# ------------------------
# Regime Detection Engine
# ------------------------

class RegimeDetector:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.history: List[MarketRegime] = []
        self.window_size: int = self.config.get("window_size", 20)
        self.min_confidence: float = self.config.get("min_confidence", 0.6)

    def _load_config(self, path: Optional[str]) -> Dict:
        if not path:
            return {}
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Unable to load config from {path}: {e}")
            return {}

    def detect(self, price_series: List[float], volume_series: List[float]) -> Optional[MarketRegime]:
        if len(price_series) < self.window_size or len(volume_series) < self.window_size:
            return None

        vol = self._calculate_volatility(price_series)
        trend = self._calculate_trend(price_series)
        volume = self._calculate_volume(volume_series)
        confidence = self._calculate_confidence(vol, trend, volume)

        if confidence < self.min_confidence:
            return None

        regime = MarketRegime(
            regime_id=f"regime_{len(self.history)}",
            timestamp=datetime.now(),
            volatility=vol,
            trend=trend,
            volume=volume,
            confidence=confidence
        )

        self.history.append(regime)
        return regime

    # ------------------------
    # Metrics
    # ------------------------

    def _calculate_volatility(self, prices: List[float]) -> float:
        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns))

    def _calculate_trend(self, prices: List[float]) -> float:
        x = np.arange(len(prices))
        y = np.array(prices)
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)

    def _calculate_volume(self, volumes: List[float]) -> float:
        return float(np.mean(volumes))

    def _calculate_confidence(self, vol: float, trend: float, volume: float) -> float:
        weights = self.config.get("weights", {
            'volatility': 0.4,
            'trend': 0.4,
            'volume': 0.2
        })
        norm = abs(vol) + abs(trend) + abs(volume)
        if norm == 0:
            return 0.0
        return (
            abs(vol) * weights['volatility'] +
            abs(trend) * weights['trend'] +
            abs(volume) * weights['volume']
        ) / norm

    def get_recent_regimes(self, n: int = 5) -> List[MarketRegime]:
        return self.history[-n:]
