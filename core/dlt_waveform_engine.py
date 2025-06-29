"""
DLT Waveform Engine
===================

Implements the Delta Lock Transform (DLT) waveform for recursive, memory-locked,
observer-aware, phase-corrected signal confirmation in Schwabot.

Mathematical Core:
    - Delta:           Δx_t = x_t - x_{t-1}
    - Memory:          M_n = {Δx_{t-k} | k in [1, H]}
    - Triplet Lock:    ∃k1,k2,k3: Δx_{t-k1} ≈ Δx_{t-k2} ≈ Δx_{t-k3}
    - Phase Projection: Π_t (phase overlay operator)
    - Drift Correction: θ(t) = ∫₀ᵗ ∇DLT_n(τ) dτ
    - Greyscale:       sigmoid-weighted confidence for soft fade-out
    - Observer Lock:   Only collapse waveform if observer confirms
    - Hash State:      Output for system-wide routing
"""

import numpy as np
import hashlib
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class DLTState(Enum):
    UNLOCKED = 0
    LOCKED = 1
    FADED = 2
    WAITING = 3

@dataclass
class DLTMemory:
    """Stores delta memory for recursive lock confirmation."""
    horizon: int = 30
    deltas: List[float] = field(default_factory=list)
    
    def update(self, value: float):
        self.deltas.append(value)
        if len(self.deltas) > self.horizon:
            self.deltas.pop(0)
    
    def get_triplets(self) -> List[Tuple[float, float, float]]:
        """Return all possible triplets in memory."""
        if len(self.deltas) < 3:
            return []
        return [
            (self.deltas[i], self.deltas[i+1], self.deltas[i+2])
            for i in range(len(self.deltas) - 2)
        ]

@dataclass
class DLTWaveformResult:
    state: DLTState
    lock_triplet: Optional[Tuple[float, float, float]] = None
    phase_projection: float = 0.0
    drift_correction: float = 0.0
    confidence: float = 0.0
    hash_state: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    observer_confirmed: bool = False
    greyscale: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

class DLTWaveformEngine:
    """
    Delta Lock Transform (DLT) Waveform Engine for recursive, memory-locked,
    observer-aware, phase-corrected signal confirmation.
    """
    def __init__(self, memory_horizon: int = 30, triplet_tol: float = 1e-6, greyscale_threshold: float = 0.5):
        self.memory = DLTMemory(horizon=memory_horizon)
        self.triplet_tol = triplet_tol
        self.greyscale_threshold = greyscale_threshold
        self.last_phase_projection = 0.0
        self.last_drift_correction = 0.0
        self.last_hash_state = ""
        self.last_confidence = 0.0
        self.last_greyscale = 0.0
        self.last_state = DLTState.WAITING
        self.last_triplet = None
        self.last_observer_confirmed = False
        self.history: List[DLTWaveformResult] = []

    def update(self, x_t: float, x_prev: float, observer: Optional[bool] = None) -> DLTWaveformResult:
        """
        Update the DLT engine with a new value and compute lock state.
        Args:
            x_t: Current observed value
            x_prev: Previous observed value
            observer: Optional observer confirmation (True/False)
        Returns:
            DLTWaveformResult with all computed fields
        """
        delta = x_t - x_prev
        self.memory.update(delta)
        triplet, locked = self._check_triplet_lock()
        phase_proj = self._phase_projection()
        drift_corr = self._drift_correction()
        confidence = self._confidence(triplet, locked)
        greyscale = self._greyscale(confidence)
        hash_state = self._hash_state(triplet, phase_proj, drift_corr, confidence)
        observer_confirmed = observer if observer is not None else False

        # Determine state
        if locked and observer_confirmed and greyscale > self.greyscale_threshold:
            state = DLTState.LOCKED
        elif greyscale < self.greyscale_threshold:
            state = DLTState.FADED
        else:
            state = DLTState.WAITING

        result = DLTWaveformResult(
            state=state,
            lock_triplet=triplet,
            phase_projection=phase_proj,
            drift_correction=drift_corr,
            confidence=confidence,
            hash_state=hash_state,
            timestamp=datetime.utcnow(),
            observer_confirmed=observer_confirmed,
            greyscale=greyscale,
            meta={
                "delta": delta,
                "locked": locked,
                "observer": observer,
            }
        )
        self._update_last(result)
        self.history.append(result)
        return result

    def _check_triplet_lock(self) -> Tuple[Optional[Tuple[float, float, float]], bool]:
        """
        Check for triplet lock in memory.
        Returns:
            (triplet, locked)
        """
        triplets = self.memory.get_triplets()
        for triplet in triplets[::-1]:  # Check most recent first
            a, b, c = triplet
            if abs(a - b) < self.triplet_tol and abs(b - c) < self.triplet_tol:
                return triplet, True
        return None, False

    def _phase_projection(self) -> float:
        """
        Project phase overlay (simple running mean as placeholder).
        Returns:
            Projected phase value
        """
        if not self.memory.deltas:
            return 0.0
        return float(np.mean(self.memory.deltas))

    def _drift_correction(self) -> float:
        """
        Integrate drift correction (cumulative sum of delta changes).
        Returns:
            Drift correction value
        """
        if not self.memory.deltas:
            return 0.0
        grad = np.gradient(self.memory.deltas)
        return float(np.sum(grad))

    def _confidence(self, triplet: Optional[Tuple[float, float, float]], locked: bool) -> float:
        """
        Compute confidence score based on triplet and lock state.
        Returns:
            Confidence value in [0, 1]
        """
        if not self.memory.deltas:
            return 0.0
        if locked and triplet:
            # Lower variance = higher confidence
            var = np.var(triplet)
            conf = max(0.0, 1.0 - var / (abs(np.mean(triplet)) + 1e-8))
            return min(1.0, conf)
        # Otherwise, use normalized std of recent deltas
        std = np.std(self.memory.deltas[-5:]) if len(self.memory.deltas) >= 5 else np.std(self.memory.deltas)
        conf = max(0.0, 1.0 - std / (abs(np.mean(self.memory.deltas)) + 1e-8))
        return min(1.0, conf)

    def _greyscale(self, confidence: float) -> float:
        """
        Sigmoid-weighted greyscale confidence for soft fade-out.
        Returns:
            Greyscale value in [0, 1]
        """
        return 1.0 / (1.0 + np.exp(-8 * (confidence - 0.5)))

    def _hash_state(self, triplet, phase_proj, drift_corr, confidence) -> str:
        """
        Hash the current state for system-wide routing.
        Returns:
            Hex digest string
        """
        state_str = f"{triplet}_{phase_proj:.6f}_{drift_corr:.6f}_{confidence:.6f}"
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]

    def _update_last(self, result: DLTWaveformResult):
        self.last_state = result.state
        self.last_triplet = result.lock_triplet
        self.last_phase_projection = result.phase_projection
        self.last_drift_correction = result.drift_correction
        self.last_confidence = result.confidence
        self.last_greyscale = result.greyscale
        self.last_hash_state = result.hash_state
        self.last_observer_confirmed = result.observer_confirmed

    def get_last_result(self) -> DLTWaveformResult:
        return self.history[-1] if self.history else None

    def get_history(self) -> List[DLTWaveformResult]:
        return self.history.copy()

# Export main class
__all__ = ["DLTWaveformEngine", "DLTWaveformResult", "DLTState"] 