#!/usr/bin/env python3
""""""
Drift-Phase Weighter Module
===========================

Implements weighted λ drift scoring via exponential smoothing.
Tracks Δp/Δt patterns to signal transition tension for Schwabot v0.5.
""""""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List
from numpy.typing import NDArray
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Drift type enumeration."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    OSCILLATORY = "oscillatory"
    CHAOTIC = "chaotic"


@dataclass
class DriftMetrics:
    """Drift metrics data."""
    drift_weight: float
    drift_type: DriftType
    lambda_decay: float
    transition_tension: float
    phase_stability: float
    entropy_score: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseTransition:
    """Phase transition data."""
    transition_id: str
    from_phase: str
    to_phase: str
    drift_weight: float
    tension_score: float
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class DriftPhaseWeighter:
    """"""
    Drift-Phase Weighter for Schwabot v0.5.

    Implements weighted λ drift scoring via exponential smoothing.
    Tracks Δp/Δt patterns to signal transition tension.
    """"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the drift-phase weighter."""
        self.config = config or self._default_config()

        # Drift tracking
        self.drift_history: List[DriftMetrics] = []
        self.max_history_size = self.config.get('max_history_size', 1000)

        # Phase transitions
        self.phase_transitions: List[PhaseTransition] = []
        self.max_transition_history = self.config.get('max_transition_history', 100)

        # Lambda decay parameters
        self.lambda_decay_rate = self.config.get('lambda_decay_rate', 0.1)
        self.exponential_smoothing_alpha = self.config.get('exponential_smoothing_alpha', 0.3)

        # Transition tension thresholds
        self.tension_threshold_high = self.config.get('tension_threshold_high', 0.8)
        self.tension_threshold_medium = self.config.get('tension_threshold_medium', 0.5)
        self.tension_threshold_low = self.config.get('tension_threshold_low', 0.2)

        # Performance tracking
        self.total_drift_calculations = 0
        self.total_transitions_detected = 0

        logger.info("🌊 Drift-Phase Weighter initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {}
            'max_history_size': 1000,
                'max_transition_history': 100,
                    'lambda_decay_rate': 0.1,
                    'exponential_smoothing_alpha': 0.3,
                    'tension_threshold_high': 0.8,
                    'tension_threshold_medium': 0.5,
                    'tension_threshold_low': 0.2,
                    'min_trace_length': 10,
                    'drift_window_size': 20,
                    'phase_detection_sensitivity': 0.7,
                    'entropy_calculation_window': 50
}
    def calculate_phase_drift_weight(self, trace: NDArray) -> float:
        """"""
        Calculate phase drift weight from trace data.

        Args:
            trace: Input trace array (price/time series)

        Returns:
            Drift weight score
        """"""
        try:
            if len(trace) < self.config['min_trace_length']:
                logger.warning(f"Trace too short: {len(trace)} < {self.config['min_trace_length']}")
                return 0.0

            # Calculate Δp/Δt (price change over time)
            price_changes = np.diff(trace)
            time_steps = np.arange(1, len(trace))

            # Apply exponential smoothing to price changes
            smoothed_changes = self._apply_exponential_smoothing(price_changes)

            # Calculate drift weight using weighted mean
            weights = np.exp(-self.lambda_decay_rate * time_steps)
            drift_weight = np.average(smoothed_changes, weights=weights)

            # Normalize to [0, 1] range
            drift_weight = self._normalize_drift_weight(drift_weight)

            self.total_drift_calculations += 1
            logger.debug(f"Calculated drift weight: {drift_weight:.4f}")

            return drift_weight

        except Exception as e:
            logger.error(f"Error calculating drift weight: {e}")
            return 0.0

    def normalize_drift_curve(self, curve: NDArray) -> NDArray:
        """"""
        Normalize drift curve using min-max normalization.

        Args:
            curve: Input curve array

        Returns:
            Normalized curve array
        """"""
        try:
            if len(curve) == 0:
                return curve

            min_val = np.min(curve)
            max_val = np.max(curve)

            if max_val == min_val:
                return np.zeros_like(curve)

            normalized = (curve - min_val) / (max_val - min_val)
            return normalized

        except Exception as e:
            logger.error(f"Error normalizing drift curve: {e}")
            return curve

    def apply_lambda_decay(self, signal: NDArray, λ: float) -> NDArray:
        """"""
        Apply lambda decay to signal using exponential decay.

        Args:
            signal: Input signal array
            λ: Lambda decay rate

        Returns:
            Decayed signal array
        """"""
        try:
            if len(signal) == 0:
                return signal

            # Create time vector
            time_vector = np.arange(len(signal))

            # Apply exponential decay: exp(-λ * t)
            decay_factor = np.exp(-λ * time_vector)

            # Apply decay to signal
            decayed_signal = signal * decay_factor

            return decayed_signal

        except Exception as e:
            logger.error(f"Error applying lambda decay: {e}")
            return signal

    def _apply_exponential_smoothing(self, data: NDArray) -> NDArray:
        """Apply exponential smoothing to data."""
        try:
            alpha = self.exponential_smoothing_alpha
            smoothed = np.zeros_like(data)
            smoothed[0] = data[0]

            for i in range(1, len(data)):
                smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]

            return smoothed

        except Exception as e:
            logger.error(f"Error applying exponential smoothing: {e}")
            return data

    def _normalize_drift_weight(self, weight: float) -> float:
        """Normalize drift weight to [0, 1] range."""
        # Use sigmoid normalization for smooth mapping
        return 1.0 / (1.0 + np.exp(-weight))

    def detect_phase_transition(self, current_trace: NDArray, )
                              previous_phase: str) -> Optional[PhaseTransition]:
        """"""
        Detect phase transition based on drift patterns.

        Args:
            current_trace: Current trace data
            previous_phase: Previous phase identifier

        Returns:
            Phase transition if detected
        """"""
        try:
            # Calculate current drift weight
            current_drift = self.calculate_phase_drift_weight(current_trace)

            # Get previous drift weight
            previous_drift = 0.0
            if self.drift_history:
                previous_drift = self.drift_history[-1].drift_weight

            # Calculate drift change
            drift_change = abs(current_drift - previous_drift)

            # Calculate transition tension
            tension_score = self._calculate_transition_tension()
                current_drift, previous_drift, drift_change
            )

            # Determine if transition is likely
            if tension_score > self.config['phase_detection_sensitivity']:
                # Determine new phase based on drift characteristics
                new_phase = self._determine_phase_from_drift(current_drift)

                # Create transition record
                transition = PhaseTransition()
                    transition_id=f"transition_{int(time.time() * 1000)}",
                        from_phase=previous_phase,
                            to_phase=new_phase,
                            drift_weight=current_drift,
                            tension_score=tension_score,
                            confidence=self._calculate_transition_confidence(tension_score),
                            timestamp=time.time()
                )

                self.phase_transitions.append(transition)
                if len(self.phase_transitions) > self.max_transition_history:
                    self.phase_transitions.pop(0)

                self.total_transitions_detected += 1
                logger.info(f"Phase transition detected: {previous_phase} -> {new_phase} (tension: {tension_score:.3f})")

                return transition

            return None

        except Exception as e:
            logger.error(f"Error detecting phase transition: {e}")
            return None

    def _calculate_transition_tension(self, current_drift: float, )
                                    previous_drift: float,
                                        drift_change: float) -> float:
        """Calculate transition tension score."""
        try:
            # Base tension from drift change
            base_tension = drift_change

            # Add volatility component
            volatility_component = self._calculate_volatility_component(current_drift)

            # Add momentum component
            momentum_component = self._calculate_momentum_component(current_drift, previous_drift)

            # Combine components
            tension = base_tension + 0.3 * volatility_component + 0.2 * momentum_component

            # Normalize to [0, 1]
            return min(1.0, max(0.0, tension))

        except Exception as e:
            logger.error(f"Error calculating transition tension: {e}")
            return 0.0

    def _calculate_volatility_component(self, drift: float) -> float:
        """Calculate volatility component of tension."""
        # Higher drift values indicate higher volatility
        return abs(drift - 0.5) * 2.0  # Scale to [0, 1]

    def _calculate_momentum_component(self, current_drift: float, )
                                    previous_drift: float) -> float:
        """Calculate momentum component of tension."""
        # Momentum is the rate of change
        return abs(current_drift - previous_drift)

    def _determine_phase_from_drift(self, drift: float) -> str:
        """Determine phase from drift characteristics."""
        if drift < 0.25:
            return "accumulation"
        elif drift < 0.5:
            return "consolidation"
        elif drift < 0.75:
            return "expansion"
        else:
            return "distribution"

    def _calculate_transition_confidence(self, tension_score: float) -> float:
        """Calculate confidence in transition detection."""
        # Higher tension = higher confidence
        return min(1.0, tension_score * 1.2)

    def analyze_drift_pattern(self, trace: NDArray) -> DriftMetrics:
        """"""
        Analyze drift pattern and return comprehensive metrics.

        Args:
            trace: Input trace array

        Returns:
            Drift metrics
        """"""
        try:
            # Calculate basic drift weight
            drift_weight = self.calculate_phase_drift_weight(trace)

            # Determine drift type
            drift_type = self._classify_drift_type(trace)

            # Calculate lambda decay
            lambda_decay = self._calculate_lambda_decay(trace)

            # Calculate transition tension
            transition_tension = self._calculate_overall_tension(trace)

            # Calculate phase stability
            phase_stability = self._calculate_phase_stability(trace)

            # Calculate entropy score
            entropy_score = self._calculate_entropy_score(trace)

            # Create metrics object
            metrics = DriftMetrics()
                drift_weight=drift_weight,
                    drift_type=drift_type,
                        lambda_decay=lambda_decay,
                        transition_tension=transition_tension,
                        phase_stability=phase_stability,
                        entropy_score=entropy_score,
                        timestamp=time.time()
            )

            # Add to history
            self.drift_history.append(metrics)
            if len(self.drift_history) > self.max_history_size:
                self.drift_history.pop(0)

            return metrics

        except Exception as e:
            logger.error(f"Error analyzing drift pattern: {e}")
            return self._create_default_metrics()

    def _classify_drift_type(self, trace: NDArray) -> DriftType:
        """Classify drift type based on pattern analysis."""
        try:
            if len(trace) < 10:
                return DriftType.LINEAR

            # Calculate autocorrelation
            autocorr = np.correlate(trace, trace, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Calculate spectral density
            fft = np.fft.fft(trace)
            spectral_density = np.abs(fft) ** 2

            # Analyze patterns
            if np.std(trace) < 0.1:
                return DriftType.LINEAR
            elif np.max(spectral_density[1:]) > np.mean(spectral_density) * 2:
                return DriftType.OSCILLATORY
            elif np.std(autocorr) > np.std(trace) * 0.5:
                return DriftType.CHAOTIC
            else:
                return DriftType.EXPONENTIAL

        except Exception as e:
            logger.error(f"Error classifying drift type: {e}")
            return DriftType.LINEAR

    def _calculate_lambda_decay(self, trace: NDArray) -> float:
        """Calculate lambda decay rate from trace."""
        try:
            if len(trace) < 2:
                return 0.0

            # Fit exponential decay model
            time_steps = np.arange(len(trace))
            log_trace = np.log(np.abs(trace) + 1e-10)

            # Linear fit to log data
            coeffs = np.polyfit(time_steps, log_trace, 1)
            lambda_decay = -coeffs[0]  # Negative slope = decay

            return max(0.0, lambda_decay)

        except Exception as e:
            logger.error(f"Error calculating lambda decay: {e}")
            return 0.0

    def _calculate_overall_tension(self, trace: NDArray) -> float:
        """Calculate overall transition tension."""
        try:
            # Calculate multiple tension components
            drift_tension = self.calculate_phase_drift_weight(trace)
            volatility_tension = np.std(trace) / (np.mean(trace) + 1e-10)
            momentum_tension = np.mean(np.abs(np.diff(trace)))

            # Combine tensions
            overall_tension = (drift_tension + volatility_tension + momentum_tension) / 3.0

            return min(1.0, overall_tension)

        except Exception as e:
            logger.error(f"Error calculating overall tension: {e}")
            return 0.0

    def _calculate_phase_stability(self, trace: NDArray) -> float:
        """Calculate phase stability score."""
        try:
            if len(trace) < 10:
                return 1.0

            # Calculate rolling variance
            window_size = min(10, len(trace) // 2)
            rolling_var = []

            for i in range(window_size, len(trace)):
                window = trace[i-window_size:i]
                rolling_var.append(np.var(window))

            if not rolling_var:
                return 1.0

            # Stability is inverse of variance
            avg_variance = np.mean(rolling_var)
            stability = 1.0 / (1.0 + avg_variance)

            return min(1.0, stability)

        except Exception as e:
            logger.error(f"Error calculating phase stability: {e}")
            return 1.0

    def _calculate_entropy_score(self, trace: NDArray) -> float:
        """Calculate entropy score of trace."""
        try:
            if len(trace) < 2:
                return 0.0

            # Discretize trace into bins
            bins = np.histogram(trace, bins=min(20, len(trace)//2))[0]
            bins = bins[bins > 0]  # Remove zero bins

            if len(bins) == 0:
                return 0.0

            # Calculate Shannon entropy
            probabilities = bins / np.sum(bins)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

            # Normalize to [0, 1]
            max_entropy = np.log2(len(bins))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            return normalized_entropy

        except Exception as e:
            logger.error(f"Error calculating entropy score: {e}")
            return 0.0

    def _create_default_metrics(self) -> DriftMetrics:
        """Create default drift metrics."""
        return DriftMetrics()
            drift_weight=0.0,
                drift_type=DriftType.LINEAR,
                    lambda_decay=0.0,
                    transition_tension=0.0,
                    phase_stability=1.0,
                    entropy_score=0.0,
                    timestamp=time.time()
        )

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get drift analysis summary."""
        try:
            if not self.drift_history:
                return {}
                    "total_calculations": 0,
                        "total_transitions": 0,
                            "average_drift_weight": 0.0,
                            "most_common_drift_type": "linear",
                            "average_tension": 0.0,
                            "average_stability": 1.0
}
            # Calculate statistics
            drift_weights = [m.drift_weight for m in self.drift_history]
            tension_scores = [m.transition_tension for m in self.drift_history]
            stability_scores = [m.phase_stability for m in self.drift_history]

            # Count drift types
            drift_type_counts = {}
            for metrics in self.drift_history:
                drift_type = metrics.drift_type.value
                drift_type_counts[drift_type] = drift_type_counts.get(drift_type, 0) + 1

            most_common_type = max(drift_type_counts.items(), key=lambda x: x[1])[0]

            return {}
                "total_calculations": self.total_drift_calculations,
                    "total_transitions": self.total_transitions_detected,
                        "average_drift_weight": np.mean(drift_weights),
                        "most_common_drift_type": most_common_type,
                        "average_tension": np.mean(tension_scores),
                        "average_stability": np.mean(stability_scores),
                        "recent_transitions": len(self.phase_transitions[-5:]) if self.phase_transitions else 0
}
        except Exception as e:
            logger.error(f"Error getting drift summary: {e}")
            return {}

    def get_recent_transitions(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent phase transitions."""
        recent_transitions = self.phase_transitions[-count:]
        return []
            {}
                "transition_id": t.transition_id,
                    "from_phase": t.from_phase,
                        "to_phase": t.to_phase,
                        "drift_weight": t.drift_weight,
                        "tension_score": t.tension_score,
                        "confidence": t.confidence,
                        "timestamp": t.timestamp
}
            for t in recent_transitions
]
    def export_drift_data(self, filepath: str) -> bool:
        """"""
        Export drift data to JSON file.

        Args:
            filepath: Output file path

        Returns:
            True if export was successful
        """"""
        try:
            import json

            data = {
                "export_timestamp": time.time(),
                "config": self.config,
                "summary": self.get_drift_summary(),
                "recent_metrics": []
}
                    {}
                        "drift_weight": m.drift_weight,
                            "drift_type": m.drift_type.value,
                                "lambda_decay": m.lambda_decay,
                                "transition_tension": m.transition_tension,
                                "phase_stability": m.phase_stability,
                                "entropy_score": m.entropy_score,
                                "timestamp": m.timestamp
}
                    for m in self.drift_history[-100:]  # Last 100 metrics
                ],
                    "recent_transitions": self.get_recent_transitions(50)
}
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Exported drift data to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting drift data: {e}")
            return False