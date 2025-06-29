#!/usr/bin/env python3
"""
Phase Transition Monitor Module
==============================

Implements phase-state volatility estimation via entropy differential 
and lambda-weighted drift for Schwabot v0.05.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any, Tuple
from numpy.typing import NDArray
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


class PhaseState(Enum):
    """Phase state enumeration."""
    ACCUMULATION = "accumulation"
    CONSOLIDATION = "consolidation"
    EXPANSION = "expansion"
    DISTRIBUTION = "distribution"
    TRANSITION = "transition"


@dataclass
class PhaseTransitionEvent:
    """Phase transition event data."""
    event_id: str
    timestamp: float
    from_state: PhaseState
    to_state: PhaseState
    entropy_differential: float
    drift_weight: float
    volatility_estimate: float
    transition_probability: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseMetrics:
    """Phase metrics data."""
    metrics_id: str
    timestamp: float
    current_state: PhaseState
    entropy_trace: NDArray
    drift_weight: float
    volatility_estimate: float
    stability_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PhaseTransitionMonitor:
    """
    Phase Transition Monitor for Schwabot v0.05.
    
    Implements phase-state volatility estimation via entropy differential 
    and lambda-weighted drift.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the phase transition monitor."""
        self.config = config or self._default_config()
        
        # Phase tracking
        self.phase_metrics: List[PhaseMetrics] = []
        self.max_metrics_history = self.config.get('max_metrics_history', 1000)
        
        # Transition events
        self.transition_events: List[PhaseTransitionEvent] = []
        self.max_event_history = self.config.get('max_event_history', 100)
        
        # Current state
        self.current_phase_state = PhaseState.CONSOLIDATION
        
        # Monitoring parameters
        self.entropy_window_size = self.config.get('entropy_window_size', 50)
        self.drift_lambda = self.config.get('drift_lambda', 0.1)
        self.volatility_threshold = self.config.get('volatility_threshold', 0.7)
        self.transition_threshold = self.config.get('transition_threshold', 0.8)
        
        # Performance tracking
        self.total_evaluations = 0
        self.total_transitions = 0
        
        logger.info("🔄 Phase Transition Monitor initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'max_metrics_history': 1000,
            'max_event_history': 100,
            'entropy_window_size': 50,
            'drift_lambda': 0.1,
            'volatility_threshold': 0.7,
            'transition_threshold': 0.8,
            'min_signal_length': 10,
            'entropy_bin_count': 20,
            'drift_smoothing_factor': 0.3,
            'volatility_calculation_window': 20
        }
    
    def evaluate_phase_state(self, entropy_trace: NDArray, 
                           drift_weight: float) -> PhaseState:
        """
        Evaluate current phase state from entropy trace and drift weight.
        
        Args:
            entropy_trace: Entropy trace array
            drift_weight: Drift weight value
            
        Returns:
            Current phase state
        """
        try:
            if len(entropy_trace) < self.config['min_signal_length']:
                logger.warning(f"Entropy trace too short: {len(entropy_trace)}")
                return self.current_phase_state
            
            # Calculate volatility estimate
            volatility_estimate = self._calculate_volatility_estimate(entropy_trace)
            
            # Calculate stability score
            stability_score = self._calculate_stability_score(entropy_trace, drift_weight)
            
            # Determine phase state
            new_phase_state = self._determine_phase_state(entropy_trace, drift_weight, volatility_estimate)
            
            # Store phase metrics
            self._store_phase_metrics(entropy_trace, drift_weight, volatility_estimate, stability_score, new_phase_state)
            
            # Check for transition
            if new_phase_state != self.current_phase_state:
                self._record_transition_event(self.current_phase_state, new_phase_state, 
                                           entropy_trace, drift_weight, volatility_estimate)
                self.current_phase_state = new_phase_state
                self.total_transitions += 1
            
            self.total_evaluations += 1
            logger.debug(f"Phase state evaluation: {new_phase_state.value} (volatility: {volatility_estimate:.3f})")
            
            return new_phase_state
            
        except Exception as e:
            logger.error(f"Error evaluating phase state: {e}")
            return self.current_phase_state
    
    def is_phase_transition_likely(self, prev_phase: PhaseState, 
                                 current_score: float) -> bool:
        """
        Check if phase transition is likely based on current score.
        
        Args:
            prev_phase: Previous phase state
            current_score: Current transition score
            
        Returns:
            True if transition is likely
        """
        try:
            # Check if score exceeds threshold
            transition_likely = current_score >= self.transition_threshold
            
            # Additional checks based on phase characteristics
            if prev_phase == PhaseState.ACCUMULATION:
                # Accumulation to consolidation is common
                transition_likely = transition_likely or current_score >= 0.6
            elif prev_phase == PhaseState.EXPANSION:
                # Expansion to distribution is common
                transition_likely = transition_likely or current_score >= 0.6
            
            logger.debug(f"Phase transition likelihood: {transition_likely} (score: {current_score:.3f})")
            
            return transition_likely
            
        except Exception as e:
            logger.error(f"Error checking phase transition likelihood: {e}")
            return False
    
    def _calculate_volatility_estimate(self, entropy_trace: NDArray) -> float:
        """Calculate volatility estimate from entropy trace."""
        try:
            if len(entropy_trace) < 2:
                return 0.0
            
            # Calculate rolling volatility
            window_size = min(self.config['volatility_calculation_window'], len(entropy_trace) // 2)
            if window_size < 2:
                return np.std(entropy_trace)
            
            rolling_volatilities = []
            for i in range(window_size, len(entropy_trace)):
                window = entropy_trace[i-window_size:i]
                volatility = np.std(window)
                rolling_volatilities.append(volatility)
            
            if not rolling_volatilities:
                return np.std(entropy_trace)
            
            # Calculate weighted average (recent values have higher weight)
            weights = np.exp(-self.drift_lambda * np.arange(len(rolling_volatilities)))
            weighted_volatility = np.average(rolling_volatilities, weights=weights)
            
            # Normalize to [0, 1] range
            normalized_volatility = min(1.0, weighted_volatility / np.mean(entropy_trace) if np.mean(entropy_trace) > 0 else 0.0)
            
            return normalized_volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility estimate: {e}")
            return 0.0
    
    def _calculate_stability_score(self, entropy_trace: NDArray, 
                                 drift_weight: float) -> float:
        """Calculate stability score from entropy trace and drift weight."""
        try:
            if len(entropy_trace) < 2:
                return 1.0
            
            # Calculate entropy stability
            entropy_variance = np.var(entropy_trace)
            entropy_stability = 1.0 / (1.0 + entropy_variance)
            
            # Calculate drift stability
            drift_stability = 1.0 - min(1.0, abs(drift_weight - 0.5) * 2)
            
            # Combine stability scores
            stability_score = (entropy_stability + drift_stability) / 2.0
            
            return min(1.0, max(0.0, stability_score))
            
        except Exception as e:
            logger.error(f"Error calculating stability score: {e}")
            return 1.0
    
    def _determine_phase_state(self, entropy_trace: NDArray, 
                             drift_weight: float, 
                             volatility_estimate: float) -> PhaseState:
        """Determine phase state based on metrics."""
        try:
            # Calculate entropy characteristics
            entropy_mean = np.mean(entropy_trace)
            entropy_trend = np.polyfit(range(len(entropy_trace)), entropy_trace, 1)[0]
            
            # Determine state based on characteristics
            if volatility_estimate < 0.3 and drift_weight < 0.3:
                return PhaseState.ACCUMULATION
            elif volatility_estimate < 0.5 and abs(drift_weight - 0.5) < 0.2:
                return PhaseState.CONSOLIDATION
            elif volatility_estimate > 0.7 and drift_weight > 0.7:
                return PhaseState.EXPANSION
            elif volatility_estimate > 0.5 and drift_weight < 0.3:
                return PhaseState.DISTRIBUTION
            elif abs(entropy_trend) > 0.01:
                return PhaseState.TRANSITION
            else:
                return PhaseState.CONSOLIDATION
                
        except Exception as e:
            logger.error(f"Error determining phase state: {e}")
            return PhaseState.CONSOLIDATION
    
    def _store_phase_metrics(self, entropy_trace: NDArray, 
                           drift_weight: float, 
                           volatility_estimate: float, 
                           stability_score: float, 
                           phase_state: PhaseState):
        """Store phase metrics for historical analysis."""
        try:
            metrics = PhaseMetrics(
                metrics_id=f"metrics_{int(time.time() * 1000)}",
                timestamp=time.time(),
                current_state=phase_state,
                entropy_trace=entropy_trace.copy(),
                drift_weight=drift_weight,
                volatility_estimate=volatility_estimate,
                stability_score=stability_score
            )
            
            self.phase_metrics.append(metrics)
            if len(self.phase_metrics) > self.max_metrics_history:
                self.phase_metrics.pop(0)
                
        except Exception as e:
            logger.error(f"Error storing phase metrics: {e}")
    
    def _record_transition_event(self, from_state: PhaseState, 
                               to_state: PhaseState, 
                               entropy_trace: NDArray, 
                               drift_weight: float, 
                               volatility_estimate: float):
        """Record phase transition event."""
        try:
            # Calculate transition metrics
            entropy_differential = self._calculate_entropy_differential(entropy_trace)
            transition_probability = self._calculate_transition_probability(from_state, to_state)
            confidence = self._calculate_transition_confidence(entropy_differential, drift_weight, volatility_estimate)
            
            # Create transition event
            event = PhaseTransitionEvent(
                event_id=f"transition_{int(time.time() * 1000)}",
                timestamp=time.time(),
                from_state=from_state,
                to_state=to_state,
                entropy_differential=entropy_differential,
                drift_weight=drift_weight,
                volatility_estimate=volatility_estimate,
                transition_probability=transition_probability,
                confidence=confidence
            )
            
            self.transition_events.append(event)
            if len(self.transition_events) > self.max_event_history:
                self.transition_events.pop(0)
            
            logger.info(f"Phase transition recorded: {from_state.value} -> {to_state.value} (confidence: {confidence:.3f})")
            
        except Exception as e:
            logger.error(f"Error recording transition event: {e}")
    
    def _calculate_entropy_differential(self, entropy_trace: NDArray) -> float:
        """Calculate entropy differential."""
        try:
            if len(entropy_trace) < 2:
                return 0.0
            
            # Calculate rate of change
            entropy_diff = np.diff(entropy_trace)
            differential = np.mean(np.abs(entropy_diff))
            
            return differential
            
        except Exception as e:
            logger.error(f"Error calculating entropy differential: {e}")
            return 0.0
    
    def _calculate_transition_probability(self, from_state: PhaseState, 
                                        to_state: PhaseState) -> float:
        """Calculate transition probability between states."""
        try:
            # Define transition probabilities (simplified model)
            transition_matrix = {
                PhaseState.ACCUMULATION: {
                    PhaseState.CONSOLIDATION: 0.6,
                    PhaseState.EXPANSION: 0.3,
                    PhaseState.DISTRIBUTION: 0.1
                },
                PhaseState.CONSOLIDATION: {
                    PhaseState.ACCUMULATION: 0.3,
                    PhaseState.EXPANSION: 0.4,
                    PhaseState.DISTRIBUTION: 0.3
                },
                PhaseState.EXPANSION: {
                    PhaseState.CONSOLIDATION: 0.3,
                    PhaseState.DISTRIBUTION: 0.6,
                    PhaseState.ACCUMULATION: 0.1
                },
                PhaseState.DISTRIBUTION: {
                    PhaseState.ACCUMULATION: 0.5,
                    PhaseState.CONSOLIDATION: 0.4,
                    PhaseState.EXPANSION: 0.1
                }
            }
            
            probability = transition_matrix.get(from_state, {}).get(to_state, 0.25)
            return probability
            
        except Exception as e:
            logger.error(f"Error calculating transition probability: {e}")
            return 0.25
    
    def _calculate_transition_confidence(self, entropy_differential: float, 
                                       drift_weight: float, 
                                       volatility_estimate: float) -> float:
        """Calculate confidence in transition detection."""
        try:
            # Confidence based on multiple factors
            differential_confidence = min(1.0, entropy_differential * 10)  # Scale differential
            drift_confidence = 1.0 - abs(drift_weight - 0.5) * 2  # Center around 0.5
            volatility_confidence = volatility_estimate
            
            # Weighted combination
            confidence = (0.4 * differential_confidence + 
                         0.3 * drift_confidence + 
                         0.3 * volatility_confidence)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating transition confidence: {e}")
            return 0.5
    
    def get_phase_summary(self) -> Dict[str, Any]:
        """Get phase analysis summary."""
        try:
            if not self.phase_metrics:
                return {
                    "total_evaluations": 0,
                    "total_transitions": 0,
                    "current_phase": "consolidation",
                    "average_volatility": 0.0,
                    "average_stability": 1.0,
                    "most_common_phase": "consolidation"
                }
            
            # Calculate statistics
            volatilities = [m.volatility_estimate for m in self.phase_metrics]
            stabilities = [m.stability_score for m in self.phase_metrics]
            
            # Count phase states
            phase_counts = {}
            for metrics in self.phase_metrics:
                phase = metrics.current_state.value
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            
            most_common_phase = max(phase_counts.items(), key=lambda x: x[1])[0] if phase_counts else "consolidation"
            
            return {
                "total_evaluations": self.total_evaluations,
                "total_transitions": self.total_transitions,
                "current_phase": self.current_phase_state.value,
                "average_volatility": np.mean(volatilities),
                "average_stability": np.mean(stabilities),
                "most_common_phase": most_common_phase,
                "transition_rate": self.total_transitions / self.total_evaluations if self.total_evaluations > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting phase summary: {e}")
            return {}
    
    def get_recent_transitions(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent phase transitions."""
        recent_transitions = self.transition_events[-count:]
        return [
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp,
                "from_state": e.from_state.value,
                "to_state": e.to_state.value,
                "entropy_differential": e.entropy_differential,
                "drift_weight": e.drift_weight,
                "volatility_estimate": e.volatility_estimate,
                "transition_probability": e.transition_probability,
                "confidence": e.confidence
            }
            for e in recent_transitions
        ]
    
    def export_phase_data(self, filepath: str) -> bool:
        """
        Export phase data to JSON file.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if export was successful
        """
        try:
            import json
            
            data = {
                "export_timestamp": time.time(),
                "config": self.config,
                "summary": self.get_phase_summary(),
                "recent_transitions": self.get_recent_transitions(20)
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported phase data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting phase data: {e}")
            return False 