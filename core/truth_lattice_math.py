#!/usr/bin/env python3
"""
Truth Lattice Math (Consensus Engine) Module
============================================

Implements weighted mean collapse logic with sigmoid dampening for 
volatility tolerance for Schwabot v0.05.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib

logger = logging.getLogger(__name__)


class ConsensusState(Enum):
    """Consensus state enumeration."""
    DISSENT = "dissent"
    PARTIAL = "partial"
    CONVERGING = "converging"
    CONSENSUS = "consensus"
    UNANIMOUS = "unanimous"


@dataclass
class ConsensusResult:
    """Consensus result data."""
    consensus_id: str
    timestamp: float
    consensus_state: ConsensusState
    collapse_score: float
    signal_count: int
    agreement_ratio: float
    volatility_tolerance: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalWeight:
    """Signal weight data."""
    signal_id: str
    weight: float
    source: str
    timestamp: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class TruthLatticeMath:
    """
    Truth Lattice Math (Consensus Engine) for Schwabot v0.05.
    
    Implements weighted mean collapse logic with sigmoid dampening 
    for volatility tolerance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the truth lattice math engine."""
        self.config = config or self._default_config()
        
        # Consensus tracking
        self.consensus_history: List[ConsensusResult] = []
        self.max_history_size = self.config.get('max_history_size', 1000)
        
        # Signal weights
        self.signal_weights: List[SignalWeight] = []
        self.max_weight_history = self.config.get('max_weight_history', 100)
        
        # Consensus parameters
        self.consensus_threshold = self.config.get('consensus_threshold', 0.85)
        self.sigmoid_steepness = self.config.get('sigmoid_steepness', 5.0)
        self.volatility_dampening = self.config.get('volatility_dampening', 0.1)
        self.weight_decay_rate = self.config.get('weight_decay_rate', 0.05)
        
        # Performance tracking
        self.total_collapses = 0
        self.successful_consensus = 0
        
        logger.info("🔮 Truth Lattice Math (Consensus Engine) initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'max_history_size': 1000,
            'max_weight_history': 100,
            'consensus_threshold': 0.85,
            'sigmoid_steepness': 5.0,
            'volatility_dampening': 0.1,
            'weight_decay_rate': 0.05,
            'min_signals': 3,
            'max_signals': 100,
            'agreement_tolerance': 0.1,
            'volatility_window': 20,
            'confidence_decay': 0.02
        }
    
    def collapse_score(self, signals: List[float]) -> float:
        """
        Calculate collapse score using weighted mean collapse logic.
        
        Args:
            signals: List of signal values
            
        Returns:
            Collapse score between 0 and 1
        """
        try:
            if len(signals) < self.config['min_signals']:
                logger.warning(f"Too few signals: {len(signals)} < {self.config['min_signals']}")
                return 0.0
            
            if len(signals) > self.config['max_signals']:
                signals = signals[:self.config['max_signals']]
                logger.warning(f"Truncated signals to {self.config['max_signals']}")
            
            # Calculate signal statistics
            signal_mean = np.mean(signals)
            signal_std = np.std(signals)
            signal_variance = np.var(signals)
            
            # Calculate agreement ratio (how close signals are to mean)
            agreement_ratios = []
            for signal in signals:
                if signal_std > 0:
                    agreement = 1.0 - min(1.0, abs(signal - signal_mean) / (signal_std * 2))
                else:
                    agreement = 1.0
                agreement_ratios.append(agreement)
            
            # Calculate weighted agreement
            weights = self._calculate_signal_weights(signals)
            weighted_agreement = np.average(agreement_ratios, weights=weights)
            
            # Apply sigmoid dampening for volatility tolerance
            volatility_factor = self._calculate_volatility_factor(signal_variance)
            dampened_agreement = self._apply_sigmoid_dampening(weighted_agreement, volatility_factor)
            
            # Final collapse score
            collapse_score = min(1.0, max(0.0, dampened_agreement))
            
            self.total_collapses += 1
            logger.debug(f"Collapse score: {collapse_score:.4f} (signals: {len(signals)}, agreement: {weighted_agreement:.3f})")
            
            return collapse_score
            
        except Exception as e:
            logger.error(f"Error calculating collapse score: {e}")
            return 0.0
    
    def is_consensus_reached(self, score: float, threshold: float = 0.85) -> bool:
        """
        Check if consensus is reached based on collapse score.
        
        Args:
            score: Collapse score
            threshold: Consensus threshold (default: 0.85)
            
        Returns:
            True if consensus is reached
        """
        try:
            consensus_reached = score >= threshold
            
            if consensus_reached:
                self.successful_consensus += 1
            
            logger.debug(f"Consensus check: {consensus_reached} (score: {score:.3f}, threshold: {threshold:.3f})")
            
            return consensus_reached
            
        except Exception as e:
            logger.error(f"Error checking consensus: {e}")
            return False
    
    def evaluate_consensus(self, signals: List[float], 
                          signal_sources: Optional[List[str]] = None) -> ConsensusResult:
        """
        Evaluate consensus from multiple signals.
        
        Args:
            signals: List of signal values
            signal_sources: List of signal sources (optional)
            
        Returns:
            Consensus result
        """
        try:
            if signal_sources is None:
                signal_sources = [f"source_{i}" for i in range(len(signals))]
            
            # Calculate collapse score
            collapse_score = self.collapse_score(signals)
            
            # Determine consensus state
            consensus_state = self._determine_consensus_state(collapse_score, len(signals))
            
            # Calculate agreement ratio
            agreement_ratio = self._calculate_agreement_ratio(signals)
            
            # Calculate volatility tolerance
            volatility_tolerance = self._calculate_volatility_tolerance(signals)
            
            # Calculate confidence
            confidence = self._calculate_consensus_confidence(collapse_score, len(signals), agreement_ratio)
            
            # Create consensus result
            result = ConsensusResult(
                consensus_id=f"consensus_{int(time.time() * 1000)}",
                timestamp=time.time(),
                consensus_state=consensus_state,
                collapse_score=collapse_score,
                signal_count=len(signals),
                agreement_ratio=agreement_ratio,
                volatility_tolerance=volatility_tolerance,
                confidence=confidence,
                metadata={
                    'signal_sources': signal_sources,
                    'signal_mean': np.mean(signals),
                    'signal_std': np.std(signals)
                }
            )
            
            # Add to history
            self.consensus_history.append(result)
            if len(self.consensus_history) > self.max_history_size:
                self.consensus_history.pop(0)
            
            # Update signal weights
            self._update_signal_weights(signals, signal_sources, collapse_score)
            
            logger.info(f"Consensus evaluation: {consensus_state.value} (score: {collapse_score:.3f}, confidence: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating consensus: {e}")
            return self._create_default_consensus_result()
    
    def _calculate_signal_weights(self, signals: List[float]) -> List[float]:
        """Calculate weights for signals based on historical performance."""
        try:
            if not self.signal_weights:
                # Equal weights if no history
                return [1.0] * len(signals)
            
            # Get recent weights for each source
            source_weights = {}
            current_time = time.time()
            
            for weight_data in self.signal_weights:
                source = weight_data.source
                age = current_time - weight_data.timestamp
                decay_factor = np.exp(-self.weight_decay_rate * age)
                
                if source not in source_weights:
                    source_weights[source] = []
                
                source_weights[source].append(weight_data.weight * decay_factor * weight_data.confidence)
            
            # Calculate average weight for each source
            avg_weights = {}
            for source, weights in source_weights.items():
                if weights:
                    avg_weights[source] = np.mean(weights)
            
            # Assign weights to signals (assume sources in order)
            signal_weights = []
            for i, signal in enumerate(signals):
                source = f"source_{i}" if i < len(signals) else "unknown"
                weight = avg_weights.get(source, 1.0)
                signal_weights.append(weight)
            
            # Normalize weights
            total_weight = sum(signal_weights)
            if total_weight > 0:
                signal_weights = [w / total_weight for w in signal_weights]
            
            return signal_weights
            
        except Exception as e:
            logger.error(f"Error calculating signal weights: {e}")
            return [1.0] * len(signals)
    
    def _calculate_volatility_factor(self, variance: float) -> float:
        """Calculate volatility factor for dampening."""
        try:
            # Higher variance = higher volatility factor
            volatility_factor = min(1.0, variance / self.config['volatility_dampening'])
            return volatility_factor
            
        except Exception as e:
            logger.error(f"Error calculating volatility factor: {e}")
            return 0.0
    
    def _apply_sigmoid_dampening(self, agreement: float, volatility_factor: float) -> float:
        """Apply sigmoid dampening for volatility tolerance."""
        try:
            # Sigmoid function: 1 / (1 + exp(-k * (x - 0.5)))
            k = self.config['sigmoid_steepness']
            
            # Adjust agreement based on volatility
            adjusted_agreement = agreement * (1.0 - volatility_factor * 0.5)
            
            # Apply sigmoid transformation
            sigmoid_input = k * (adjusted_agreement - 0.5)
            dampened_agreement = 1.0 / (1.0 + np.exp(-sigmoid_input))
            
            return dampened_agreement
            
        except Exception as e:
            logger.error(f"Error applying sigmoid dampening: {e}")
            return agreement
    
    def _determine_consensus_state(self, collapse_score: float, signal_count: int) -> ConsensusState:
        """Determine consensus state based on collapse score and signal count."""
        try:
            if signal_count < self.config['min_signals']:
                return ConsensusState.DISSENT
            
            if collapse_score >= self.consensus_threshold:
                if collapse_score >= 0.95:
                    return ConsensusState.UNANIMOUS
                else:
                    return ConsensusState.CONSENSUS
            elif collapse_score >= 0.7:
                return ConsensusState.CONVERGING
            elif collapse_score >= 0.4:
                return ConsensusState.PARTIAL
            else:
                return ConsensusState.DISSENT
                
        except Exception as e:
            logger.error(f"Error determining consensus state: {e}")
            return ConsensusState.DISSENT
    
    def _calculate_agreement_ratio(self, signals: List[float]) -> float:
        """Calculate agreement ratio among signals."""
        try:
            if len(signals) < 2:
                return 1.0
            
            signal_mean = np.mean(signals)
            signal_std = np.std(signals)
            
            if signal_std == 0:
                return 1.0
            
            # Calculate how many signals are within tolerance
            tolerance = self.config['agreement_tolerance']
            within_tolerance = sum(1 for s in signals if abs(s - signal_mean) <= tolerance * signal_std)
            
            agreement_ratio = within_tolerance / len(signals)
            return agreement_ratio
            
        except Exception as e:
            logger.error(f"Error calculating agreement ratio: {e}")
            return 0.0
    
    def _calculate_volatility_tolerance(self, signals: List[float]) -> float:
        """Calculate volatility tolerance score."""
        try:
            if len(signals) < 2:
                return 1.0
            
            # Calculate rolling volatility
            if len(signals) >= self.config['volatility_window']:
                recent_signals = signals[-self.config['volatility_window']:]
            else:
                recent_signals = signals
            
            volatility = np.std(recent_signals)
            mean_signal = np.mean(recent_signals)
            
            # Normalize volatility
            if mean_signal != 0:
                normalized_volatility = volatility / abs(mean_signal)
            else:
                normalized_volatility = volatility
            
            # Tolerance is inverse of normalized volatility
            tolerance = 1.0 / (1.0 + normalized_volatility)
            
            return min(1.0, tolerance)
            
        except Exception as e:
            logger.error(f"Error calculating volatility tolerance: {e}")
            return 1.0
    
    def _calculate_consensus_confidence(self, collapse_score: float, 
                                      signal_count: int, 
                                      agreement_ratio: float) -> float:
        """Calculate confidence in consensus result."""
        try:
            # Base confidence from collapse score
            base_confidence = collapse_score
            
            # Signal count factor (more signals = higher confidence)
            count_factor = min(1.0, signal_count / self.config['max_signals'])
            
            # Agreement factor
            agreement_factor = agreement_ratio
            
            # Combine factors
            confidence = (base_confidence + count_factor + agreement_factor) / 3.0
            
            # Apply confidence decay over time
            if self.consensus_history:
                time_since_last = time.time() - self.consensus_history[-1].timestamp
                decay = np.exp(-self.config['confidence_decay'] * time_since_last)
                confidence *= decay
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating consensus confidence: {e}")
            return 0.5
    
    def _update_signal_weights(self, signals: List[float], 
                             signal_sources: List[str], 
                             consensus_score: float):
        """Update signal weights based on consensus performance."""
        try:
            current_time = time.time()
            
            for i, (signal, source) in enumerate(zip(signals, signal_sources)):
                # Calculate weight based on consensus contribution
                signal_mean = np.mean(signals)
                signal_contribution = 1.0 - abs(signal - signal_mean) / (np.std(signals) + 1e-10)
                
                # Weight is combination of contribution and consensus score
                weight = (signal_contribution + consensus_score) / 2.0
                
                # Create weight record
                weight_record = SignalWeight(
                    signal_id=f"weight_{int(current_time * 1000)}_{i}",
                    weight=weight,
                    source=source,
                    timestamp=current_time,
                    confidence=consensus_score
                )
                
                self.signal_weights.append(weight_record)
            
            # Trim weight history
            if len(self.signal_weights) > self.max_weight_history:
                self.signal_weights = self.signal_weights[-self.max_weight_history:]
                
        except Exception as e:
            logger.error(f"Error updating signal weights: {e}")
    
    def _create_default_consensus_result(self) -> ConsensusResult:
        """Create default consensus result."""
        return ConsensusResult(
            consensus_id="default",
            timestamp=time.time(),
            consensus_state=ConsensusState.DISSENT,
            collapse_score=0.0,
            signal_count=0,
            agreement_ratio=0.0,
            volatility_tolerance=1.0,
            confidence=0.0
        )
    
    def get_consensus_summary(self) -> Dict[str, Any]:
        """Get consensus analysis summary."""
        try:
            if not self.consensus_history:
                return {
                    "total_collapses": 0,
                    "successful_consensus": 0,
                    "consensus_rate": 0.0,
                    "average_collapse_score": 0.0,
                    "most_common_state": "dissent",
                    "average_confidence": 0.0
                }
            
            # Calculate statistics
            collapse_scores = [r.collapse_score for r in self.consensus_history]
            confidences = [r.confidence for r in self.consensus_history]
            
            # Count consensus states
            state_counts = {}
            for result in self.consensus_history:
                state = result.consensus_state.value
                state_counts[state] = state_counts.get(state, 0) + 1
            
            most_common_state = max(state_counts.items(), key=lambda x: x[1])[0]
            
            # Calculate consensus rate
            consensus_rate = self.successful_consensus / self.total_collapses if self.total_collapses > 0 else 0.0
            
            return {
                "total_collapses": self.total_collapses,
                "successful_consensus": self.successful_consensus,
                "consensus_rate": consensus_rate,
                "average_collapse_score": np.mean(collapse_scores),
                "most_common_state": most_common_state,
                "average_confidence": np.mean(confidences),
                "recent_consensus_trend": self._calculate_recent_trend()
            }
            
        except Exception as e:
            logger.error(f"Error getting consensus summary: {e}")
            return {}
    
    def _calculate_recent_trend(self) -> str:
        """Calculate recent consensus trend."""
        try:
            recent_results = self.consensus_history[-10:]
            if len(recent_results) < 2:
                return "stable"
            
            recent_scores = [r.collapse_score for r in recent_results]
            trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            
            if trend_slope > 0.01:
                return "improving"
            elif trend_slope < -0.01:
                return "declining"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating recent trend: {e}")
            return "stable"
    
    def get_recent_consensus(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent consensus results."""
        recent_results = self.consensus_history[-count:]
        return [
            {
                "consensus_id": r.consensus_id,
                "timestamp": r.timestamp,
                "consensus_state": r.consensus_state.value,
                "collapse_score": r.collapse_score,
                "signal_count": r.signal_count,
                "agreement_ratio": r.agreement_ratio,
                "volatility_tolerance": r.volatility_tolerance,
                "confidence": r.confidence
            }
            for r in recent_results
        ]
    
    def detect_consensus_anomalies(self, threshold: float = 0.8) -> List[ConsensusResult]:
        """Detect consensus anomalies."""
        try:
            anomalies = []
            
            for result in self.consensus_history:
                if (result.collapse_score > threshold and 
                    result.consensus_state == ConsensusState.DISSENT):
                    anomalies.append(result)
                elif (result.collapse_score < (1.0 - threshold) and 
                      result.consensus_state == ConsensusState.UNANIMOUS):
                    anomalies.append(result)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting consensus anomalies: {e}")
            return []
    
    def export_consensus_data(self, filepath: str) -> bool:
        """
        Export consensus data to JSON file.
        
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
                "summary": self.get_consensus_summary(),
                "recent_consensus": self.get_recent_consensus(50),
                "anomalies": [
                    {
                        "consensus_id": r.consensus_id,
                        "timestamp": r.timestamp,
                        "consensus_state": r.consensus_state.value,
                        "collapse_score": r.collapse_score,
                        "confidence": r.confidence
                    }
                    for r in self.detect_consensus_anomalies()
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported consensus data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting consensus data: {e}")
            return False 