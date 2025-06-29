#!/usr/bin/env python3
"""
Ghost Field Stabilizer Module
=============================

Implements Shannon entropy delta slope estimator with NDArray-based 
volatility surface detection and bounding for Schwabot v0.05.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List
from numpy.typing import NDArray
from dataclasses import dataclass, field
from enum import Enum
import time
from scipy import signal
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class StabilityLevel(Enum):
    """Stability level enumeration."""
    CRITICAL = "critical"
    UNSTABLE = "unstable"
    MARGINAL = "marginal"
    STABLE = "stable"
    HIGHLY_STABLE = "highly_stable"


@dataclass
class StabilityReport:
    """Stability report data."""
    report_id: str
    timestamp: float
    stability_level: StabilityLevel
    entropy_score: float
    entropy_bounds: Tuple[float, float]
    volatility_surface: NDArray
    slope_estimate: float
    field_integrity: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FieldState:
    """Field state data."""
    state_id: str
    timestamp: float
    entropy_trace: NDArray
    volatility_matrix: NDArray
    stability_metrics: Dict[str, float]
    field_energy: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class GhostFieldStabilizer:
    """
    Ghost Field Stabilizer for Schwabot v0.05.
    
    Implements Shannon entropy delta slope estimator with NDArray-based 
    volatility surface detection and bounding.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the ghost field stabilizer."""
        self.config = config or self._default_config()
        
        # Field tracking
        self.field_history: List[FieldState] = []
        self.max_history_size = self.config.get('max_history_size', 1000)
        
        # Stability reports
        self.stability_reports: List[StabilityReport] = []
        self.max_report_history = self.config.get('max_report_history', 100)
        
        # Entropy analysis parameters
        self.entropy_window_size = self.config.get('entropy_window_size', 50)
        self.slope_calculation_window = self.config.get('slope_calculation_window', 20)
        self.volatility_surface_size = self.config.get('volatility_surface_size', 32)
        
        # Stability thresholds
        self.critical_threshold = self.config.get('critical_threshold', 0.9)
        self.unstable_threshold = self.config.get('unstable_threshold', 0.7)
        self.marginal_threshold = self.config.get('marginal_threshold', 0.5)
        self.stable_threshold = self.config.get('stable_threshold', 0.3)
        
        # Performance tracking
        self.total_evaluations = 0
        self.total_stability_events = 0
        
        logger.info("👻 Ghost Field Stabilizer initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'max_history_size': 1000,
            'max_report_history': 100,
            'entropy_window_size': 50,
            'slope_calculation_window': 20,
            'volatility_surface_size': 32,
            'critical_threshold': 0.9,
            'unstable_threshold': 0.7,
            'marginal_threshold': 0.5,
            'stable_threshold': 0.3,
            'min_signal_length': 10,
            'entropy_bin_count': 20,
            'volatility_kernel_size': 3,
            'field_energy_decay_rate': 0.1
        }
    
    def evaluate_stability(self, signal: NDArray) -> StabilityReport:
        """
        Evaluate field stability using Shannon entropy analysis.
        
        Args:
            signal: Input signal array
            
        Returns:
            Stability report
        """
        try:
            if len(signal) < self.config['min_signal_length']:
                logger.warning(f"Signal too short: {len(signal)} < {self.config['min_signal_length']}")
                return self._create_default_report()
            
            # Calculate entropy bounds
            entropy_bounds = self.compute_entropy_bounds(signal)
            
            # Calculate entropy score
            entropy_score = self._calculate_entropy_score(signal)
            
            # Generate volatility surface
            volatility_surface = self._generate_volatility_surface(signal)
            
            # Calculate slope estimate
            slope_estimate = self._calculate_entropy_slope(signal)
            
            # Calculate field integrity
            field_integrity = self._calculate_field_integrity(signal, volatility_surface)
            
            # Determine stability level
            stability_level = self._determine_stability_level(entropy_score, slope_estimate, field_integrity)
            
            # Calculate confidence
            confidence = self._calculate_stability_confidence(entropy_score, slope_estimate, field_integrity)
            
            # Create stability report
            report = StabilityReport(
                report_id=f"stability_{int(time.time() * 1000)}",
                timestamp=time.time(),
                stability_level=stability_level,
                entropy_score=entropy_score,
                entropy_bounds=entropy_bounds,
                volatility_surface=volatility_surface,
                slope_estimate=slope_estimate,
                field_integrity=field_integrity,
                confidence=confidence
            )
            
            # Store field state
            self._store_field_state(signal, volatility_surface, {
                'entropy_score': entropy_score,
                'slope_estimate': slope_estimate,
                'field_integrity': field_integrity,
                'stability_level': stability_level.value
            })
            
            # Add to report history
            self.stability_reports.append(report)
            if len(self.stability_reports) > self.max_report_history:
                self.stability_reports.pop(0)
            
            self.total_evaluations += 1
            if stability_level in [StabilityLevel.CRITICAL, StabilityLevel.UNSTABLE]:
                self.total_stability_events += 1
            
            logger.info(f"Stability evaluation: {stability_level.value} (entropy: {entropy_score:.3f}, integrity: {field_integrity:.3f})")
            
            return report
            
        except Exception as e:
            logger.error(f"Error evaluating stability: {e}")
            return self._create_default_report()
    
    def compute_entropy_bounds(self, trace: NDArray) -> Tuple[float, float]:
        """
        Compute entropy bounds using Shannon entropy analysis.
        
        Args:
            trace: Input trace array
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        try:
            if len(trace) < 2:
                return (0.0, 0.0)
            
            # Calculate rolling entropy over window
            window_size = min(self.entropy_window_size, len(trace) // 2)
            if window_size < 2:
                return (0.0, 0.0)
            
            rolling_entropies = []
            
            for i in range(window_size, len(trace)):
                window = trace[i-window_size:i]
                
                # Discretize window into bins
                bins = np.histogram(window, bins=self.config['entropy_bin_count'])[0]
                bins = bins[bins > 0]  # Remove zero bins
                
                if len(bins) > 1:
                    # Calculate Shannon entropy
                    probabilities = bins / np.sum(bins)
                    entropy_val = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                    rolling_entropies.append(entropy_val)
            
            if not rolling_entropies:
                return (0.0, 0.0)
            
            # Calculate bounds
            lower_bound = np.percentile(rolling_entropies, 5)  # 5th percentile
            upper_bound = np.percentile(rolling_entropies, 95)  # 95th percentile
            
            return (float(lower_bound), float(upper_bound))
            
        except Exception as e:
            logger.error(f"Error computing entropy bounds: {e}")
            return (0.0, 0.0)
    
    def _calculate_entropy_score(self, signal: NDArray) -> float:
        """Calculate normalized entropy score."""
        try:
            # Discretize signal into bins
            bins = np.histogram(signal, bins=self.config['entropy_bin_count'])[0]
            bins = bins[bins > 0]  # Remove zero bins
            
            if len(bins) < 2:
                return 0.0
            
            # Calculate Shannon entropy
            probabilities = bins / np.sum(bins)
            entropy_val = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            # Normalize to [0, 1] range
            max_entropy = np.log2(len(bins))
            normalized_entropy = entropy_val / max_entropy if max_entropy > 0 else 0.0
            
            return normalized_entropy
            
        except Exception as e:
            logger.error(f"Error calculating entropy score: {e}")
            return 0.0
    
    def _generate_volatility_surface(self, signal: NDArray) -> NDArray:
        """Generate volatility surface using NDArray operations."""
        try:
            surface_size = self.config['volatility_surface_size']
            kernel_size = self.config['volatility_kernel_size']
            
            # Calculate rolling volatility
            if len(signal) < kernel_size:
                return np.zeros((surface_size, surface_size))
            
            # Calculate rolling standard deviation
            rolling_vol = []
            for i in range(kernel_size, len(signal)):
                window = signal[i-kernel_size:i]
                vol = np.std(window)
                rolling_vol.append(vol)
            
            if len(rolling_vol) < surface_size * surface_size:
                # Pad with zeros if not enough data
                padding = surface_size * surface_size - len(rolling_vol)
                rolling_vol.extend([0.0] * padding)
            
            # Reshape to surface
            surface_data = rolling_vol[:surface_size * surface_size]
            surface = np.array(surface_data).reshape(surface_size, surface_size)
            
            # Apply smoothing kernel
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
            smoothed_surface = signal.convolve2d(surface, kernel, mode='same', boundary='wrap')
            
            return smoothed_surface
            
        except Exception as e:
            logger.error(f"Error generating volatility surface: {e}")
            return np.zeros((self.config['volatility_surface_size'], self.config['volatility_surface_size']))
    
    def _calculate_entropy_slope(self, signal: NDArray) -> float:
        """Calculate entropy slope estimate."""
        try:
            window_size = self.slope_calculation_window
            if len(signal) < window_size * 2:
                return 0.0
            
            # Calculate entropy over sliding windows
            entropies = []
            for i in range(0, len(signal) - window_size, window_size // 2):
                window = signal[i:i+window_size]
                
                # Calculate entropy for window
                bins = np.histogram(window, bins=self.config['entropy_bin_count'])[0]
                bins = bins[bins > 0]
                
                if len(bins) > 1:
                    probabilities = bins / np.sum(bins)
                    entropy_val = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                    entropies.append(entropy_val)
            
            if len(entropies) < 2:
                return 0.0
            
            # Calculate slope using linear regression
            x = np.arange(len(entropies))
            slope, _ = np.polyfit(x, entropies, 1)
            
            return float(slope)
            
        except Exception as e:
            logger.error(f"Error calculating entropy slope: {e}")
            return 0.0
    
    def _calculate_field_integrity(self, signal: NDArray, volatility_surface: NDArray) -> float:
        """Calculate field integrity score."""
        try:
            # Calculate signal coherence
            signal_coherence = self._calculate_signal_coherence(signal)
            
            # Calculate surface stability
            surface_stability = self._calculate_surface_stability(volatility_surface)
            
            # Calculate energy conservation
            energy_conservation = self._calculate_energy_conservation(signal)
            
            # Combine metrics
            integrity = (signal_coherence + surface_stability + energy_conservation) / 3.0
            
            return min(1.0, max(0.0, integrity))
            
        except Exception as e:
            logger.error(f"Error calculating field integrity: {e}")
            return 0.0
    
    def _calculate_signal_coherence(self, signal: NDArray) -> float:
        """Calculate signal coherence score."""
        try:
            if len(signal) < 10:
                return 1.0
            
            # Calculate autocorrelation
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Normalize autocorrelation
            autocorr = autocorr / autocorr[0]
            
            # Calculate coherence as decay rate
            decay_rate = np.mean(np.abs(np.diff(autocorr[:10])))
            coherence = 1.0 / (1.0 + decay_rate)
            
            return min(1.0, coherence)
            
        except Exception as e:
            logger.error(f"Error calculating signal coherence: {e}")
            return 1.0
    
    def _calculate_surface_stability(self, volatility_surface: NDArray) -> float:
        """Calculate volatility surface stability."""
        try:
            # Calculate surface variance
            surface_variance = np.var(volatility_surface)
            
            # Calculate surface gradient magnitude
            gradient_x = np.gradient(volatility_surface, axis=1)
            gradient_y = np.gradient(volatility_surface, axis=0)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            avg_gradient = np.mean(gradient_magnitude)
            
            # Stability is inverse of variance and gradient
            stability = 1.0 / (1.0 + surface_variance + avg_gradient)
            
            return min(1.0, stability)
            
        except Exception as e:
            logger.error(f"Error calculating surface stability: {e}")
            return 1.0
    
    def _calculate_energy_conservation(self, signal: NDArray) -> float:
        """Calculate energy conservation score."""
        try:
            if len(signal) < 2:
                return 1.0
            
            # Calculate signal energy
            energy = np.sum(signal**2)
            
            # Calculate energy variation
            energy_variation = np.std(signal**2)
            
            # Conservation is inverse of variation
            conservation = 1.0 / (1.0 + energy_variation / (energy + 1e-10))
            
            return min(1.0, conservation)
            
        except Exception as e:
            logger.error(f"Error calculating energy conservation: {e}")
            return 1.0
    
    def _determine_stability_level(self, entropy_score: float, 
                                 slope_estimate: float, 
                                 field_integrity: float) -> StabilityLevel:
        """Determine stability level based on metrics."""
        try:
            # Calculate composite stability score
            stability_score = (entropy_score + abs(slope_estimate) + (1.0 - field_integrity)) / 3.0
            
            # Determine level based on thresholds
            if stability_score >= self.critical_threshold:
                return StabilityLevel.CRITICAL
            elif stability_score >= self.unstable_threshold:
                return StabilityLevel.UNSTABLE
            elif stability_score >= self.marginal_threshold:
                return StabilityLevel.MARGINAL
            elif stability_score >= self.stable_threshold:
                return StabilityLevel.STABLE
            else:
                return StabilityLevel.HIGHLY_STABLE
                
        except Exception as e:
            logger.error(f"Error determining stability level: {e}")
            return StabilityLevel.STABLE
    
    def _calculate_stability_confidence(self, entropy_score: float,
                                      slope_estimate: float,
                                      field_integrity: float) -> float:
        """Calculate confidence in stability assessment."""
        try:
            # Confidence based on consistency of metrics
            metric_consistency = 1.0 - np.std([entropy_score, abs(slope_estimate), 1.0 - field_integrity])
            
            # Base confidence on field integrity
            base_confidence = field_integrity
            
            # Combine factors
            confidence = (base_confidence + metric_consistency) / 2.0
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating stability confidence: {e}")
            return 0.5
    
    def _store_field_state(self, signal: NDArray, volatility_surface: NDArray, 
                          metrics: Dict[str, float]):
        """Store field state for historical analysis."""
        try:
            # Calculate field energy
            field_energy = np.sum(signal**2) / len(signal)
            
            field_state = FieldState(
                state_id=f"field_{int(time.time() * 1000)}",
                timestamp=time.time(),
                entropy_trace=signal.copy(),
                volatility_matrix=volatility_surface.copy(),
                stability_metrics=metrics,
                field_energy=field_energy
            )
            
            self.field_history.append(field_state)
            if len(self.field_history) > self.max_history_size:
                self.field_history.pop(0)
                
        except Exception as e:
            logger.error(f"Error storing field state: {e}")
    
    def _create_default_report(self) -> StabilityReport:
        """Create default stability report."""
        return StabilityReport(
            report_id="default",
            timestamp=time.time(),
            stability_level=StabilityLevel.STABLE,
            entropy_score=0.0,
            entropy_bounds=(0.0, 0.0),
            volatility_surface=np.zeros((self.config['volatility_surface_size'], 
                                       self.config['volatility_surface_size'])),
            slope_estimate=0.0,
            field_integrity=1.0,
            confidence=1.0
        )
    
    def get_stability_summary(self) -> Dict[str, Any]:
        """Get stability analysis summary."""
        try:
            if not self.stability_reports:
                return {
                    "total_evaluations": 0,
                    "total_stability_events": 0,
                    "average_entropy_score": 0.0,
                    "average_field_integrity": 1.0,
                    "most_common_stability_level": "stable",
                    "recent_stability_trend": "stable"
                }
            
            # Calculate statistics
            entropy_scores = [r.entropy_score for r in self.stability_reports]
            field_integrities = [r.field_integrity for r in self.stability_reports]
            
            # Count stability levels
            level_counts = {}
            for report in self.stability_reports:
                level = report.stability_level.value
                level_counts[level] = level_counts.get(level, 0) + 1
            
            most_common_level = max(level_counts.items(), key=lambda x: x[1])[0]
            
            # Calculate recent trend
            recent_reports = self.stability_reports[-10:]
            if len(recent_reports) >= 2:
                recent_trend = "improving" if recent_reports[-1].field_integrity > recent_reports[0].field_integrity else "declining"
            else:
                recent_trend = "stable"
            
            return {
                "total_evaluations": self.total_evaluations,
                "total_stability_events": self.total_stability_events,
                "average_entropy_score": np.mean(entropy_scores),
                "average_field_integrity": np.mean(field_integrities),
                "most_common_stability_level": most_common_level,
                "recent_stability_trend": recent_trend,
                "current_stability_level": self.stability_reports[-1].stability_level.value if self.stability_reports else "stable"
            }
            
        except Exception as e:
            logger.error(f"Error getting stability summary: {e}")
            return {}
    
    def get_recent_reports(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent stability reports."""
        recent_reports = self.stability_reports[-count:]
        return [
            {
                "report_id": r.report_id,
                "timestamp": r.timestamp,
                "stability_level": r.stability_level.value,
                "entropy_score": r.entropy_score,
                "entropy_bounds": r.entropy_bounds,
                "slope_estimate": r.slope_estimate,
                "field_integrity": r.field_integrity,
                "confidence": r.confidence
            }
            for r in recent_reports
        ]
    
    def detect_stability_anomalies(self, threshold: float = 0.8) -> List[StabilityReport]:
        """Detect stability anomalies above threshold."""
        try:
            anomalies = []
            
            for report in self.stability_reports:
                if (report.entropy_score > threshold or 
                    report.field_integrity < (1.0 - threshold) or
                    abs(report.slope_estimate) > threshold):
                    anomalies.append(report)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting stability anomalies: {e}")
            return []
    
    def export_stability_data(self, filepath: str) -> bool:
        """
        Export stability data to JSON file.
        
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
                "summary": self.get_stability_summary(),
                "recent_reports": self.get_recent_reports(50),
                "anomalies": [
                    {
                        "report_id": r.report_id,
                        "timestamp": r.timestamp,
                        "stability_level": r.stability_level.value,
                        "entropy_score": r.entropy_score,
                        "field_integrity": r.field_integrity
                    }
                    for r in self.detect_stability_anomalies()
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported stability data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting stability data: {e}")
            return False 