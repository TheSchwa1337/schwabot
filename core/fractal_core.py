#!/usr/bin/env python3
"""
Fractal Core Module
==================

Fractal matching and error correction for Schwabot v0.05.
Provides fractal pattern recognition and correction mechanisms.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft

from core.linguistic_glyph_engine import echo_fractal

logger = logging.getLogger(__name__)


class FractalType(Enum):
    """Fractal type enumeration."""
    PRICE = "price"
    VOLUME = "volume"
    PATTERN = "pattern"
    SIGNAL = "signal"
    ERROR = "error"


class FractalState(Enum):
    """Fractal state enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    CORRECTING = "correcting"
    STABLE = "stable"
    UNSTABLE = "unstable"


@dataclass
class FractalData:
    """Fractal data structure."""
    fractal_id: str
    fractal_type: FractalType
    state: FractalState
    data: np.ndarray
    dimensions: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FractalPattern:
    """Fractal pattern structure."""
    pattern_id: str
    pattern_type: str
    fractal_data: FractalData
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorCorrection:
    """Error correction result."""
    correction_id: str
    fractal_id: str
    error_type: str
    correction_method: str
    original_data: np.ndarray
    corrected_data: np.ndarray
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class FractalCore:
    """
    Fractal Core for Schwabot v0.05.
    
    Provides fractal matching and error correction
    for pattern recognition and data correction.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the fractal core."""
        self.config = config or self._default_config()
        
        # Fractal management
        self.fractals: Dict[str, FractalData] = {}
        self.fractal_history: List[FractalData] = []
        self.max_fractal_history = self.config.get('max_fractal_history', 100)
        self.echo_fractal_cache: Dict[str, np.ndarray] = {}
        
        # Pattern recognition
        self.patterns: List[FractalPattern] = []
        self.max_patterns = self.config.get('max_patterns', 100)
        
        # Error correction
        self.corrections: List[ErrorCorrection] = []
        self.max_corrections = self.config.get('max_corrections', 100)
        
        # Performance tracking
        self.total_fractals = 0
        self.total_patterns = 0
        self.total_corrections = 0
        self.error_detections = 0
        
        # State management
        self.last_update = time.time()
        
        # Initialize default fractals
        self._initialize_default_fractals()
        
        logger.info("Fractal Core initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'max_fractal_history': 100,
            'max_patterns': 100,
            'max_corrections': 100,
            'fractal_dimensions': 100,
            'pattern_confidence_threshold': 0.7,
            'error_threshold': 0.1,
            'correction_methods': ['smoothing', 'interpolation', 'outlier_removal'],
            'fractal_types': ['price', 'volume', 'pattern'],
            'auto_correction_enabled': True,
            'pattern_detection_enabled': True
        }
    
    def _initialize_default_fractals(self):
        """Initialize default fractals."""
        # Price fractal
        price_data = np.random.rand(100)
        self.add_fractal("price_fractal", FractalType.PRICE, FractalState.ACTIVE, price_data)
        
        # Volume fractal
        volume_data = np.random.rand(100)
        self.add_fractal("volume_fractal", FractalType.VOLUME, FractalState.ACTIVE, volume_data)
        
        # Pattern fractal
        pattern_data = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
        self.add_fractal("pattern_fractal", FractalType.PATTERN, FractalState.ACTIVE, pattern_data)
    
    def add_fractal(self, fractal_id: str, fractal_type: FractalType,
                   state: FractalState, data: np.ndarray) -> FractalData:
        """
        Add a new fractal.
        
        Args:
            fractal_id: Unique fractal identifier
            fractal_type: Type of fractal
            state: Current state
            data: Fractal data as numpy array
            
        Returns:
            Created fractal data
        """
        try:
            # Validate fractal data
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            if data.ndim != 1:
                raise ValueError("Fractal data must be 1-dimensional")
            
            fractal = FractalData(
                fractal_id=fractal_id,
                fractal_type=fractal_type,
                state=state,
                data=data,
                dimensions=len(data),
                timestamp=time.time()
            )
            
            self.fractals[fractal_id] = fractal
            self.total_fractals += 1
            
            # Add to history
            self.fractal_history.append(fractal)
            if len(self.fractal_history) > self.max_fractal_history:
                self.fractal_history.pop(0)
            
            logger.info(f"Added fractal: {fractal_id} ({fractal_type.value}, {len(data)} points)")
            return fractal
            
        except Exception as e:
            logger.error(f"Error adding fractal {fractal_id}: {e}")
            return self._create_default_fractal()
    
    def _create_default_fractal(self) -> FractalData:
        """Create default fractal."""
        return FractalData(
            fractal_id="default",
            fractal_type=FractalType.PATTERN,
            state=FractalState.INACTIVE,
            data=np.zeros(50),
            dimensions=50,
            timestamp=time.time()
        )
    
    def update_fractal(self, fractal_id: str, new_data: np.ndarray) -> bool:
        """
        Update fractal data.
        
        Args:
            fractal_id: Fractal identifier
            new_data: New fractal data
            
        Returns:
            True if update was successful
        """
        try:
            if fractal_id not in self.fractals:
                logger.error(f"Fractal {fractal_id} not found")
                return False
            
            fractal = self.fractals[fractal_id]
            
            # Validate new data
            if not isinstance(new_data, np.ndarray):
                new_data = np.array(new_data)
            
            if new_data.ndim != 1:
                logger.error("New fractal data must be 1-dimensional")
                return False
            
            # Check for errors in data
            if self._detect_errors(new_data):
                logger.warning(f"Errors detected in fractal {fractal_id}, applying correction")
                corrected_data = self._apply_error_correction(fractal_id, new_data)
                if corrected_data is not None:
                    new_data = corrected_data
            
            # Update fractal
            fractal.data = new_data
            fractal.dimensions = len(new_data)
            fractal.timestamp = time.time()
            
            # Update state based on data characteristics
            fractal.state = self._determine_fractal_state(new_data)
            
            self.last_update = time.time()
            logger.debug(f"Updated fractal: {fractal_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating fractal {fractal_id}: {e}")
            return False
    
    def _detect_errors(self, data: np.ndarray) -> bool:
        """Detect errors in fractal data."""
        try:
            # Check for NaN values
            if np.any(np.isnan(data)):
                return True
            
            # Check for infinite values
            if np.any(np.isinf(data)):
                return True
            
            # Check for outliers (values beyond 3 standard deviations)
            mean_val = np.mean(data)
            std_val = np.std(data)
            outliers = np.abs(data - mean_val) > 3 * std_val
            
            if np.any(outliers):
                return True
            
            # Check for sudden jumps (large differences between consecutive points)
            if len(data) > 1:
                differences = np.abs(np.diff(data))
                large_jumps = differences > 2 * np.std(differences)
                if np.any(large_jumps):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting errors: {e}")
            return True
    
    def _apply_error_correction(self, fractal_id: str, data: np.ndarray) -> Optional[np.ndarray]:
        """Apply error correction to fractal data."""
        try:
            correction_methods = self.config.get('correction_methods', ['smoothing'])
            
            corrected_data = data.copy()
            
            for method in correction_methods:
                if method == 'smoothing':
                    corrected_data = self._apply_smoothing(corrected_data)
                elif method == 'interpolation':
                    corrected_data = self._apply_interpolation(corrected_data)
                elif method == 'outlier_removal':
                    corrected_data = self._apply_outlier_removal(corrected_data)
            
            # Create correction record
            correction = ErrorCorrection(
                correction_id=f"correction_{int(time.time() * 1000)}",
                fractal_id=fractal_id,
                error_type="data_errors",
                correction_method=",".join(correction_methods),
                original_data=data,
                corrected_data=corrected_data,
                confidence=0.8,
                timestamp=time.time()
            )
            
            self.corrections.append(correction)
            if len(self.corrections) > self.max_corrections:
                self.corrections.pop(0)
            
            self.total_corrections += 1
            self.error_detections += 1
            
            logger.info(f"Applied error correction to fractal {fractal_id}")
            return corrected_data
            
        except Exception as e:
            logger.error(f"Error applying correction: {e}")
            return None
    
    def _apply_smoothing(self, data: np.ndarray) -> np.ndarray:
        """Apply smoothing to data."""
        try:
            # Use moving average smoothing
            window_size = min(5, len(data) // 10)
            if window_size < 3:
                window_size = 3
            
            smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='same')
            return smoothed
            
        except Exception as e:
            logger.error(f"Error applying smoothing: {e}")
            return data
    
    def _apply_interpolation(self, data: np.ndarray) -> np.ndarray:
        """Apply interpolation to data."""
        try:
            # Find and interpolate NaN values
            if np.any(np.isnan(data)):
                valid_indices = ~np.isnan(data)
                if np.sum(valid_indices) > 1:
                    valid_data = data[valid_indices]
                    valid_positions = np.where(valid_indices)[0]
                    
                    # Linear interpolation
                    all_positions = np.arange(len(data))
                    interpolated = np.interp(all_positions, valid_positions, valid_data)
                    return interpolated
            
            return data
            
        except Exception as e:
            logger.error(f"Error applying interpolation: {e}")
            return data
    
    def _apply_outlier_removal(self, data: np.ndarray) -> np.ndarray:
        """Apply outlier removal to data."""
        try:
            # Remove outliers using IQR method
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Replace outliers with median
            median_val = np.median(data)
            corrected_data = data.copy()
            corrected_data[data < lower_bound] = median_val
            corrected_data[data > upper_bound] = median_val
            
            return corrected_data
            
        except Exception as e:
            logger.error(f"Error applying outlier removal: {e}")
            return data
    
    def _determine_fractal_state(self, data: np.ndarray) -> FractalState:
        """Determine fractal state based on data characteristics."""
        try:
            # Calculate stability metrics
            variance = np.var(data)
            mean_val = np.mean(data)
            
            # Check for stability
            if variance < 0.01:  # Low variance = stable
                return FractalState.STABLE
            elif variance > 1.0:  # High variance = unstable
                return FractalState.UNSTABLE
            else:
                return FractalState.ACTIVE
                
        except Exception as e:
            logger.error(f"Error determining fractal state: {e}")
            return FractalState.INACTIVE
    
    def detect_patterns(self) -> List[FractalPattern]:
        """
        Detect fractal patterns.
        
        Returns:
            List of detected patterns
        """
        try:
            patterns = []
            
            for fractal_id, fractal in self.fractals.items():
                if fractal.state == FractalState.INACTIVE:
                    continue
                
                # Detect different types of patterns
                pattern_types = self._detect_pattern_types(fractal)
                
                for pattern_type, confidence in pattern_types:
                    if confidence >= self.config.get('pattern_confidence_threshold', 0.7):
                        pattern = FractalPattern(
                            pattern_id=f"pattern_{int(time.time() * 1000)}",
                            pattern_type=pattern_type,
                            fractal_data=fractal,
                            confidence=confidence,
                            timestamp=time.time()
                        )
                        
                        patterns.append(pattern)
            
            # Update pattern history
            for pattern in patterns:
                self.patterns.append(pattern)
                if len(self.patterns) > self.max_patterns:
                    self.patterns.pop(0)
            
            self.total_patterns += len(patterns)
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []
    
    def _detect_pattern_types(self, fractal: FractalData) -> List[Tuple[str, float]]:
        """Detect different types of patterns in fractal data."""
        try:
            patterns = []
            data = fractal.data
            
            # Trend pattern
            trend_confidence = self._detect_trend_pattern(data)
            if trend_confidence > 0.5:
                patterns.append(("trend", trend_confidence))
            
            # Cyclic pattern
            cyclic_confidence = self._detect_cyclic_pattern(data)
            if cyclic_confidence > 0.5:
                patterns.append(("cyclic", cyclic_confidence))
            
            # Oscillatory pattern
            oscillatory_confidence = self._detect_oscillatory_pattern(data)
            if oscillatory_confidence > 0.5:
                patterns.append(("oscillatory", oscillatory_confidence))
            
            # Random pattern
            random_confidence = self._detect_random_pattern(data)
            if random_confidence > 0.5:
                patterns.append(("random", random_confidence))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting pattern types: {e}")
            return []
    
    def _detect_trend_pattern(self, data: np.ndarray) -> float:
        """Detect trend pattern in data."""
        try:
            if len(data) < 3:
                return 0.0
            
            # Linear regression
            x = np.arange(len(data))
            slope, intercept = np.polyfit(x, data, 1)
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_res = np.sum((data - y_pred) ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Confidence based on R-squared and slope magnitude
            slope_magnitude = abs(slope)
            confidence = r_squared * min(slope_magnitude, 1.0)
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error detecting trend pattern: {e}")
            return 0.0
    
    def _detect_cyclic_pattern(self, data: np.ndarray) -> float:
        """Detect cyclic pattern in data."""
        try:
            if len(data) < 10:
                return 0.0
            
            # FFT analysis
            fft_data = fft(data)
            power_spectrum = np.abs(fft_data) ** 2
            
            # Find dominant frequency
            freqs = np.fft.fftfreq(len(data))
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            dominant_freq = freqs[dominant_freq_idx]
            
            # Calculate confidence based on power concentration
            total_power = np.sum(power_spectrum)
            dominant_power = power_spectrum[dominant_freq_idx]
            power_ratio = dominant_power / total_power if total_power > 0 else 0
            
            # Higher confidence for clear periodic patterns
            confidence = power_ratio * 2  # Scale up for better sensitivity
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error detecting cyclic pattern: {e}")
            return 0.0
    
    def _detect_oscillatory_pattern(self, data: np.ndarray) -> float:
        """Detect oscillatory pattern in data."""
        try:
            if len(data) < 5:
                return 0.0
            
            # Count zero crossings
            zero_crossings = np.sum(np.diff(np.sign(data - np.mean(data))) != 0)
            
            # Calculate oscillation frequency
            oscillation_freq = zero_crossings / (2 * len(data))
            
            # Confidence based on oscillation frequency
            confidence = min(oscillation_freq * 5, 1.0)  # Scale for reasonable range
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error detecting oscillatory pattern: {e}")
            return 0.0
    
    def _detect_random_pattern(self, data: np.ndarray) -> float:
        """Detect random pattern in data."""
        try:
            if len(data) < 10:
                return 0.0
            
            # Calculate autocorrelation
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Normalize
            autocorr = autocorr / autocorr[0]
            
            # Check for rapid decay (characteristic of random data)
            decay_rate = np.mean(np.abs(np.diff(autocorr[:10])))
            
            # Higher decay rate indicates more random data
            confidence = min(decay_rate * 10, 1.0)
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error detecting random pattern: {e}")
            return 0.0
    
    def calculate_fractal_compression_state(self, N: int = 10, tau: int = 1) -> float:
        """
        Calculates the Fractal Memory Compression State (Λ(t)).

        Λ(t) = 1 / (1 + Σ (ℰ(t − kτ))²)
        Where:
        - ℰ(t − kτ) is the echo fractal state from k previous cycles.
        - N is the number of previous cycles to consider.
        - τ is the time constant/lag between cycles.

        Returns:
            The calculated fractal memory compression state Λ(t).
        """
        if not self.fractal_history:
            return 1.0  # No history, assume perfect compression (no errors/drift)

        sum_of_squared_echoes = 0.0
        
        # Iterate over the N most recent fractal history items, with a lag of tau
        for k in range(N):
            # Calculate index for (t - kτ)
            history_index = -(k * tau + 1) # Start from the most recent, going backwards

            if abs(history_index) <= len(self.fractal_history):
                # Get the fractal data from k previous cycles
                # Assuming 'data' in FractalData is a numpy array for echo_fractal
                # For simplicity, we'll use a portion of the data
                relevant_fractal_data = self.fractal_history[history_index].data
                
                # Apply echo_fractal to a sample of data
                # Using a fixed size for echo_fractal input for consistency
                if relevant_fractal_data.size > 0:
                    x_input = np.linspace(0, 10, min(128, relevant_fractal_data.size)) # Use up to 128 points
                    echo_val = echo_fractal(x_input)
                    sum_of_squared_echoes += np.sum(echo_val**2)
                else:
                    sum_of_squared_echoes += 0.0 # No data, no echo contribution
            else:
                break # Ran out of history

        lambda_t = 1.0 / (1.0 + sum_of_squared_echoes)
        
        # Ensure lambda_t is within a reasonable range (0 to 1)
        lambda_t = np.clip(lambda_t, 0.0, 1.0)
        
        logger.debug(f"Calculated Fractal Compression State (Λ): {lambda_t:.4f}")
        return float(lambda_t)

    def get_fractal_summary(self) -> Dict[str, Any]:
        """
        Get a summary of fractal core status, including compression state.
        """
        summary = {
            'total_fractals': self.total_fractals,
            'active_fractals': len([f for f in self.fractals.values() if f.state == FractalState.ACTIVE]),
            'fractal_history_size': len(self.fractal_history),
            'total_patterns_detected': self.total_patterns,
            'total_errors_corrected': self.total_corrections,
            'last_update': datetime.fromtimestamp(self.last_update).isoformat(),
            'current_fractal_compression_state_lambda': self.calculate_fractal_compression_state(), # Add to summary
        }
        return summary
    
    def get_fractal_status(self) -> List[Dict[str, Any]]:
        """Get current status of all fractals."""
        return [
            {
                "fractal_id": fractal.fractal_id,
                "fractal_type": fractal.fractal_type.value,
                "state": fractal.state.value,
                "dimensions": fractal.dimensions,
                "timestamp": fractal.timestamp
            }
            for fractal in self.fractals.values()
        ]
    
    def get_recent_patterns(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent detected patterns."""
        recent_patterns = self.patterns[-count:]
        return [
            {
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type,
                "fractal_id": pattern.fractal_data.fractal_id,
                "confidence": pattern.confidence,
                "timestamp": pattern.timestamp
            }
            for pattern in recent_patterns
        ]
    
    def get_recent_corrections(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent error corrections."""
        recent_corrections = self.corrections[-count:]
        return [
            {
                "correction_id": correction.correction_id,
                "fractal_id": correction.fractal_id,
                "error_type": correction.error_type,
                "correction_method": correction.correction_method,
                "confidence": correction.confidence,
                "timestamp": correction.timestamp
            }
            for correction in recent_corrections
        ]
    
    def export_fractal_data(self, filepath: str) -> bool:
        """
        Export fractal data to file.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if export was successful
        """
        try:
            import json
            
            data = {
                "export_timestamp": time.time(),
                "fractal_summary": self.get_fractal_summary(),
                "fractal_status": self.get_fractal_status(),
                "recent_patterns": self.get_recent_patterns(50),
                "recent_corrections": self.get_recent_corrections(50)
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported fractal data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting fractal data: {e}")
            return False 