#!/usr/bin/env python3
"""
Analysis Engine - Advanced Mathematical Analysis and Signal Processing
====================================================================

This module implements a comprehensive analysis engine for Schwabot,
performing advanced mathematical analysis, signal processing, and pattern recognition.

Core Mathematical Functions:
- Signal Analysis: S(t) = Σ(aₙ * cos(ωₙt + φₙ)) + noise
- Pattern Recognition: P(x) = argmax(P(C|x)) where C are pattern classes
- Technical Indicators: RSI = 100 - (100 / (1 + RS)) where RS = avg_gain / avg_loss
- Fourier Analysis: X(ω) = ∫x(t)e^(-jωt)dt
- Wavelet Transform: W(a,b) = (1/√|a|)∫x(t)ψ*((t-b)/a)dt

Core Functionality:
- Real-time market data analysis
- Technical indicator calculation
- Pattern recognition and classification
- Signal processing and filtering
- Statistical analysis and modeling
- Altitude logic and advanced strategies
- Performance tracking and optimization
"""

import logging
import json
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import queue
import weakref
import traceback
import math
import statistics
from scipy import signal, stats
from scipy.fft import fft, fftfreq, ifft
import pandas as pd

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    TECHNICAL = "technical"
    STATISTICAL = "statistical"
    PATTERN = "pattern"
    SIGNAL = "signal"
    FOURIER = "fourier"
    WAVELET = "wavelet"
    ALTITUDE = "altitude"
    MACHINE_LEARNING = "machine_learning"

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    WEAK_BUY = "weak_buy"
    WEAK_SELL = "weak_sell"

class PatternType(Enum):
    TREND = "trend"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"

@dataclass
class MarketData:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    interval: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TechnicalIndicator:
    name: str
    value: float
    timestamp: datetime
    signal_type: SignalType
    confidence: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Pattern:
    pattern_type: PatternType
    confidence: float
    start_time: datetime
    end_time: datetime
    strength: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisResult:
    analysis_id: str
    timestamp: datetime
    symbol: str
    indicators: List[TechnicalIndicator]
    patterns: List[Pattern]
    signals: List[SignalType]
    confidence_score: float
    risk_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class SignalProcessor:
    """Advanced signal processing and filtering."""
    
    def __init__(self, sample_rate: float = 1.0, filter_type: str = "butterworth"):
        self.sample_rate = sample_rate
        self.filter_type = filter_type
        self.filter_order = 4
        self.cutoff_freq = 0.1
    
    def apply_low_pass_filter(self, data: np.ndarray, cutoff: float = None) -> np.ndarray:
        """Apply low-pass filter to remove high-frequency noise."""
        try:
            if cutoff is None:
                cutoff = self.cutoff_freq
            
            # Normalize cutoff frequency
            nyquist = self.sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            
            # Design filter
            b, a = signal.butter(self.filter_order, normalized_cutoff, btype='low')
            
            # Apply filter
            filtered_data = signal.filtfilt(b, a, data)
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error applying low-pass filter: {e}")
            return data
    
    def apply_high_pass_filter(self, data: np.ndarray, cutoff: float = None) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency trends."""
        try:
            if cutoff is None:
                cutoff = self.cutoff_freq
            
            # Normalize cutoff frequency
            nyquist = self.sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            
            # Design filter
            b, a = signal.butter(self.filter_order, normalized_cutoff, btype='high')
            
            # Apply filter
            filtered_data = signal.filtfilt(b, a, data)
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error applying high-pass filter: {e}")
            return data
    
    def apply_band_pass_filter(self, data: np.ndarray, low_cutoff: float, high_cutoff: float) -> np.ndarray:
        """Apply band-pass filter."""
        try:
            # Normalize cutoff frequencies
            nyquist = self.sample_rate / 2
            low_norm = low_cutoff / nyquist
            high_norm = high_cutoff / nyquist
            
            # Design filter
            b, a = signal.butter(self.filter_order, [low_norm, high_norm], btype='band')
            
            # Apply filter
            filtered_data = signal.filtfilt(b, a, data)
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error applying band-pass filter: {e}")
            return data
    
    def compute_fft(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Fast Fourier Transform."""
        try:
            # Apply window function to reduce spectral leakage
            windowed_data = data * signal.hann(len(data))
            
            # Compute FFT
            fft_result = fft(windowed_data)
            frequencies = fftfreq(len(data), 1/self.sample_rate)
            
            # Return positive frequencies only
            positive_freq_mask = frequencies >= 0
            return frequencies[positive_freq_mask], np.abs(fft_result[positive_freq_mask])
            
        except Exception as e:
            logger.error(f"Error computing FFT: {e}")
            return np.array([]), np.array([])
    
    def compute_power_spectral_density(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density."""
        try:
            frequencies, psd = signal.welch(data, fs=self.sample_rate, nperseg=min(256, len(data)//4))
            return frequencies, psd
            
        except Exception as e:
            logger.error(f"Error computing PSD: {e}")
            return np.array([]), np.array([])

class TechnicalIndicators:
    """Technical indicator calculations."""
    
    def __init__(self):
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bollinger_period = 20
        self.bollinger_std = 2
    
    def calculate_rsi(self, prices: np.ndarray, period: int = None) -> float:
        """Calculate Relative Strength Index."""
        try:
            if period is None:
                period = self.rsi_period
            
            if len(prices) < period + 1:
                return 50.0  # Neutral value
            
            # Calculate price changes
            deltas = np.diff(prices)
            
            # Separate gains and losses
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Calculate average gains and losses
            avg_gains = np.mean(gains[-period:])
            avg_losses = np.mean(losses[-period:])
            
            if avg_losses == 0:
                return 100.0
            
            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def calculate_macd(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            if len(prices) < self.macd_slow:
                return 0.0, 0.0, 0.0
            
            # Calculate EMAs
            ema_fast = self._calculate_ema(prices, self.macd_fast)
            ema_slow = self._calculate_ema(prices, self.macd_slow)
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line (EMA of MACD)
            macd_values = []
            for i in range(len(prices)):
                if i >= self.macd_slow - 1:
                    fast_ema = self._calculate_ema(prices[:i+1], self.macd_fast)
                    slow_ema = self._calculate_ema(prices[:i+1], self.macd_slow)
                    macd_values.append(fast_ema - slow_ema)
                else:
                    macd_values.append(0.0)
            
            signal_line = self._calculate_ema(np.array(macd_values), self.macd_signal)
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return 0.0, 0.0, 0.0
    
    def calculate_bollinger_bands(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        try:
            if len(prices) < self.bollinger_period:
                return prices[-1], prices[-1], prices[-1]
            
            # Calculate moving average
            ma = np.mean(prices[-self.bollinger_period:])
            
            # Calculate standard deviation
            std = np.std(prices[-self.bollinger_period:])
            
            # Calculate bands
            upper_band = ma + (self.bollinger_std * std)
            lower_band = ma - (self.bollinger_std * std)
            
            return upper_band, ma, lower_band
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return prices[-1], prices[-1], prices[-1]
    
    def calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator."""
        try:
            if len(close) < period:
                return 50.0, 50.0
            
            # Calculate %K
            highest_high = np.max(high[-period:])
            lowest_low = np.min(low[-period:])
            
            if highest_high == lowest_low:
                k_percent = 50.0
            else:
                k_percent = ((close[-1] - lowest_low) / (highest_high - lowest_low)) * 100
            
            # Calculate %D (3-period SMA of %K)
            k_values = []
            for i in range(period, len(close)):
                hh = np.max(high[i-period:i])
                ll = np.min(low[i-period:i])
                if hh == ll:
                    k_val = 50.0
                else:
                    k_val = ((close[i] - ll) / (hh - ll)) * 100
                k_values.append(k_val)
            
            if len(k_values) >= 3:
                d_percent = np.mean(k_values[-3:])
            else:
                d_percent = k_percent
            
            return k_percent, d_percent
            
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return 50.0, 50.0
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        try:
            if len(prices) < period:
                return prices[-1]
            
            # Calculate smoothing factor
            alpha = 2 / (period + 1)
            
            # Initialize EMA with SMA
            ema = np.mean(prices[:period])
            
            # Calculate EMA
            for price in prices[period:]:
                ema = alpha * price + (1 - alpha) * ema
            
            return ema
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return prices[-1] if len(prices) > 0 else 0.0

class PatternRecognizer:
    """Advanced pattern recognition and classification."""
    
    def __init__(self):
        self.pattern_threshold = 0.7
        self.min_pattern_length = 5
        self.max_pattern_length = 50
    
    def detect_trend_patterns(self, prices: np.ndarray, volumes: np.ndarray = None) -> List[Pattern]:
        """Detect trend patterns in price data."""
        try:
            patterns = []
            
            # Detect uptrend
            if self._is_uptrend(prices):
                patterns.append(Pattern(
                    pattern_type=PatternType.TREND,
                    confidence=self._calculate_trend_confidence(prices, "up"),
                    start_time=datetime.now() - timedelta(days=len(prices)),
                    end_time=datetime.now(),
                    strength=self._calculate_trend_strength(prices),
                    description="Uptrend detected"
                ))
            
            # Detect downtrend
            elif self._is_downtrend(prices):
                patterns.append(Pattern(
                    pattern_type=PatternType.TREND,
                    confidence=self._calculate_trend_confidence(prices, "down"),
                    start_time=datetime.now() - timedelta(days=len(prices)),
                    end_time=datetime.now(),
                    strength=self._calculate_trend_strength(prices),
                    description="Downtrend detected"
                ))
            
            # Detect consolidation
            elif self._is_consolidation(prices):
                patterns.append(Pattern(
                    pattern_type=PatternType.CONSOLIDATION,
                    confidence=self._calculate_consolidation_confidence(prices),
                    start_time=datetime.now() - timedelta(days=len(prices)),
                    end_time=datetime.now(),
                    strength=0.5,
                    description="Consolidation pattern detected"
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting trend patterns: {e}")
            return []
    
    def detect_reversal_patterns(self, prices: np.ndarray, volumes: np.ndarray = None) -> List[Pattern]:
        """Detect reversal patterns."""
        try:
            patterns = []
            
            # Detect double top
            if self._is_double_top(prices):
                patterns.append(Pattern(
                    pattern_type=PatternType.REVERSAL,
                    confidence=0.8,
                    start_time=datetime.now() - timedelta(days=len(prices)),
                    end_time=datetime.now(),
                    strength=0.8,
                    description="Double top reversal pattern"
                ))
            
            # Detect double bottom
            elif self._is_double_bottom(prices):
                patterns.append(Pattern(
                    pattern_type=PatternType.REVERSAL,
                    confidence=0.8,
                    start_time=datetime.now() - timedelta(days=len(prices)),
                    end_time=datetime.now(),
                    strength=0.8,
                    description="Double bottom reversal pattern"
                ))
            
            # Detect head and shoulders
            elif self._is_head_and_shoulders(prices):
                patterns.append(Pattern(
                    pattern_type=PatternType.REVERSAL,
                    confidence=0.9,
                    start_time=datetime.now() - timedelta(days=len(prices)),
                    end_time=datetime.now(),
                    strength=0.9,
                    description="Head and shoulders pattern"
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting reversal patterns: {e}")
            return []
    
    def _is_uptrend(self, prices: np.ndarray) -> bool:
        """Check if prices are in uptrend."""
        try:
            if len(prices) < 3:
                return False
            
            # Calculate linear regression
            x = np.arange(len(prices))
            slope, _, r_value, _, _ = stats.linregress(x, prices)
            
            return slope > 0 and r_value > 0.7
            
        except Exception:
            return False
    
    def _is_downtrend(self, prices: np.ndarray) -> bool:
        """Check if prices are in downtrend."""
        try:
            if len(prices) < 3:
                return False
            
            # Calculate linear regression
            x = np.arange(len(prices))
            slope, _, r_value, _, _ = stats.linregress(x, prices)
            
            return slope < 0 and r_value > 0.7
            
        except Exception:
            return False
    
    def _is_consolidation(self, prices: np.ndarray) -> bool:
        """Check if prices are consolidating."""
        try:
            if len(prices) < 10:
                return False
            
            # Calculate price range
            price_range = np.max(prices) - np.min(prices)
            avg_price = np.mean(prices)
            
            # Check if range is small relative to average price
            return (price_range / avg_price) < 0.1
            
        except Exception:
            return False
    
    def _is_double_top(self, prices: np.ndarray) -> bool:
        """Detect double top pattern."""
        try:
            if len(prices) < 10:
                return False
            
            # Find local maxima
            peaks = signal.find_peaks(prices)[0]
            
            if len(peaks) < 2:
                return False
            
            # Check if last two peaks are similar in height
            last_two_peaks = peaks[-2:]
            peak_values = prices[last_two_peaks]
            
            return abs(peak_values[0] - peak_values[1]) / peak_values[0] < 0.05
            
        except Exception:
            return False
    
    def _is_double_bottom(self, prices: np.ndarray) -> bool:
        """Detect double bottom pattern."""
        try:
            if len(prices) < 10:
                return False
            
            # Find local minima
            valleys = signal.find_peaks(-prices)[0]
            
            if len(valleys) < 2:
                return False
            
            # Check if last two valleys are similar in height
            last_two_valleys = valleys[-2:]
            valley_values = prices[last_two_valleys]
            
            return abs(valley_values[0] - valley_values[1]) / valley_values[0] < 0.05
            
        except Exception:
            return False
    
    def _is_head_and_shoulders(self, prices: np.ndarray) -> bool:
        """Detect head and shoulders pattern."""
        try:
            if len(prices) < 15:
                return False
            
            # Find peaks
            peaks = signal.find_peaks(prices)[0]
            
            if len(peaks) < 3:
                return False
            
            # Check for three peaks with middle peak higher
            if len(peaks) >= 3:
                last_three_peaks = peaks[-3:]
                peak_values = prices[last_three_peaks]
                
                # Middle peak should be higher than others
                return (peak_values[1] > peak_values[0] and 
                       peak_values[1] > peak_values[2] and
                       abs(peak_values[0] - peak_values[2]) / peak_values[0] < 0.1)
            
            return False
            
        except Exception:
            return False
    
    def _calculate_trend_confidence(self, prices: np.ndarray, trend_type: str) -> float:
        """Calculate confidence in trend detection."""
        try:
            x = np.arange(len(prices))
            slope, _, r_value, _, _ = stats.linregress(x, prices)
            
            # R-squared value indicates confidence
            return r_value ** 2
            
        except Exception:
            return 0.5
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength."""
        try:
            # Calculate average directional movement
            price_changes = np.diff(prices)
            positive_changes = np.sum(price_changes > 0)
            total_changes = len(price_changes)
            
            return positive_changes / total_changes if total_changes > 0 else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_consolidation_confidence(self, prices: np.ndarray) -> float:
        """Calculate confidence in consolidation detection."""
        try:
            # Calculate coefficient of variation
            std_dev = np.std(prices)
            mean_price = np.mean(prices)
            
            cv = std_dev / mean_price if mean_price > 0 else 0
            
            # Lower CV indicates higher confidence in consolidation
            return max(0, 1 - cv * 10)
            
        except Exception:
            return 0.5

class AltitudeLogic:
    """Advanced altitude logic for trading strategies."""
    
    def __init__(self):
        self.altitude_threshold = 0.1
        self.momentum_factor = 0.8
        self.gravity_factor = 0.2
    
    def calculate_altitude(self, prices: np.ndarray, volumes: np.ndarray = None) -> float:
        """Calculate altitude (price momentum and strength)."""
        try:
            if len(prices) < 2:
                return 0.0
            
            # Calculate price momentum
            price_momentum = (prices[-1] - prices[0]) / prices[0]
            
            # Calculate volume momentum if available
            volume_momentum = 0.0
            if volumes is not None and len(volumes) > 1:
                volume_momentum = (volumes[-1] - volumes[0]) / volumes[0]
            
            # Calculate altitude as weighted combination
            altitude = (self.momentum_factor * price_momentum + 
                       (1 - self.momentum_factor) * volume_momentum)
            
            return altitude
            
        except Exception as e:
            logger.error(f"Error calculating altitude: {e}")
            return 0.0
    
    def calculate_gravity_effect(self, altitude: float, time_factor: float = 1.0) -> float:
        """Calculate gravity effect on altitude."""
        try:
            # Gravity pulls altitude towards zero over time
            gravity_effect = altitude * self.gravity_factor * time_factor
            
            return gravity_effect
            
        except Exception as e:
            logger.error(f"Error calculating gravity effect: {e}")
            return 0.0
    
    def calculate_escape_velocity(self, altitude: float, resistance_level: float) -> float:
        """Calculate escape velocity needed to break resistance."""
        try:
            # Escape velocity increases with resistance level
            escape_velocity = resistance_level * (1 + abs(altitude))
            
            return escape_velocity
            
        except Exception as e:
            logger.error(f"Error calculating escape velocity: {e}")
            return 0.0
    
    def detect_altitude_signals(self, altitudes: np.ndarray) -> List[SignalType]:
        """Detect trading signals based on altitude logic."""
        try:
            signals = []
            
            if len(altitudes) < 3:
                return [SignalType.HOLD]
            
            current_altitude = altitudes[-1]
            previous_altitude = altitudes[-2]
            
            # Strong upward momentum
            if current_altitude > self.altitude_threshold and current_altitude > previous_altitude:
                signals.append(SignalType.STRONG_BUY)
            
            # Moderate upward momentum
            elif current_altitude > 0 and current_altitude > previous_altitude:
                signals.append(SignalType.BUY)
            
            # Strong downward momentum
            elif current_altitude < -self.altitude_threshold and current_altitude < previous_altitude:
                signals.append(SignalType.STRONG_SELL)
            
            # Moderate downward momentum
            elif current_altitude < 0 and current_altitude < previous_altitude:
                signals.append(SignalType.SELL)
            
            # Neutral
            else:
                signals.append(SignalType.HOLD)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error detecting altitude signals: {e}")
            return [SignalType.HOLD]

class AnalysisEngine:
    """Main analysis engine."""
    
    def __init__(self):
        self.signal_processor = SignalProcessor()
        self.technical_indicators = TechnicalIndicators()
        self.pattern_recognizer = PatternRecognizer()
        self.altitude_logic = AltitudeLogic()
        self.analysis_history: deque = deque(maxlen=1000)
        self.is_running = False
        self.analysis_thread = None
    
    def analyze_market_data(self, market_data: List[MarketData]) -> AnalysisResult:
        """Perform comprehensive market analysis."""
        try:
            if not market_data:
                return self._create_empty_result()
            
            # Extract price and volume data
            prices = np.array([data.close for data in market_data])
            volumes = np.array([data.volume for data in market_data])
            timestamps = [data.timestamp for data in market_data]
            
            # Apply signal processing
            filtered_prices = self.signal_processor.apply_low_pass_filter(prices)
            
            # Calculate technical indicators
            indicators = self._calculate_all_indicators(filtered_prices, volumes)
            
            # Detect patterns
            patterns = self._detect_all_patterns(filtered_prices, volumes)
            
            # Calculate altitude
            altitude = self.altitude_logic.calculate_altitude(filtered_prices, volumes)
            
            # Generate signals
            signals = self._generate_signals(indicators, patterns, altitude)
            
            # Calculate confidence and risk scores
            confidence_score = self._calculate_confidence_score(indicators, patterns)
            risk_score = self._calculate_risk_score(indicators, patterns)
            
            # Create analysis result
            result = AnalysisResult(
                analysis_id=f"analysis_{int(time.time())}",
                timestamp=datetime.now(),
                symbol=market_data[-1].symbol,
                indicators=indicators,
                patterns=patterns,
                signals=signals,
                confidence_score=confidence_score,
                risk_score=risk_score,
                metadata={
                    'altitude': altitude,
                    'price_change': (prices[-1] - prices[0]) / prices[0],
                    'volume_change': (volumes[-1] - volumes[0]) / volumes[0] if len(volumes) > 1 else 0
                }
            )
            
            # Store in history
            self.analysis_history.append(result)
            
            logger.info(f"Analysis completed for {result.symbol}: {len(signals)} signals, confidence: {confidence_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return self._create_empty_result()
    
    def _calculate_all_indicators(self, prices: np.ndarray, volumes: np.ndarray) -> List[TechnicalIndicator]:
        """Calculate all technical indicators."""
        try:
            indicators = []
            timestamp = datetime.now()
            
            # RSI
            rsi_value = self.technical_indicators.calculate_rsi(prices)
            rsi_signal = self._rsi_to_signal(rsi_value)
            indicators.append(TechnicalIndicator(
                name="RSI",
                value=rsi_value,
                timestamp=timestamp,
                signal_type=rsi_signal,
                confidence=abs(rsi_value - 50) / 50,
                parameters={'period': self.technical_indicators.rsi_period}
            ))
            
            # MACD
            macd_line, signal_line, histogram = self.technical_indicators.calculate_macd(prices)
            macd_signal = self._macd_to_signal(macd_line, signal_line, histogram)
            indicators.append(TechnicalIndicator(
                name="MACD",
                value=macd_line,
                timestamp=timestamp,
                signal_type=macd_signal,
                confidence=abs(macd_line) / max(abs(macd_line), 1),
                parameters={
                    'fast': self.technical_indicators.macd_fast,
                    'slow': self.technical_indicators.macd_slow,
                    'signal': self.technical_indicators.macd_signal
                }
            ))
            
            # Bollinger Bands
            upper, middle, lower = self.technical_indicators.calculate_bollinger_bands(prices)
            bb_signal = self._bollinger_to_signal(prices[-1], upper, middle, lower)
            bb_position = (prices[-1] - lower) / (upper - lower) if upper != lower else 0.5
            indicators.append(TechnicalIndicator(
                name="Bollinger_Bands",
                value=bb_position,
                timestamp=timestamp,
                signal_type=bb_signal,
                confidence=abs(bb_position - 0.5) * 2,
                parameters={
                    'period': self.technical_indicators.bollinger_period,
                    'std': self.technical_indicators.bollinger_std
                }
            ))
            
            # Stochastic
            k_percent, d_percent = self.technical_indicators.calculate_stochastic(
                prices, prices, prices  # Using same array for high/low/close
            )
            stoch_signal = self._stochastic_to_signal(k_percent, d_percent)
            indicators.append(TechnicalIndicator(
                name="Stochastic",
                value=k_percent,
                timestamp=timestamp,
                signal_type=stoch_signal,
                confidence=abs(k_percent - 50) / 50,
                parameters={'period': 14}
            ))
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return []
    
    def _detect_all_patterns(self, prices: np.ndarray, volumes: np.ndarray) -> List[Pattern]:
        """Detect all patterns."""
        try:
            patterns = []
            
            # Detect trend patterns
            trend_patterns = self.pattern_recognizer.detect_trend_patterns(prices, volumes)
            patterns.extend(trend_patterns)
            
            # Detect reversal patterns
            reversal_patterns = self.pattern_recognizer.detect_reversal_patterns(prices, volumes)
            patterns.extend(reversal_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []
    
    def _generate_signals(self, indicators: List[TechnicalIndicator], 
                         patterns: List[Pattern], altitude: float) -> List[SignalType]:
        """Generate trading signals based on indicators and patterns."""
        try:
            signals = []
            
            # Count buy and sell signals from indicators
            buy_signals = 0
            sell_signals = 0
            
            for indicator in indicators:
                if indicator.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                    buy_signals += indicator.confidence
                elif indicator.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                    sell_signals += indicator.confidence
            
            # Consider patterns
            for pattern in patterns:
                if pattern.pattern_type == PatternType.TREND:
                    if "Uptrend" in pattern.description:
                        buy_signals += pattern.confidence
                    elif "Downtrend" in pattern.description:
                        sell_signals += pattern.confidence
                elif pattern.pattern_type == PatternType.REVERSAL:
                    if "bottom" in pattern.description.lower():
                        buy_signals += pattern.confidence
                    elif "top" in pattern.description.lower():
                        sell_signals += pattern.confidence
            
            # Consider altitude
            if altitude > 0.1:
                buy_signals += abs(altitude)
            elif altitude < -0.1:
                sell_signals += abs(altitude)
            
            # Determine final signal
            if buy_signals > sell_signals * 1.5:
                signals.append(SignalType.STRONG_BUY)
            elif buy_signals > sell_signals:
                signals.append(SignalType.BUY)
            elif sell_signals > buy_signals * 1.5:
                signals.append(SignalType.STRONG_SELL)
            elif sell_signals > buy_signals:
                signals.append(SignalType.SELL)
            else:
                signals.append(SignalType.HOLD)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return [SignalType.HOLD]
    
    def _calculate_confidence_score(self, indicators: List[TechnicalIndicator], 
                                  patterns: List[Pattern]) -> float:
        """Calculate overall confidence score."""
        try:
            if not indicators and not patterns:
                return 0.5
            
            total_confidence = 0.0
            total_weight = 0.0
            
            # Weight indicators
            for indicator in indicators:
                total_confidence += indicator.confidence * 0.7
                total_weight += 0.7
            
            # Weight patterns
            for pattern in patterns:
                total_confidence += pattern.confidence * 0.3
                total_weight += 0.3
            
            return total_confidence / total_weight if total_weight > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    def _calculate_risk_score(self, indicators: List[TechnicalIndicator], 
                            patterns: List[Pattern]) -> float:
        """Calculate risk score."""
        try:
            risk_factors = []
            
            # RSI extremes
            for indicator in indicators:
                if indicator.name == "RSI":
                    if indicator.value > 80 or indicator.value < 20:
                        risk_factors.append(0.8)
                    elif indicator.value > 70 or indicator.value < 30:
                        risk_factors.append(0.6)
                    else:
                        risk_factors.append(0.2)
            
            # Pattern risk
            for pattern in patterns:
                if pattern.pattern_type == PatternType.REVERSAL:
                    risk_factors.append(0.7)
                elif pattern.pattern_type == PatternType.CONSOLIDATION:
                    risk_factors.append(0.3)
                else:
                    risk_factors.append(0.5)
            
            return np.mean(risk_factors) if risk_factors else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5
    
    def _rsi_to_signal(self, rsi: float) -> SignalType:
        """Convert RSI value to signal."""
        if rsi > 80:
            return SignalType.STRONG_SELL
        elif rsi > 70:
            return SignalType.SELL
        elif rsi < 20:
            return SignalType.STRONG_BUY
        elif rsi < 30:
            return SignalType.BUY
        else:
            return SignalType.HOLD
    
    def _macd_to_signal(self, macd_line: float, signal_line: float, histogram: float) -> SignalType:
        """Convert MACD values to signal."""
        if macd_line > signal_line and histogram > 0:
            return SignalType.BUY
        elif macd_line < signal_line and histogram < 0:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _bollinger_to_signal(self, price: float, upper: float, middle: float, lower: float) -> SignalType:
        """Convert Bollinger Bands position to signal."""
        if price <= lower:
            return SignalType.STRONG_BUY
        elif price < middle:
            return SignalType.BUY
        elif price >= upper:
            return SignalType.STRONG_SELL
        elif price > middle:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _stochastic_to_signal(self, k_percent: float, d_percent: float) -> SignalType:
        """Convert Stochastic values to signal."""
        if k_percent < 20 and d_percent < 20:
            return SignalType.STRONG_BUY
        elif k_percent < 30 and d_percent < 30:
            return SignalType.BUY
        elif k_percent > 80 and d_percent > 80:
            return SignalType.STRONG_SELL
        elif k_percent > 70 and d_percent > 70:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _create_empty_result(self) -> AnalysisResult:
        """Create empty analysis result."""
        return AnalysisResult(
            analysis_id=f"empty_{int(time.time())}",
            timestamp=datetime.now(),
            symbol="",
            indicators=[],
            patterns=[],
            signals=[SignalType.HOLD],
            confidence_score=0.0,
            risk_score=0.5,
            metadata={}
        )
    
    def get_analysis_history(self, limit: int = 100) -> List[AnalysisResult]:
        """Get analysis history."""
        return list(self.analysis_history)[-limit:]
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get analysis summary."""
        try:
            if not self.analysis_history:
                return {'total_analyses': 0}
            
            analyses = list(self.analysis_history)
            
            # Calculate statistics
            total_analyses = len(analyses)
            avg_confidence = np.mean([a.confidence_score for a in analyses])
            avg_risk = np.mean([a.risk_score for a in analyses])
            
            # Count signals
            signal_counts = defaultdict(int)
            for analysis in analyses:
                for signal in analysis.signals:
                    signal_counts[signal.value] += 1
            
            return {
                'total_analyses': total_analyses,
                'avg_confidence': avg_confidence,
                'avg_risk': avg_risk,
                'signal_distribution': dict(signal_counts),
                'last_analysis': analyses[-1].timestamp.isoformat() if analyses else None
            }
            
        except Exception as e:
            logger.error(f"Error getting analysis summary: {e}")
            return {'total_analyses': 0, 'error': str(e)}

def main():
    """Main function for testing."""
    try:
        # Create analysis engine
        engine = AnalysisEngine()
        
        # Create sample market data
        market_data = []
        base_price = 100.0
        for i in range(50):
            # Simulate price movement
            price_change = np.random.normal(0, 1)
            base_price += price_change
            volume = np.random.uniform(1000, 10000)
            
            market_data.append(MarketData(
                timestamp=datetime.now() - timedelta(days=50-i),
                open=base_price - 0.5,
                high=base_price + 0.5,
                low=base_price - 0.5,
                close=base_price,
                volume=volume,
                symbol="BTC/USD",
                interval="1h"
            ))
        
        # Perform analysis
        result = engine.analyze_market_data(market_data)
        
        print("Analysis Result:")
        print(f"Symbol: {result.symbol}")
        print(f"Confidence Score: {result.confidence_score:.2f}")
        print(f"Risk Score: {result.risk_score:.2f}")
        print(f"Signals: {[s.value for s in result.signals]}")
        print(f"Indicators: {len(result.indicators)}")
        print(f"Patterns: {len(result.patterns)}")
        
        # Print indicator details
        for indicator in result.indicators:
            print(f"  {indicator.name}: {indicator.value:.2f} ({indicator.signal_type.value})")
        
        # Print pattern details
        for pattern in result.patterns:
            print(f"  {pattern.pattern_type.value}: {pattern.description} (confidence: {pattern.confidence:.2f})")
        
        # Get analysis summary
        summary = engine.get_analysis_summary()
        print(f"\nAnalysis Summary:")
        print(json.dumps(summary, indent=2, default=str))
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 