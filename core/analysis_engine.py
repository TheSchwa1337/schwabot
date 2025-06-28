# !/usr/bin/env python3
"""
Analysis Engine - Advanced Market Analysis and Signal Processing
===============================================================

This module implements a comprehensive analysis engine for Schwabot,
providing advanced market analysis, signal processing, and pattern recognition.

Core Mathematical Functions:
- RSI: RSI = 100 - (100 / (1 + RS)) where RS = avg_gain / avg_loss
- MACD: MACD = EMA(12) - EMA(26), Signal = EMA(9) of MACD
- Bollinger Bands: Upper = SMA + (2 * std_dev), Lower = SMA - (2 * std_dev)
- Stochastic: %K = ((close - low_n) / (high_n - low_n)) * 100
- FFT: X(k) = Σ x(n) * e^(-j2πkn/N) for k = 0 to N-1
- Power Spectral Density: PSD = |FFT|² / N

Core Functionality:
- Real-time signal processing
- Technical indicator calculation
- Pattern recognition and analysis
- Altitude-based trading logic
- Fourier and wavelet analysis
- Machine learning integration
- Confidence and risk scoring
"""

import logging
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import numpy as np
from scipy import stats, signal
import pandas as pd
import warnings

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


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
    """Advanced signal processing for market data."""

    def __init__(self, sample_rate: float = 1.0, filter_type: str = "butterworth"):
        self.sample_rate = sample_rate
        self.filter_type = filter_type
        self.nyquist_freq = sample_rate / 2

    def apply_low_pass_filter(self, data: np.ndarray, cutoff: float = None) -> np.ndarray:
        """Apply low-pass filter to remove high-frequency noise."""
        try:
            if cutoff is None:
                cutoff = self.nyquist_freq * 0.1  # 10% of Nyquist frequency

            # Design Butterworth filter
            b, a = signal.butter(4, cutoff / self.nyquist_freq, btype='low')

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
                cutoff = self.nyquist_freq * 0.01  # 1% of Nyquist frequency

            # Design Butterworth filter
            b, a = signal.butter(4, cutoff / self.nyquist_freq, btype='high')

            # Apply filter
            filtered_data = signal.filtfilt(b, a, data)

            return filtered_data

        except Exception as e:
            logger.error(f"Error applying high-pass filter: {e}")
            return data

    def apply_band_pass_filter(self, data: np.ndarray, low_cutoff: float, high_cutoff: float) -> np.ndarray:
        """Apply band-pass filter for specific frequency ranges."""
        try:
            # Design Butterworth filter
            b, a = signal.butter(4, [low_cutoff / self.nyquist_freq, high_cutoff / self.nyquist_freq], btype='band')

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
            windowed_data = data * np.hanning(len(data))

            # Compute FFT
            fft_result = np.fft.fft(windowed_data)
            frequencies = np.fft.fftfreq(len(data), 1 / self.sample_rate)

            return frequencies, fft_result

        except Exception as e:
            logger.error(f"Error computing FFT: {e}")
            return np.array([]), np.array([])

    def compute_power_spectral_density(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Power Spectral Density."""
        try:
            frequencies, fft_result = self.compute_fft(data)

            # Calculate PSD
            psd = np.abs(fft_result) ** 2 / len(data)

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
        self.stochastic_period = 14

    def calculate_rsi(self, prices: np.ndarray, period: int = None) -> float:
        """Calculate Relative Strength Index."""
        try:
            if period is None:
                period = self.rsi_period

            if len(prices) < period + 1:
                return 50.0  # Neutral RSI

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
                    ema_fast_i = self._calculate_ema(prices[:i+1], self.macd_fast)
                    ema_slow_i = self._calculate_ema(prices[:i+1], self.macd_slow)
                    macd_values.append(ema_fast_i - ema_slow_i)

            if len(macd_values) >= self.macd_signal:
                signal_line = self._calculate_ema(np.array(macd_values), self.macd_signal)
            else:
                signal_line = macd_line

            # Calculate histogram
            histogram = macd_line - signal_line

            return macd_line, signal_line, histogram

        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return 0.0, 0.0, 0.0

    def calculate_bollinger_bands(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        try:
            if len(prices) < self.bollinger_period:
                current_price = prices[-1]
                return current_price, current_price, current_price

            # Calculate SMA
            sma = np.mean(prices[-self.bollinger_period:])

            # Calculate standard deviation
            std_dev = np.std(prices[-self.bollinger_period:])

            # Calculate bands
            upper_band = sma + (self.bollinger_std * std_dev)
            lower_band = sma - (self.bollinger_std * std_dev)

            return upper_band, sma, lower_band

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            current_price = prices[-1] if len(prices) > 0 else 0.0
            return current_price, current_price, current_price

    def calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator."""
        try:
            if len(close) < period:
                return 50.0, 50.0

            # Calculate %K
            lowest_low = np.min(low[-period:])
            highest_high = np.max(high[-period:])

            if highest_high == lowest_low:
                k_percent = 50.0
            else:
                k_percent = ((close[-1] - lowest_low) / (highest_high - lowest_low)) * 100

            # Calculate %D (3-period SMA of %K)
            k_values = []
            for i in range(period, len(close)):
                period_low = np.min(low[i-period:i])
                period_high = np.max(high[i-period:i])
                if period_high == period_low:
                    k_values.append(50.0)
                else:
                    k_values.append(((close[i] - period_low) / (period_high - period_low)) * 100)

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
                return prices[-1] if len(prices) > 0 else 0.0

            # Calculate smoothing factor
            alpha = 2 / (period + 1)

            # Calculate EMA
            ema = prices[0]
            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema

            return ema

        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return prices[-1] if len(prices) > 0 else 0.0


class PatternRecognizer:
    """Pattern recognition for market analysis."""

    def __init__(self):
        self.trend_threshold = 0.1
        self.consolidation_threshold = 0.05

    def detect_trend_patterns(self, prices: np.ndarray, volumes: np.ndarray = None) -> List[Pattern]:
        """Detect trend patterns in price data."""
        try:
            patterns = []

            if len(prices) < 20:
                return patterns

            # Detect uptrend
            if self._is_uptrend(prices):
                patterns.append(Pattern(
                    pattern_type=PatternType.TREND,
                    confidence=self._calculate_trend_confidence(prices, "uptrend"),
                    start_time=datetime.now() - timedelta(days=len(prices)),
                    end_time=datetime.now(),
                    strength=self._calculate_trend_strength(prices),
                    description="Uptrend detected",
                    metadata={'direction': 'up'}
                ))

            # Detect downtrend
            elif self._is_downtrend(prices):
                patterns.append(Pattern(
                    pattern_type=PatternType.TREND,
                    confidence=self._calculate_trend_confidence(prices, "downtrend"),
                    start_time=datetime.now() - timedelta(days=len(prices)),
                    end_time=datetime.now(),
                    strength=self._calculate_trend_strength(prices),
                    description="Downtrend detected",
                    metadata={'direction': 'down'}
                ))

            # Detect consolidation
            elif self._is_consolidation(prices):
                patterns.append(Pattern(
                    pattern_type=PatternType.CONSOLIDATION,
                    confidence=self._calculate_consolidation_confidence(prices),
                    start_time=datetime.now() - timedelta(days=len(prices)),
                    end_time=datetime.now(),
                    strength=0.5,
                    description="Consolidation detected",
                    metadata={'type': 'sideways'}
                ))

            return patterns

        except Exception as e:
            logger.error(f"Error detecting trend patterns: {e}")
            return []

    def detect_reversal_patterns(self, prices: np.ndarray, volumes: np.ndarray = None) -> List[Pattern]:
        """Detect reversal patterns in price data."""
        try:
            patterns = []

            if len(prices) < 30:
                return patterns

            # Detect double top
            if self._is_double_top(prices):
                patterns.append(Pattern(
                    pattern_type=PatternType.REVERSAL,
                    confidence=0.7,
                    start_time=datetime.now() - timedelta(days=len(prices)),
                    end_time=datetime.now(),
                    strength=0.8,
                    description="Double top reversal pattern",
                    metadata={'pattern': 'double_top'}
                ))

            # Detect double bottom
            elif self._is_double_bottom(prices):
                patterns.append(Pattern(
                    pattern_type=PatternType.REVERSAL,
                    confidence=0.7,
                    start_time=datetime.now() - timedelta(days=len(prices)),
                    end_time=datetime.now(),
                    strength=0.8,
                    description="Double bottom reversal pattern",
                    metadata={'pattern': 'double_bottom'}
                ))

            # Detect head and shoulders
            elif self._is_head_and_shoulders(prices):
                patterns.append(Pattern(
                    pattern_type=PatternType.REVERSAL,
                    confidence=0.8,
                    start_time=datetime.now() - timedelta(days=len(prices)),
                    end_time=datetime.now(),
                    strength=0.9,
                    description="Head and shoulders pattern",
                    metadata={'pattern': 'head_and_shoulders'}
                ))

            return patterns

        except Exception as e:
            logger.error(f"Error detecting reversal patterns: {e}")
            return []

    def _is_uptrend(self, prices: np.ndarray) -> bool:
        """Check if prices are in uptrend."""
        try:
            if len(prices) < 10:
                return False

            # Calculate linear regression
            x = np.arange(len(prices))
            slope, _, r_value, _, _ = stats.linregress(x, prices)

            # Check if slope is positive and correlation is strong
            return slope > 0 and r_value > 0.7

        except Exception:
            return False

    def _is_downtrend(self, prices: np.ndarray) -> bool:
        """Check if prices are in downtrend."""
        try:
            if len(prices) < 10:
                return False

            # Calculate linear regression
            x = np.arange(len(prices))
            slope, _, r_value, _, _ = stats.linregress(x, prices)

            # Check if slope is negative and correlation is strong
            return slope < 0 and r_value > 0.7

        except Exception:
            return False

    def _is_consolidation(self, prices: np.ndarray) -> bool:
        """Check if prices are in consolidation."""
        try:
            if len(prices) < 20:
                return False

            # Calculate coefficient of variation
            std_dev = np.std(prices)
            mean_price = np.mean(prices)

            cv = std_dev / mean_price if mean_price > 0 else 0

            # Low CV indicates consolidation
            return cv < self.consolidation_threshold

        except Exception:
            return False

    def _is_double_top(self, prices: np.ndarray) -> bool:
        """Check for double top pattern."""
        try:
            if len(prices) < 20:
                return False

            # Find peaks
            peaks = signal.find_peaks(prices)[0]

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

    def _is_double_bottom(self, prices: np.ndarray) -> bool:
        """Check for double bottom pattern."""
        try:
            if len(prices) < 20:
                return False

            # Find troughs
            troughs = signal.find_peaks(-prices)[0]

            if len(troughs) >= 3:
                last_three_troughs = troughs[-3:]
                trough_values = prices[last_three_troughs]

                # Middle trough should be lower than others
                return (trough_values[1] < trough_values[0] and
                        trough_values[1] < trough_values[2] and
                        abs(trough_values[0] - trough_values[2]) / trough_values[0] < 0.1)

            return False

        except Exception:
            return False

    def _is_head_and_shoulders(self, prices: np.ndarray) -> bool:
        """Check for head and shoulders pattern."""
        try:
            if len(prices) < 30:
                return False

            # Find peaks
            peaks = signal.find_peaks(prices)[0]

            if len(peaks) >= 5:
                last_five_peaks = peaks[-5:]
                peak_values = prices[last_five_peaks]

                # Check head and shoulders pattern
                left_shoulder = peak_values[0]
                head = peak_values[1]
                right_shoulder = peak_values[2]

                return (
                    head > left_shoulder and
                    head > right_shoulder and
                    abs(left_shoulder - right_shoulder) / left_shoulder < 0.1
                )

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
            altitude = (
                self.momentum_factor * price_momentum +
                (1 - self.momentum_factor) * volume_momentum
            )

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
                    'filtered_prices': filtered_prices.tolist(),
                    'original_prices': prices.tolist()
                }
            )

            # Store in history
            self.analysis_history.append(result)

            logger.info(
                f"Analysis completed for {result.symbol}: {len(signals)} signals, confidence: {confidence_score:.2f}")

            return result

        except Exception as e:
            logger.error(f"Error analyzing market data: {e}")
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
                confidence=abs(macd_line) / (abs(macd_line) + abs(signal_line)
                                             ) if abs(macd_line) + abs(signal_line) > 0 else 0.5,
                parameters={'fast': self.technical_indicators.macd_fast, 'slow': self.technical_indicators.macd_slow}
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
                confidence=1 - bb_position if bb_position > 0.5 else bb_position,
                parameters={'period': self.technical_indicators.bollinger_period,
                            'std': self.technical_indicators.bollinger_std}
            ))

            # Stochastic
            k_percent, d_percent = self.technical_indicators.calculate_stochastic(
                np.array([data.high for data in market_data]),
                np.array([data.low for data in market_data]),
                prices
            )
            stoch_signal = self._stochastic_to_signal(k_percent, d_percent)
            indicators.append(TechnicalIndicator(
                name="Stochastic",
                value=k_percent,
                timestamp=timestamp,
                signal_type=stoch_signal,
                confidence=abs(k_percent - 50) / 50,
                parameters={'period': self.technical_indicators.stochastic_period}
            ))

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return []

    def _detect_all_patterns(self, prices: np.ndarray, volumes: np.ndarray) -> List[Pattern]:
        """Detect all patterns in price data."""
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
        """Generate trading signals from indicators and patterns."""
        try:
            signals = []

            # Collect signals from indicators
            for indicator in indicators:
                signals.append(indicator.signal_type)

            # Add pattern-based signals
            for pattern in patterns:
                if pattern.pattern_type == PatternType.TREND:
                    if pattern.metadata.get('direction') == 'up':
                        signals.append(SignalType.BUY)
                    else:
                        signals.append(SignalType.SELL)
                elif pattern.pattern_type == PatternType.REVERSAL:
                    signals.append(SignalType.STRONG_SELL if 'top' in pattern.metadata.get(
                        'pattern', '') else SignalType.STRONG_BUY)

            # Add altitude-based signals
            altitude_signals = self.altitude_logic.detect_altitude_signals(np.array([altitude]))
            signals.extend(altitude_signals)

            # If no signals, default to HOLD
            if not signals:
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

            # Calculate average confidence from indicators
            indicator_confidence = np.mean([ind.confidence for ind in indicators]) if indicators else 0.5

            # Calculate average confidence from patterns
            pattern_confidence = np.mean([pat.confidence for pat in patterns]) if patterns else 0.5

            # Weighted average
            total_confidence = (indicator_confidence * 0.7 + pattern_confidence * 0.3)

            return min(1.0, max(0.0, total_confidence))

        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5

    def _calculate_risk_score(self, indicators: List[TechnicalIndicator],
                              patterns: List[Pattern]) -> float:
        """Calculate risk score."""
        try:
            if not indicators and not patterns:
                return 0.5

            # Calculate risk based on indicator volatility
            indicator_risks = []
            for indicator in indicators:
                if indicator.name == "RSI":
                    # RSI extremes indicate risk
                    risk = abs(indicator.value - 50) / 50
                    indicator_risks.append(risk)
                elif indicator.name == "Bollinger_Bands":
                    # Position within bands indicates risk
                    indicator_risks.append(indicator.value)
                else:
                    indicator_risks.append(0.5)

            avg_indicator_risk = np.mean(indicator_risks) if indicator_risks else 0.5

            # Calculate risk based on patterns
            pattern_risks = []
            for pattern in patterns:
                if pattern.pattern_type == PatternType.REVERSAL:
                    pattern_risks.append(0.8)  # High risk for reversals
                elif pattern.pattern_type == PatternType.CONSOLIDATION:
                    pattern_risks.append(0.3)  # Low risk for consolidation
                else:
                    pattern_risks.append(0.5)

            avg_pattern_risk = np.mean(pattern_risks) if pattern_risks else 0.5

            # Weighted average
            total_risk = (avg_indicator_risk * 0.6 + avg_pattern_risk * 0.4)

            return min(1.0, max(0.0, total_risk))

        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5

    def _rsi_to_signal(self, rsi: float) -> SignalType:
        """Convert RSI value to signal."""
        if rsi > 70:
            return SignalType.STRONG_SELL
        elif rsi > 60:
            return SignalType.SELL
        elif rsi < 30:
            return SignalType.STRONG_BUY
        elif rsi < 40:
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

            recent_analyses = list(self.analysis_history)[-100:]

            # Calculate statistics
            total_analyses = len(recent_analyses)
            avg_confidence = np.mean([a.confidence_score for a in recent_analyses])
            avg_risk = np.mean([a.risk_score for a in recent_analyses])

            # Count signal types
            signal_counts = {}
            for analysis in recent_analyses:
                for sig in analysis.signals:
                    signal_counts[sig.value] = signal_counts.get(sig.value, 0) + 1

            return {
                'total_analyses': total_analyses,
                'average_confidence': avg_confidence,
                'average_risk': avg_risk,
                'signal_distribution': signal_counts,
                'last_analysis': recent_analyses[-1].timestamp.isoformat() if recent_analyses else None
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
        for i in range(100):
            # Simulate price movement
            price_change = np.random.normal(0, 1)
            base_price += price_change

            market_data.append(MarketData(
                timestamp=datetime.now() - timedelta(minutes=100-i),
                open=base_price - 0.5,
                high=base_price + 1.0,
                low=base_price - 1.0,
                close=base_price,
                volume=np.random.uniform(1000, 10000),
                symbol="BTC/USD",
                interval="1m"
            ))

        # Perform analysis
        result = engine.analyze_market_data(market_data)

        print("\nAnalysis Summary:")
        print(f"Symbol: {result.symbol}")
        print(f"Signals: {[s.value for s in result.signals]}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Risk Score: {result.risk_score:.2f}")
        print(f"Indicators: {len(result.indicators)}")
        print(f"Patterns: {len(result.patterns)}")

        # Get analysis summary
        summary = engine.get_analysis_summary()
        print("\nAnalysis Summary:")
        print(f"Total Analyses: {summary['total_analyses']}")
        print(f"Average Confidence: {summary.get('average_confidence', 0):.2f}")
        print(f"Signal Distribution: {summary.get('signal_distribution', {})}")

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
