"""
TA-Lib Fallback Module

Provides basic technical analysis functions when ta-lib is not available.
Uses numpy and pandas for calculations to maintain mathematical integrity.
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import ta-lib, fall back to custom implementations
try:
    import talib

    TALIB_AVAILABLE = True
    logger.info("TA-Lib successfully imported")
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available, using fallback implementations")


class TalibFallback:
    """Fallback implementations for TA-Lib functions."""

    @staticmethod
    def SMA(data: Union[np.ndarray, pd.Series], timeperiod: int = 30) -> np.ndarray:
        """Simple Moving Average fallback."""
        if TALIB_AVAILABLE:
            return talib.SMA(data, timeperiod)

        # Fallback implementation
        if isinstance(data, pd.Series):
            return data.rolling(window=timeperiod).mean().values
        else:
            return pd.Series(data).rolling(window=timeperiod).mean().values

    @staticmethod
    def EMA(data: Union[np.ndarray, pd.Series], timeperiod: int = 30) -> np.ndarray:
        """Exponential Moving Average fallback."""
        if TALIB_AVAILABLE:
            return talib.EMA(data, timeperiod)

        # Fallback implementation
        if isinstance(data, pd.Series):
            return data.ewm(span=timeperiod).mean().values
        else:
            return pd.Series(data).ewm(span=timeperiod).mean().values

    @staticmethod
    def RSI(data: Union[np.ndarray, pd.Series], timeperiod: int = 14) -> np.ndarray:
        """Relative Strength Index fallback."""
        if TALIB_AVAILABLE:
            return talib.RSI(data, timeperiod)

        # Fallback implementation
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values

    @staticmethod
    def MACD(
        data: Union[np.ndarray, pd.Series], fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9
    ) -> tuple:
        """MACD fallback."""
        if TALIB_AVAILABLE:
            return talib.MACD(data, fastperiod, slowperiod, signalperiod)

        # Fallback implementation
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        ema_fast = data.ewm(span=fastperiod).mean()
        ema_slow = data.ewm(span=slowperiod).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signalperiod).mean()
        histogram = macd_line - signal_line

        return macd_line.values, signal_line.values, histogram.values

    @staticmethod
    def BBANDS(
        data: Union[np.ndarray, pd.Series], timeperiod: int = 20, nbdevup: float = 2, nbdevdn: float = 2
    ) -> tuple:
        """Bollinger Bands fallback."""
        if TALIB_AVAILABLE:
            return talib.BBANDS(data, timeperiod, nbdevup, nbdevdn)

        # Fallback implementation
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        sma = data.rolling(window=timeperiod).mean()
        std = data.rolling(window=timeperiod).std()
        upper_band = sma + (std * nbdevup)
        lower_band = sma - (std * nbdevdn)

        return upper_band.values, sma.values, lower_band.values


# Create module-level functions that match ta-lib API
SMA = TalibFallback.SMA
EMA = TalibFallback.EMA
RSI = TalibFallback.RSI
MACD = TalibFallback.MACD
BBANDS = TalibFallback.BBANDS

# Mathematical preservation notice
logger.info("TA-Lib fallback module loaded - mathematical integrity maintained")
logger.info("Supporting ZPE/ZBE/Ghost Core/Ferris wheel calculations")
