from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Utilities - Mathematical Helper Functions and System Utilities
============================================================

This module implements comprehensive utility functions for Schwabot,
providing mathematical helper functions, data processing utilities,
and system utilities.

Core Mathematical Functions:
- Statistical Functions: mean, std, correlation, etc.
- Financial Functions: returns, volatility, Sharpe ratio, etc.
- Data Processing: normalization, scaling, filtering, etc.
- System Utilities: timing, memory, file operations, etc.

Core Functionality:
- Mathematical and statistical calculations
- Financial metrics and indicators
- Data processing and transformation
- System monitoring and utilities
- File and data handling
- Performance optimization
"""

import logging
import json
import time
from core.unified_math_system import unified_math
import hashlib
import os
import sys
import platform
import psutil
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from core.unified_math_system import unified_math
from collections import defaultdict, deque
import queue
import weakref
import threading
import asyncio

logger = logging.getLogger(__name__)


class DataType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TIMESTAMP = "timestamp"
    TEXT = "text"
    BINARY = "binary"


@dataclass
class PerformanceMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MathematicalUtils:
    """Mathematical utility functions."""

    @staticmethod
def calculate_returns(prices: List[float] -> List[float):    """Calculate percentage returns from price series."""
    try:
    pass
    if len(prices] < 2:
    return []

    returns = []
    for i in range(1, len(prices)):
    if prices[i-1] != 0:
    return_pct = (prices[i] - prices[i-1]] / prices[i-1)
    returns.append(return_pct)
    else:
    returns.append(0.0)

    return returns

    except Exception as e:
    logger.error(f"Error calculating returns: {e}")
    return []

    @ staticmethod
def calculate_volatility(returns: List[float], window: int=None) -> float:
    """Calculate volatility (standard deviation) of returns."""
    try:
    pass
    if not returns:
    return 0.0

    if window and window < len(returns):
    returns = returns[-window:]

    return float(unified_math.unified_math.std(returns))

    except Exception as e:
    logger.error(f"Error calculating volatility: {e}")
    return 0.0

    @ staticmethod
def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float=0.02) -> float:
    """Calculate Sharpe ratio."""
    try:
    pass
    if not returns:
    return 0.0

    avg_return = unified_math.unified_math.mean(returns)
    volatility = unified_math.unified_math.std(returns)

    if volatility == 0:
    return 0.0

    # Annualize if needed (assuming daily returns)
    sharpe = (avg_return - risk_free_rate/252) / volatility * unified_math.unified_math.sqrt(252)
    return float(sharpe)

    except Exception as e:
    logger.error(f"Error calculating Sharpe ratio: {e}")
    return 0.0

    @ staticmethod
def calculate_max_drawdown(returns: List[float]) -> float:
    """Calculate maximum drawdown."""
    try:
    pass
    if not returns:
    return 0.0

    cumulative_returns = np.cumprod(1 + np.array(returns))
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max

    max_drawdown = float(unified_math.unified_math.min(drawdowns))
    return unified_math.abs(max_drawdown)

    except Exception as e:
    logger.error(f"Error calculating max drawdown: {e}")
    return 0.0

    @ staticmethod
def calculate_correlation(series1: List[float], series2: List[float]) -> float:
    """Calculate correlation between two series."""
    try:
    pass
    if len(series1) != len(series2) or len(series1) < 2:
    return 0.0

    correlation = unified_math.unified_math.correlation(series1, series2][0, 1)
    return float(correlation) if not np.isnan(correlation) else 0.0

    except Exception as e:
    logger.error(f"Error calculating correlation: {e}")
    return 0.0

    @ staticmethod
def normalize_data(data: List[float], method: str='minmax'] -> List[float]:
    """Normalize data using specified method."""
    try:
    pass
    if not data:
    return [)

    data_array = np.array(data)

    if method == 'minmax':
    min_val = unified_math.unified_math.min(data_array)
    max_val = unified_math.unified_math.max(data_array)
    if max_val != min_val:
    normalized = (data_array - min_val) / (max_val - min_val)
    else:
    normalized = np.zeros_like(data_array)

    elif method == 'zscore':
    mean_val = unified_math.unified_math.mean(data_array)
    std_val = unified_math.unified_math.std(data_array)
    if std_val != 0:
    normalized = (data_array - mean_val) / std_val
    else:
    normalized = np.zeros_like(data_array)

    elif method == 'log':
    normalized = np.log1p(unified_math.unified_math.abs(data_array))

    else:
    logger.warning(f"Unknown normalization method: {method}")
    return data

    return normalized.tolist()

    except Exception as e:
    logger.error(f"Error normalizing data: {e}")
    return data

    @ staticmethod
def calculate_moving_average(data: List[float], window: int] -> List[float):
    """Calculate moving average."""
    try:
    pass
    if len(data] < window:
    return []

    moving_avg = []
    for i in range(window - 1, len(data)):
    avg = unified_math.unified_math.mean(data[i - window + 1:i + 1])
    moving_avg.append(float(avg))

    return moving_avg

    except Exception as e:
    logger.error(f"Error calculating moving average: {e}")
    return []

    @ staticmethod
def calculate_exponential_moving_average(data: List[float], alpha: float] -> List[float]:
    """Calculate exponential moving average."""
    try:
    pass
    if not data:
    return []

    ema = [data[0])
    for i in range(1, len(data)):
    ema_val = alpha * data[i] + (1 - alpha) * ema[i-1]
    ema.append(float(ema_val))

    return ema

    except Exception as e:
    logger.error(f"Error calculating EMA: {e}")
    return []

    @ staticmethod
def calculate_rsi(prices: List[float], period: int=14] -> List[float):
    """Calculate Relative Strength Index."""
    try:
    pass
    if len(prices] < period + 1:
    return []

    returns = MathematicalUtils.calculate_returns(prices)
    if len(returns) < period:
    return []

    rsi_values = []
    for i in range(period, len(returns)):
    gains = [r for r in returns[i-period:i] if r > 0]
    losses = [unified_math.abs(r] for r in returns[i-period:i) if r < 0)

    avg_gain = unified_math.unified_math.mean(gains) if gains else 0
    avg_loss = unified_math.unified_math.mean(losses) if losses else 0

    if avg_loss == 0:
    rsi = 100
    else:
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi_values.append(float(rsi))

    return rsi_values

    except Exception as e:
    logger.error(f"Error calculating RSI: {e}")
    return []

class FinancialUtils:
    """Financial utility functions."""

    @ staticmethod
def calculate_present_value(future_value: float, rate: float, periods: int) -> float:
    """Calculate present value."""
    try:
    pass
    if rate == 0:
    return future_value

    pv = future_value / ((1 + rate) ** periods)
    return float(pv)

    except Exception as e:
    logger.error(f"Error calculating present value: {e}")
    return 0.0

    @ staticmethod
def calculate_future_value(present_value: float, rate: float, periods: int) -> float:
    """Calculate future value."""
    try:
    pass
    fv = present_value * ((1 + rate) ** periods)
    return float(fv)

    except Exception as e:
    logger.error(f"Error calculating future value: {e}")
    return 0.0

    @ staticmethod
def calculate_compound_annual_growth_rate(beginning_value: float, ending_value: float, years: float) -> float:
    """Calculate Compound Annual Growth Rate."""
    try:
    pass
    if beginning_value <= 0 or years <= 0:
    return 0.0

    cagr = (ending_value / beginning_value) ** (1 / years) - 1
    return float(cagr)

    except Exception as e:
    logger.error(f"Error calculating CAGR: {e}")
    return 0.0

    @ staticmethod
def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Calculate Kelly Criterion for position sizing."""
    try:
    pass
    if avg_loss == 0:
    return 0.0

    kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    return unified_math.max(0.0, unified_math.min(1.0, float(kelly)))  # Clamp between 0 and 1

    except Exception as e:
    logger.error(f"Error calculating Kelly Criterion: {e}")
    return 0.0

    @ staticmethod
def calculate_value_at_risk(returns: List[float], confidence_level: float=0.95) -> float:
    """Calculate Value at Risk."""
    try:
    pass
    if not returns:
    return 0.0

    sorted_returns = sorted(returns)
    index = int((1 - confidence_level) * len(sorted_returns)]
    var = sorted_returns[index] if index < len(sorted_returns) else sorted_returns[-1]

    return float(var)

    except Exception as e:
    logger.error(f"Error calculating VaR: {e}")
    return 0.0

    @ staticmethod
def calculate_conditional_value_at_risk(returns: List[float], confidence_level: float=0.95) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)."""
    try:
    pass
    if not returns:
    return 0.0

    var = FinancialUtils.calculate_value_at_risk(returns, confidence_level)
    tail_returns = [r for r in (returns if r <= var]

    for returns if r <= var)
    pass

    in ((returns if r <= var)

    for (returns if r <= var)
    pass

    in (((returns if r <= var)

    for ((returns if r <= var)
    pass

    in ((((returns if r <= var)

    for (((returns if r <= var)
    pass

    in (((((returns if r <= var)

    for ((((returns if r <= var)

    in (((((returns if r <= var)

    if not tail_returns)))))))))):
    return var

    cvar=unified_math.unified_math.mean(tail_returns)
    return float(cvar)

    except Exception as e:
    logger.error(f"Error calculating CVaR: {e}")
    return 0.0

class DataProcessingUtils:
    """Data processing utility functions."""

    @ staticmethod
def detect_outliers(data: List[float], method: str='iqr', threshold: float=1.5] -> List[bool]:
    """Detect outliers in data."""
    try:
    pass
    if not data:
    return [)

    data_array=np.array(data)
    outliers=[]

    if method == 'iqr':
    q1=np.percentile(data_array, 25)
    q3=np.percentile(data_array, 75)
    iqr=q3 - q1

    lower_bound=q1 - threshold * iqr
    upper_bound=q3 + threshold * iqr

    outliers=(data_array < lower_bound) | (data_array > upper_bound)

    elif method == 'zscore':
    mean_val=unified_math.unified_math.mean(data_array)
    std_val=unified_math.unified_math.std(data_array)

    if std_val > 0:
    z_scores=unified_math.abs((data_array - mean_val) / std_val)
    outliers=z_scores > threshold
    else:
    outliers=np.zeros_like(data_array, dtype=bool)

    return outliers.tolist()

    except Exception as e:
    logger.error(f"Error detecting outliers: {e}")
    return [False] * len(data) if data else []

    @ staticmethod
def remove_outliers(data: List[float], method: str='iqr', threshold: float=1.5] -> List[float]:
    """Remove outliers from data."""
    try:
    pass
    if not data:
    return [)

    outliers=DataProcessingUtils.detect_outliers(data, method, threshold)
    cleaned_data=[val for val, is_outlier in (zip(data, outliers) for zip(data, outliers) in ((zip(data, outliers) for (zip(data, outliers) in (((zip(data, outliers] for ((zip(data, outliers] in ((((zip(data, outliers] for (((zip(data, outliers] in (((((zip(data, outliers] for ((((zip(data, outliers] in (((((zip(data, outliers) if not is_outlier)

    return cleaned_data

    except Exception as e)))))))))):
    logger.error(f"Error removing outliers: {e}")
    return data

    @ staticmethod
def interpolate_missing_values(data: List[Optional[float]], method: str='linear'] -> List[float]:
    """Interpolate missing values in data."""
    try:
    pass
    if not data:
    return []

    # Convert to numpy array with NaN for missing values
    data_array=np.array([val if val is not None else np.nan for val in (data]]

    for data))
    pass

    in ((data))

    for (data))
    pass

    in (((data))

    for ((data))
    pass

    in ((((data))

    for (((data))
    pass

    in (((((data))

    for ((((data))

    in (((((data))

    if method == 'linear')))))))))):
    # Linear interpolation
    mask=np.isnan(data_array)
    if not mask.any():
    return data_array.tolist()

    indices=np.arange(len(data_array))
    data_array[mask]=np.interp(indices[mask), indices[~mask), data_array[~mask]]

    elif method == 'forward_fill':
    # Forward fill
    mask=np.isnan(data_array)
    if not mask.any():
    return data_array.tolist()

    for i in range(1, len(data_array)):
    if np.isnan(data_array[i]]:
    data_array[i]=data_array[i-1)

    elif method == 'backward_fill':
    # Backward fill
    mask=np.isnan(data_array)
    if not mask.any():
    return data_array.tolist()

    for i in range(len(data_array)-2, -1, -1):
    if np.isnan(data_array[i]]:
    data_array[i]=data_array[i+1)

    return data_array.tolist()

    except Exception as e:
    logger.error(f"Error interpolating missing values: {e}")
    return [0.0 if val is None else val for val in data]

    @ staticmethod
def calculate_rolling_statistics(data: List[float], window: int,
    statistic: str='mean'] -> List[float):
    """Calculate rolling statistics."""
    try:
    pass
    if len(data] < window:
    return []

    results=[]
    for i in range(window - 1, len(data)):
    window_data=data[i - window + 1:i + 1]

    if statistic == 'mean':
    result=unified_math.unified_math.mean(window_data)
    elif statistic == 'std':
    result=unified_math.unified_math.std(window_data)
    elif statistic == 'min':
    result=unified_math.unified_math.min(window_data)
    elif statistic == 'max':
    result=unified_math.unified_math.max(window_data)
    elif statistic == 'median':
    result=np.median(window_data)
    else:
    logger.warning(f"Unknown statistic: {statistic}")
    result=unified_math.unified_math.mean(window_data)

    results.append(float(result))

    return results

    except Exception as e:
    logger.error(f"Error calculating rolling statistics: {e}")
    return []

    @ staticmethod
def calculate_percentiles(data: List[float], percentiles: List[float] -> Dict[float, float):
    """Calculate percentiles of data."""
    try:
    pass
    if not data:
    return {}

    data_array=np.array(data)
    results={}

    for percentile in percentiles:
    if 0 <= percentile <= 100:
    value=np.percentile(data_array, percentile]
    results[percentile]=float(value)

    return results

    except Exception as e:
    logger.error(f"Error calculating percentiles: {e}")
    return {}

class SystemUtils:
    """System utility functions."""

    @ staticmethod
def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    try:
    pass
    info={
    'platform': platform.platform(),
    'system': platform.system(),
    'release': platform.release(),
    'version': platform.version(),
    'machine': platform.machine(),
    'processor': platform.processor(),
    'python_version': sys.version,
    'python_implementation': platform.python_implementation()
    }

    return info

    except Exception as e:
    logger.error(f"Error getting system info: {e}")
    return {}

    @ staticmethod
def get_memory_usage() -> Dict[str, float]:
    """Get memory usage information."""
    try:
    pass
    memory=psutil.virtual_memory()
    swap=psutil.swap_memory()

    return {
    'total_memory_mb': memory.total / (1024 * 1024),
    'available_memory_mb': memory.available / (1024 * 1024),
    'used_memory_mb': memory.used / (1024 * 1024),
    'memory_percent': memory.percent,
    'swap_total_mb': swap.total / (1024 * 1024),
    'swap_used_mb': swap.used / (1024 * 1024),
    'swap_percent': swap.percent
    }

    except Exception as e:
    logger.error(f"Error getting memory usage: {e}")
    return {}

    @ staticmethod
def get_cpu_usage() -> Dict[str, float]:
    """Get CPU usage information."""
    try:
    pass
    cpu_percent=psutil.cpu_percent(interval=1)
    cpu_count=psutil.cpu_count()
    cpu_freq=psutil.cpu_freq()

    return {
    'cpu_percent': cpu_percent,
    'cpu_count': cpu_count,
    'cpu_freq_mhz': cpu_freq.current if cpu_freq else 0,
    'cpu_freq_min_mhz': cpu_freq.min if cpu_freq else 0,
    'cpu_freq_max_mhz': cpu_freq.max if cpu_freq else 0
    }

    except Exception as e:
    logger.error(f"Error getting CPU usage: {e}")
    return {}

    @ staticmethod
def get_disk_usage(path: str="/") -> Dict[str, float]:
    """Get disk usage information."""
    try:
    pass
    disk=psutil.disk_usage(path)

    return {
    'total_gb': disk.total / (1024 * 1024 * 1024),
    'used_gb': disk.used / (1024 * 1024 * 1024),
    'free_gb': disk.free / (1024 * 1024 * 1024),
    'percent': disk.percent
    }

    except Exception as e:
    logger.error(f"Error getting disk usage: {e}")
    return {}

    @ staticmethod
def get_network_io() -> Dict[str, float]:
    """Get network I/O information."""
    try:
    pass
    network=psutil.net_io_counters()

    return {
    'bytes_sent_mb': network.bytes_sent / (1024 * 1024),
    'bytes_recv_mb': network.bytes_recv / (1024 * 1024),
    'packets_sent': network.packets_sent,
    'packets_recv': network.packets_recv,
    'errin': network.errin,
    'errout': network.errout,
    'dropin': network.dropin,
    'dropout': network.dropout
    }

    except Exception as e:
    logger.error(f"Error getting network I/O: {e}")
    return {}

    @ staticmethod
def get_performance_metrics() -> PerformanceMetrics:
    """Get comprehensive performance metrics."""
    try:
    pass
    memory_usage=SystemUtils.get_memory_usage()
    cpu_usage=SystemUtils.get_cpu_usage()
    disk_usage=SystemUtils.get_disk_usage()
    network_io=SystemUtils.get_network_io()

    metrics=PerformanceMetrics(
    timestamp=datetime.now(),
    cpu_usage=cpu_usage.get('cpu_percent', 0.0),
    memory_usage=memory_usage.get('memory_percent', 0.0),
    disk_usage=disk_usage.get('percent', 0.0),
    network_io=network_io
    )

    return metrics

    except Exception as e:
    logger.error(f"Error getting performance metrics: {e}")
    return PerformanceMetrics(
    timestamp=datetime.now(),
    cpu_usage=0.0,
    memory_usage=0.0,
    disk_usage=0.0,
    network_io={}
    )

class FileUtils:
    """File utility functions."""

    @ staticmethod
def ensure_directory(path: str) -> bool:
    """Ensure directory exists, create if it doesn't."""
    try:
    pass
    if not os.path.exists(path):
    os.makedirs(path)
    return True
    except Exception as e:
    logger.error(f"Error ensuring directory: {e}")
    return False

    @ staticmethod
def save_json(data: Any, file_path: str, indent: int=2) -> bool:
    """Save data to JSON file."""
    try:
    pass
    # Ensure directory exists
    directory=os.path.dirname(file_path)
    if directory and not FileUtils.ensure_directory(directory):
    return False

    with open(file_path, 'w') as f:
    json.dump(data, f, indent=indent, default=str)

    return True

    except Exception as e:
    logger.error(f"Error saving JSON: {e}")
    return False

    @ staticmethod
def load_json(file_path: str) -> Optional[Any]:
    """Load data from JSON file."""
    try:
    pass
    if not os.path.exists(file_path):
    logger.warning(f"File not found: {file_path}")
    return None

    with open(file_path, 'r') as f:
    data=json.load(f)

    return data

    except Exception as e:
    logger.error(f"Error loading JSON: {e}")
    return None

    @ staticmethod
def get_file_hash(file_path: str, algorithm: str='md5') -> Optional[str]:
    """Calculate file hash."""
    try:
    pass
    if not os.path.exists(file_path):
    return None

    hash_func=hashlib.new(algorithm)

    with open(file_path, 'rb') as f:
    for chunk in iter(lambda: f.read(4096), b""):
    hash_func.update(chunk)

    return hash_func.hexdigest()

    except Exception as e:
    logger.error(f"Error calculating file hash: {e}")
    return None

    @ staticmethod
def get_file_size(file_path: str) -> Optional[int]:
    """Get file size in bytes."""
    try:
    pass
    if not os.path.exists(file_path):
    return None

    return os.path.getsize(file_path)

    except Exception as e:
    logger.error(f"Error getting file size: {e}")
    return None

class TimingUtils:
    """Timing utility functions."""

    @ staticmethod
def time_function(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Time a function execution."""
    try:
    pass
    start_time=time.time()
    result=func(*args, **kwargs)
    end_time=time.time()

    execution_time=end_time - start_time
    return result, execution_time

    except Exception as e:
    logger.error(f"Error timing function: {e}")
    return None, 0.0

    @ staticmethod
def create_timer() -> Callable:
    """Create a timer function."""
    start_time=time.time()

def timer():
    return time.time() - start_time

    return timer

    @ staticmethod
def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    try:
    pass
    if seconds < 60:
    return f"{seconds:.2f}s"
    elif seconds < 3600:
    minutes=seconds / 60
    return f"{minutes:.1f}m"
    else:
    hours=seconds / 3600
    return f"{hours:.1f}h"

    except Exception as e:
    logger.error(f"Error formatting duration: {e}")
    return f"{seconds}s"

def main():
    """Main function for testing."""
    try:
    pass
    # Set up logging
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test mathematical utilities
    test_prices=[100, 105, 103, 108, 110, 107, 112, 115, 113, 118]
    returns=MathematicalUtils.calculate_returns(test_prices)
    volatility=MathematicalUtils.calculate_volatility(returns)
    sharpe=MathematicalUtils.calculate_sharpe_ratio(returns)
    max_dd=MathematicalUtils.calculate_max_drawdown(returns)

    safe_print("Mathematical Utils Test:")
    safe_print(f"Returns: {returns}")
    safe_print(f"Volatility: {volatility:.4f}")
    safe_print(f"Sharpe Ratio: {sharpe:.4f}")
    safe_print(f"Max Drawdown: {max_dd:.4f}")

    # Test financial utilities
    pv=FinancialUtils.calculate_present_value(1000, 0.05, 5)
    fv=FinancialUtils.calculate_future_value(1000, 0.05, 5)
    cagr=FinancialUtils.calculate_compound_annual_growth_rate(1000, 1500, 3)

    safe_print("\nFinancial Utils Test:")
    safe_print(f"Present Value: ${pv:.2f}")
    safe_print(f"Future Value: ${fv:.2f}")
    safe_print(f"CAGR: {cagr:.2%}")

    # Test data processing utilities
    outliers=DataProcessingUtils.detect_outliers(test_prices)
    cleaned_data=DataProcessingUtils.remove_outliers(test_prices)

    safe_print("\nData Processing Utils Test:")
    safe_print(f"Outliers detected: {sum(outliers)}")
    safe_print(f"Cleaned data length: {len(cleaned_data)}")

    # Test system utilities
    system_info=SystemUtils.get_system_info()
    memory_usage=SystemUtils.get_memory_usage()
    cpu_usage=SystemUtils.get_cpu_usage()

    safe_print("\nSystem Utils Test:")
    safe_print(f"Platform: {system_info.get('platform', 'Unknown')}")
    safe_print(f"Memory Usage: {memory_usage.get('memory_percent', 0):.1f}%")
    safe_print(f"CPU Usage: {cpu_usage.get('cpu_percent', 0):.1f}%")

    # Test file utilities
    test_data={'test': 'data', 'numbers': [1, 2, 3, 4, 5]}
    FileUtils.save_json(test_data, 'test_utils.json')
    loaded_data=FileUtils.load_json('test_utils.json')

    safe_print("\nFile Utils Test:")
    safe_print(f"Data saved and loaded: {loaded_data == test_data}")

    # Test timing utilities
def test_function():
    time.sleep(0.1)
    return "test"

    result, execution_time=TimingUtils.time_function(test_function)
    formatted_time=TimingUtils.format_duration(execution_time)

    safe_print("\nTiming Utils Test:")
    safe_print(f"Function result: {result}")
    safe_print(f"Execution time: {formatted_time}")

    safe_print("\nAll utility tests completed successfully!")

    except Exception as e:
    safe_print(f"Error in main: {e}")
import traceback
traceback.print_exc()

if __name__ == "__main__":
    main()
