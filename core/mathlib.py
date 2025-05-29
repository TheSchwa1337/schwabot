"""
Core Math Library for Schwabot System
Implements core mathematical functions for trading/automation
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import json
from datetime import datetime
from pathlib import Path

def profit_decay(t: float, base_value: float = 0.618, curve_factor: float = 0.777) -> float:
    """Calculate profit decay over time"""
    return base_value * np.exp(-curve_factor * t)

def detect_tick_sequence(ticks: np.ndarray) -> str:
    """Detect tick sequence trend"""
    signal = np.mean(np.gradient(ticks[-5:]))
    if signal > 0.002:
        return "UPTREND"
    elif signal < -0.002:
        return "DOWNTREND"
    return "NEUTRAL"

def xor_trigger(hash_a: str, hash_b: str) -> int:
    """Calculate XOR trigger value"""
    xor_val = int(hash_a, 16) ^ int(hash_b, 16)
    return xor_val % 23

def shard_hold_balance(trades: List[Dict], base_ratio: float = 0.02) -> float:
    """Calculate shard hold balance"""
    rebal = [t['profit'] * base_ratio for t in trades if t['profit'] > 0]
    return sum(rebal)

def recursive_matrix_growth(matrix: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Calculate recursive matrix growth"""
    for i in range(iterations):
        matrix = np.dot(matrix, matrix.T) + np.eye(len(matrix))
    return matrix

def calculate_hash_storage(tokens: int = 5, scans_per_hour: int = 16) -> Dict[str, Union[int, float]]:
    """Calculate hash storage metrics"""
    hours_per_day = 24
    days_per_year = 365
    size_per_hash = 500  # bytes
    compression_factor = 4  # average compression
    
    # Daily calculations
    hashes_per_day = tokens * scans_per_hour * hours_per_day
    daily_raw_size = hashes_per_day * size_per_hash
    daily_compressed = daily_raw_size / compression_factor
    
    # Yearly calculations
    yearly_compressed = daily_compressed * days_per_year
    
    return {
        "hashes_per_day": hashes_per_day,
        "daily_raw_bytes": daily_raw_size,
        "daily_compressed_bytes": daily_compressed,
        "yearly_compressed_bytes": yearly_compressed
    }

def calculate_rotation_index(timestamp: datetime) -> int:
    """Calculate rotation index from timestamp"""
    day_of_year = timestamp.timetuple().tm_yday
    hour = timestamp.hour
    return day_of_year * 24 + hour

def calculate_trade_grade(
    entry_delta: float,
    exit_velocity: float,
    echo_weight: float,
    confidence: float,
    market_trend: str
) -> str:
    """Calculate trade grade"""
    # Base score calculation
    base_score = (
        entry_delta * 0.3 +
        exit_velocity * 0.2 +
        echo_weight * 0.2 +
        confidence * 0.2 +
        (1.0 if market_trend == "UPTREND" else 0.5) * 0.1
    )
    
    # Map to letter grade
    if base_score >= 0.9:
        return "A+"
    elif base_score >= 0.8:
        return "A"
    elif base_score >= 0.7:
        return "B+"
    elif base_score >= 0.6:
        return "B"
    elif base_score >= 0.5:
        return "C+"
    elif base_score >= 0.4:
        return "C"
    elif base_score >= 0.3:
        return "D"
    else:
        return "F"

def calculate_echo_weight(
    recurrence_score: float,
    confidence_modifier: float,
    outcome_factor: float
) -> float:
    """Calculate echo weight"""
    return recurrence_score * confidence_modifier * outcome_factor

def create_ferris_wheel_payload(
    token: str,
    price: float,
    volume: float,
    hash_value: str,
    confidence: float,
    delta: float,
    cycle_tags: List[str]
) -> Dict:
    """Create Ferris wheel hash payload"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "token": token,
        "price": price,
        "volume": volume,
        "hash": hash_value,
        "confidence": confidence,
        "delta": delta,
        "cycle": {
            "r1": True if "r1" in cycle_tags else False,
            "r2": True if "r2" in cycle_tags else False,
            "tags": cycle_tags
        }
    }

def should_purge_memory(days_old: int, echo_weight: float) -> bool:
    """Determine if memory entry should be purged"""
    return days_old > 180 and echo_weight < 0.2 