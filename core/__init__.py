"""
Core package for Schwabot System
"""

from .mathlib import (
    profit_decay,
    detect_tick_sequence,
    xor_trigger,
    shard_hold_balance,
    recursive_matrix_growth,
    calculate_hash_storage,
    calculate_rotation_index,
    calculate_trade_grade,
    calculate_echo_weight,
    create_ferris_wheel_payload,
    should_purge_memory
)

__all__ = [
    'profit_decay',
    'detect_tick_sequence',
    'xor_trigger',
    'shard_hold_balance',
    'recursive_matrix_growth',
    'calculate_hash_storage',
    'calculate_rotation_index',
    'calculate_trade_grade',
    'calculate_echo_weight',
    'create_ferris_wheel_payload',
    'should_purge_memory'
] 