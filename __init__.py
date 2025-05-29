"""
Schwabot System
A highly modular, mathematically rigorous, and fault-tolerant core for trading/automation
"""

__version__ = "0.1.0"

from .core import (
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

from .scaling import (
    MonitorPortal,
    DeviceMetrics,
    ThrottleManager,
    SystemState,
    ThrottleConfig,
    HashDispatcher
)

__all__ = [
    # Core functions
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
    'should_purge_memory',
    
    # Scaling components
    'MonitorPortal',
    'DeviceMetrics',
    'ThrottleManager',
    'SystemState',
    'ThrottleConfig',
    'HashDispatcher'
] 