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

from .quantum_mathlib import (
    QuantumState,
    calculate_drem_delta,
    calculate_entropy,
    calculate_kappa,
    calculate_phase_oscillation,
    calculate_kelly_criterion,
    detect_fork,
    calculate_value_evolution,
    calculate_hash_zeta,
    calculate_quantum_state,
    calculate_entanglement_entropy,
    calculate_correlation_strength,
    calculate_lyapunov_exponent,
    calculate_attention,
    calculate_policy_gradient,
    calculate_mandelbrot_iteration,
    calculate_hausdorff_dimension,
    calculate_path_integral,
    calculate_renyi_entropy,
    calculate_mutual_information,
    calculate_overlap,
    calculate_persistence_diagram,
    calculate_compression_gain,
    calculate_trigger_velocity,
    calculate_optimal_yield,
    calculate_portal_strength,
    calculate_ghost_coupling,
    calculate_buyback_probability,
    calculate_ghost_position
)

from .qutrit_processor import QutritProcessor, QutritState, QutritGate
from .edos_processor import EDOSProcessor, EDOSProfile, EDOSState

__all__ = [
    # Core math functions
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
    
    # Quantum math functions
    'QuantumState',
    'calculate_drem_delta',
    'calculate_entropy',
    'calculate_kappa',
    'calculate_phase_oscillation',
    'calculate_kelly_criterion',
    'detect_fork',
    'calculate_value_evolution',
    'calculate_hash_zeta',
    'calculate_quantum_state',
    'calculate_entanglement_entropy',
    'calculate_correlation_strength',
    'calculate_lyapunov_exponent',
    'calculate_attention',
    'calculate_policy_gradient',
    'calculate_mandelbrot_iteration',
    'calculate_hausdorff_dimension',
    'calculate_path_integral',
    'calculate_renyi_entropy',
    'calculate_mutual_information',
    'calculate_overlap',
    'calculate_persistence_diagram',
    'calculate_compression_gain',
    'calculate_trigger_velocity',
    'calculate_optimal_yield',
    'calculate_portal_strength',
    'calculate_ghost_coupling',
    'calculate_buyback_probability',
    'calculate_ghost_position',
    
    # Processors
    'QutritProcessor',
    'QutritState',
    'QutritGate',
    'EDOSProcessor',
    'EDOSProfile',
    'EDOSState'
] 