"""
Greyscale Collapse Matrix
========================

Implements the Greyscale Collapse Matrix for recursive sigmoid logic in Schwabot.
Day 27: Transcendent Logic Modeling - Full implementation of Greyscale Collapse Matrix

Mathematical Core:
    C(t) = ∑ [C_raw(t)] / (1 + e^(−Ωt))
    
    M_sigmoid = ∇C · collapse_hash(t)
    
    Greyscale Collapse:
    - Sigmoid-weighted confidence for soft fade-out
    - Observer-aware collapse functions
    - Recursive sigmoid logic to avoid poor decisions
    - Gradient-based collapse validation
"""

import numpy as np
import hashlib
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class CollapseState(Enum):
    """Greyscale collapse states."""
    UNCOLLAPSED = 0
    PARTIAL = 1
    FULL = 2
    FADED = 3
    OBSERVER_LOCKED = 4
    SIGMOID_STABLE = 5


@dataclass
class GreyscaleState:
    """Individual greyscale state for collapse tracking."""
    state_id: str
    raw_value: float
    sigmoid_value: float
    confidence: float
    collapse_state: CollapseState
    omega: float = 1.0  # Collapse frequency scaler
    timestamp: datetime = field(default_factory=datetime.utcnow)
    gradient: float = 0.0
    fade_factor: float = 1.0
    
    def compute_sigmoid(self, t: float) -> float:
        """Compute sigmoid-weighted value."""
        return self.raw_value / (1 + np.exp(-self.omega * t))
    
    def get_hash(self) -> str:
        """Generate hash of current greyscale state."""
        state_str = (f"{self.raw_value:.6f}_{self.sigmoid_value:.6f}_"
                    f"{self.confidence:.6f}_{self.omega:.6f}")
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]


@dataclass
class CollapseMatrix:
    """Core greyscale collapse matrix."""
    states: List[GreyscaleState]
    omega_matrix: np.ndarray
    sigmoid_weights: np.ndarray
    collapse_hash: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Initialize collapse matrix."""
        self.matrix_size = len(self.states)
        self.gradient_matrix = self._compute_gradient_matrix()
        self.collapse_hash = self._generate_collapse_hash()
    
    def _compute_gradient_matrix(self) -> np.ndarray:
        """Compute gradient matrix for collapse validation."""
        if not self.states:
            return np.array([])
        
        gradients = np.array([state.gradient for state in self.states])
        return gradients.reshape(-1, 1)
    
    def _generate_collapse_hash(self) -> str:
        """Generate collapse hash from current state."""
        if not self.states:
            return hashlib.sha256("empty".encode()).hexdigest()[:16]
        
        state_str = "".join([state.get_hash() for state in self.states])
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]


class GreyscaleCollapseMatrix:
    """
    Greyscale Collapse Matrix implementing recursive sigmoid logic.
    
    Core Functions:
    - Sigmoid-weighted confidence computation
    - Observer-aware collapse functions
    - Gradient-based collapse validation
    - Recursive sigmoid logic for decision avoidance
    """
    
    def __init__(self, matrix_size: int = 8, omega_base: float = 1.0, 
                 sigmoid_threshold: float = 0.5):
        self.matrix_size = matrix_size
        self.omega_base = omega_base
        self.sigmoid_threshold = sigmoid_threshold
        self.states: List[GreyscaleState] = []
        self.collapse_history: List[CollapseMatrix] = []
        self.current_matrix: Optional[CollapseMatrix] = None
        self.observer_lock = False
        self.collapse_count = 0
        self.fade_cycle = 0
        
    def initialize_matrix(self) -> CollapseMatrix:
        """Initialize a new greyscale collapse matrix."""
        # Create initial greyscale states
        states = []
        for i in range(self.matrix_size):
            raw_value = np.random.rand()
            omega = self.omega_base * (1 + 0.1 * np.random.randn())
            
            state = GreyscaleState(
                state_id=f"state_{i:03d}",
                raw_value=raw_value,
                sigmoid_value=0.0,  # Will be computed
                confidence=raw_value,
                collapse_state=CollapseState.UNCOLLAPSED,
                omega=omega,
                gradient=0.0,
                fade_factor=1.0
            )
            states.append(state)
        
        # Initialize matrices
        omega_matrix = np.diag([state.omega for state in states])
        sigmoid_weights = np.random.rand(self.matrix_size, self.matrix_size)
        
        matrix = CollapseMatrix(
            states=states,
            omega_matrix=omega_matrix,
            sigmoid_weights=sigmoid_weights,
            collapse_hash=""
        )
        
        self.current_matrix = matrix
        self.collapse_history.append(matrix)
        return matrix
    
    def update_greyscale_state(self, state_id: str, raw_value: float, 
                              observer_data: Optional[Dict[str, Any]] = None) -> bool:
        """Update a greyscale state with new raw value."""
        if not self.current_matrix:
            self.initialize_matrix()
        
        # Find state
        state = None
        for s in self.current_matrix.states:
            if s.state_id == state_id:
                state = s
                break
        
        if not state:
            return False
        
        # Update raw value and compute sigmoid
        state.raw_value = raw_value
        t = self.collapse_count + self.fade_cycle * 0.1
        state.sigmoid_value = state.compute_sigmoid(t)
        
        # Update confidence based on observer data
        if observer_data:
            observer_strength = observer_data.get('strength', 0.5)
            state.confidence = raw_value * observer_strength
        else:
            state.confidence = raw_value
        
        # Compute gradient
        if len(self.collapse_history) > 1:
            prev_state = None
            for s in self.collapse_history[-2].states:
                if s.state_id == state_id:
                    prev_state = s
                    break
            
            if prev_state:
                state.gradient = state.sigmoid_value - prev_state.sigmoid_value
        
        return True
    
    def collapse_state(self, state_id: str, observer_confirmation: bool = False) -> bool:
        """Collapse a greyscale state based on sigmoid logic."""
        if not self.current_matrix:
            return False
        
        # Find state
        state = None
        for s in self.current_matrix.states:
            if s.state_id == state_id:
                state = s
                break
        
        if not state:
            return False
        
        # Observer lock check
        if self.observer_lock and not observer_confirmation:
            state.collapse_state = CollapseState.OBSERVER_LOCKED
            return False
        
        # Sigmoid collapse logic
        sigmoid_threshold = self.sigmoid_threshold
        
        if state.sigmoid_value > sigmoid_threshold and observer_confirmation:
            state.collapse_state = CollapseState.FULL
        elif state.sigmoid_value > sigmoid_threshold:
            state.collapse_state = CollapseState.PARTIAL
        elif state.sigmoid_value < sigmoid_threshold * 0.3:
            state.collapse_state = CollapseState.FADED
        else:
            state.collapse_state = CollapseState.SIGMOID_STABLE
        
        self.collapse_count += 1
        return True
    
    def compute_sigmoid_matrix(self, t: float) -> np.ndarray:
        """Compute sigmoid-weighted matrix for all states."""
        if not self.current_matrix:
            return np.array([])
        
        sigmoid_values = []
        for state in self.current_matrix.states:
            sigmoid_val = state.compute_sigmoid(t)
            sigmoid_values.append(sigmoid_val)
        
        return np.array(sigmoid_values).reshape(-1, 1)
    
    def gradient_collapse_validation(self) -> Dict[str, float]:
        """Validate collapse using gradient analysis."""
        if not self.current_matrix:
            return {'error': 'No matrix initialized'}
        
        gradients = [state.gradient for state in self.current_matrix.states]
        gradient_norm = np.linalg.norm(gradients)
        gradient_mean = np.mean(gradients)
        gradient_std = np.std(gradients)
        
        # Collapse validation based on gradient stability
        stable_gradients = sum(1 for g in gradients if abs(g) < 0.1)
        stability_ratio = stable_gradients / len(gradients) if gradients else 0.0
        
        return {
            'gradient_norm': gradient_norm,
            'gradient_mean': gradient_mean,
            'gradient_std': gradient_std,
            'stability_ratio': stability_ratio,
            'stable_gradients': stable_gradients,
            'total_states': len(gradients)
        }
    
    def recursive_sigmoid_logic(self, depth: int = 3) -> np.ndarray:
        """Apply recursive sigmoid logic for decision avoidance."""
        if not self.current_matrix:
            return np.array([])
        
        # Start with current sigmoid values
        current_sigmoid = np.array([state.sigmoid_value for state in self.current_matrix.states])
        
        for _ in range(depth):
            # Apply recursive sigmoid transformation
            # S_new = sigmoid(S_old * omega_matrix)
            transformed = np.matmul(self.current_matrix.omega_matrix, current_sigmoid)
            current_sigmoid = 1.0 / (1.0 + np.exp(-transformed))
            
            # Apply sigmoid weights
            current_sigmoid = np.matmul(self.current_matrix.sigmoid_weights, current_sigmoid)
            
            # Normalize to prevent explosion
            current_sigmoid = current_sigmoid / (np.linalg.norm(current_sigmoid) + 1e-8)
        
        return current_sigmoid
    
    def fade_out_logic(self, fade_factor: float = 0.9) -> bool:
        """Apply fade-out logic to greyscale states."""
        if not self.current_matrix:
            return False
        
        for state in self.current_matrix.states:
            # Apply fade factor
            state.fade_factor *= fade_factor
            state.sigmoid_value *= state.fade_factor
            
            # Update confidence based on fade
            state.confidence *= state.fade_factor
        
        self.fade_cycle += 1
        return True
    
    def observer_collapse_matrix(self, observer_data: Dict[str, Any]) -> bool:
        """Perform observer-aware collapse of greyscale matrix."""
        if not self.current_matrix:
            return False
        
        observer_strength = observer_data.get('strength', 0.5)
        observer_purpose = observer_data.get('purpose', 'validation')
        
        # Update states based on observer
        for state in self.current_matrix.states:
            if observer_purpose == 'validation':
                state.confidence *= observer_strength
            elif observer_purpose == 'collapse':
                state.sigmoid_value *= observer_strength
        
        # Update omega matrix based on observer strength
        self.current_matrix.omega_matrix *= observer_strength
        
        # Regenerate collapse hash
        self.current_matrix.collapse_hash = self.current_matrix._generate_collapse_hash()
        
        return True
    
    def get_collapse_summary(self) -> Dict[str, Any]:
        """Get comprehensive greyscale collapse summary."""
        if not self.current_matrix:
            return {'error': 'No matrix initialized'}
        
        gradient_stats = self.gradient_collapse_validation()
        
        # Count collapse states
        state_counts = {}
        for state_enum in CollapseState:
            state_counts[state_enum.name] = sum(
                1 for state in self.current_matrix.states 
                if state.collapse_state == state_enum
            )
        
        # Compute average sigmoid values
        avg_sigmoid = np.mean([state.sigmoid_value for state in self.current_matrix.states])
        avg_confidence = np.mean([state.confidence for state in self.current_matrix.states])
        
        return {
            'matrix_size': self.matrix_size,
            'collapse_count': self.collapse_count,
            'fade_cycle': self.fade_cycle,
            'observer_lock': self.observer_lock,
            'collapse_hash': self.current_matrix.collapse_hash,
            'state_counts': state_counts,
            'average_sigmoid': avg_sigmoid,
            'average_confidence': avg_confidence,
            'gradient_validation': gradient_stats,
            'omega_base': self.omega_base,
            'sigmoid_threshold': self.sigmoid_threshold,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def export_state(self) -> Dict[str, Any]:
        """Export current state for persistence."""
        state = {
            'matrix_size': self.matrix_size,
            'omega_base': self.omega_base,
            'sigmoid_threshold': self.sigmoid_threshold,
            'observer_lock': self.observer_lock,
            'collapse_count': self.collapse_count,
            'fade_cycle': self.fade_cycle,
            'states': {
                s.state_id: {
                    'raw_value': s.raw_value,
                    'sigmoid_value': s.sigmoid_value,
                    'confidence': s.confidence,
                    'collapse_state': s.collapse_state.value,
                    'omega': s.omega,
                    'gradient': s.gradient,
                    'fade_factor': s.fade_factor,
                    'timestamp': s.timestamp.isoformat()
                }
                for s in self.current_matrix.states if self.current_matrix
            }
        }
        
        if self.current_matrix:
            state['current_matrix'] = {
                'omega_matrix': self.current_matrix.omega_matrix.tolist(),
                'sigmoid_weights': self.current_matrix.sigmoid_weights.tolist(),
                'collapse_hash': self.current_matrix.collapse_hash,
                'timestamp': self.current_matrix.timestamp.isoformat()
            }
        
        return state


# Export main classes
__all__ = ["GreyscaleCollapseMatrix", "CollapseMatrix", "GreyscaleState", "CollapseState"] 