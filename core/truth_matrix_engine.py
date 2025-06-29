"""
Truth Matrix Engine
==================

Implements the U = R·C·P formal recursion model for Schwabot's identity-layer math.
Day 18: Stacked Math Lattice - Structured symbolic math into truth_matrix_engine.py

Mathematical Core:
    U = R·C·P where:
    - R = Recursive Superposition
    - C = Conscious Observer (CPU/GPU)
    - P = Purposeful Collapse
    
    Extended: U = R·C·P(E=mc²)
    
    Recursive Identity Logic:
    - Truth gates for decision validation
    - Observer-aware collapse functions
    - Purposeful state transitions
    - Energy-mass equivalence in trading context
"""

import numpy as np
import hashlib
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class TruthState(Enum):
    """Truth matrix states for recursive validation."""
    UNKNOWN = 0
    VALID = 1
    INVALID = 2
    SUPERPOSITION = 3
    COLLAPSED = 4
    OBSERVER_LOCKED = 5

@dataclass
class TruthMatrix:
    """Core truth matrix with R·C·P components."""
    R: np.ndarray  # Recursive Superposition
    C: np.ndarray  # Conscious Observer
    P: np.ndarray  # Purposeful Collapse
    E: float = 1.0  # Energy-mass equivalence (E=mc²)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate matrix dimensions and compute U."""
        if self.R.shape != self.C.shape or self.C.shape != self.P.shape:
            raise ValueError("All matrices must have same dimensions")
        self.U = self.compute_U()
    
    def compute_U(self) -> np.ndarray:
        """Compute U = R·C·P(E=mc²)."""
        # U = R·C·P·E where E is the energy-mass scaling factor
        U = np.matmul(np.matmul(self.R, self.C), self.P) * self.E
        return U
    
    def get_hash(self) -> str:
        """Generate hash of current truth state."""
        state_str = f"{self.R.tobytes()}{self.C.tobytes()}{self.P.tobytes()}{self.E}"
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]

@dataclass
class TruthGate:
    """Individual truth gate for decision validation."""
    gate_id: str
    input_hash: str
    output_hash: str
    confidence: float
    state: TruthState
    energy: float = 1.0
    mass: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def validate_emc2(self) -> bool:
        """Validate E=mc² equivalence."""
        c_squared = 299792458 ** 2  # Speed of light squared
        return abs(self.energy - self.mass * c_squared) < 1e-6

class TruthMatrixEngine:
    """
    Truth Matrix Engine implementing U = R·C·P formal recursion model.
    
    Core Functions:
    - Recursive superposition management
    - Observer-aware collapse functions
    - Purposeful state transitions
    - Energy-mass equivalence validation
    """
    
    def __init__(self, matrix_size: int = 8, energy_threshold: float = 0.5):
        self.matrix_size = matrix_size
        self.energy_threshold = energy_threshold
        self.truth_gates: Dict[str, TruthGate] = {}
        self.matrix_history: List[TruthMatrix] = []
        self.current_matrix: Optional[TruthMatrix] = None
        self.observer_lock = False
        self.collapse_count = 0
        
    def initialize_matrix(self) -> TruthMatrix:
        """Initialize a new truth matrix with random superposition."""
        # R: Recursive Superposition (random complex values)
        R = np.random.rand(self.matrix_size, self.matrix_size) + \
            1j * np.random.rand(self.matrix_size, self.matrix_size)
        
        # C: Conscious Observer (identity matrix with noise)
        C = np.eye(self.matrix_size) + 0.1 * np.random.randn(self.matrix_size, self.matrix_size)
        
        # P: Purposeful Collapse (diagonal with purpose weights)
        P = np.diag(np.random.rand(self.matrix_size))
        
        # E: Energy-mass equivalence (E=mc²)
        E = 1.0  # Base energy unit
        
        matrix = TruthMatrix(R=R, C=C, P=P, E=E)
        self.current_matrix = matrix
        self.matrix_history.append(matrix)
        return matrix
    
    def create_truth_gate(self, input_data: str, purpose: str = "validation") -> TruthGate:
        """Create a new truth gate for decision validation."""
        timestamp_str = datetime.utcnow().isoformat()
        gate_id = hashlib.sha256(
            f"{input_data}{purpose}{timestamp_str}".encode()
        ).hexdigest()[:8]
        
        # Generate input and output hashes
        input_hash = hashlib.sha256(input_data.encode()).hexdigest()[:16]
        output_hash = hashlib.sha256(f"{input_hash}{purpose}".encode()).hexdigest()[:16]
        
        # Compute confidence based on matrix state
        confidence = self._compute_gate_confidence(input_hash, output_hash)
        
        # Determine initial state
        state = TruthState.SUPERPOSITION if confidence < self.energy_threshold else TruthState.VALID
        
        # Energy-mass equivalence (simplified trading context)
        energy = confidence * 1.0  # 1.0 = 1 unit of trading energy
        mass = energy / (299792458 ** 2)  # E = mc²
        
        gate = TruthGate(
            gate_id=gate_id,
            input_hash=input_hash,
            output_hash=output_hash,
            confidence=confidence,
            state=state,
            energy=energy,
            mass=mass
        )
        
        self.truth_gates[gate_id] = gate
        return gate
    
    def _compute_gate_confidence(self, input_hash: str, output_hash: str) -> float:
        """Compute confidence score for truth gate."""
        if not self.current_matrix:
            return 0.0
        
        # Use matrix U to compute confidence
        U = self.current_matrix.U
        
        # Convert hashes to numerical values
        input_val = int(input_hash[:8], 16) / (16**8)
        output_val = int(output_hash[:8], 16) / (16**8)
        
        # Compute confidence using matrix trace and hash correlation
        trace_U = np.trace(U)
        hash_correlation = abs(input_val - output_val)
        
        confidence = (trace_U.real + trace_U.imag) / (2 * self.matrix_size) * (1 - hash_correlation)
        return max(0.0, min(1.0, confidence))
    
    def collapse_gate(self, gate_id: str, observer_confirmation: bool = False) -> bool:
        """Collapse a truth gate based on observer confirmation."""
        if gate_id not in self.truth_gates:
            return False
        
        gate = self.truth_gates[gate_id]
        
        if gate.state == TruthState.COLLAPSED:
            return True  # Already collapsed
        
        # Observer lock check
        if self.observer_lock and not observer_confirmation:
            gate.state = TruthState.OBSERVER_LOCKED
            return False
        
        # Collapse based on confidence and observer
        if gate.confidence > self.energy_threshold and observer_confirmation:
            gate.state = TruthState.VALID
        elif gate.confidence > self.energy_threshold:
            gate.state = TruthState.COLLAPSED
        else:
            gate.state = TruthState.INVALID
        
        self.collapse_count += 1
        return True
    
    def recursive_superposition(self, depth: int = 3) -> np.ndarray:
        """Apply recursive superposition to current matrix."""
        if not self.current_matrix:
            self.initialize_matrix()
        
        R = self.current_matrix.R.copy()
        
        for _ in range(depth):
            # Recursive application: R = R·R + noise
            R = np.matmul(R, R) + 0.1 * np.random.randn(*R.shape)
            # Normalize to prevent explosion
            R = R / np.linalg.norm(R)
        
        return R
    
    def observer_collapse(self, observer_data: Dict[str, Any]) -> bool:
        """Perform observer-aware collapse of truth matrix."""
        if not self.current_matrix:
            return False
        
        # Observer data influences collapse
        observer_strength = observer_data.get('strength', 0.5)
        observer_purpose = observer_data.get('purpose', 'validation')
        
        # Update conscious observer matrix C
        C_new = self.current_matrix.C.copy()
        C_new *= observer_strength
        
        # Update purposeful collapse matrix P
        P_new = self.current_matrix.P.copy()
        purpose_weight = 1.0 if observer_purpose == 'validation' else 0.5
        P_new *= purpose_weight
        
        # Create new matrix with observer influence
        new_matrix = TruthMatrix(
            R=self.current_matrix.R,
            C=C_new,
            P=P_new,
            E=self.current_matrix.E
        )
        
        self.current_matrix = new_matrix
        self.matrix_history.append(new_matrix)
        
        return True
    
    def energy_mass_validation(self) -> Dict[str, float]:
        """Validate energy-mass equivalence across all gates."""
        total_energy = 0.0
        total_mass = 0.0
        valid_gates = 0
        
        for gate in self.truth_gates.values():
            if gate.validate_emc2():
                total_energy += gate.energy
                total_mass += gate.mass
                valid_gates += 1
        
        return {
            'total_energy': total_energy,
            'total_mass': total_mass,
            'valid_gates': valid_gates,
            'emc2_ratio': total_energy / (total_mass * (299792458 ** 2)) if total_mass > 0 else 0.0
        }
    
    def get_truth_summary(self) -> Dict[str, Any]:
        """Get comprehensive truth matrix summary."""
        if not self.current_matrix:
            return {'error': 'No matrix initialized'}
        
        emc2_stats = self.energy_mass_validation()
        
        return {
            'matrix_size': self.matrix_size,
            'current_hash': self.current_matrix.get_hash(),
            'collapse_count': self.collapse_count,
            'observer_lock': self.observer_lock,
            'truth_gates_count': len(self.truth_gates),
            'matrix_history_length': len(self.matrix_history),
            'emc2_validation': emc2_stats,
            'U_trace': np.trace(self.current_matrix.U),
            'U_determinant': np.linalg.det(self.current_matrix.U),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def export_state(self) -> Dict[str, Any]:
        """Export current state for persistence."""
        state = {
            'matrix_size': self.matrix_size,
            'energy_threshold': self.energy_threshold,
            'observer_lock': self.observer_lock,
            'collapse_count': self.collapse_count,
            'truth_gates': {
                gate_id: {
                    'input_hash': gate.input_hash,
                    'output_hash': gate.output_hash,
                    'confidence': gate.confidence,
                    'state': gate.state.value,
                    'energy': gate.energy,
                    'mass': gate.mass,
                    'timestamp': gate.timestamp.isoformat()
                }
                for gate_id, gate in self.truth_gates.items()
            }
        }
        
        if self.current_matrix:
            state['current_matrix'] = {
                'R': self.current_matrix.R.tolist(),
                'C': self.current_matrix.C.tolist(),
                'P': self.current_matrix.P.tolist(),
                'E': self.current_matrix.E,
                'U': self.current_matrix.U.tolist(),
                'timestamp': self.current_matrix.timestamp.isoformat()
            }
        
        return state

# Export main classes
__all__ = ["TruthMatrixEngine", "TruthMatrix", "TruthGate", "TruthState"] 