import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

class LoopMemoryRing:
    """
    Circular buffer with exponential decay weighting for SHA pattern memory.
    
    Memory weight: W(t,τ) = exp(-λ(t-τ))
    Effective memory: M_eff = Σ[W(t,τ_i) × M_i]
    """
    
    def __init__(self, size: int = 1000, decay_rate: float = 0.01) -> None:
        self.size = size
        self.decay_rate = decay_rate
        # [tick, idx42, idx81, profit, entropy]
        self.buffer = np.zeros((size, 5))
        self.pointer = 0
        self.tick_counter = 0
        
    def add(self, idx42: int, idx81: int, profit: float, entropy: float) -> None:
        """Add new entry to ring buffer."""
        self.buffer[self.pointer] = [
            self.tick_counter,
            idx42,
            idx81,
            profit,
            entropy
        ]
        self.pointer = (self.pointer + 1) % self.size
        self.tick_counter += 1
        
    def get_weighted_memory(self, idx42: int, idx81: int) -> Optional[Dict[str, Any]]:
        """Get exponentially weighted memory for SHA pair."""
        matches = self.buffer[
            (self.buffer[:, 1] == idx42) & 
            (self.buffer[:, 2] == idx81)
        ]
        
        if matches.shape[0] == 0:
            return None
            
        current_tick = self.tick_counter
        # Filter out future ticks if any (shouldn't happen but for safety)
        valid_matches = matches[matches[:, 0] < current_tick]
        if valid_matches.shape[0] == 0:
            return None

        weights = np.exp(-self.decay_rate * (current_tick - valid_matches[:, 0]))
        
        # Avoid division by zero if all weights are effectively zero
        sum_weights = np.sum(weights)
        if sum_weights < 1e-9:
            return None

        weighted_profit = np.sum(weights * valid_matches[:, 3]) / sum_weights
        weighted_entropy = np.sum(weights * valid_matches[:, 4]) / sum_weights
        
        return {
            'profit': weighted_profit,
            'entropy': weighted_entropy,
            'occurrences': valid_matches.shape[0],
            'total_weight': sum_weights
        }

class OrbitRing:
    """
    Models profit trajectories as orbital paths for SHA families.
    
    Orbit equation: r(θ) = a(1 - e²) / (1 + e×cos(θ - θ₀))
    """
    
    def __init__(self, ring_id: str, sha_family: List[str]) -> None:
        self.ring_id = ring_id
        self.sha_family = sha_family  # List of related SHA keys
        self.a: float = 0.0  # Semi-major axis (mean profit radius)
        self.e: float = 0.0  # Eccentricity (profit volatility)
        self.theta_0: float = 0.0  # Phase offset (optimal entry angle)
        
    def update_orbital_parameters(self, profit_events: List[Dict[str, Any]]) -> None:
        """Fit orbital parameters to profit distribution."""
        if not profit_events:
            self.a = 0.0
            self.e = 0.0
            self.theta_0 = 0.0
            return

        profits = [e['profit'] for e in profit_events if 'profit' in e]
        phases = [e['phase'] for e in profit_events if 'phase' in e]
        
        if not profits:
            self.a = 0.0
            self.e = 0.0
            self.theta_0 = 0.0
            return

        # Semi-major axis = mean profit radius
        self.a = float(np.mean(profits))
        
        # Eccentricity = normalized std deviation
        if self.a > 1e-6:  # Avoid division by zero
            self.e = float(np.std(profits) / self.a)
            self.e = np.clip(self.e, 0.0, 0.99)  # Keep elliptical
        else:
            self.e = 0.0
        
        # Phase offset = circular mean
        if phases:
            self.theta_0 = float(np.angle(np.mean(np.exp(1j * np.array(phases)))))
        else:
            self.theta_0 = 0.0
        
    def predict_profit_at_phase(self, theta: float) -> float:
        """Predict profit magnitude at given phase."""
        denominator = 1 + self.e * np.cos(theta - self.theta_0)
        if denominator < 1e-9:  # Avoid division by near-zero, indicates hyperbolic trajectory or bad params
            return 0.0 # Effectively no predictable profit
        
        r = self.a * (1 - self.e**2) / denominator
        return max(0.0, r)  # Profit can't be negative

class RecursiveRegistry:
    """
    Tracks hash_signature → echo_map → decay trajectory.
    Acts as a memory-bound state machine for retrigger threshold calibration.
    """
    def __init__(self, memory_ring_size: int = 1000, memory_decay_rate: float = 0.01) -> None:
        self.loop_memory_ring = LoopMemoryRing(size=memory_ring_size, decay_rate=memory_decay_rate)
        self.orbit_rings: Dict[str, OrbitRing] = {} # ring_id -> OrbitRing instance
        self.sha_to_ring_map: Dict[str, str] = {} # sha_key -> ring_id
        self.historical_profit_events: Dict[str, List[Dict[str, Any]]] = {} # sha_key -> list of profit events

    def register_sha_event(self, idx42: int, idx81: int, profit: float, entropy: float, phase: float) -> None:
        """Registers a new SHA event in the loop memory ring and updates historical profit events."""
        sha_key = f"{idx42}-{idx81}"
        self.loop_memory_ring.add(idx42, idx81, profit, entropy)
        
        if sha_key not in self.historical_profit_events:
            self.historical_profit_events[sha_key] = []
        self.historical_profit_events[sha_key].append({'profit': profit, 'phase': phase, 'timestamp': datetime.now().isoformat()})

    def get_sha_memory(self, idx42: int, idx81: int) -> Optional[Dict[str, Any]]:
        """Retrieves weighted memory for a given SHA pair."""
        return self.loop_memory_ring.get_weighted_memory(idx42, idx81)

    def create_or_update_orbit_ring(self, ring_id: str, sha_family: List[str]) -> OrbitRing:
        """Creates or updates an OrbitRing for a family of SHA keys."""
        if ring_id not in self.orbit_rings:
            self.orbit_rings[ring_id] = OrbitRing(ring_id, sha_family)
        
        # Aggregate profit events for this family
        family_profit_events: List[Dict[str, Any]] = []
        for sha_key in sha_family:
            if sha_key in self.historical_profit_events:
                family_profit_events.extend(self.historical_profit_events[sha_key])
            self.sha_to_ring_map[sha_key] = ring_id # Map SHA to its ring

        self.orbit_rings[ring_id].update_orbital_parameters(family_profit_events)
        return self.orbit_rings[ring_id]

    def get_orbit_ring(self, ring_id: str) -> Optional[OrbitRing]:
        """Retrieves an OrbitRing by its ID."""
        return self.orbit_rings.get(ring_id)

    def get_ring_for_sha(self, sha_key: str) -> Optional[OrbitRing]:
        """Retrieves the OrbitRing associated with a specific SHA key."""
        ring_id = self.sha_to_ring_map.get(sha_key)
        if ring_id:
            return self.get_orbit_ring(ring_id)
        return None

    def get_all_orbit_rings(self) -> Dict[str, OrbitRing]:
        """Returns all registered OrbitRings."""
        return self.orbit_rings 