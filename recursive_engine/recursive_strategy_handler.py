import numpy as np
from typing import Dict, Any, List, Optional

from schwabot.recursive_engine.recursive_registry import RecursiveRegistry, OrbitRing
from schwabot.recursive_engine.echo_hash_engine import EchoHashEngine, SHAGravitationField

class RecursiveStrategyHandler:
    """
    Central decision-making unit for recursive trading strategies.
    Leverages recursive metrics, orbital mechanics, and gravitational fields to select optimal strategies.
    """

    def __init__(self, recursive_registry: RecursiveRegistry, echo_hash_engine: EchoHashEngine) -> None:
        self.recursive_registry = recursive_registry
        self.echo_hash_engine = echo_hash_engine
        # Thresholds for strategy activation, can be made dynamic or configurable
        self.confidence_threshold: float = 0.75
        self.recall_bias_threshold: float = 0.5
        self.reentry_profit_multiplier: float = 1.2
        self.gravitational_influence_threshold: float = 0.1

    def _evaluate_ghost_shell_resonance(self, sha_key: str, gravitational_force: List[float]) -> float:
        """
        Evaluates the 'ghost shell resonance' for a given SHA key.
        This is a hypothetical metric representing the historical alignment and predictive strength
        of a pattern. Could involve: 
        - Past profit consistency (from registry)
        - Orbit eccentricity (from registry)
        - Strength of gravitational field from similar patterns (from echo_hash_engine)
        - Long-term memory decay characteristics
        """
        resonance_score = 0.0

        sha_memory = self.recursive_registry.get_sha_memory(int(sha_key.split('-')[0]), int(sha_key.split('-')[1]))
        if sha_memory:
            # Incorporate profit consistency and total weight
            resonance_score += sha_memory['profit'] * sha_memory['total_weight'] * 0.5

        orbit_ring = self.recursive_registry.get_ring_for_sha(sha_key)
        if orbit_ring and orbit_ring.a > 0:
            # Lower eccentricity (more stable orbit) implies higher resonance
            resonance_score += (1.0 - orbit_ring.e) * orbit_ring.a * 0.3
            
        # Influence from gravitational field (magnitude of force)
        resonance_score += np.linalg.norm(gravitational_force) * 0.2

        return resonance_score

    def get_strategy_recommendation(self,
                                    new_fingerprint: Dict[str, Any],
                                    current_entropy: float,
                                    current_coherence: float,
                                    current_profit_delta: float) -> Dict[str, Any]:
        """
        Provides strategy recommendations based on recursive metrics.
        This function should be called for each new market tick/event.
        """
        echo_metrics = self.echo_hash_engine.detect_echo_and_gravitate(
            new_fingerprint, current_entropy, current_coherence, current_profit_delta
        )

        sha_key = echo_metrics["sha_key"]
        echo_match_confidence = echo_metrics["echo_match_confidence"]
        recall_bias = echo_metrics["recall_bias"]
        reentry_signal = echo_metrics["reentry_signal"]
        gravitational_force = echo_metrics["gravitational_force"]

        strategy_signal = "HOLD"
        activation_score = 0.0
        recommended_action = "NONE"
        predicted_profit_potential = 0.0
        soft_loop_break_signal = False

        # Evaluate Ghost Shell Resonance (hypothetical, needs concrete definition)
        ghost_resonance = self._evaluate_ghost_shell_resonance(sha_key, gravitational_force)
        activation_score += ghost_resonance * 0.4

        # --- Core Recursive Logic & Strategy Selection ---

        # 1. High Confidence Echo Reentry
        if reentry_signal and echo_match_confidence >= self.confidence_threshold:
            strategy_signal = "REENTRY_PROFIT_LOOP"
            activation_score += echo_match_confidence * self.reentry_profit_multiplier * 0.3
            recommended_action = "BUY_OR_SELL"
            # Predict profit based on orbital mechanics if available
            orbit_ring = self.recursive_registry.get_ring_for_sha(sha_key)
            if orbit_ring:
                predicted_profit_potential = orbit_ring.predict_profit_at_phase(new_fingerprint["phase"])
            
            # Soft loop break if resonance is extremely high, indicating a strong, predictable pattern
            if ghost_resonance > 0.8: # Arbitrary high threshold
                soft_loop_break_signal = True

        # 2. Gravitational Pull towards high-profit attractors
        elif np.linalg.norm(gravitational_force) >= self.gravitational_influence_threshold and recall_bias > self.recall_bias_threshold:
            strategy_signal = "GRAVITATIONAL_DIVE"
            activation_score += np.linalg.norm(gravitational_force) * recall_bias * 0.25
            recommended_action = "ADJUST_POSITION"
            # Here, you might predict profit based on the 'mass' of the strongest attractor
            # For simplicity, we'll use a derived value based on gravitational force
            predicted_profit_potential = np.linalg.norm(gravitational_force) * 100 # Example scaling

        # 3. New Pattern Exploration (low confidence, low bias)
        elif echo_match_confidence < self.confidence_threshold and recall_bias < self.recall_bias_threshold:
            strategy_signal = "EXPLORE_NEW_PATTERN"
            activation_score += (1.0 - echo_match_confidence) * 0.1 # Encourage exploration
            recommended_action = "OBSERVE"
            predicted_profit_potential = 0.0 # Unknown profit potential

        # 4. Pattern Decay / Exit (low weighted memory, negative bias)
        elif sha_key and self.recursive_registry.get_sha_memory(int(sha_key.split('-')[0]), int(sha_key.split('-')[1])) and \
             self.recursive_registry.get_sha_memory(int(sha_key.split('-')[0]), int(sha_key.split('-')[1]))['total_weight'] < 0.1 and \
             recall_bias < 0:
            strategy_signal = "DECAY_EXIT"
            activation_score += abs(recall_bias) * 0.1
            recommended_action = "REDUCE_POSITION"
            predicted_profit_potential = 0.0
        
        # Default/Fallback
        else:
            strategy_signal = "CONSISTENCY_CHECK"
            activation_score += 0.05 # Small baseline score
            recommended_action = "HOLD"

        return {
            "strategy_signal": strategy_signal,
            "activation_score": activation_score,
            "recommended_action": recommended_action,
            "predicted_profit_potential": predicted_profit_potential,
            "soft_loop_break_signal": soft_loop_break_signal,
            "echo_metrics": echo_metrics # Include full echo metrics for detailed analysis
}