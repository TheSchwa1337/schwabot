"""
Resonance Loop Controller (Meta-Layer)

Coordinates the meta-layer decision feedback loop per timeband. This module
implements the core recursive intelligence function:

S_{t+1} = F(S_t, G_t, F_t, g_i(t), Entropy Shift, Anomaly Phase)

It orchestrates the sync, fit, and glyph modules to drive strategy shifts.
"""

from schwabot.sync.temporal_graph_sync import TemporalConsensusGraph
from schwabot.fit.profit_gradient_oracle import ProfitGradientOracle
from schwabot.fit.glyph_lattice_core import GlyphLatticeCore, Glyph
from schwabot.fit.entropy_alignment_filter import EntropyAlignmentFilter
from schwabot.core.recursive_market_oracle import RecursiveMarketOracle
from schwabot.utils.oracle_logger import logger, load_oracle_config

import numpy as np
from typing import Dict, Any, Optional

class ResonanceLoopController:
    """
    Implements the main recursive loop F(...) for Schwabot's intelligence core.
    """
    def __init__(self, ai_nodes: list, initial_strategy: str = "default"):
        """
        Initializes the controller and its sub-modules.
        
        Args:
            ai_nodes (list): List of AI agent identifiers.
            initial_strategy (str): The starting strategy state.
        """
        self.state = {"strategy": initial_strategy, "tick": 0}
        
        # Initialize all component modules
        self.sync = TemporalConsensusGraph(initial_nodes=ai_nodes)
        self.fit = ProfitGradientOracle(ai_nodes=ai_nodes)
        self.glyph_core = GlyphLatticeCore()
        self.entropy_filter = EntropyAlignmentFilter(ai_nodes=ai_nodes)
        
        # Load Oracle configuration
        self.oracle_config = load_oracle_config()
        if not self.oracle_config:
            logger.warning("Failed to load Oracle config - running in fallback mode")
            self.oracle = None
        else:
            # Initialize the recursive market oracle
            self.oracle = RecursiveMarketOracle(
                manifold_dim=self.oracle_config['manifold']['dim'],
                max_homology_dim=self.oracle_config['topology']['max_dim'],
                num_strategies=self.oracle_config['quantum']['num_strategies']
            )
            logger.info("Oracle initialized successfully")
        
        # Add the controller itself as a node in the graph
        self.sync.add_node("ResonanceController", node_type="meta_controller")
        print("\nMETA: Resonance Loop Controller is online.")

    def process_tick(self, tick_data: Dict[str, Any]):
        """
        Processes a single time step (tick) of the entire system.
        This is the main entry point for the recursive loop.

        Args:
            tick_data (Dict[str, Any]): A dictionary containing all data for this
                                       tick, e.g., market window, AI votes,
                                       profit outcome, market entropy.
        """
        # S_t: The current state
        s_t = self.state
        current_tick = s_t['tick']
        
        # Update the recursive market oracle if available
        oracle_output = None
        if self.oracle:
            try:
                # Prepare market data for Oracle
                market_data = {
                    "volatility": tick_data.get("volatility", 0.0),
                    "drift": tick_data.get("drift", 0.0),
                    "entropy": tick_data.get("market_entropy", 0.5),
                    "timestamp": current_tick
                }
                
                # Get Oracle insights
                oracle_output = self.oracle.recursive_update(market_data)
                insight = self.oracle.get_market_insights()
                
                # Log Oracle insights
                logger.info(f"[ORACLE] Strategy Coherence: {insight.get('strategy', {}).get('coherence', 0):.3f}")
                logger.info(f"[ORACLE] Stable Topo Features: {insight.get('topology', {}).get('stable_features', 0)}")
                
                # Store Oracle metadata in tick data
                tick_data['oracle_metadata'] = {
                    "strategy_coherence": insight.get('strategy', {}).get('coherence', 0),
                    "entropy": insight.get('manifold_state', {}).get('distribution', []),
                    "regime_shift": bool(insight.get('topology', {}).get('regime_change', False))
                }
                
            except Exception as e:
                logger.error(f"Oracle update failed: {str(e)}")
        
        # F_t: Encode the current market state into a glyph
        market_window = tick_data["market_window"]
        profit = tick_data["profit_outcome"]
        glyph_metadata = {
            "profit": profit,
            "strategy": s_t["strategy"],
            "oracle_insights": insight if oracle_output else None
        }
        f_t = self.glyph_core.create_and_store_glyph(current_tick, market_window, glyph_metadata)
        
        # g_i(t): Update trust weights based on profit outcome and AI votes
        ai_votes = tick_data["ai_votes"]  # e.g., {'Claude': 0.9, '4o': 0.6}
        self.fit.update_trust_weights(profit, ai_votes)
        
        # G_t: Update the temporal graph with this tick's events
        self.sync.add_temporal_edge(
            "ResonanceController", s_t['strategy'], "strategy_execution", 
            metadata={"tick": current_tick, "profit": profit}
        )
        for ai, confidence in ai_votes.items():
            self.sync.add_temporal_edge(
                ai, s_t['strategy'], "ai_vote",
                metadata={"confidence": confidence}
            )

        # Entropy Shift: Update trust based on entropy divergence
        market_entropy = tick_data["market_entropy"]
        strategy_entropies = tick_data["strategy_entropies"]  # {'Claude': H_c, '4o': H_4o}
        entropy_divergences = self.entropy_filter.calculate_divergences(market_entropy, strategy_entropies)
        self.entropy_filter.apply_trust_decay(entropy_divergences)
        
        # Get optimal strategy from oracle if available
        optimal_strategy = None
        if self.oracle and oracle_output:
            try:
                optimal_strategy = self.oracle.get_optimal_strategy()
            except Exception as e:
                logger.error(f"Failed to get optimal strategy: {str(e)}")
        
        # Combine all inputs to decide the next state, S_{t+1}
        next_strategy = self._decide_next_strategy(
            s_t, f_t, entropy_divergences, optimal_strategy
        )
        
        # Update the state for the next iteration
        self.state = {"strategy": next_strategy, "tick": current_tick + 1}
        
        print(f"META: Tick {current_tick} processed. New strategy: '{next_strategy}'.")

    def _decide_next_strategy(
        self,
        s_t: Dict,
        f_t: Glyph,
        entropy_divergences: Dict,
        optimal_strategy: Optional[callable]
    ) -> str:
        """
        The core decision function F. Determines the next strategy.
        This is where the "intelligence" happens.
        
        Args:
            s_t (Dict): Current state S_t.
            f_t (Glyph): Current glyph F_t.
            entropy_divergences (Dict): Entropy divergence data.
            optimal_strategy (Optional[callable]): Strategy from the oracle.

        Returns:
            str: The identifier for the next strategy to be used.
        """
        # 1. Look for similar historical glyphs (market patterns)
        similar_glyphs = self.glyph_core.find_similar_glyphs(f_t, top_k=3)
        
        # 2. Analyze outcomes of those similar patterns
        if similar_glyphs:
            best_past_glyph = max(similar_glyphs, key=lambda g: g.metadata.get("profit", -np.inf))
            if best_past_glyph.metadata.get("profit", 0) > 0:
                # If a similar past pattern was profitable, consider reusing its strategy
                logger.info(f"Found similar profitable glyph (tick {best_past_glyph.tick_id}). Leaning towards strategy '{best_past_glyph.metadata['strategy']}'.")
                return best_past_glyph.metadata['strategy']

        # 3. If no clear signal from glyphs and Oracle is available, use its strategy
        if optimal_strategy and self.oracle:
            coherence = self.oracle.get_market_insights().get('strategy', {}).get('coherence', 0)
            if coherence > self.oracle_config['quantum']['coherence_threshold']:
                logger.info(f"Using Oracle strategy with coherence {coherence:.3f}")
                return f"strategy_oracle_{optimal_strategy.__name__}"
        
        # 4. Fall back to default strategy
        return s_t["strategy"]

# Example Usage:
if __name__ == '__main__':
    controller = ResonanceLoopController(ai_nodes=['Claude', '4o', 'R1'])
    
    # Simulate a tick
    tick_1_data = {
        "market_window": np.random.rand(10, 4),
        "ai_votes": {'Claude': 0.9, 'R1': 0.75},
        "profit_outcome": 300.0,
        "market_entropy": 1.5,
        "strategy_entropies": {'Claude': 1.4, '4o': 1.8, 'R1': 1.45},
        "volatility": 0.02,
        "drift": 0.001
    }
    
    controller.process_tick(tick_1_data) 