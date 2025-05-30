"""
DREM Router: Handles execution routing and chunk size selection
Integrates with DremController and ScalarLaws for profit-optimized execution
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from .drem_controller import DremController
from .scalar_laws import ScalarLaws

class DremRouter:
    """
    Routes execution based on DREM memory and profit corridors.
    Integrates with ScalarLaws for chunk size optimization.
    """
    
    def __init__(
        self,
        drem_controller: DremController,
        scalar_laws: ScalarLaws
    ):
        """
        Initialize DREM router.
        
        Args:
            drem_controller: DREM controller instance
            scalar_laws: Scalar laws instance
        """
        self.drem = drem_controller
        self.scalar_laws = scalar_laws
        self.execution_history: List[Dict] = []
        
    def route_execution(
        self,
        hash_pattern: str,
        entropy: float,
        profit_likelihood: float,
        memkey_confidence: float,
        current_time: datetime
    ) -> Tuple[List[int], Optional[str]]:
        """
        Route execution based on DREM memory and current conditions.
        
        Args:
            hash_pattern: Current hash pattern
            entropy: Current entropy value
            profit_likelihood: Probability of profitable trade
            memkey_confidence: Confidence score from memory key
            current_time: Current timestamp
            
        Returns:
            Tuple[List[int], Optional[str]]: Chunk sizes and corridor ID if found
        """
        # First check DREM memory for preferred chunks
        preferred_chunks = self.drem.get_preferred_chunks(
            hash_pattern,
            entropy,
            current_time
        )
        
        if preferred_chunks:
            # Found a matching corridor, use its preferred chunks
            corridor_id = self._find_matching_corridor(hash_pattern, entropy)
            self._log_execution(
                preferred_chunks,
                corridor_id,
                profit_likelihood,
                entropy
            )
            return preferred_chunks, corridor_id
            
        # No matching corridor, use scalar laws
        chunks = self.scalar_laws.allocate_chunk_bandwidth(
            profit_likelihood=profit_likelihood,
            entropy_score=entropy,
            memkey_confidence=memkey_confidence
        )
        
        self._log_execution(chunks, None, profit_likelihood, entropy)
        return chunks, None
        
    def _find_matching_corridor(
        self,
        hash_pattern: str,
        entropy: float
    ) -> Optional[str]:
        """
        Find matching corridor for current conditions.
        
        Args:
            hash_pattern: Current hash pattern
            entropy: Current entropy value
            
        Returns:
            Optional[str]: Corridor ID if found
        """
        for corridor_id in self.drem.active_corridors:
            corridor = self.drem.corridors[corridor_id]
            
            if hash_pattern in corridor.hash_families:
                min_entropy, max_entropy = corridor.entropy_band
                if min_entropy <= entropy <= max_entropy:
                    return corridor_id
                    
        return None
        
    def _log_execution(
        self,
        chunks: List[int],
        corridor_id: Optional[str],
        profit_likelihood: float,
        entropy: float
    ):
        """Log execution details"""
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "chunks": chunks,
            "corridor_id": corridor_id,
            "profit_likelihood": profit_likelihood,
            "entropy": entropy
        })
        
    def get_execution_metrics(self) -> Dict:
        """
        Get execution metrics.
        
        Returns:
            Dict: Execution metrics
        """
        if not self.execution_history:
            return {}
            
        recent_history = self.execution_history[-100:]  # Last 100 executions
        
        metrics = {
            "total_executions": len(recent_history),
            "drem_routed": sum(1 for h in recent_history if h["corridor_id"] is not None),
            "avg_profit_likelihood": np.mean([h["profit_likelihood"] for h in recent_history]),
            "avg_entropy": np.mean([h["entropy"] for h in recent_history]),
            "chunk_distribution": {}
        }
        
        # Calculate chunk size distribution
        for history in recent_history:
            for chunk in history["chunks"]:
                metrics["chunk_distribution"][chunk] = (
                    metrics["chunk_distribution"].get(chunk, 0) + 1
                )
                
        return metrics
        
    def update_corridor_performance(
        self,
        corridor_id: str,
        profit: float,
        success: bool
    ):
        """
        Update corridor performance after execution.
        
        Args:
            corridor_id: Corridor identifier
            profit: Actual profit achieved
            success: Whether execution was successful
        """
        self.drem.update_corridor_stats(corridor_id, profit, success) 