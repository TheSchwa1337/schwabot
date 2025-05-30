"""
Scalar Laws: Implementation of chunk size optimization rules for Schwabot
Implements the Parse Cost Timeband Law (PCTL) and related optimizations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ChunkProfile:
    """Profile for chunk size performance characteristics"""
    size: int
    parse_time: float
    max_runs_per_tick: int
    confidence_threshold: float

class ScalarLaws:
    """
    Implements the Parse Cost Timeband Law (PCTL) and chunk size optimization
    for Schwabot's 3.75-minute tick windows.
    """
    
    def __init__(self):
        """Initialize scalar laws with chunk profiles"""
        self.chunk_profiles = {
            128: ChunkProfile(128, 1.55, 145, 0.65),
            256: ChunkProfile(256, 1.57, 143, 0.70),
            512: ChunkProfile(512, 1.68, 134, 0.75),
            1024: ChunkProfile(1024, 1.68, 134, 0.80),
            2048: ChunkProfile(2048, 1.72, 130, 0.85)
        }
        
        self.tick_window = 225.0  # 3.75 minutes in seconds
        self.current_tick_start = datetime.now()
        self.chunk_usage_log: List[Dict] = []
        
    def allocate_chunk_bandwidth(
        self,
        profit_likelihood: float,
        entropy_score: float,
        memkey_confidence: float
    ) -> List[int]:
        """
        Allocate chunk sizes based on profit likelihood and available bandwidth.
        
        Args:
            profit_likelihood: Probability of profitable trade
            entropy_score: Current entropy score from AlephUnitizer
            memkey_confidence: Confidence score from memory key system
            
        Returns:
            List[int]: List of chunk sizes to use in current tick
        """
        # Calculate effective confidence
        effective_confidence = (
            0.4 * profit_likelihood +
            0.3 * entropy_score +
            0.3 * memkey_confidence
        )
        
        # Determine optimal chunk size
        optimal_chunk = self._get_optimal_chunk_size(effective_confidence)
        
        # Calculate how many runs we can do in this tick
        max_runs = int(self.tick_window / self.chunk_profiles[optimal_chunk].parse_time)
        
        # Log the allocation
        self._log_chunk_allocation(
            optimal_chunk,
            max_runs,
            effective_confidence,
            profit_likelihood
        )
        
        return [optimal_chunk] * max_runs
    
    def _get_optimal_chunk_size(self, confidence: float) -> int:
        """
        Determine optimal chunk size based on confidence level.
        
        Args:
            confidence: Combined confidence score
            
        Returns:
            int: Optimal chunk size
        """
        for size, profile in sorted(
            self.chunk_profiles.items(),
            key=lambda x: x[1].confidence_threshold
        ):
            if confidence >= profile.confidence_threshold:
                return size
        return 128  # Default to smallest chunk
    
    def _log_chunk_allocation(
        self,
        chunk_size: int,
        runs: int,
        confidence: float,
        profit_likelihood: float
    ):
        """Log chunk allocation for analysis"""
        self.chunk_usage_log.append({
            "timestamp": datetime.now().isoformat(),
            "chunk_size": chunk_size,
            "runs": runs,
            "confidence": confidence,
            "profit_likelihood": profit_likelihood,
            "tick_window": self.tick_window
        })
    
    def get_chunk_metrics(self) -> Dict:
        """
        Get metrics about chunk usage.
        
        Returns:
            Dict: Chunk usage metrics
        """
        if not self.chunk_usage_log:
            return {}
            
        recent_logs = self.chunk_usage_log[-100:]  # Last 100 allocations
        
        metrics = {
            "total_allocations": len(recent_logs),
            "chunk_distribution": {},
            "avg_confidence": np.mean([log["confidence"] for log in recent_logs]),
            "avg_profit_likelihood": np.mean([log["profit_likelihood"] for log in recent_logs])
        }
        
        # Calculate chunk size distribution
        for log in recent_logs:
            size = log["chunk_size"]
            metrics["chunk_distribution"][size] = (
                metrics["chunk_distribution"].get(size, 0) + 1
            )
            
        return metrics
    
    def reset_tick_window(self):
        """Reset the tick window timer"""
        self.current_tick_start = datetime.now()
        
    def get_remaining_tick_time(self) -> float:
        """
        Get remaining time in current tick window.
        
        Returns:
            float: Remaining seconds in tick window
        """
        elapsed = (datetime.now() - self.current_tick_start).total_seconds()
        return max(0.0, self.tick_window - elapsed) 