"""
DREM Controller: Dynamic Recursive Execution Memory system
Manages profit-aware memory overlays and execution routing for Schwabot
"""

import json
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

@dataclass
class DremCorridor:
    """Profit corridor definition for DREM"""
    corridor_id: str
    valid_for: List[str]
    hash_families: List[str]
    bandwidth_approved: bool
    profit_threshold: float
    entropy_band: Tuple[float, float]
    preferred_chunks: List[int]
    last_triggered: datetime
    success_count: int
    total_attempts: int

@dataclass
class DremSlot:
    """Temporal slot definition for DREM"""
    entropy_band: float
    expected_profit: float
    hash_seed: str
    execution_cost: float
    preferred_chunks: List[int]
    drem_link: bool
    last_updated: datetime
    profit_history: List[float]

class DremController:
    """
    Manages DREM memory overlays and profit corridors.
    Integrates with AlephUnitizer and ScalarLaws for profit-optimized execution.
    """
    
    def __init__(self, memory_log_path: Optional[Union[str, Path]] = None):
        """
        Initialize DREM controller.
        
        Args:
            memory_log_path: Path to DREM memory log file
        """
        self.corridors: Dict[str, DremCorridor] = {}
        self.temporal_slots: Dict[str, DremSlot] = {}
        self.memory_log_path = memory_log_path
        self.active_corridors: List[str] = []
        
        if memory_log_path:
            self.load_memory_log()
            
    def register_corridor(
        self,
        corridor_id: str,
        valid_for: List[str],
        hash_families: List[str],
        profit_threshold: float,
        entropy_band: Tuple[float, float],
        preferred_chunks: List[int]
    ) -> DremCorridor:
        """
        Register a new profit corridor.
        
        Args:
            corridor_id: Unique identifier for corridor
            valid_for: Time periods where corridor is valid
            hash_families: List of hash patterns that trigger corridor
            profit_threshold: Minimum profit to consider successful
            entropy_band: Valid entropy range (min, max)
            preferred_chunks: Preferred chunk sizes for this corridor
            
        Returns:
            DremCorridor: Created corridor
        """
        corridor = DremCorridor(
            corridor_id=corridor_id,
            valid_for=valid_for,
            hash_families=hash_families,
            bandwidth_approved=True,
            profit_threshold=profit_threshold,
            entropy_band=entropy_band,
            preferred_chunks=preferred_chunks,
            last_triggered=datetime.now(),
            success_count=0,
            total_attempts=0
        )
        
        self.corridors[corridor_id] = corridor
        self._save_memory_log()
        return corridor
        
    def register_temporal_slot(
        self,
        slot_id: str,
        entropy_band: float,
        expected_profit: float,
        hash_seed: str,
        execution_cost: float,
        preferred_chunks: List[int]
    ) -> DremSlot:
        """
        Register a new temporal slot.
        
        Args:
            slot_id: Unique identifier for slot
            entropy_band: Current entropy value
            expected_profit: Expected profit for this slot
            hash_seed: Hash pattern for this slot
            execution_cost: Expected execution time
            preferred_chunks: Preferred chunk sizes
            
        Returns:
            DremSlot: Created slot
        """
        slot = DremSlot(
            entropy_band=entropy_band,
            expected_profit=expected_profit,
            hash_seed=hash_seed,
            execution_cost=execution_cost,
            preferred_chunks=preferred_chunks,
            drem_link=True,
            last_updated=datetime.now(),
            profit_history=[]
        )
        
        self.temporal_slots[slot_id] = slot
        self._save_memory_log()
        return slot
        
    def update_corridor_stats(
        self,
        corridor_id: str,
        profit: float,
        success: bool
    ):
        """
        Update corridor statistics after execution.
        
        Args:
            corridor_id: Corridor identifier
            profit: Actual profit achieved
            success: Whether execution was successful
        """
        if corridor_id not in self.corridors:
            return
            
        corridor = self.corridors[corridor_id]
        corridor.total_attempts += 1
        if success:
            corridor.success_count += 1
        corridor.last_triggered = datetime.now()
        
        # Update bandwidth approval based on success rate
        success_rate = corridor.success_count / corridor.total_attempts
        corridor.bandwidth_approved = success_rate >= 0.6
        
        self._save_memory_log()
        
    def get_preferred_chunks(
        self,
        hash_pattern: str,
        entropy: float,
        current_time: datetime
    ) -> Optional[List[int]]:
        """
        Get preferred chunk sizes based on DREM memory.
        
        Args:
            hash_pattern: Current hash pattern
            entropy: Current entropy value
            current_time: Current timestamp
            
        Returns:
            Optional[List[int]]: Preferred chunk sizes if found
        """
        # Check active corridors
        for corridor_id in self.active_corridors:
            corridor = self.corridors[corridor_id]
            
            # Check if hash pattern matches
            if hash_pattern in corridor.hash_families:
                # Check entropy band
                min_entropy, max_entropy = corridor.entropy_band
                if min_entropy <= entropy <= max_entropy:
                    return corridor.preferred_chunks
                    
        return None
        
    def activate_corridor(self, corridor_id: str):
        """Activate a profit corridor"""
        if corridor_id in self.corridors:
            self.active_corridors.append(corridor_id)
            
    def deactivate_corridor(self, corridor_id: str):
        """Deactivate a profit corridor"""
        if corridor_id in self.active_corridors:
            self.active_corridors.remove(corridor_id)
            
    def _save_memory_log(self):
        """Save current state to memory log"""
        if not self.memory_log_path:
            return
            
        state = {
            "corridors": {
                cid: {
                    "corridor_id": c.corridor_id,
                    "valid_for": c.valid_for,
                    "hash_families": c.hash_families,
                    "bandwidth_approved": c.bandwidth_approved,
                    "profit_threshold": c.profit_threshold,
                    "entropy_band": c.entropy_band,
                    "preferred_chunks": c.preferred_chunks,
                    "last_triggered": c.last_triggered.isoformat(),
                    "success_count": c.success_count,
                    "total_attempts": c.total_attempts
                }
                for cid, c in self.corridors.items()
            },
            "temporal_slots": {
                sid: {
                    "entropy_band": s.entropy_band,
                    "expected_profit": s.expected_profit,
                    "hash_seed": s.hash_seed,
                    "execution_cost": s.execution_cost,
                    "preferred_chunks": s.preferred_chunks,
                    "drem_link": s.drem_link,
                    "last_updated": s.last_updated.isoformat(),
                    "profit_history": s.profit_history
                }
                for sid, s in self.temporal_slots.items()
            },
            "active_corridors": self.active_corridors
        }
        
        with open(self.memory_log_path, 'w') as f:
            json.dump(state, f, indent=2)
            
    def load_memory_log(self):
        """Load state from memory log"""
        if not self.memory_log_path or not Path(self.memory_log_path).exists():
            return
            
        with open(self.memory_log_path, 'r') as f:
            state = json.load(f)
            
        # Load corridors
        self.corridors = {
            cid: DremCorridor(
                corridor_id=data["corridor_id"],
                valid_for=data["valid_for"],
                hash_families=data["hash_families"],
                bandwidth_approved=data["bandwidth_approved"],
                profit_threshold=data["profit_threshold"],
                entropy_band=tuple(data["entropy_band"]),
                preferred_chunks=data["preferred_chunks"],
                last_triggered=datetime.fromisoformat(data["last_triggered"]),
                success_count=data["success_count"],
                total_attempts=data["total_attempts"]
            )
            for cid, data in state["corridors"].items()
        }
        
        # Load temporal slots
        self.temporal_slots = {
            sid: DremSlot(
                entropy_band=data["entropy_band"],
                expected_profit=data["expected_profit"],
                hash_seed=data["hash_seed"],
                execution_cost=data["execution_cost"],
                preferred_chunks=data["preferred_chunks"],
                drem_link=data["drem_link"],
                last_updated=datetime.fromisoformat(data["last_updated"]),
                profit_history=data["profit_history"]
            )
            for sid, data in state["temporal_slots"].items()
        }
        
        self.active_corridors = state["active_corridors"] 