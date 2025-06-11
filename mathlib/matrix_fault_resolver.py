"""
Matrix Fault Resolver

Handles the resolution of matrix faults and provides retry logic with exponential backoff.
"""

import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import random
from core.config import ConfigLoader, ConfigError

@dataclass
class FaultState:
    """Represents the state of a matrix fault."""
    fault_type: str
    timestamp: float
    retry_count: int
    last_retry: float
    resolved: bool

class MatrixFaultResolver:
    """Resolver for matrix faults with exponential backoff retry logic."""
    
    def __init__(self, config_path: str = "config/matrix_fault.yaml"):
        """Initialize the matrix fault resolver.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_loader = ConfigLoader()
        try:
            self.config = self.config_loader.load_yaml(config_path)
        except ConfigError as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            self.config = self.config_loader.load_yaml("config/defaults.yaml")
        
        self.max_retries = self.config.get("max_retries", 5)
        self.base_delay = self.config.get("base_delay", 1.0)
        self.max_delay = self.config.get("max_delay", 30.0)
        self.jitter_factor = self.config.get("jitter_factor", 0.1)
        
        self.active_faults: Dict[str, FaultState] = {}
        
    def register_fault(self, fault_id: str, fault_type: str) -> None:
        """Register a new matrix fault.
        
        Args:
            fault_id: Unique identifier for the fault
            fault_type: Type of fault (e.g., "collision", "drift", "timing")
        """
        self.active_faults[fault_id] = FaultState(
            fault_type=fault_type,
            timestamp=datetime.now().timestamp(),
            retry_count=0,
            last_retry=0.0,
            resolved=False
        )
        
    def calculate_retry_delay(self, fault_id: str) -> float:
        """Calculate the delay before the next retry attempt.
        
        Args:
            fault_id: ID of the fault to calculate delay for
            
        Returns:
            float: Delay in seconds before next retry
        """
        if fault_id not in self.active_faults:
            raise ValueError(f"Fault {fault_id} not found")
            
        fault = self.active_faults[fault_id]
        if fault.retry_count >= self.max_retries:
            return float('inf')
            
        # Exponential backoff with jitter
        delay = min(
            self.base_delay * (2 ** fault.retry_count),
            self.max_delay
        )
        
        # Add jitter to prevent thundering herd
        jitter = delay * self.jitter_factor
        delay += random.uniform(-jitter, jitter)
        
        return max(0.0, delay)
        
    def should_retry(self, fault_id: str) -> bool:
        """Check if a fault should be retried.
        
        Args:
            fault_id: ID of the fault to check
            
        Returns:
            bool: True if the fault should be retried
        """
        if fault_id not in self.active_faults:
            return False
            
        fault = self.active_faults[fault_id]
        if fault.resolved:
            return False
            
        return fault.retry_count < self.max_retries
        
    def mark_resolved(self, fault_id: str) -> None:
        """Mark a fault as resolved.
        
        Args:
            fault_id: ID of the fault to mark as resolved
        """
        if fault_id in self.active_faults:
            self.active_faults[fault_id].resolved = True
            
    def increment_retry(self, fault_id: str) -> None:
        """Increment the retry count for a fault.
        
        Args:
            fault_id: ID of the fault to increment retry count for
        """
        if fault_id in self.active_faults:
            fault = self.active_faults[fault_id]
            fault.retry_count += 1
            fault.last_retry = datetime.now().timestamp()
            
    def get_active_faults(self) -> Dict[str, FaultState]:
        """Get all currently active faults.
        
        Returns:
            Dict[str, FaultState]: Dictionary of active faults
        """
        return {k: v for k, v in self.active_faults.items() if not v.resolved}
        
    def reset(self) -> None:
        """Reset the fault resolver state."""
        self.active_faults.clear() 