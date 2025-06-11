"""
Matrix Fault Resolver for Schwabot v0.3
Handles safe transitions between matrix states during faults
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from core.config import load_yaml_config, ConfigError
from pathlib import Path
import time
import random

@dataclass
class FaultState:
    """State of a matrix fault"""
    timestamp: str
    fault_type: str
    severity: float
    current_matrix: str
    target_matrix: str
    transition_started: bool = False
    transition_complete: bool = False
    retry_count: int = 0

class MatrixFaultResolver:
    """
    Implements safe matrix state transitions during faults
    """
    
    def __init__(self, config_path: str = "matrix_response_paths.yaml"):
        """Initialize matrix fault resolver"""
        self.config_path = config_path
        self.current_matrix = "default"
        self.fault_history: List[FaultState] = []
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML"""
        try:
            self.config = load_yaml_config(Path(self.config_path).name)
        except ConfigError:
            self.config = {
                "retry_config": {
                    "base_delay": 1000,
                    "max_delay": 30000,
                    "backoff_factor": 2,
                    "jitter": True,
                    "jitter_factor": 0.1
                }
            }
    
    def calculate_retry_delay(self, retry_count: int) -> int:
        """
        Calculate retry delay with exponential backoff and jitter
        
        Args:
            retry_count: Number of retries attempted
            
        Returns:
            Delay in milliseconds
        """
        base_delay = self.config["retry_config"]["base_delay"]
        max_delay = self.config["retry_config"]["max_delay"]
        backoff_factor = self.config["retry_config"]["backoff_factor"]
        
        # Calculate exponential backoff
        delay = min(base_delay * (backoff_factor ** retry_count), max_delay)
        
        # Add jitter if enabled
        if self.config["retry_config"].get("jitter", False):
            jitter_factor = self.config["retry_config"].get("jitter_factor", 0.1)
            jitter = delay * jitter_factor
            delay += random.uniform(-jitter, jitter)
            
        return int(delay) 