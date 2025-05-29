"""
Throttle Manager for Schwabot System
Handles temperature-based throttling and state management
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from .monitor_portals import MonitorPortal, DeviceMetrics

class SystemState(Enum):
    """System operating states"""
    NORMAL = "normal"
    WARM = "warm"
    HOT = "hot"
    CRITICAL = "critical"

@dataclass
class ThrottleConfig:
    """Configuration for throttling thresholds"""
    cpu_temp_thresholds: Dict[SystemState, Tuple[float, float]]
    gpu_temp_thresholds: Dict[SystemState, Tuple[float, float]]
    cpu_load_thresholds: Dict[SystemState, Tuple[float, float]]
    gpu_load_thresholds: Dict[SystemState, Tuple[float, float]]
    
    # Throttling factors for each state
    throttle_factors: Dict[SystemState, float]

class ThrottleManager:
    """Manages system throttling based on temperature and load"""
    
    def __init__(self):
        self.monitor = MonitorPortal()
        self.current_state = SystemState.NORMAL
        self.config = self._default_config()
        
    def _default_config(self) -> ThrottleConfig:
        """Create default throttling configuration"""
        return ThrottleConfig(
            cpu_temp_thresholds={
                SystemState.NORMAL: (0, 60),
                SystemState.WARM: (60, 70),
                SystemState.HOT: (70, 80),
                SystemState.CRITICAL: (80, 100)
            },
            gpu_temp_thresholds={
                SystemState.NORMAL: (0, 65),
                SystemState.WARM: (65, 75),
                SystemState.HOT: (75, 85),
                SystemState.CRITICAL: (85, 100)
            },
            cpu_load_thresholds={
                SystemState.NORMAL: (0, 60),
                SystemState.WARM: (60, 75),
                SystemState.HOT: (75, 85),
                SystemState.CRITICAL: (85, 100)
            },
            gpu_load_thresholds={
                SystemState.NORMAL: (0, 65),
                SystemState.WARM: (65, 75),
                SystemState.HOT: (75, 85),
                SystemState.CRITICAL: (85, 100)
            },
            throttle_factors={
                SystemState.NORMAL: 1.0,
                SystemState.WARM: 0.75,
                SystemState.HOT: 0.5,
                SystemState.CRITICAL: 0.25
            }
        )
        
    def _determine_state(self, metrics: Dict[str, DeviceMetrics]) -> SystemState:
        """Determine system state based on current metrics"""
        cpu_metrics = metrics["cpu"]
        gpu_metrics = metrics["gpu"]
        
        # Check CPU conditions
        for state in reversed(list(SystemState)):
            cpu_temp_lo, cpu_temp_hi = self.config.cpu_temp_thresholds[state]
            cpu_load_lo, cpu_load_hi = self.config.cpu_load_thresholds[state]
            
            if (cpu_temp_lo <= cpu_metrics.temperature < cpu_temp_hi and
                cpu_load_lo <= cpu_metrics.load < cpu_load_hi):
                return state
                
        # Check GPU conditions
        for state in reversed(list(SystemState)):
            gpu_temp_lo, gpu_temp_hi = self.config.gpu_temp_thresholds[state]
            gpu_load_lo, gpu_load_hi = self.config.gpu_load_thresholds[state]
            
            if (gpu_temp_lo <= gpu_metrics.temperature < gpu_temp_hi and
                gpu_load_lo <= gpu_metrics.load < gpu_load_hi):
                return state
                
        return SystemState.CRITICAL
        
    def update_state(self) -> Tuple[SystemState, float]:
        """Update system state and return throttle factor"""
        metrics = self.monitor.get_all_metrics()
        new_state = self._determine_state(metrics)
        
        if new_state != self.current_state:
            self.current_state = new_state
            
        throttle_factor = self.config.throttle_factors[self.current_state]
        return self.current_state, throttle_factor
        
    def get_throttle_factor(self) -> float:
        """Get current throttle factor"""
        return self.config.throttle_factors[self.current_state]
        
    def is_critical(self) -> bool:
        """Check if system is in critical state"""
        return self.current_state == SystemState.CRITICAL 