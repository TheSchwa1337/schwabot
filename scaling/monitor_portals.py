"""
Monitor Portals for Schwabot System
Handles CPU/GPU temperature, load, and resource monitoring
"""

import os
import psutil
import subprocess
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DeviceMetrics:
    """Container for device metrics"""
    temperature: float
    load: float
    memory_used: float
    memory_total: float
    power_usage: Optional[float] = None

class MonitorPortal:
    """Core monitoring portal for system metrics"""
    
    def __init__(self):
        self.cpu_metrics = DeviceMetrics(0.0, 0.0, 0.0, 0.0)
        self.gpu_metrics = DeviceMetrics(0.0, 0.0, 0.0, 0.0)
        self._init_sysfs_paths()
        
    def _init_sysfs_paths(self):
        """Initialize sysfs paths for monitoring"""
        self.thermal_path = Path("/sys/class/thermal")
        self.power_path = Path("/sys/class/powercap")
        
    def get_cpu_metrics(self) -> DeviceMetrics:
        """Get current CPU metrics"""
        # CPU temperature from thermal zone
        try:
            temp_raw = (self.thermal_path / "thermal_zone0/temp").read_text().strip()
            temp = float(temp_raw) / 1000.0
        except:
            temp = 0.0
            
        # CPU load
        load = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        mem = psutil.virtual_memory()
        
        self.cpu_metrics = DeviceMetrics(
            temperature=temp,
            load=load,
            memory_used=mem.used / (1024**3),  # GB
            memory_total=mem.total / (1024**3)  # GB
        )
        
        return self.cpu_metrics
        
    def get_gpu_metrics(self) -> DeviceMetrics:
        """Get current GPU metrics using nvidia-smi"""
        try:
            # Query GPU metrics
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                temp, util, mem_used, mem_total, power = map(float, result.stdout.strip().split(", "))
                
                self.gpu_metrics = DeviceMetrics(
                    temperature=temp,
                    load=util,
                    memory_used=mem_used / 1024,  # Convert MB to GB
                    memory_total=mem_total / 1024,
                    power_usage=power
                )
                
        except Exception as e:
            print(f"Error getting GPU metrics: {e}")
            
        return self.gpu_metrics
        
    def get_all_metrics(self) -> Dict[str, DeviceMetrics]:
        """Get metrics for all devices"""
        return {
            "cpu": self.get_cpu_metrics(),
            "gpu": self.get_gpu_metrics()
        }
        
    def is_safe_zone(self) -> Tuple[bool, str]:
        """Check if system is in safe operating zone"""
        metrics = self.get_all_metrics()
        
        # CPU safety checks
        if metrics["cpu"].temperature > 80:
            return False, "CPU temperature critical"
        if metrics["cpu"].load > 90:
            return False, "CPU load critical"
            
        # GPU safety checks
        if metrics["gpu"].temperature > 85:
            return False, "GPU temperature critical"
        if metrics["gpu"].load > 95:
            return False, "GPU load critical"
            
        return True, "System in safe zone" 