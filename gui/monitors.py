"""
System monitoring components for Schwabot GUI
"""

import streamlit as st
import psutil
import GPUtil
from typing import Dict, List
import time
from datetime import datetime

class SystemMonitor:
    """System resource monitoring and visualization"""
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.history = {
            'cpu': [],
            'gpu': [],
            'mem': [],
            'disk': [],
            'net': [],
            'timestamps': []
        }
    
    def update(self):
        """Update system metrics and history"""
        current_time = datetime.now()
        
        # Get current metrics
        metrics = {
            'cpu': psutil.cpu_percent(interval=0.5),
            'gpu': self._get_gpu_load(),
            'mem': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('/').percent,
            'net': self._get_network_load()
        }
        
        # Update history
        for key, value in metrics.items():
            self.history[key].append(value)
            if len(self.history[key]) > self.history_length:
                self.history[key].pop(0)
        
        self.history['timestamps'].append(current_time)
        if len(self.history['timestamps']) > self.history_length:
            self.history['timestamps'].pop(0)
        
        return metrics
    
    def render(self):
        """Render system monitoring dashboard"""
        st.subheader("System Resource Monitor")
        
        # Create columns for metrics
        cols = st.columns(5)
        
        # CPU Usage
        with cols[0]:
            st.metric(
                "CPU Usage",
                f"{self.history['cpu'][-1]:.1f}%",
                delta=f"{self.history['cpu'][-1] - self.history['cpu'][-2]:.1f}%" if len(self.history['cpu']) > 1 else None
            )
        
        # GPU Usage
        with cols[1]:
            st.metric(
                "GPU Usage",
                f"{self.history['gpu'][-1]:.1f}%",
                delta=f"{self.history['gpu'][-1] - self.history['gpu'][-2]:.1f}%" if len(self.history['gpu']) > 1 else None
            )
        
        # Memory Usage
        with cols[2]:
            st.metric(
                "Memory Usage",
                f"{self.history['mem'][-1]:.1f}%",
                delta=f"{self.history['mem'][-1] - self.history['mem'][-2]:.1f}%" if len(self.history['mem']) > 1 else None
            )
        
        # Disk Usage
        with cols[3]:
            st.metric(
                "Disk Usage",
                f"{self.history['disk'][-1]:.1f}%",
                delta=f"{self.history['disk'][-1] - self.history['disk'][-2]:.1f}%" if len(self.history['disk']) > 1 else None
            )
        
        # Network Usage
        with cols[4]:
            st.metric(
                "Network Usage",
                f"{self.history['net'][-1]:.1f}%",
                delta=f"{self.history['net'][-1] - self.history['net'][-2]:.1f}%" if len(self.history['net']) > 1 else None
            )
        
        # Progress bars
        st.progress(self.history['cpu'][-1] / 100, text="CPU Load")
        st.progress(self.history['gpu'][-1] / 100, text="GPU Load")
        st.progress(self.history['mem'][-1] / 100, text="Memory Usage")
        st.progress(self.history['disk'][-1] / 100, text="Disk Usage")
        st.progress(self.history['net'][-1] / 100, text="Network Load")
    
    def _get_gpu_load(self) -> float:
        """Get current GPU load percentage"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except Exception:
            pass
        return 0.0
    
    def _get_network_load(self) -> float:
        """Get current network load percentage"""
        io = psutil.net_io_counters()
        total = io.bytes_sent + io.bytes_recv
        return min(total / (1024 * 1024), 100)  # Convert to MB and cap at 100% 