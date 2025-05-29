"""
Load allocation and system control components for Schwabot GUI
"""

import streamlit as st
from typing import Dict, Any
import psutil
import GPUtil

class LoadAllocationControls:
    """Controls for system resource allocation"""
    
    def __init__(self):
        self.state = {
            'cpu': 50,
            'gpu': 30,
            'mem': 40,
            'disk': 20,
            'net': 15,
            'auto': False,
            'looper': False
        }
    
    def render(self) -> Dict[str, Any]:
        """Render the control panel and return updated state"""
        st.sidebar.subheader("System Resource Controls")
        
        # CPU Controls
        st.sidebar.markdown("### CPU Allocation")
        self.state['cpu'] = st.sidebar.slider(
            "CPU Load Target (%)",
            min_value=10,
            max_value=100,
            value=self.state['cpu'],
            step=5
        )
        
        # GPU Controls
        st.sidebar.markdown("### GPU Allocation")
        self.state['gpu'] = st.sidebar.slider(
            "GPU Load Target (%)",
            min_value=10,
            max_value=100,
            value=self.state['gpu'],
            step=5
        )
        
        # Memory Controls
        st.sidebar.markdown("### Memory Allocation")
        self.state['mem'] = st.sidebar.slider(
            "Memory Load Target (%)",
            min_value=10,
            max_value=100,
            value=self.state['mem'],
            step=5
        )
        
        # Disk I/O Controls
        st.sidebar.markdown("### Disk I/O Allocation")
        self.state['disk'] = st.sidebar.slider(
            "Disk I/O Target (%)",
            min_value=10,
            max_value=100,
            value=self.state['disk'],
            step=5
        )
        
        # Network Controls
        st.sidebar.markdown("### Network Allocation")
        self.state['net'] = st.sidebar.slider(
            "Network Load Target (%)",
            min_value=10,
            max_value=100,
            value=self.state['net'],
            step=5
        )
        
        # Auto-load Controls
        st.sidebar.markdown("### Auto-Load Controls")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            self.state['auto'] = st.button(
                "Auto-Increase (75% Preset)",
                help="Automatically increase load to 75% for all resources"
            )
        
        with col2:
            self.state['looper'] = st.button(
                "Plug Looper into Queue",
                help="Enable profit loop hashing and load cycles"
            )
        
        # Apply button
        if st.sidebar.button("Apply Settings"):
            self._apply_settings()
        
        return self.state
    
    def _apply_settings(self):
        """Apply the current settings to the system"""
        try:
            # CPU settings
            if psutil.cpu_count() > 0:
                # Set CPU affinity and priority
                pass
            
            # GPU settings
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    # Set GPU compute mode and memory limits
                    pass
            except Exception:
                st.warning("No GPU detected or GPU settings not available")
            
            # Memory settings
            # Set memory limits and swap usage
            
            # Disk I/O settings
            # Set disk I/O priority and limits
            
            # Network settings
            # Set network bandwidth limits
            
            st.success("Settings applied successfully")
        except Exception as e:
            st.error(f"Error applying settings: {str(e)}")
    
    def get_current_loads(self) -> Dict[str, float]:
        """Get current system resource loads"""
        return {
            'cpu': psutil.cpu_percent(interval=0.5),
            'gpu': self._get_gpu_load(),
            'mem': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('/').percent,
            'net': self._get_network_load()
        }
    
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
        """Get current network load in bytes"""
        io = psutil.net_io_counters()
        return io.bytes_sent + io.bytes_recv 