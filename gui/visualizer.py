"""
Main visualization components for Schwabot GUI
"""

import streamlit as st
import time
from datetime import datetime
from typing import Dict, Any
import numpy as np

from .controls import LoadAllocationControls
from .monitors import SystemMonitor
from .ring_analysis import RingAnalysisChart

class TradingDashboard:
    """Main trading dashboard component"""
    
    def __init__(self):
        self.controls = LoadAllocationControls()
        self.monitor = SystemMonitor()
        self.ring_analysis = RingAnalysisChart()
        
        # Initialize state
        self.state = {
            'auto_increase': False,
            'looper_active': False,
            'last_update': datetime.now()
        }
    
    def render(self):
        """Render the main dashboard"""
        st.title("Schwabot Trading System Monitor")
        
        # Sidebar controls
        with st.sidebar:
            st.header("System Controls")
            control_state = self.controls.render()
            
            # Handle auto-increase
            if control_state['auto']:
                self._handle_auto_increase()
            
            # Handle looper
            if control_state['looper']:
                self._handle_looper()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Ring analysis
            self.ring_analysis.render()
            
            # System monitor
            self.monitor.render()
        
        with col2:
            # Status indicators
            self._render_status_indicators()
            
            # Performance metrics
            self._render_performance_metrics()
        
        # Update data
        self._update_data()
    
    def _handle_auto_increase(self):
        """Handle auto-increase button press"""
        if not self.state['auto_increase']:
            self.state['auto_increase'] = True
            # Set all resources to 75%
            self.controls.state.update({
                'cpu': 75,
                'gpu': 75,
                'mem': 75,
                'disk': 75,
                'net': 75
            })
            self.controls._apply_settings()
    
    def _handle_looper(self):
        """Handle looper button press"""
        if not self.state['looper_active']:
            self.state['looper_active'] = True
            # Start profit loop hashing
            self._start_profit_loop()
    
    def _render_status_indicators(self):
        """Render system status indicators"""
        st.subheader("System Status")
        
        # Ring Buffer Status
        st.markdown("### Ring Buffer")
        ring_status = "normal" if self.ring_analysis.data['efficiency'][-1] > 80 else "warning"
        st.markdown(f"Status: {ring_status}")
        
        # Hash Chain Status
        st.markdown("### Hash Chain")
        hash_status = "normal" if self.state['looper_active'] else "warning"
        st.markdown(f"Status: {hash_status}")
        
        # Memory Pool Status
        st.markdown("### Memory Pool")
        mem_status = "normal" if self.monitor.history['mem'][-1] < 80 else "warning"
        st.markdown(f"Status: {mem_status}")
    
    def _render_performance_metrics(self):
        """Render performance metrics"""
        st.subheader("Performance Metrics")
        
        # Profit Ratio
        profit_ratio = np.mean(self.ring_analysis.data['efficiency']) if self.ring_analysis.data['efficiency'] else 0
        st.metric("Profit Ratio", f"{profit_ratio:.1f}%")
        
        # Volatility
        volatility = np.std(self.ring_analysis.data['values']) if self.ring_analysis.data['values'] else 0
        st.metric("Volatility", f"{volatility:.2f}")
        
        # Hash Stability
        hash_stability = np.mean(self.ring_analysis.data['shell_distance']) if self.ring_analysis.data['shell_distance'] else 0
        st.metric("Hash Stability", f"{hash_stability:.2f}")
        
        # System Load
        system_load = np.mean([
            self.monitor.history['cpu'][-1],
            self.monitor.history['gpu'][-1],
            self.monitor.history['mem'][-1]
        ])
        st.metric("System Load", f"{system_load:.1f}%")
    
    def _update_data(self):
        """Update all data sources"""
        # Update system monitor
        self.monitor.update()
        
        # Update ring analysis with simulated data
        if self.state['looper_active']:
            new_data = {
                'timestamps': datetime.now(),
                'values': np.random.normal(0, 1),
                'upper_band': 2,
                'lower_band': -2,
                'shell_distance': np.random.uniform(0, 1),
                'efficiency': np.random.uniform(80, 100)
            }
            self.ring_analysis.update(new_data)
    
    def _start_profit_loop(self):
        """Start the profit loop hashing process"""
        # TODO: Implement actual profit loop hashing
        pass

class AdvancedTradingDashboard(TradingDashboard):
    """Advanced trading dashboard with additional features"""
    
    def __init__(self):
        super().__init__()
        self.cross_section_data = {
            'rittleEfficiency': 0,
            'shellConvergence': 0,
            'dropZoneIntensity': 0,
            'bandCrossings': 0,
            'profitRatio': 0,
            'systemStability': 0
        }
    
    def render(self):
        """Render the advanced dashboard"""
        st.title("Advanced Trading Analysis")
        
        # Sidebar controls
        with st.sidebar:
            st.header("Advanced Controls")
            control_state = self.controls.render()
            
            # Additional controls
            st.subheader("Analysis Settings")
            self.cross_section_data['rittleEfficiency'] = st.slider(
                "Rittle Efficiency",
                min_value=0.0,
                max_value=1.0,
                value=self.cross_section_data['rittleEfficiency']
            )
            
            self.cross_section_data['shellConvergence'] = st.slider(
                "Shell Convergence",
                min_value=0.0,
                max_value=1.0,
                value=self.cross_section_data['shellConvergence']
            )
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Ring analysis with advanced features
            self.ring_analysis.render()
            
            # Add drop zones
            self.ring_analysis.add_drop_zone(-5, -2)
            self.ring_analysis.add_drop_zone(12, 15)
            
            # System monitor
            self.monitor.render()
        
        with col2:
            # Cross-section analysis
            self._render_cross_section()
            
            # Advanced metrics
            self._render_advanced_metrics()
        
        # Update data
        self._update_data()
    
    def _render_cross_section(self):
        """Render cross-section analysis"""
        st.subheader("Cross-Section Analysis")
        
        # Create grid of metrics
        cols = st.columns(3)
        
        with cols[0]:
            st.metric("Rittle Efficiency", f"{self.cross_section_data['rittleEfficiency']:.2f}")
            st.metric("Shell Convergence", f"{self.cross_section_data['shellConvergence']:.2f}")
        
        with cols[1]:
            st.metric("Drop Zone Intensity", f"{self.cross_section_data['dropZoneIntensity']:.2f}")
            st.metric("Band Crossings", f"{self.cross_section_data['bandCrossings']}")
        
        with cols[2]:
            st.metric("Profit Ratio", f"{self.cross_section_data['profitRatio']:.2f}")
            st.metric("System Stability", f"{self.cross_section_data['systemStability']:.2f}")
    
    def _render_advanced_metrics(self):
        """Render advanced performance metrics"""
        st.subheader("Advanced Metrics")
        
        # Efficiency metrics
        st.markdown("### Efficiency Metrics")
        efficiency = np.mean([
            self.cross_section_data['rittleEfficiency'],
            self.cross_section_data['shellConvergence'],
            self.cross_section_data['systemStability']
        ])
        st.metric("Overall Efficiency", f"{efficiency:.1%}")
        
        # Risk metrics
        st.markdown("### Risk Metrics")
        risk = np.mean([
            self.cross_section_data['dropZoneIntensity'],
            1 - self.cross_section_data['systemStability']
        ])
        st.metric("Risk Level", f"{risk:.1%}")
        
        # Performance metrics
        st.markdown("### Performance Metrics")
        performance = np.mean([
            self.cross_section_data['profitRatio'],
            self.cross_section_data['bandCrossings'] / 10
        ])
        st.metric("Performance Score", f"{performance:.1%}")
    
    def _update_data(self):
        """Update all data sources with advanced metrics"""
        super()._update_data()
        
        # Update cross-section data
        if self.state['looper_active']:
            self.cross_section_data.update({
                'rittleEfficiency': np.random.uniform(0.8, 0.95),
                'shellConvergence': np.random.uniform(0.85, 0.98),
                'dropZoneIntensity': np.random.uniform(0.1, 0.3),
                'bandCrossings': np.random.randint(0, 5),
                'profitRatio': np.random.uniform(0.6, 0.9),
                'systemStability': np.random.uniform(0.8, 0.95)
            }) 