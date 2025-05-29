"""
Ring analysis and visualization components for Schwabot GUI
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

class RingAnalysisChart:
    """Ring analysis visualization component"""
    
    def __init__(self):
        self.data = {
            'timestamps': [],
            'values': [],
            'upper_band': [],
            'lower_band': [],
            'shell_distance': [],
            'efficiency': []
        }
    
    def update(self, new_data: Dict[str, Any]):
        """Update ring analysis data"""
        for key in self.data.keys():
            if key in new_data:
                self.data[key].append(new_data[key])
                if len(self.data[key]) > 100:  # Keep last 100 points
                    self.data[key].pop(0)
    
    def render(self):
        """Render ring analysis visualization"""
        st.subheader("Ring Value Analysis")
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add main value line
        fig.add_trace(
            go.Scatter(
                x=self.data['timestamps'],
                y=self.data['values'],
                name="Value",
                line=dict(color='#8884d8')
            ),
            secondary_y=False
        )
        
        # Add upper and lower bands
        fig.add_trace(
            go.Scatter(
                x=self.data['timestamps'],
                y=self.data['upper_band'],
                name="Upper Band",
                line=dict(color='#82ca9d', dash='dash')
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.data['timestamps'],
                y=self.data['lower_band'],
                name="Lower Band",
                line=dict(color='#ff7f0e', dash='dash')
            ),
            secondary_y=False
        )
        
        # Add shell distance
        fig.add_trace(
            go.Scatter(
                x=self.data['timestamps'],
                y=self.data['shell_distance'],
                name="Shell Distance",
                line=dict(color='#ff0000')
            ),
            secondary_y=True
        )
        
        # Add efficiency
        fig.add_trace(
            go.Scatter(
                x=self.data['timestamps'],
                y=self.data['efficiency'],
                name="Efficiency",
                line=dict(color='#00ff00')
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Ring Value Analysis with Shell Metrics",
            xaxis_title="Time",
            hovermode="x unified",
            showlegend=True
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Value", secondary_y=False)
        fig.update_yaxes(title_text="Shell Metrics", secondary_y=True)
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Value",
                f"{self.data['values'][-1]:.2f}" if self.data['values'] else "N/A"
            )
        
        with col2:
            st.metric(
                "Shell Distance",
                f"{self.data['shell_distance'][-1]:.2f}" if self.data['shell_distance'] else "N/A"
            )
        
        with col3:
            st.metric(
                "Efficiency",
                f"{self.data['efficiency'][-1]:.1f}%" if self.data['efficiency'] else "N/A"
            )
    
    def add_drop_zone(self, start: float, end: float, color: str = 'rgba(255,0,0,0.2)'):
        """Add a drop zone to the visualization"""
        fig = go.Figure()
        
        # Add drop zone
        fig.add_shape(
            type="rect",
            x0=min(start, end),
            x1=max(start, end),
            y0=min(self.data['values']),
            y1=max(self.data['values']),
            fillcolor=color,
            opacity=0.2,
            layer="below",
            line_width=0
        )
        
        st.plotly_chart(fig, use_container_width=True) 