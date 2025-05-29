"""
Schwabot GUI Application
Main entry point for the Schwabot trading system visualization
"""

import streamlit as st
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schwabot.gui.visualizer import TradingDashboard, AdvancedTradingDashboard

def main():
    """Main entry point for the Schwabot GUI application"""
    st.set_page_config(
        page_title="Schwabot Trading System",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize the dashboard
    dashboard = AdvancedTradingDashboard()
    
    # Render the dashboard
    dashboard.render()

if __name__ == "__main__":
    main() 