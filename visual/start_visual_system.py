"""
Start Script for Schwabot Visual System

This script initializes and starts both the visual system and the WebSocket bridge
for connecting with the React dashboard.
"""

import asyncio
import threading
from schwabot.visual.visual_app import SchwabotVisualApp
from schwabot.visual.websocket_bridge import start_websocket_bridge
from schwabot.core.recursive_market_oracle import RecursiveMarketOracle

def start_visual_system():
    """Start the visual system and WebSocket bridge"""
    # Initialize the recursive market oracle
    oracle = RecursiveMarketOracle(
        manifold_dim=10,
        max_homology_dim=2,
        num_strategies=5
    )
    
    # Initialize the visual app
    visual_app = SchwabotVisualApp(oracle)
    
    # Start the WebSocket bridge in a separate thread
    bridge_thread = threading.Thread(
        target=start_websocket_bridge,
        args=(visual_app, oracle),
        daemon=True
    )
    bridge_thread.start()
    
    # Start the visual app
    visual_app.run()

if __name__ == '__main__':
    start_visual_system() 