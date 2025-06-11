"""
WebSocket Bridge for Schwabot Visual Integration

This module creates a WebSocket server that bridges the Python visual system
with the React dashboard, enabling real-time data streaming and control.
"""

import asyncio
import json
import websockets
from typing import Dict, Any, Set
from schwabot.visual.visual_app import SchwabotVisualApp
from schwabot.core.recursive_market_oracle import RecursiveMarketOracle

class VisualWebSocketBridge:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.visual_app = None
        self.oracle = None
        
    async def register(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client connection"""
        self.clients.add(websocket)
        try:
            # Send initial state
            if self.visual_app and self.oracle:
                await self._send_state(websocket)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            
    async def _send_state(self, websocket: websockets.WebSocketServerProtocol):
        """Send current state to a client"""
        if not self.visual_app or not self.oracle:
            return
            
        state = {
            "patternData": self._get_pattern_data(),
            "entropyLattice": self._get_entropy_lattice(),
            "smartMoneyFlow": self._get_smart_money_flow(),
            "hookPerformance": self._get_hook_performance(),
            "tetragramMatrix": self._get_tetragram_matrix(),
            "profitTrajectory": self._get_profit_trajectory(),
            "basketState": self._get_basket_state(),
            "patternMetrics": self._get_pattern_metrics(),
            "hashMetrics": self._get_hash_metrics()
        }
        
        await websocket.send(json.dumps(state))
        
    def _get_pattern_data(self) -> list:
        """Get pattern recognition data"""
        if not self.oracle:
            return []
            
        insights = self.oracle.get_market_insights()
        return [{
            "timestamp": insights.get("timestamp", 0),
            "confidence": insights.get("confidence", 0),
            "patternType": insights.get("pattern_type", "unknown"),
            "nodes": len(insights.get("nodes", []))
        }]
        
    def _get_entropy_lattice(self) -> list:
        """Get entropy lattice data"""
        if not self.oracle:
            return []
            
        insights = self.oracle.get_market_insights()
        return [{
            "timestamp": insights.get("timestamp", 0),
            "entropy": insights.get("market_state", {}).get("entropy", 0),
            "coherence": insights.get("quantum", {}).get("coherence", 0)
        }]
        
    def _get_smart_money_flow(self) -> list:
        """Get smart money flow data"""
        if not self.oracle:
            return []
            
        insights = self.oracle.get_market_insights()
        return [{
            "timestamp": insights.get("timestamp", 0),
            "flow": insights.get("market_state", {}).get("flow", 0),
            "direction": insights.get("market_state", {}).get("direction", "neutral")
        }]
        
    def _get_hook_performance(self) -> list:
        """Get hook performance data"""
        if not self.oracle:
            return []
            
        insights = self.oracle.get_market_insights()
        return [{
            "timestamp": insights.get("timestamp", 0),
            "latency": insights.get("performance", {}).get("latency", 0),
            "accuracy": insights.get("performance", {}).get("accuracy", 0)
        }]
        
    def _get_tetragram_matrix(self) -> list:
        """Get tetragram matrix data"""
        if not self.oracle:
            return []
            
        insights = self.oracle.get_market_insights()
        return insights.get("tetragram_matrix", [])
        
    def _get_profit_trajectory(self) -> list:
        """Get profit trajectory data"""
        if not self.oracle:
            return []
            
        insights = self.oracle.get_market_insights()
        return [{
            "timestamp": insights.get("timestamp", 0),
            "entryPrice": insights.get("market_state", {}).get("entry_price", 0),
            "currentPrice": insights.get("market_state", {}).get("current_price", 0),
            "targetPrice": insights.get("market_state", {}).get("target_price", 0),
            "stopLoss": insights.get("market_state", {}).get("stop_loss", 0),
            "confidence": insights.get("confidence", 0),
            "latticePhase": insights.get("lattice_phase", "unknown")
        }]
        
    def _get_basket_state(self) -> Dict[str, float]:
        """Get basket state data"""
        if not self.oracle:
            return {"xrp": 0, "usdc": 0, "btc": 0, "eth": 0}
            
        insights = self.oracle.get_market_insights()
        return insights.get("basket_state", {"xrp": 0, "usdc": 0, "btc": 0, "eth": 0})
        
    def _get_pattern_metrics(self) -> Dict[str, float]:
        """Get pattern metrics data"""
        if not self.oracle:
            return {
                "successRate": 0,
                "averageProfit": 0,
                "patternFrequency": 0,
                "cooldownEfficiency": 0
            }
            
        insights = self.oracle.get_market_insights()
        return insights.get("pattern_metrics", {
            "successRate": 0,
            "averageProfit": 0,
            "patternFrequency": 0,
            "cooldownEfficiency": 0
        })
        
    def _get_hash_metrics(self) -> Dict[str, float]:
        """Get hash metrics data"""
        if not self.oracle:
            return {
                "hashCount": 0,
                "patternConfidence": 0,
                "collisionRate": 0,
                "tetragramDensity": 0,
                "gpuUtilization": 0,
                "bitPatternStrength": 0,
                "longDensity": 0,
                "midDensity": 0,
                "shortDensity": 0,
                "currentTier": 0
            }
            
        insights = self.oracle.get_market_insights()
        return insights.get("hash_metrics", {
            "hashCount": 0,
            "patternConfidence": 0,
            "collisionRate": 0,
            "tetragramDensity": 0,
            "gpuUtilization": 0,
            "bitPatternStrength": 0,
            "longDensity": 0,
            "midDensity": 0,
            "shortDensity": 0,
            "currentTier": 0
        })
        
    async def broadcast_state(self):
        """Broadcast current state to all connected clients"""
        if not self.clients:
            return
            
        websockets_to_remove = set()
        for websocket in self.clients:
            try:
                await self._send_state(websocket)
            except websockets.exceptions.ConnectionClosed:
                websockets_to_remove.add(websocket)
                
        # Remove closed connections
        self.clients -= websockets_to_remove
        
    def set_visual_app(self, app: SchwabotVisualApp):
        """Set the visual app instance"""
        self.visual_app = app
        
    def set_oracle(self, oracle: RecursiveMarketOracle):
        """Set the oracle instance"""
        self.oracle = oracle
        
    async def start(self):
        """Start the WebSocket server"""
        async with websockets.serve(self.register, self.host, self.port):
            print(f"WebSocket server started at ws://{self.host}:{self.port}")
            while True:
                await self.broadcast_state()
                await asyncio.sleep(0.1)  # Update every 100ms
                
def start_websocket_bridge(visual_app: SchwabotVisualApp, oracle: RecursiveMarketOracle):
    """Start the WebSocket bridge"""
    bridge = VisualWebSocketBridge()
    bridge.set_visual_app(visual_app)
    bridge.set_oracle(oracle)
    
    asyncio.get_event_loop().run_until_complete(bridge.start()) 