"""
Strategy Manager for Schwabot
Handles integration and management of multiple trading strategies
"""

from typing import Dict, List, Callable, Any
from dataclasses import dataclass
import threading
import time
from .drem_strategy import DREMStrategy, DREMState

@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    enabled: bool = False
    weight: float = 1.0
    parameters: Dict[str, Any] = None

class StrategyManager:
    """
    Manages multiple trading strategies and their integration
    """
    
    def __init__(self):
        self.strategies = {
            'drem': {
                'instance': DREMStrategy(),
                'config': StrategyConfig(
                    enabled=False,
                    weight=1.0,
                    parameters={
                        'dimensions': (50, 50),
                        'entropy_threshold': 0.5
                    }
                )
            }
        }
        
        self.subscribers = {
            'drem': []
        }
        
        self.running = False
        self.update_thread = None
    
    def start(self):
        """Start the strategy manager"""
        if self.running:
            return
            
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def stop(self):
        """Stop the strategy manager"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
    
    def _update_loop(self):
        """Main update loop for strategies"""
        while self.running:
            for strategy_name, strategy_data in self.strategies.items():
                if not strategy_data['config'].enabled:
                    continue
                    
                if strategy_name == 'drem':
                    state = strategy_data['instance'].apply_recursion(
                        int(time.time() * 1000)
                    )
                    self._notify_subscribers('drem', state)
            
            time.sleep(0.1)  # 100ms update interval
    
    def subscribe_to_drem(self, callback: Callable[[DREMState], None]):
        """Subscribe to DREM strategy updates"""
        self.subscribers['drem'].append(callback)
        return Subscription(self.subscribers['drem'], callback)
    
    def _notify_subscribers(self, strategy_name: str, state: Any):
        """Notify subscribers of strategy updates"""
        for callback in self.subscribers[strategy_name]:
            try:
                callback(state)
            except Exception as e:
                print(f"Error in strategy subscriber: {e}")
    
    def get_drem_dimensions(self) -> int:
        """Get current DREM field dimensions"""
        return self.strategies['drem']['config'].parameters['dimensions'][0]
    
    def set_drem_dimensions(self, dim: int):
        """Set DREM field dimensions"""
        self.strategies['drem']['config'].parameters['dimensions'] = (dim, dim)
        self.strategies['drem']['instance'] = DREMStrategy(dimensions=(dim, dim))
    
    def get_drem_entropy_threshold(self) -> float:
        """Get current DREM entropy threshold"""
        return self.strategies['drem']['config'].parameters['entropy_threshold']
    
    def set_drem_entropy_threshold(self, threshold: float):
        """Set DREM entropy threshold"""
        self.strategies['drem']['config'].parameters['entropy_threshold'] = threshold
    
    def reset_drem(self):
        """Reset DREM strategy state"""
        dims = self.strategies['drem']['config'].parameters['dimensions']
        self.strategies['drem']['instance'] = DREMStrategy(dimensions=dims)
    
    def toggle_drem(self):
        """Toggle DREM strategy enabled state"""
        self.strategies['drem']['config'].enabled = not self.strategies['drem']['config'].enabled
    
    def is_drem_enabled(self) -> bool:
        """Check if DREM strategy is enabled"""
        return self.strategies['drem']['config'].enabled
    
    def get_combined_signal(self) -> Dict[str, Any]:
        """Get combined trading signal from all enabled strategies"""
        signals = []
        total_weight = 0.0
        
        for strategy_name, strategy_data in self.strategies.items():
            if not strategy_data['config'].enabled:
                continue
                
            if strategy_name == 'drem':
                signal = strategy_data['instance'].get_strategy_signal()
                signals.append((signal, strategy_data['config'].weight))
                total_weight += strategy_data['config'].weight
        
        if not signals:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reason': ['No active strategies']
            }
        
        # Weighted average of signals
        weighted_action = 0.0
        weighted_confidence = 0.0
        reasons = []
        
        for signal, weight in signals:
            action_value = {
                'BUY': 1.0,
                'SELL': -1.0,
                'HOLD': 0.0
            }[signal['action']]
            
            weighted_action += action_value * weight
            weighted_confidence += signal['confidence'] * weight
            reasons.extend(signal['reason'])
        
        weighted_action /= total_weight
        weighted_confidence /= total_weight
        
        # Determine final action
        if abs(weighted_action) < 0.3:
            final_action = 'HOLD'
        else:
            final_action = 'BUY' if weighted_action > 0 else 'SELL'
        
        return {
            'action': final_action,
            'confidence': weighted_confidence,
            'reason': reasons
        }

class Subscription:
    """Subscription object for strategy updates"""
    
    def __init__(self, subscribers: List[Callable], callback: Callable):
        self.subscribers = subscribers
        self.callback = callback
    
    def unsubscribe(self):
        """Unsubscribe from strategy updates"""
        if self.callback in self.subscribers:
            self.subscribers.remove(self.callback) 