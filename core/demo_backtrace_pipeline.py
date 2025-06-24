#!/usr/bin/env python3
"""
Demo Backtrace Pipeline - Trade Hash Replay and Recursive Path Logic
==================================================================

This module implements advanced backtrace functionality for Schwabot,
including trade hash replay, recursive path analysis, and tick window
rebuilding for demo and testing scenarios.

Core Mathematical Functions:
- Trade Hash Replay: H(t) = hash(price_t, vector_state_t)
- Recursive Path Logic: Ψ_backtrace = ∑(exit_vector_i · reentry_signal_i)
- Tick Window Rebuild: τ(t) = tₙ - t₀; for n in replay range
"""

import numpy as np
import hashlib
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from schwabot.core.multi_bit_btc_processor import MultiBitBTCProcessor
    from schwabot.mathlib.sfsss_tensor import SFSSTensor
    from schwabot.mathlib.ufs_tensor import UFSTensor
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    # Create mock classes for testing
    MultiBitBTCProcessor = type('MultiBitBTCProcessor', (), {})
    SFSSTensor = type('SFSSTensor', (), {})
    UFSTensor = type('UFSTensor', (), {})

logger = logging.getLogger(__name__)

@dataclass
class TradeEvent:
    """Trade event structure for backtrace analysis."""
    timestamp: datetime
    price: float
    volume: float
    event_type: str  # 'entry', 'exit', 'signal'
    vector_state: Dict[str, float]
    hash_value: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktraceConfig:
    """Backtrace configuration."""
    replay_window: int = 1000  # Number of ticks to replay
    hash_precision: int = 8    # Decimal places for hash calculation
    vector_dimensions: int = 16  # Dimensions for vector state
    recursion_depth: int = 5   # Maximum recursion depth
    similarity_threshold: float = 0.85  # Similarity threshold for pattern matching
    cache_size: int = 10000    # Cache size for hash lookups

class TradeHashReplay:
    """Trade hash replay engine."""
    
    def __init__(self, config: BacktraceConfig):
        self.config = config
        self.hash_cache: Dict[str, List[TradeEvent]] = {}
        self.vector_cache: Dict[str, np.ndarray] = {}
        self.replay_history: List[TradeEvent] = []
        
    def calculate_trade_hash(self, price: float, vector_state: Dict[str, float]) -> str:
        """
        Calculate trade hash: H(t) = hash(price_t, vector_state_t)
        
        Args:
            price: Current price
            vector_state: Current vector state
            
        Returns:
            Hash string representing the trade state
        """
        try:
            # Normalize price to config precision
            normalized_price = round(price, self.config.hash_precision)
            
            # Create hash input string
            hash_input = f"{normalized_price}"
            for key in sorted(vector_state.keys()):
                hash_input += f":{key}={vector_state[key]:.6f}"
            
            # Calculate SHA-256 hash
            hash_object = hashlib.sha256(hash_input.encode())
            hash_value = hash_object.hexdigest()[:16]  # Use first 16 chars
            
            return hash_value
            
        except Exception as e:
            logger.error(f"Error calculating trade hash: {e}")
            return "0000000000000000"
    
    def add_trade_event(self, event: TradeEvent):
        """Add trade event to replay history."""
        try:
            # Calculate hash if not provided
            if not event.hash_value:
                event.hash_value = self.calculate_trade_hash(event.price, event.vector_state)
            
            # Add to replay history
            self.replay_history.append(event)
            
            # Add to hash cache
            if event.hash_value not in self.hash_cache:
                self.hash_cache[event.hash_value] = []
            self.hash_cache[event.hash_value].append(event)
            
            # Maintain cache size
            if len(self.replay_history) > self.config.cache_size:
                oldest_event = self.replay_history.pop(0)
                # Remove from hash cache if it's the only instance
                if oldest_event.hash_value in self.hash_cache:
                    self.hash_cache[oldest_event.hash_value] = [
                        e for e in self.hash_cache[oldest_event.hash_value] 
                        if e.timestamp != oldest_event.timestamp
                    ]
                    if not self.hash_cache[oldest_event.hash_value]:
                        del self.hash_cache[oldest_event.hash_value]
            
            logger.debug(f"Added trade event: {event.event_type} at {event.timestamp}")
            
        except Exception as e:
            logger.error(f"Error adding trade event: {e}")
    
    def find_similar_events(self, target_hash: str, max_results: int = 10) -> List[TradeEvent]:
        """Find similar trade events based on hash similarity."""
        try:
            if target_hash not in self.hash_cache:
                return []
            
            events = self.hash_cache[target_hash]
            return events[-max_results:]  # Return most recent events
            
        except Exception as e:
            logger.error(f"Error finding similar events: {e}")
            return []

class RecursivePathLogic:
    """Recursive path analysis engine."""
    
    def __init__(self, config: BacktraceConfig):
        self.config = config
        self.path_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.signal_weights: Dict[str, float] = {}
        
    def calculate_recursive_path(self, exit_vectors: List[np.ndarray], 
                               reentry_signals: List[np.ndarray]) -> float:
        """
        Calculate recursive path logic: Ψ_backtrace = ∑(exit_vector_i · reentry_signal_i)
        
        Args:
            exit_vectors: List of exit vectors
            reentry_signals: List of reentry signals
            
        Returns:
            Recursive path strength
        """
        try:
            if len(exit_vectors) != len(reentry_signals):
                logger.warning("Mismatch in exit vectors and reentry signals length")
                return 0.0
            
            path_strength = 0.0
            for i, (exit_vec, reentry_sig) in enumerate(zip(exit_vectors, reentry_signals)):
                # Calculate dot product
                dot_product = np.dot(exit_vec, reentry_sig)
                
                # Apply weight based on position in sequence
                weight = 1.0 / (i + 1)  # Decay weight
                
                path_strength += dot_product * weight
            
            return float(path_strength)
            
        except Exception as e:
            logger.error(f"Error calculating recursive path: {e}")
            return 0.0
    
    def analyze_path_patterns(self, events: List[TradeEvent]) -> Dict[str, float]:
        """Analyze patterns in trade events for recursive logic."""
        try:
            patterns = {
                'entry_exit_ratio': 0.0,
                'signal_strength': 0.0,
                'vector_correlation': 0.0,
                'temporal_consistency': 0.0
            }
            
            if len(events) < 2:
                return patterns
            
            # Calculate entry/exit ratio
            entry_events = [e for e in events if e.event_type == 'entry']
            exit_events = [e for e in events if e.event_type == 'exit']
            
            if entry_events and exit_events:
                patterns['entry_exit_ratio'] = len(exit_events) / len(entry_events)
            
            # Calculate signal strength
            signal_events = [e for e in events if e.event_type == 'signal']
            if signal_events:
                signal_strengths = [sum(e.vector_state.values()) for e in signal_events]
                patterns['signal_strength'] = np.mean(signal_strengths)
            
            # Calculate vector correlation
            if len(events) >= 2:
                vectors = [list(e.vector_state.values()) for e in events]
                correlation_matrix = np.corrcoef(vectors)
                patterns['vector_correlation'] = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            
            # Calculate temporal consistency
            if len(events) >= 2:
                timestamps = [e.timestamp for e in events]
                intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                           for i in range(len(timestamps)-1)]
                patterns['temporal_consistency'] = 1.0 / (1.0 + np.std(intervals))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing path patterns: {e}")
            return {'entry_exit_ratio': 0.0, 'signal_strength': 0.0, 
                   'vector_correlation': 0.0, 'temporal_consistency': 0.0}

class TickWindowRebuild:
    """Tick window rebuilding engine."""
    
    def __init__(self, config: BacktraceConfig):
        self.config = config
        self.tick_buffer: List[Dict[str, Any]] = []
        self.window_cache: Dict[str, List[Dict[str, Any]]] = {}
        
    def rebuild_tick_window(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Rebuild tick window: τ(t) = tₙ - t₀; for n in replay range
        
        Args:
            start_time: Start of tick window
            end_time: End of tick window
            
        Returns:
            List of tick data for the window
        """
        try:
            window_key = f"{start_time.isoformat()}_{end_time.isoformat()}"
            
            # Check cache first
            if window_key in self.window_cache:
                return self.window_cache[window_key]
            
            # Filter ticks in the window
            window_ticks = [
                tick for tick in self.tick_buffer
                if start_time <= tick['timestamp'] <= end_time
            ]
            
            # Sort by timestamp
            window_ticks.sort(key=lambda x: x['timestamp'])
            
            # Cache the result
            self.window_cache[window_key] = window_ticks
            
            # Maintain cache size
            if len(self.window_cache) > 100:
                oldest_key = min(self.window_cache.keys())
                del self.window_cache[oldest_key]
            
            return window_ticks
            
        except Exception as e:
            logger.error(f"Error rebuilding tick window: {e}")
            return []
    
    def add_tick_data(self, timestamp: datetime, price: float, volume: float, 
                     metadata: Dict[str, Any] = None):
        """Add tick data to the buffer."""
        try:
            tick_data = {
                'timestamp': timestamp,
                'price': price,
                'volume': volume,
                'metadata': metadata or {}
            }
            
            self.tick_buffer.append(tick_data)
            
            # Maintain buffer size
            if len(self.tick_buffer) > self.config.replay_window:
                self.tick_buffer = self.tick_buffer[-self.config.replay_window:]
            
        except Exception as e:
            logger.error(f"Error adding tick data: {e}")
    
    def get_tick_statistics(self, window_ticks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate statistics for a tick window."""
        try:
            if not window_ticks:
                return {}
            
            prices = [tick['price'] for tick in window_ticks]
            volumes = [tick['volume'] for tick in window_ticks]
            
            stats = {
                'price_mean': float(np.mean(prices)),
                'price_std': float(np.std(prices)),
                'price_min': float(np.min(prices)),
                'price_max': float(np.max(prices)),
                'volume_mean': float(np.mean(volumes)),
                'volume_std': float(np.std(volumes)),
                'tick_count': len(window_ticks),
                'price_range': float(np.max(prices) - np.min(prices)),
                'price_volatility': float(np.std(prices) / np.mean(prices)) if np.mean(prices) > 0 else 0.0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating tick statistics: {e}")
            return {}

class DemoBacktracePipeline:
    """Main demo backtrace pipeline."""
    
    def __init__(self, config: Optional[BacktraceConfig] = None):
        self.config = config or BacktraceConfig()
        self.hash_replay = TradeHashReplay(self.config)
        self.path_logic = RecursivePathLogic(self.config)
        self.tick_rebuild = TickWindowRebuild(self.config)
        self.is_running = False
        self.analysis_thread = None
        
    def start_backtrace_analysis(self):
        """Start the backtrace analysis pipeline."""
        try:
            self.is_running = True
            self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self.analysis_thread.start()
            logger.info("Demo backtrace pipeline started")
            
        except Exception as e:
            logger.error(f"Error starting backtrace analysis: {e}")
    
    def stop_backtrace_analysis(self):
        """Stop the backtrace analysis pipeline."""
        try:
            self.is_running = False
            if self.analysis_thread:
                self.analysis_thread.join(timeout=5)
            logger.info("Demo backtrace pipeline stopped")
            
        except Exception as e:
            logger.error(f"Error stopping backtrace analysis: {e}")
    
    def _analysis_loop(self):
        """Main analysis loop."""
        while self.is_running:
            try:
                # Analyze recent events
                if len(self.hash_replay.replay_history) > 0:
                    recent_events = self.hash_replay.replay_history[-100:]  # Last 100 events
                    
                    # Analyze patterns
                    patterns = self.path_logic.analyze_path_patterns(recent_events)
                    
                    # Generate insights
                    insights = self._generate_insights(patterns, recent_events)
                    
                    # Log insights
                    if insights:
                        logger.info(f"Backtrace insights: {insights}")
                
                # Sleep for analysis interval
                import time
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                time.sleep(5)
    
    def _generate_insights(self, patterns: Dict[str, float], 
                          events: List[TradeEvent]) -> Dict[str, Any]:
        """Generate insights from patterns and events."""
        try:
            insights = {
                'pattern_strength': 0.0,
                'recommendations': [],
                'risk_indicators': [],
                'opportunity_signals': []
            }
            
            # Calculate pattern strength
            pattern_strength = sum(patterns.values()) / len(patterns)
            insights['pattern_strength'] = pattern_strength
            
            # Generate recommendations
            if patterns['entry_exit_ratio'] < 0.8:
                insights['recommendations'].append("Consider reducing entry frequency")
            
            if patterns['signal_strength'] > 0.7:
                insights['recommendations'].append("Strong signals detected - increase position sizes")
            
            if patterns['vector_correlation'] < 0.3:
                insights['risk_indicators'].append("Low vector correlation - potential market instability")
            
            if patterns['temporal_consistency'] > 0.8:
                insights['opportunity_signals'].append("High temporal consistency - good for trend following")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {}
    
    def replay_trade_sequence(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Replay a complete trade sequence for analysis."""
        try:
            # Rebuild tick window
            window_ticks = self.tick_rebuild.rebuild_tick_window(start_time, end_time)
            
            # Get events in the window
            window_events = [
                event for event in self.hash_replay.replay_history
                if start_time <= event.timestamp <= end_time
            ]
            
            # Calculate statistics
            tick_stats = self.tick_rebuild.get_tick_statistics(window_ticks)
            patterns = self.path_logic.analyze_path_patterns(window_events)
            
            # Generate replay report
            replay_report = {
                'window_start': start_time.isoformat(),
                'window_end': end_time.isoformat(),
                'tick_count': len(window_ticks),
                'event_count': len(window_events),
                'tick_statistics': tick_stats,
                'pattern_analysis': patterns,
                'hash_distribution': self._analyze_hash_distribution(window_events),
                'vector_analysis': self._analyze_vector_patterns(window_events)
            }
            
            return replay_report
            
        except Exception as e:
            logger.error(f"Error replaying trade sequence: {e}")
            return {}
    
    def _analyze_hash_distribution(self, events: List[TradeEvent]) -> Dict[str, Any]:
        """Analyze hash distribution in events."""
        try:
            hash_counts = {}
            for event in events:
                hash_value = event.hash_value
                hash_counts[hash_value] = hash_counts.get(hash_value, 0) + 1
            
            return {
                'unique_hashes': len(hash_counts),
                'most_common_hash': max(hash_counts.items(), key=lambda x: x[1])[0] if hash_counts else None,
                'hash_entropy': self._calculate_entropy(list(hash_counts.values()))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing hash distribution: {e}")
            return {}
    
    def _analyze_vector_patterns(self, events: List[TradeEvent]) -> Dict[str, Any]:
        """Analyze vector patterns in events."""
        try:
            if not events:
                return {}
            
            # Extract vectors
            vectors = [list(event.vector_state.values()) for event in events]
            vectors_array = np.array(vectors)
            
            return {
                'vector_mean': float(np.mean(vectors_array)),
                'vector_std': float(np.std(vectors_array)),
                'vector_correlation': float(np.corrcoef(vectors_array.T)[0, 1]) if vectors_array.shape[1] > 1 else 0.0,
                'vector_dimensions': vectors_array.shape[1] if len(vectors_array.shape) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing vector patterns: {e}")
            return {}
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate entropy of a distribution."""
        try:
            if not values:
                return 0.0
            
            total = sum(values)
            if total == 0:
                return 0.0
            
            probabilities = [v / total for v in values]
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            return float(entropy)
            
        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            return 0.0

def main():
    """Main function for testing."""
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create pipeline
        config = BacktraceConfig()
        pipeline = DemoBacktracePipeline(config)
        
        # Start analysis
        pipeline.start_backtrace_analysis()
        
        # Simulate some trade events
        for i in range(50):
            timestamp = datetime.now() + timedelta(seconds=i)
            price = 50000 + np.random.normal(0, 100)
            volume = np.random.uniform(0.1, 10.0)
            
            # Create vector state
            vector_state = {
                'trend': np.random.uniform(-1, 1),
                'volatility': np.random.uniform(0, 0.1),
                'momentum': np.random.uniform(-0.5, 0.5),
                'volume_ratio': np.random.uniform(0.5, 2.0)
            }
            
            # Create trade event
            event_type = np.random.choice(['entry', 'exit', 'signal'])
            event = TradeEvent(
                timestamp=timestamp,
                price=price,
                volume=volume,
                event_type=event_type,
                vector_state=vector_state,
                hash_value=""
            )
            
            # Add to pipeline
            pipeline.hash_replay.add_trade_event(event)
            pipeline.tick_rebuild.add_tick_data(timestamp, price, volume)
        
        # Wait for analysis
        import time
        time.sleep(5)
        
        # Generate replay report
        start_time = datetime.now() - timedelta(minutes=1)
        end_time = datetime.now()
        replay_report = pipeline.replay_trade_sequence(start_time, end_time)
        
        print("Replay Report:")
        print(json.dumps(replay_report, indent=2, default=str))
        
        # Stop pipeline
        pipeline.stop_backtrace_analysis()
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 