from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Multi-Bit BTC Processor - 16-bit Tickstream Compression and Logic Handler
=======================================================================

This module implements the multi-bit BTC processor for Schwabot, providing
comprehensive 16-bit tickstream compression, multi-bit frame collapse,
profit drift detection, and compression hash mapping for Bitcoin trading.

Core Mathematical Functions:
- Multi-bit Frame Collapse: Φ(t) = Σ [μₙ(t) × ωₙ] over n ∈ {2,4,8,16}
- Profit Drift Detection: δₚ = |P(t) - P(t - Δτ)| / Δτ
- Compression Hash Mapping: Hₙ = SHA256([tickₙ, Δtₙ, volₙ])

Core Functionality:
- 16-bit tickstream compression and analysis
- Multi-bit frame collapse processing
- Profit drift detection and analysis
- Compression hash mapping and pattern recognition
- Entropy-weighted bit collapse
- XOR pattern matching for BTC ticks
"""

import logging
import json
import time
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from core.unified_math_system import unified_math
from collections import defaultdict, deque
import os

logger = logging.getLogger(__name__)

class BitFrameSize(Enum):
    BIT_2 = 2
    BIT_4 = 4
    BIT_8 = 8
    BIT_16 = 16

class CompressionMode(Enum):
    FRAME_COLLAPSE = "frame_collapse"
    DRIFT_DETECTION = "drift_detection"
    HASH_MAPPING = "hash_mapping"
    XOR_PATTERN = "xor_pattern"

@dataclass
class TickData:
    tick_id: str
    timestamp: datetime
    price: float
    volume: float
    bit_frame: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MultiBitFrame:
    frame_id: str
    frame_size: BitFrameSize
    entropy_modulation: float
    frame_collapse: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProfitDrift:
    drift_id: str
    current_profit: float
    previous_profit: float
    time_delta: float
    drift_velocity: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CompressionHash:
    hash_id: str
    tick_data: Dict[str, Any]
    time_delta: float
    volume: float
    hash_value: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class MultiBitBTCProcessor:
    def __init__(self, config_path: str = "./config/multi_bit_btc_config.json"):
        self.config_path = config_path
        self.tick_data: Dict[str, TickData] = {}
        self.multi_bit_frames: Dict[str, MultiBitFrame] = {}
        self.profit_drifts: Dict[str, ProfitDrift] = {}
        self.compression_hashes: Dict[str, CompressionHash] = {}
        self.entropy_weights: Dict[int, float] = {}
        self.frame_history: deque = deque(maxlen=10000)
        self.profit_history: deque = deque(maxlen=1000)
        self._load_configuration()
        self._initialize_processor()
        self._start_tick_processing()
        logger.info("Multi-Bit BTC Processor initialized")

    def _load_configuration(self) -> None:
        """Load multi-bit BTC processor configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                logger.info(f"Loaded multi-bit BTC processor configuration")
            else:
                self._create_default_configuration()
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._create_default_configuration()

    def _create_default_configuration(self) -> None:
        """Create default multi-bit BTC processor configuration."""
        config = {
            "frame_processing": {
                "default_frame_size": 16,
                "entropy_threshold": 0.5,
                "collapse_factor": 0.8
            },
            "drift_detection": {
                "drift_threshold": 0.01,
                "time_window": 60,
                "velocity_threshold": 0.001
            },
            "hash_mapping": {
                "hash_algorithm": "sha256",
                "compression_factor": 0.9,
                "pattern_threshold": 0.7
            },
            "xor_pattern": {
                "pattern_length": 16,
                "matching_threshold": 0.8,
                "update_frequency": 100
            }
        }
        
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def _initialize_processor(self) -> None:
        """Initialize the multi-bit BTC processor."""
        # Initialize entropy weights for different bit frames
        self._initialize_entropy_weights()
        
        # Initialize frame processors
        self._initialize_frame_processors()
        
        logger.info("Multi-bit BTC processor initialized successfully")

    def _initialize_entropy_weights(self) -> None:
        """Initialize entropy weights for different bit frame sizes."""
        try:
            # Entropy weights based on bit frame size
            # Higher bit frames get higher weights due to more information content
            self.entropy_weights = {
                2: 0.1,   # 2-bit frame weight
                4: 0.2,   # 4-bit frame weight
                8: 0.4,   # 8-bit frame weight
                16: 0.8   # 16-bit frame weight
            }
            
            logger.info(f"Initialized entropy weights for {len(self.entropy_weights)} bit frame sizes")
            
        except Exception as e:
            logger.error(f"Error initializing entropy weights: {e}")

    def _initialize_frame_processors(self) -> None:
        """Initialize frame processing components."""
        try:
            self.frame_processors = {
                BitFrameSize.BIT_2: self._process_2bit_frame,
                BitFrameSize.BIT_4: self._process_4bit_frame,
                BitFrameSize.BIT_8: self._process_8bit_frame,
                BitFrameSize.BIT_16: self._process_16bit_frame
            }
            
            logger.info(f"Initialized {len(self.frame_processors)} frame processors")
            
        except Exception as e:
            logger.error(f"Error initializing frame processors: {e}")

    def _start_tick_processing(self) -> None:
        """Start the tick processing system."""
        # This would start background processing tasks
        logger.info("Tick processing started")

    def process_tick(self, price: float, volume: float, timestamp: Optional[datetime] = None) -> TickData:
        """Process a single BTC tick."""
        try:
            tick_id = f"tick_{int(time.time())}"
            if timestamp is None:
                timestamp = datetime.now()
            
            # Determine bit frame based on price precision
            bit_frame = self._determine_bit_frame(price)
            
            # Create tick data object
            tick_data = TickData(
                tick_id=tick_id,
                timestamp=timestamp,
                price=price,
                volume=volume,
                bit_frame=bit_frame,
                metadata={
                    "price_precision": len(str(price).split('.')[-1]) if '.' in str(price) else 0,
                    "volume_normalized": volume / 1000  # Normalize volume
                }
            )
            
            # Store tick data
            self.tick_data[tick_id] = tick_data
            self.frame_history.append(tick_data)
            
            logger.info(f"Processed tick: {tick_id} at price {price}")
            return tick_data
            
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            return None

    def _determine_bit_frame(self, price: float) -> int:
        """Determine appropriate bit frame size based on price."""
        try:
            # Convert price to integer representation for bit analysis
            price_int = int(price * 1000000)  # Convert to micro-units
            
            # Count significant bits
            if price_int == 0:
                return 2
            
            bit_count = price_int.bit_length()
            
            # Map to available bit frames
            if bit_count <= 2:
                return 2
            elif bit_count <= 4:
                return 4
            elif bit_count <= 8:
                return 8
            else:
                return 16
                
        except Exception as e:
            logger.error(f"Error determining bit frame: {e}")
            return 16

    def calculate_frame_collapse(self, frame_size: BitFrameSize, time_window: float = 60.0) -> MultiBitFrame:
        """
        Calculate Multi-bit Frame Collapse.
        
        Mathematical Formula:
        Φ(t) = Σ [μₙ(t) × ωₙ] over n ∈ {2,4,8,16}
        
        Where:
        - μₙ(t) is the entropy modulation at time t for bit frame n
        - ωₙ is the entropy weight for bit frame n
        """
        try:
            frame_id = f"frame_{frame_size.value}_{int(time.time())}"
            
            # Get recent ticks within time window
            cutoff_time = datetime.now() - timedelta(seconds=time_window)
            recent_ticks = [
                tick for tick in self.frame_history
                if tick.timestamp >= cutoff_time and tick.bit_frame == frame_size.value
            ]
            
            if not recent_ticks:
                logger.warning(f"No recent ticks found for frame size {frame_size.value}")
                return None
            
            # Calculate entropy modulation for each tick
            entropy_modulations = []
            for tick in recent_ticks:
                entropy = self._calculate_entropy_modulation(tick)
                entropy_modulations.append(entropy)
            
            # Calculate weighted sum using the mathematical formula
            entropy_weight = self.entropy_weights.get(frame_size.value, 0.5)
            frame_collapse = sum(entropy_modulations) * entropy_weight
            
            # Calculate average entropy modulation
            avg_entropy_modulation = unified_math.unified_math.mean(entropy_modulations) if entropy_modulations else 0.0
            
            # Create multi-bit frame object
            multi_bit_frame = MultiBitFrame(
                frame_id=frame_id,
                frame_size=frame_size,
                entropy_modulation=avg_entropy_modulation,
                frame_collapse=frame_collapse,
                timestamp=datetime.now(),
                metadata={
                    "num_ticks": len(recent_ticks),
                    "time_window": time_window,
                    "entropy_weight": entropy_weight,
                    "entropy_modulations": entropy_modulations
                }
            )
            
            # Store frame
            self.multi_bit_frames[frame_id] = multi_bit_frame
            
            logger.info(f"Frame collapse calculated for {frame_size.value}-bit: {frame_collapse:.6f}")
            return multi_bit_frame
            
        except Exception as e:
            logger.error(f"Error calculating frame collapse: {e}")
            return None

    def _calculate_entropy_modulation(self, tick: TickData) -> float:
        """Calculate entropy modulation for a tick."""
        try:
            # Calculate entropy based on price and volume
            price_entropy = unified_math.abs(tick.price - 50000) / 50000  # Normalize around typical BTC price
            volume_entropy = unified_math.min(tick.volume / 1000, 1.0)  # Normalize volume
            
            # Combine entropy components
            total_entropy = (price_entropy + volume_entropy) / 2
            
            # Apply bit frame scaling
            frame_scale = tick.bit_frame / 16.0
            entropy_modulation = total_entropy * frame_scale
            
            return unified_math.max(0.0, unified_math.min(1.0, entropy_modulation))
            
        except Exception as e:
            logger.error(f"Error calculating entropy modulation: {e}")
            return 0.0

    def detect_profit_drift(self, current_profit: float, previous_profit: float,
                           time_delta: float) -> ProfitDrift:
        """
        Detect Profit Drift.
        
        Mathematical Formula:
        δₚ = |P(t) - P(t - Δτ)| / Δτ
        
        Where:
        - P(t) is the current profit
        - P(t - Δτ) is the previous profit
        - Δτ is the time delta
        """
        try:
            drift_id = f"drift_{int(time.time())}"
            
            # Calculate drift velocity using the mathematical formula
            profit_difference = unified_math.abs(current_profit - previous_profit)
            drift_velocity = profit_difference / time_delta if time_delta > 0 else 0.0
            
            # Create profit drift object
            profit_drift = ProfitDrift(
                drift_id=drift_id,
                current_profit=current_profit,
                previous_profit=previous_profit,
                time_delta=time_delta,
                drift_velocity=drift_velocity,
                timestamp=datetime.now(),
                metadata={
                    "profit_difference": profit_difference,
                    "drift_direction": "positive" if current_profit > previous_profit else "negative"
                }
            )
            
            # Store drift
            self.profit_drifts[drift_id] = profit_drift
            self.profit_history.append(profit_drift)
            
            logger.info(f"Profit drift detected: {drift_velocity:.6f}")
            return profit_drift
            
        except Exception as e:
            logger.error(f"Error detecting profit drift: {e}")
            return None

    def create_compression_hash(self, tick_data: Dict[str, Any], time_delta: float,
                              volume: float) -> CompressionHash:
        """
        Create Compression Hash Mapping.
        
        Mathematical Formula:
        Hₙ = SHA256([tickₙ, Δtₙ, volₙ])
        
        Where:
        - tickₙ is the tick data
        - Δtₙ is the time delta
        - volₙ is the volume
        """
        try:
            hash_id = f"hash_{int(time.time())}"
            
            # Prepare data for hashing
            hash_data = {
                "tick": tick_data,
                "time_delta": time_delta,
                "volume": volume,
                "timestamp": datetime.now().isoformat()
            }
            
            # Convert to string and hash
            hash_string = json.dumps(hash_data, sort_keys=True)
            hash_value = hashlib.sha256(hash_string.encode()).hexdigest()
            
            # Create compression hash object
            compression_hash = CompressionHash(
                hash_id=hash_id,
                tick_data=tick_data,
                time_delta=time_delta,
                volume=volume,
                hash_value=hash_value,
                timestamp=datetime.now(),
                metadata={
                    "hash_algorithm": "sha256",
                    "hash_length": len(hash_value),
                    "data_size": len(hash_string)
                }
            )
            
            # Store hash
            self.compression_hashes[hash_id] = compression_hash
            
            logger.info(f"Compression hash created: {hash_value[:16]}...")
            return compression_hash
            
        except Exception as e:
            logger.error(f"Error creating compression hash: {e}")
            return None

    def detect_xor_patterns(self, pattern_length: int = 16) -> Dict[str, Any]:
        """Detect XOR patterns in recent tick data."""
        try:
            if len(self.frame_history) < pattern_length:
                return {"error": "Insufficient tick data for pattern detection"}
            
            # Get recent ticks
            recent_ticks = list(self.frame_history)[-pattern_length:]
            
            # Extract bit frames
            bit_frames = [tick.bit_frame for tick in recent_ticks]
            
            # Calculate XOR patterns
            xor_patterns = []
            for i in range(len(bit_frames) - 1):
                xor_result = bit_frames[i] ^ bit_frames[i + 1]
                xor_patterns.append(xor_result)
            
            # Calculate pattern statistics
            pattern_mean = unified_math.unified_math.mean(xor_patterns)
            pattern_std = unified_math.unified_math.std(xor_patterns)
            pattern_entropy = self._calculate_pattern_entropy(xor_patterns)
            
            # Detect repeating patterns
            repeating_patterns = self._find_repeating_patterns(xor_patterns)
            
            return {
                "xor_patterns": xor_patterns,
                "pattern_mean": pattern_mean,
                "pattern_std": pattern_std,
                "pattern_entropy": pattern_entropy,
                "repeating_patterns": repeating_patterns,
                "pattern_length": pattern_length
            }
            
        except Exception as e:
            logger.error(f"Error detecting XOR patterns: {e}")
            return {"error": str(e)}

    def _calculate_pattern_entropy(self, patterns: List[int]) -> float:
        """Calculate entropy of XOR patterns."""
        try:
            if not patterns:
                return 0.0
            
            # Count pattern frequencies
            pattern_counts = defaultdict(int)
            for pattern in patterns:
                pattern_counts[pattern] += 1
            
            # Calculate entropy
            total_patterns = len(patterns)
            entropy = 0.0
            
            for count in pattern_counts.values():
                probability = count / total_patterns
                if probability > 0:
                    entropy -= probability * np.log2(probability)
            
            return entropy
            
        except Exception as e:
            logger.error(f"Error calculating pattern entropy: {e}")
            return 0.0

    def _find_repeating_patterns(self, patterns: List[int]) -> List[Dict[str, Any]]:
        """Find repeating patterns in XOR sequence."""
        try:
            repeating_patterns = []
            
            # Look for patterns of different lengths
            for pattern_length in range(2, unified_math.min(8, len(patterns) // 2)):
                for start_idx in range(len(patterns) - pattern_length * 2):
                    pattern = patterns[start_idx:start_idx + pattern_length]
                    next_pattern = patterns[start_idx + pattern_length:start_idx + pattern_length * 2]
                    
                    if pattern == next_pattern:
                        repeating_patterns.append({
                            "pattern": pattern,
                            "length": pattern_length,
                            "start_index": start_idx,
                            "repetitions": 2
                        })
            
            return repeating_patterns
            
        except Exception as e:
            logger.error(f"Error finding repeating patterns: {e}")
            return []

    def _process_2bit_frame(self, ticks: List[TickData]) -> Dict[str, Any]:
        """Process 2-bit frame."""
        try:
            if not ticks:
                return {"error": "No ticks provided"}
            
            # Extract 2-bit frame data
            frame_data = [tick.bit_frame for tick in ticks if tick.bit_frame == 2]
            
            return {
                "frame_size": 2,
                "num_ticks": len(frame_data),
                "frame_values": frame_data,
                "frame_mean": unified_math.unified_math.mean(frame_data) if frame_data else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error processing 2-bit frame: {e}")
            return {"error": str(e)}

    def _process_4bit_frame(self, ticks: List[TickData]) -> Dict[str, Any]:
        """Process 4-bit frame."""
        try:
            if not ticks:
                return {"error": "No ticks provided"}
            
            # Extract 4-bit frame data
            frame_data = [tick.bit_frame for tick in ticks if tick.bit_frame == 4]
            
            return {
                "frame_size": 4,
                "num_ticks": len(frame_data),
                "frame_values": frame_data,
                "frame_mean": unified_math.unified_math.mean(frame_data) if frame_data else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error processing 4-bit frame: {e}")
            return {"error": str(e)}

    def _process_8bit_frame(self, ticks: List[TickData]) -> Dict[str, Any]:
        """Process 8-bit frame."""
        try:
            if not ticks:
                return {"error": "No ticks provided"}
            
            # Extract 8-bit frame data
            frame_data = [tick.bit_frame for tick in ticks if tick.bit_frame == 8]
            
            return {
                "frame_size": 8,
                "num_ticks": len(frame_data),
                "frame_values": frame_data,
                "frame_mean": unified_math.unified_math.mean(frame_data) if frame_data else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error processing 8-bit frame: {e}")
            return {"error": str(e)}

    def _process_16bit_frame(self, ticks: List[TickData]) -> Dict[str, Any]:
        """Process 16-bit frame."""
        try:
            if not ticks:
                return {"error": "No ticks provided"}
            
            # Extract 16-bit frame data
            frame_data = [tick.bit_frame for tick in ticks if tick.bit_frame == 16]
            
            return {
                "frame_size": 16,
                "num_ticks": len(frame_data),
                "frame_values": frame_data,
                "frame_mean": unified_math.unified_math.mean(frame_data) if frame_data else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error processing 16-bit frame: {e}")
            return {"error": str(e)}

    def get_processor_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processor statistics."""
        total_ticks = len(self.tick_data)
        total_frames = len(self.multi_bit_frames)
        total_drifts = len(self.profit_drifts)
        total_hashes = len(self.compression_hashes)
        
        # Calculate frame size distribution
        frame_distribution = defaultdict(int)
        for tick in self.tick_data.values():
            frame_distribution[tick.bit_frame] += 1
        
        # Calculate average drift velocity
        if total_drifts > 0:
            avg_drift_velocity = unified_math.mean([d.drift_velocity for d in self.profit_drifts.values()])
        else:
            avg_drift_velocity = 0.0
        
        return {
            "total_ticks": total_ticks,
            "total_frames": total_frames,
            "total_drifts": total_drifts,
            "total_hashes": total_hashes,
            "frame_distribution": dict(frame_distribution),
            "average_drift_velocity": avg_drift_velocity,
            "frame_history_size": len(self.frame_history),
            "profit_history_size": len(self.profit_history)
        }

def main() -> None:
    """Main function for testing and demonstration."""
    processor = MultiBitBTCProcessor("./test_multi_bit_btc_config.json")
    
    # Process some sample ticks
    for i in range(10):
        price = 50000 + np.random.random() * 1000
        volume = np.random.random() * 100
        tick = processor.process_tick(price, volume)
    
    # Calculate frame collapse for different bit sizes
    for frame_size in [BitFrameSize.BIT_2, BitFrameSize.BIT_4, BitFrameSize.BIT_8, BitFrameSize.BIT_16]:
        frame_result = processor.calculate_frame_collapse(frame_size)
    
    # Detect profit drift
    drift_result = processor.detect_profit_drift(
        current_profit=0.05,
        previous_profit=0.03,
        time_delta=60.0
    )
    
    # Create compression hash
    hash_result = processor.create_compression_hash(
        tick_data={"price": 50000, "volume": 100},
        time_delta=1.0,
        volume=100
    )
    
    safe_print("Multi-Bit BTC Processor initialized successfully")
    
    # Get statistics
    stats = processor.get_processor_statistics()
    safe_print(f"Processor Statistics: {stats}")

if __name__ == "__main__":
    main() 