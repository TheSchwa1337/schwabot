#!/usr/bin/env python3
"""
Strategy Mapper Module
======================

Compliant + layered hash/bit strategy routing for Schwabot v0.05.
Provides strategy selection, routing, and execution coordination.
"""

import hashlib
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Strategy type enumeration."""
    HASH_BASED = "hash_based"
    BIT_BASED = "bit_based"
    HYBRID = "hybrid"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    SCALPING = "scalping"
    SWING = "swing"
    ARBITRAGE = "arbitrage"


class StrategyState(Enum):
    """Strategy state enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ERROR = "error"
    OPTIMIZING = "optimizing"


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    strategy_id: str
    strategy_type: StrategyType
    name: str
    description: str
    risk_level: float  # 0.0 to 1.0
    min_confidence: float
    max_position_size: float
    stop_loss: float
    take_profit: float
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    """Strategy execution result."""
    strategy_id: str
    timestamp: float
    signal_type: str  # "buy", "sell", "hold"
    confidence: float
    position_size: float
    stop_loss: float
    take_profit: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HashStrategy:
    """Hash-based strategy definition."""
    hash_pattern: str
    strategy_type: StrategyType
    confidence_threshold: float
    execution_params: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyMapper:
    """
    Strategy Mapper for Schwabot v0.05.
    
    Provides compliant + layered hash/bit strategy routing
    with advanced pattern matching and execution coordination.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the strategy mapper."""
        self.config = config or self._default_config()
        
        # Strategy registry
        self.strategies: Dict[str, StrategyConfig] = {}
        self.hash_strategies: Dict[str, HashStrategy] = {}
        self.bit_strategies: Dict[str, Dict[str, Any]] = {}
        
        # Execution tracking
        self.execution_history: List[StrategyResult] = []
        self.max_history_size = self.config.get('max_history_size', 1000)
        
        # Performance metrics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        
        # State management
        self.current_strategy = None
        self.last_update = time.time()
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
        logger.info("🗺️ Strategy Mapper initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'max_history_size': 1000,
            'default_confidence_threshold': 0.7,
            'max_position_size': 0.1,  # 10% of portfolio
            'default_stop_loss': 0.02,  # 2%
            'default_take_profit': 0.05,  # 5%
            'hash_pattern_length': 64,
            'bit_pattern_length': 32,
            'strategy_rotation_enabled': True,
            'adaptive_confidence': True
        }
    
    def _initialize_default_strategies(self):
        """Initialize default strategies."""
        # Hash-based strategies
        self.add_hash_strategy(
            "conservative_hash",
            "0000000000000000000000000000000000000000000000000000000000000000",
            StrategyType.CONSERVATIVE,
            0.8,
            {"position_size": 0.05, "stop_loss": 0.015, "take_profit": 0.03}
        )
        
        self.add_hash_strategy(
            "aggressive_hash",
            "1111111111111111111111111111111111111111111111111111111111111111",
            StrategyType.AGGRESSIVE,
            0.6,
            {"position_size": 0.15, "stop_loss": 0.03, "take_profit": 0.08}
        )
        
        # Bit-based strategies
        self.add_bit_strategy(
            "scalping_bits",
            "10101010101010101010101010101010",
            StrategyType.SCALPING,
            0.7,
            {"position_size": 0.08, "stop_loss": 0.01, "take_profit": 0.02}
        )
        
        # Hybrid strategies
        self.add_hybrid_strategy(
            "swing_hybrid",
            StrategyType.SWING,
            0.75,
            {"position_size": 0.12, "stop_loss": 0.025, "take_profit": 0.06}
        )
    
    def add_hash_strategy(self, strategy_id: str, hash_pattern: str, 
                         strategy_type: StrategyType, confidence_threshold: float,
                         execution_params: Dict[str, Any]) -> bool:
        """Add a hash-based strategy."""
        try:
            hash_strategy = HashStrategy(
                hash_pattern=hash_pattern,
                strategy_type=strategy_type,
                confidence_threshold=confidence_threshold,
                execution_params=execution_params
            )
            
            self.hash_strategies[strategy_id] = hash_strategy
            
            # Create corresponding strategy config
            strategy_config = StrategyConfig(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                name=f"Hash Strategy {strategy_id}",
                description=f"Hash-based {strategy_type.value} strategy",
                risk_level=self._get_risk_level(strategy_type),
                min_confidence=confidence_threshold,
                max_position_size=execution_params.get('position_size', 0.1),
                stop_loss=execution_params.get('stop_loss', 0.02),
                take_profit=execution_params.get('take_profit', 0.05)
            )
            
            self.strategies[strategy_id] = strategy_config
            logger.info(f"Added hash strategy: {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding hash strategy {strategy_id}: {e}")
            return False
    
    def add_bit_strategy(self, strategy_id: str, bit_pattern: str,
                        strategy_type: StrategyType, confidence_threshold: float,
                        execution_params: Dict[str, Any]) -> bool:
        """Add a bit-based strategy."""
        try:
            bit_strategy = {
                "bit_pattern": bit_pattern,
                "strategy_type": strategy_type,
                "confidence_threshold": confidence_threshold,
                "execution_params": execution_params
            }
            
            self.bit_strategies[strategy_id] = bit_strategy
            
            # Create corresponding strategy config
            strategy_config = StrategyConfig(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                name=f"Bit Strategy {strategy_id}",
                description=f"Bit-based {strategy_type.value} strategy",
                risk_level=self._get_risk_level(strategy_type),
                min_confidence=confidence_threshold,
                max_position_size=execution_params.get('position_size', 0.1),
                stop_loss=execution_params.get('stop_loss', 0.02),
                take_profit=execution_params.get('take_profit', 0.05)
            )
            
            self.strategies[strategy_id] = strategy_config
            logger.info(f"Added bit strategy: {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding bit strategy {strategy_id}: {e}")
            return False
    
    def add_hybrid_strategy(self, strategy_id: str, strategy_type: StrategyType,
                           confidence_threshold: float, execution_params: Dict[str, Any]) -> bool:
        """Add a hybrid strategy."""
        try:
            strategy_config = StrategyConfig(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                name=f"Hybrid Strategy {strategy_id}",
                description=f"Hybrid {strategy_type.value} strategy",
                risk_level=self._get_risk_level(strategy_type),
                min_confidence=confidence_threshold,
                max_position_size=execution_params.get('position_size', 0.1),
                stop_loss=execution_params.get('stop_loss', 0.02),
                take_profit=execution_params.get('take_profit', 0.05)
            )
            
            self.strategies[strategy_id] = strategy_config
            logger.info(f"Added hybrid strategy: {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding hybrid strategy {strategy_id}: {e}")
            return False
    
    def _get_risk_level(self, strategy_type: StrategyType) -> float:
        """Get risk level for strategy type."""
        risk_levels = {
            StrategyType.CONSERVATIVE: 0.2,
            StrategyType.SCALPING: 0.4,
            StrategyType.SWING: 0.6,
            StrategyType.HYBRID: 0.7,
            StrategyType.AGGRESSIVE: 0.8,
            StrategyType.ARBITRAGE: 0.9
        }
        return risk_levels.get(strategy_type, 0.5)
    
    def select_strategy(self, market_data: Dict[str, Any], 
                       portfolio_state: Dict[str, Any]) -> Optional[StrategyConfig]:
        """
        Select the best strategy based on market data and portfolio state.
        
        Args:
            market_data: Current market data
            portfolio_state: Current portfolio state
            
        Returns:
            Selected strategy configuration
        """
        try:
            # Generate hash from market data
            market_hash = self._generate_market_hash(market_data)
            
            # Find matching hash strategy
            for strategy_id, hash_strategy in self.hash_strategies.items():
                if self._hash_matches_pattern(market_hash, hash_strategy.hash_pattern):
                    strategy_config = self.strategies.get(strategy_id)
                    if strategy_config and strategy_config.enabled:
                        return strategy_config
            
            # Check bit patterns
            bit_sequence = self._generate_bit_sequence(market_data)
            for strategy_id, bit_strategy in self.bit_strategies.items():
                if self._bit_matches_pattern(bit_sequence, bit_strategy["bit_pattern"]):
                    strategy_config = self.strategies.get(strategy_id)
                    if strategy_config and strategy_config.enabled:
                        return strategy_config
            
            # Fallback to hybrid strategy
            for strategy_id, strategy_config in self.strategies.items():
                if (strategy_config.strategy_type == StrategyType.HYBRID and 
                    strategy_config.enabled):
                    return strategy_config
            
            return None
            
        except Exception as e:
            logger.error(f"Error selecting strategy: {e}")
            return None
    
    def _generate_market_hash(self, market_data: Dict[str, Any]) -> str:
        """Generate hash from market data."""
        try:
            # Create hashable string from market data
            hash_data = {
                'price': market_data.get('price', 0),
                'volume': market_data.get('volume', 0),
                'timestamp': market_data.get('timestamp', time.time()),
                'volatility': market_data.get('volatility', 0)
            }
            
            hash_string = json.dumps(hash_data, sort_keys=True)
            return hashlib.sha256(hash_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating market hash: {e}")
            return "0" * 64
    
    def _generate_bit_sequence(self, market_data: Dict[str, Any]) -> str:
        """Generate bit sequence from market data."""
        try:
            # Convert market data to binary representation
            price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            
            # Simple bit conversion (can be enhanced)
            price_bits = format(int(price * 1000) % 256, '08b')
            volume_bits = format(int(volume) % 256, '08b')
            
            # Combine and repeat to match pattern length
            combined = price_bits + volume_bits
            target_length = self.config['bit_pattern_length']
            
            while len(combined) < target_length:
                combined += combined
            
            return combined[:target_length]
            
        except Exception as e:
            logger.error(f"Error generating bit sequence: {e}")
            return "0" * self.config['bit_pattern_length']
    
    def _hash_matches_pattern(self, market_hash: str, pattern: str) -> bool:
        """Check if market hash matches pattern."""
        try:
            # Simple pattern matching (can be enhanced with fuzzy matching)
            pattern_length = len(pattern)
            if len(market_hash) < pattern_length:
                return False
            
            # Check for exact match or significant similarity
            similarity = sum(1 for i in range(pattern_length) 
                           if market_hash[i] == pattern[i])
            similarity_ratio = similarity / pattern_length
            
            return similarity_ratio > 0.8  # 80% similarity threshold
            
        except Exception as e:
            logger.error(f"Error checking hash pattern: {e}")
            return False
    
    def _bit_matches_pattern(self, bit_sequence: str, pattern: str) -> bool:
        """Check if bit sequence matches pattern."""
        try:
            if len(bit_sequence) != len(pattern):
                return False
            
            # Check for exact match or significant similarity
            similarity = sum(1 for i in range(len(pattern)) 
                           if bit_sequence[i] == pattern[i])
            similarity_ratio = similarity / len(pattern)
            
            return similarity_ratio > 0.7  # 70% similarity threshold
            
        except Exception as e:
            logger.error(f"Error checking bit pattern: {e}")
            return False
    
    def execute_strategy(self, strategy_config: StrategyConfig,
                        market_data: Dict[str, Any],
                        portfolio_state: Dict[str, Any]) -> Optional[StrategyResult]:
        """
        Execute a strategy and return the result.
        
        Args:
            strategy_config: Strategy configuration
            market_data: Current market data
            portfolio_state: Current portfolio state
            
        Returns:
            Strategy execution result
        """
        try:
            # Calculate signal based on strategy type
            signal_type, confidence = self._calculate_signal(
                strategy_config, market_data, portfolio_state
            )
            
            if confidence < strategy_config.min_confidence:
                return None
            
            # Calculate position size
            position_size = self._calculate_position_size(
                strategy_config, portfolio_state, confidence
            )
            
            # Create strategy result
            result = StrategyResult(
                strategy_id=strategy_config.strategy_id,
                timestamp=time.time(),
                signal_type=signal_type,
                confidence=confidence,
                position_size=position_size,
                stop_loss=strategy_config.stop_loss,
                take_profit=strategy_config.take_profit,
                metadata={
                    "strategy_type": strategy_config.strategy_type.value,
                    "risk_level": strategy_config.risk_level
                }
            )
            
            # Update execution history
            self.execution_history.append(result)
            if len(self.execution_history) > self.max_history_size:
                self.execution_history.pop(0)
            
            self.total_executions += 1
            self.current_strategy = strategy_config.strategy_id
            self.last_update = time.time()
            
            logger.info(f"Executed strategy {strategy_config.strategy_id}: {signal_type} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error executing strategy {strategy_config.strategy_id}: {e}")
            self.failed_executions += 1
            return None
    
    def _calculate_signal(self, strategy_config: StrategyConfig,
                         market_data: Dict[str, Any],
                         portfolio_state: Dict[str, Any]) -> Tuple[str, float]:
        """Calculate trading signal based on strategy."""
        try:
            # Simple signal calculation (can be enhanced with ML models)
            price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            volatility = market_data.get('volatility', 0)
            
            # Base confidence calculation
            base_confidence = 0.5
            
            # Adjust based on strategy type
            if strategy_config.strategy_type == StrategyType.CONSERVATIVE:
                if volatility < 0.02:  # Low volatility
                    base_confidence += 0.2
                    signal_type = "buy" if price > 0 else "hold"
                else:
                    signal_type = "hold"
            elif strategy_config.strategy_type == StrategyType.AGGRESSIVE:
                if volatility > 0.05:  # High volatility
                    base_confidence += 0.3
                    signal_type = "buy" if volume > 0 else "sell"
                else:
                    signal_type = "hold"
            else:
                # Default logic
                signal_type = "hold"
                if price > 0 and volume > 0:
                    base_confidence += 0.1
            
            # Normalize confidence
            confidence = min(base_confidence, 1.0)
            
            return signal_type, confidence
            
        except Exception as e:
            logger.error(f"Error calculating signal: {e}")
            return "hold", 0.0
    
    def _calculate_position_size(self, strategy_config: StrategyConfig,
                                portfolio_state: Dict[str, Any],
                                confidence: float) -> float:
        """Calculate position size based on strategy and confidence."""
        try:
            # Base position size from strategy
            base_size = strategy_config.max_position_size
            
            # Adjust based on confidence
            adjusted_size = base_size * confidence
            
            # Check portfolio constraints
            available_capital = portfolio_state.get('available_capital', 0)
            total_capital = portfolio_state.get('total_capital', 1)
            
            if total_capital > 0:
                max_allowed = available_capital / total_capital
                adjusted_size = min(adjusted_size, max_allowed)
            
            return max(adjusted_size, 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of strategy mapping."""
        return {
            "total_strategies": len(self.strategies),
            "hash_strategies": len(self.hash_strategies),
            "bit_strategies": len(self.bit_strategies),
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "current_strategy": self.current_strategy,
            "last_update": self.last_update,
            "execution_history_size": len(self.execution_history)
        }
    
    def get_recent_executions(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent strategy executions."""
        recent_executions = self.execution_history[-count:]
        return [
            {
                "strategy_id": result.strategy_id,
                "timestamp": result.timestamp,
                "signal_type": result.signal_type,
                "confidence": result.confidence,
                "position_size": result.position_size
            }
            for result in recent_executions
        ] 