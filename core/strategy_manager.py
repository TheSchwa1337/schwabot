#!/usr/bin/env python3
"""
Strategy Manager - Mathematical Strategy Optimization and Execution Engine
=======================================================================

This module implements a comprehensive strategy management system for Schwabot,
providing mathematical strategy optimization, real-time execution, and performance
analytics.

Core Mathematical Functions:
- Strategy Performance: P(s) = Σ(wᵢ × pᵢ) where wᵢ are performance weights
- Risk-Adjusted Return: RAR = (return - risk_free_rate) / volatility
- Sharpe Ratio: SR = (μ - r_f) / σ
- Strategy Correlation: ρ(s₁, s₂) = cov(s₁, s₂) / (σ₁ × σ₂)

Core Functionality:
- Multi-strategy portfolio management
- Real-time strategy execution
- Performance monitoring and optimization
- Risk management and allocation
- Strategy correlation analysis
- Dynamic strategy switching
"""

import logging
import json
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import os
import queue
import weakref

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    TREND_FOLLOWING = "trend_following"
    CONTRARIAN = "contrarian"
    QUANTITATIVE = "quantitative"

class StrategyStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ERROR = "error"
    OPTIMIZING = "optimizing"

@dataclass
class StrategyConfig:
    strategy_id: str
    strategy_type: StrategyType
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk_limits: Dict[str, float] = field(default_factory=dict)
    performance_targets: Dict[str, float] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 1

@dataclass
class StrategyPerformance:
    strategy_id: str
    timestamp: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategySignal:
    strategy_id: str
    timestamp: datetime
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    price: float
    volume: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class StrategyExecutor:
    """Strategy execution engine."""
    
    def __init__(self, strategy_config: StrategyConfig):
        self.config = strategy_config
        self.status = StrategyStatus.INACTIVE
        self.performance_history: deque = deque(maxlen=10000)
        self.signal_history: deque = deque(maxlen=10000)
        self.trade_history: deque = deque(maxlen=10000)
        self.is_running = False
        self.execution_thread = None
        self._initialize_strategy()
    
    def _initialize_strategy(self):
        """Initialize the strategy."""
        try:
            # Initialize strategy-specific parameters
            if self.config.strategy_type == StrategyType.MOMENTUM:
                self._initialize_momentum_strategy()
            elif self.config.strategy_type == StrategyType.MEAN_REVERSION:
                self._initialize_mean_reversion_strategy()
            elif self.config.strategy_type == StrategyType.ARBITRAGE:
                self._initialize_arbitrage_strategy()
            else:
                self._initialize_generic_strategy()
            
            logger.info(f"Strategy {self.config.name} initialized")
            
        except Exception as e:
            logger.error(f"Error initializing strategy: {e}")
            self.status = StrategyStatus.ERROR
    
    def _initialize_momentum_strategy(self):
        """Initialize momentum strategy."""
        try:
            # Set default parameters for momentum strategy
            default_params = {
                'lookback_period': 20,
                'momentum_threshold': 0.02,
                'position_size': 0.1,
                'stop_loss': 0.05,
                'take_profit': 0.10
            }
            
            # Update with custom parameters
            for key, value in default_params.items():
                if key not in self.config.parameters:
                    self.config.parameters[key] = value
            
        except Exception as e:
            logger.error(f"Error initializing momentum strategy: {e}")
    
    def _initialize_mean_reversion_strategy(self):
        """Initialize mean reversion strategy."""
        try:
            # Set default parameters for mean reversion strategy
            default_params = {
                'lookback_period': 50,
                'std_dev_threshold': 2.0,
                'position_size': 0.1,
                'stop_loss': 0.03,
                'take_profit': 0.06
            }
            
            # Update with custom parameters
            for key, value in default_params.items():
                if key not in self.config.parameters:
                    self.config.parameters[key] = value
            
        except Exception as e:
            logger.error(f"Error initializing mean reversion strategy: {e}")
    
    def _initialize_arbitrage_strategy(self):
        """Initialize arbitrage strategy."""
        try:
            # Set default parameters for arbitrage strategy
            default_params = {
                'spread_threshold': 0.001,
                'position_size': 0.05,
                'execution_speed': 0.1,
                'max_hold_time': 300  # 5 minutes
            }
            
            # Update with custom parameters
            for key, value in default_params.items():
                if key not in self.config.parameters:
                    self.config.parameters[key] = value
            
        except Exception as e:
            logger.error(f"Error initializing arbitrage strategy: {e}")
    
    def _initialize_generic_strategy(self):
        """Initialize generic strategy."""
        try:
            # Set default parameters for generic strategy
            default_params = {
                'position_size': 0.1,
                'stop_loss': 0.05,
                'take_profit': 0.10
            }
            
            # Update with custom parameters
            for key, value in default_params.items():
                if key not in self.config.parameters:
                    self.config.parameters[key] = value
            
        except Exception as e:
            logger.error(f"Error initializing generic strategy: {e}")
    
    def start_execution(self):
        """Start strategy execution."""
        try:
            if self.status == StrategyStatus.ERROR:
                logger.error(f"Cannot start strategy {self.config.name} - status is ERROR")
                return False
            
            self.is_running = True
            self.status = StrategyStatus.ACTIVE
            self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
            self.execution_thread.start()
            logger.info(f"Strategy {self.config.name} execution started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting strategy execution: {e}")
            return False
    
    def stop_execution(self):
        """Stop strategy execution."""
        try:
            self.is_running = False
            self.status = StrategyStatus.INACTIVE
            if self.execution_thread:
                self.execution_thread.join(timeout=5)
            logger.info(f"Strategy {self.config.name} execution stopped")
            
        except Exception as e:
            logger.error(f"Error stopping strategy execution: {e}")
    
    def _execution_loop(self):
        """Main execution loop."""
        while self.is_running:
            try:
                # Generate strategy signal
                signal = self._generate_signal()
                if signal:
                    self.signal_history.append(signal)
                    
                    # Execute signal if confidence is high enough
                    if signal.confidence > 0.7:
                        self._execute_signal(signal)
                
                # Update performance metrics
                self._update_performance()
                
                # Sleep for execution interval
                time.sleep(1)  # 1 second interval
                
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                self.status = StrategyStatus.ERROR
                time.sleep(5)
    
    def _generate_signal(self) -> Optional[StrategySignal]:
        """Generate trading signal based on strategy type."""
        try:
            if self.config.strategy_type == StrategyType.MOMENTUM:
                return self._generate_momentum_signal()
            elif self.config.strategy_type == StrategyType.MEAN_REVERSION:
                return self._generate_mean_reversion_signal()
            elif self.config.strategy_type == StrategyType.ARBITRAGE:
                return self._generate_arbitrage_signal()
            else:
                return self._generate_generic_signal()
                
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def _generate_momentum_signal(self) -> Optional[StrategySignal]:
        """Generate momentum strategy signal."""
        try:
            # Simulate momentum signal generation
            # In practice, this would use real market data
            current_price = 50000.0 + np.random.normal(0, 100)
            
            # Calculate momentum (simplified)
            momentum = np.random.normal(0, 0.02)
            threshold = self.config.parameters.get('momentum_threshold', 0.02)
            
            if abs(momentum) > threshold:
                signal_type = 'buy' if momentum > 0 else 'sell'
                confidence = min(abs(momentum) / threshold, 1.0)
            else:
                signal_type = 'hold'
                confidence = 0.5
            
            return StrategySignal(
                strategy_id=self.config.strategy_id,
                timestamp=datetime.now(),
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                volume=100.0,
                metadata={'momentum': momentum}
            )
            
        except Exception as e:
            logger.error(f"Error generating momentum signal: {e}")
            return None
    
    def _generate_mean_reversion_signal(self) -> Optional[StrategySignal]:
        """Generate mean reversion strategy signal."""
        try:
            # Simulate mean reversion signal generation
            current_price = 50000.0 + np.random.normal(0, 100)
            
            # Calculate deviation from mean (simplified)
            deviation = np.random.normal(0, 0.01)
            threshold = self.config.parameters.get('std_dev_threshold', 2.0)
            
            if abs(deviation) > threshold * 0.01:
                signal_type = 'sell' if deviation > 0 else 'buy'
                confidence = min(abs(deviation) / (threshold * 0.01), 1.0)
            else:
                signal_type = 'hold'
                confidence = 0.5
            
            return StrategySignal(
                strategy_id=self.config.strategy_id,
                timestamp=datetime.now(),
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                volume=100.0,
                metadata={'deviation': deviation}
            )
            
        except Exception as e:
            logger.error(f"Error generating mean reversion signal: {e}")
            return None
    
    def _generate_arbitrage_signal(self) -> Optional[StrategySignal]:
        """Generate arbitrage strategy signal."""
        try:
            # Simulate arbitrage signal generation
            price1 = 50000.0 + np.random.normal(0, 50)
            price2 = 50000.0 + np.random.normal(0, 50)
            
            spread = abs(price1 - price2) / min(price1, price2)
            threshold = self.config.parameters.get('spread_threshold', 0.001)
            
            if spread > threshold:
                signal_type = 'buy'  # Buy lower, sell higher
                confidence = min(spread / threshold, 1.0)
            else:
                signal_type = 'hold'
                confidence = 0.5
            
            return StrategySignal(
                strategy_id=self.config.strategy_id,
                timestamp=datetime.now(),
                signal_type=signal_type,
                confidence=confidence,
                price=(price1 + price2) / 2,
                volume=100.0,
                metadata={'spread': spread, 'price1': price1, 'price2': price2}
            )
            
        except Exception as e:
            logger.error(f"Error generating arbitrage signal: {e}")
            return None
    
    def _generate_generic_signal(self) -> Optional[StrategySignal]:
        """Generate generic strategy signal."""
        try:
            # Generic signal generation
            current_price = 50000.0 + np.random.normal(0, 100)
            
            # Random signal generation for testing
            signal_types = ['buy', 'sell', 'hold']
            signal_type = np.random.choice(signal_types, p=[0.3, 0.3, 0.4])
            confidence = np.random.uniform(0.5, 1.0) if signal_type != 'hold' else 0.5
            
            return StrategySignal(
                strategy_id=self.config.strategy_id,
                timestamp=datetime.now(),
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                volume=100.0,
                metadata={'strategy_type': 'generic'}
            )
            
        except Exception as e:
            logger.error(f"Error generating generic signal: {e}")
            return None
    
    def _execute_signal(self, signal: StrategySignal):
        """Execute a trading signal."""
        try:
            # Simulate trade execution
            trade = {
                'strategy_id': signal.strategy_id,
                'timestamp': signal.timestamp,
                'signal_type': signal.signal_type,
                'price': signal.price,
                'volume': signal.volume,
                'confidence': signal.confidence,
                'executed': True,
                'metadata': signal.metadata
            }
            
            self.trade_history.append(trade)
            logger.info(f"Executed {signal.signal_type} signal for strategy {signal.strategy_id}")
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    def _update_performance(self):
        """Update strategy performance metrics."""
        try:
            if len(self.trade_history) < 2:
                return
            
            # Calculate performance metrics
            trades = list(self.trade_history)
            
            # Calculate returns (simplified)
            returns = []
            for i in range(1, len(trades)):
                if trades[i]['signal_type'] == 'sell' and trades[i-1]['signal_type'] == 'buy':
                    return_pct = (trades[i]['price'] - trades[i-1]['price']) / trades[i-1]['price']
                    returns.append(return_pct)
            
            if not returns:
                return
            
            # Calculate performance metrics
            total_return = sum(returns)
            volatility = np.std(returns) if len(returns) > 1 else 0
            sharpe_ratio = (np.mean(returns) - 0.02/252) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
            
            # Calculate win rate
            winning_trades = sum(1 for r in returns if r > 0)
            total_trades = len(returns)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate profit factor
            gross_profit = sum(r for r in returns if r > 0)
            gross_loss = abs(sum(r for r in returns if r < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate max drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
            
            # Create performance record
            performance = StrategyPerformance(
                strategy_id=self.config.strategy_id,
                timestamp=datetime.now(),
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=total_trades - winning_trades
            )
            
            self.performance_history.append(performance)
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary."""
        try:
            if not self.performance_history:
                return {'strategy_id': self.config.strategy_id, 'status': 'No performance data'}
            
            latest_performance = self.performance_history[-1]
            
            summary = {
                'strategy_id': self.config.strategy_id,
                'strategy_name': self.config.name,
                'status': self.status.value,
                'total_return': latest_performance.total_return,
                'sharpe_ratio': latest_performance.sharpe_ratio,
                'max_drawdown': latest_performance.max_drawdown,
                'volatility': latest_performance.volatility,
                'win_rate': latest_performance.win_rate,
                'profit_factor': latest_performance.profit_factor,
                'total_trades': latest_performance.total_trades,
                'winning_trades': latest_performance.winning_trades,
                'losing_trades': latest_performance.losing_trades
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'strategy_id': self.config.strategy_id, 'status': 'Error'}

class StrategyManager:
    """Main strategy manager."""
    
    def __init__(self):
        self.strategies: Dict[str, StrategyExecutor] = {}
        self.strategy_configs: Dict[str, StrategyConfig] = {}
        self.portfolio_performance: deque = deque(maxlen=10000)
        self.correlation_matrix: Optional[np.ndarray] = None
        self.is_initialized = False
        self._initialize_manager()
    
    def _initialize_manager(self):
        """Initialize the strategy manager."""
        try:
            self.is_initialized = True
            logger.info("Strategy manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing strategy manager: {e}")
    
    def add_strategy(self, config: StrategyConfig) -> bool:
        """Add a strategy to the manager."""
        try:
            if not self.is_initialized:
                logger.error("Strategy manager not initialized")
                return False
            
            # Create strategy executor
            executor = StrategyExecutor(config)
            
            # Store strategy
            self.strategies[config.strategy_id] = executor
            self.strategy_configs[config.strategy_id] = config
            
            logger.info(f"Strategy {config.name} added to manager")
            return True
            
        except Exception as e:
            logger.error(f"Error adding strategy: {e}")
            return False
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy from the manager."""
        try:
            if strategy_id in self.strategies:
                # Stop execution if running
                executor = self.strategies[strategy_id]
                if executor.is_running:
                    executor.stop_execution()
                
                # Remove from collections
                del self.strategies[strategy_id]
                del self.strategy_configs[strategy_id]
                
                logger.info(f"Strategy {strategy_id} removed from manager")
                return True
            else:
                logger.warning(f"Strategy {strategy_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error removing strategy: {e}")
            return False
    
    def start_strategy(self, strategy_id: str) -> bool:
        """Start a strategy execution."""
        try:
            if strategy_id not in self.strategies:
                logger.error(f"Strategy {strategy_id} not found")
                return False
            
            executor = self.strategies[strategy_id]
            return executor.start_execution()
            
        except Exception as e:
            logger.error(f"Error starting strategy: {e}")
            return False
    
    def stop_strategy(self, strategy_id: str) -> bool:
        """Stop a strategy execution."""
        try:
            if strategy_id not in self.strategies:
                logger.error(f"Strategy {strategy_id} not found")
                return False
            
            executor = self.strategies[strategy_id]
            executor.stop_execution()
            return True
            
        except Exception as e:
            logger.error(f"Error stopping strategy: {e}")
            return False
    
    def start_all_strategies(self) -> Dict[str, bool]:
        """Start all enabled strategies."""
        try:
            results = {}
            
            for strategy_id, config in self.strategy_configs.items():
                if config.enabled:
                    results[strategy_id] = self.start_strategy(strategy_id)
                else:
                    results[strategy_id] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Error starting all strategies: {e}")
            return {}
    
    def stop_all_strategies(self) -> Dict[str, bool]:
        """Stop all strategies."""
        try:
            results = {}
            
            for strategy_id in self.strategies.keys():
                results[strategy_id] = self.stop_strategy(strategy_id)
            
            return results
            
        except Exception as e:
            logger.error(f"Error stopping all strategies: {e}")
            return {}
    
    def get_strategy_performance(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get performance summary for a specific strategy."""
        try:
            if strategy_id not in self.strategies:
                return None
            
            executor = self.strategies[strategy_id]
            return executor.get_performance_summary()
            
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return None
    
    def get_portfolio_performance(self) -> Dict[str, Any]:
        """Get overall portfolio performance."""
        try:
            if not self.strategies:
                return {'total_strategies': 0, 'status': 'No strategies'}
            
            # Aggregate performance across all strategies
            total_return = 0.0
            total_trades = 0
            active_strategies = 0
            
            strategy_performances = {}
            
            for strategy_id, executor in self.strategies.items():
                performance = executor.get_performance_summary()
                if performance and 'total_return' in performance:
                    total_return += performance['total_return']
                    total_trades += performance.get('total_trades', 0)
                    if executor.status == StrategyStatus.ACTIVE:
                        active_strategies += 1
                    
                    strategy_performances[strategy_id] = performance
            
            portfolio_performance = {
                'total_strategies': len(self.strategies),
                'active_strategies': active_strategies,
                'total_return': total_return,
                'total_trades': total_trades,
                'strategy_performances': strategy_performances
            }
            
            # Store portfolio performance
            self.portfolio_performance.append({
                'timestamp': datetime.now(),
                'performance': portfolio_performance
            })
            
            return portfolio_performance
            
        except Exception as e:
            logger.error(f"Error getting portfolio performance: {e}")
            return {'total_strategies': 0, 'status': 'Error'}
    
    def calculate_strategy_correlations(self) -> Optional[np.ndarray]:
        """Calculate correlation matrix between strategies."""
        try:
            if len(self.strategies) < 2:
                return None
            
            # Get performance data for all strategies
            performance_data = []
            strategy_ids = []
            
            for strategy_id, executor in self.strategies.items():
                if len(executor.performance_history) > 0:
                    returns = [p.total_return for p in executor.performance_history]
                    if len(returns) > 1:
                        performance_data.append(returns)
                        strategy_ids.append(strategy_id)
            
            if len(performance_data) < 2:
                return None
            
            # Pad arrays to same length
            max_length = max(len(data) for data in performance_data)
            padded_data = []
            
            for data in performance_data:
                if len(data) < max_length:
                    padded_data.append(data + [0.0] * (max_length - len(data)))
                else:
                    padded_data.append(data[:max_length])
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(padded_data)
            self.correlation_matrix = correlation_matrix
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating strategy correlations: {e}")
            return None
    
    def optimize_portfolio_weights(self) -> Dict[str, float]:
        """Optimize portfolio weights based on performance and correlations."""
        try:
            if len(self.strategies) < 2:
                return {strategy_id: 1.0 for strategy_id in self.strategies.keys()}
            
            # Get performance metrics
            performances = {}
            for strategy_id, executor in self.strategies.items():
                performance = executor.get_performance_summary()
                if performance and 'sharpe_ratio' in performance:
                    performances[strategy_id] = performance['sharpe_ratio']
            
            if not performances:
                return {strategy_id: 1.0/len(self.strategies) for strategy_id in self.strategies.keys()}
            
            # Simple weight optimization based on Sharpe ratio
            total_sharpe = sum(max(0, sharpe) for sharpe in performances.values())
            
            if total_sharpe > 0:
                weights = {strategy_id: max(0, sharpe) / total_sharpe 
                          for strategy_id, sharpe in performances.items()}
            else:
                # Equal weights if no positive Sharpe ratios
                weights = {strategy_id: 1.0/len(performances) for strategy_id in performances.keys()}
            
            return weights
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio weights: {e}")
            return {strategy_id: 1.0/len(self.strategies) for strategy_id in self.strategies.keys()}

def main():
    """Main function for testing."""
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create strategy manager
        manager = StrategyManager()
        
        # Create test strategies
        strategies = [
            StrategyConfig(
                strategy_id="momentum_001",
                strategy_type=StrategyType.MOMENTUM,
                name="Momentum Strategy",
                description="Simple momentum-based trading strategy",
                parameters={'lookback_period': 20, 'momentum_threshold': 0.02},
                risk_limits={'max_position_size': 0.2},
                performance_targets={'min_sharpe_ratio': 1.0}
            ),
            StrategyConfig(
                strategy_id="mean_reversion_001",
                strategy_type=StrategyType.MEAN_REVERSION,
                name="Mean Reversion Strategy",
                description="Mean reversion trading strategy",
                parameters={'lookback_period': 50, 'std_dev_threshold': 2.0},
                risk_limits={'max_position_size': 0.15},
                performance_targets={'min_sharpe_ratio': 0.8}
            ),
            StrategyConfig(
                strategy_id="arbitrage_001",
                strategy_type=StrategyType.ARBITRAGE,
                name="Arbitrage Strategy",
                description="Statistical arbitrage strategy",
                parameters={'spread_threshold': 0.001},
                risk_limits={'max_position_size': 0.1},
                performance_targets={'min_sharpe_ratio': 1.5}
            )
        ]
        
        # Add strategies to manager
        for config in strategies:
            manager.add_strategy(config)
        
        # Start all strategies
        start_results = manager.start_all_strategies()
        print("Strategy start results:", start_results)
        
        # Let strategies run for a while
        time.sleep(10)
        
        # Get performance summaries
        for strategy_id in manager.strategies.keys():
            performance = manager.get_strategy_performance(strategy_id)
            print(f"Strategy {strategy_id} performance:")
            print(json.dumps(performance, indent=2, default=str))
            print("-" * 50)
        
        # Get portfolio performance
        portfolio_performance = manager.get_portfolio_performance()
        print("Portfolio Performance:")
        print(json.dumps(portfolio_performance, indent=2, default=str))
        
        # Calculate correlations
        correlations = manager.calculate_strategy_correlations()
        if correlations is not None:
            print("Strategy Correlations:")
            print(correlations)
        
        # Optimize weights
        weights = manager.optimize_portfolio_weights()
        print("Optimized Portfolio Weights:")
        print(json.dumps(weights, indent=2))
        
        # Stop all strategies
        stop_results = manager.stop_all_strategies()
        print("Strategy stop results:", stop_results)
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 