#!/usr/bin/env python3
"""
Portfolio Integration Module
===========================

Connects wallet tracker to entire Schwabot strategy system.
Provides integration between CCXT/Coinbase API, strategy mapper,
Ferris RDE, profit allocator, and other core modules.
"""

import time
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

# Import Schwabot core modules
from .wallet_tracker import WalletTracker, AssetType, PositionType
from .strategy_mapper import StrategyMapper, StrategyConfig
from .ferris_rde import FerrisRDE, FerrisPhase
from .profit_cycle_allocator import ProfitCycleAllocator, AllocationStage
from .fallback_logic import FallbackLogic, FallbackType
from .matrix_map_logic import MatrixMapLogic, MatrixType
from .glyph_vm import GlyphVM, GlyphType

logger = logging.getLogger(__name__)


@dataclass
class IntegrationState:
    """Integration state data."""
    timestamp: float
    wallet_synced: bool
    strategy_active: bool
    ferris_phase: str
    portfolio_value: float
    cash_ratio: float
    pnl_ratio: float
    asset_diversity: int
    strategy_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeSignal:
    """Trade signal from integrated system."""
    signal_id: str
    timestamp: float
    asset: AssetType
    action: str  # "buy", "sell", "hold", "rebalance"
    quantity: float
    price: float
    confidence: float
    source: str  # "strategy", "ferris", "fallback", "rebalance"
    metadata: Dict[str, Any] = field(default_factory=dict)


class PortfolioIntegration:
    """
    Portfolio Integration for Schwabot v0.05.
    
    Connects wallet tracker to entire Schwabot strategy system,
    providing seamless integration between CCXT/Coinbase API,
    strategy mapper, Ferris RDE, profit allocator, and other core modules.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the portfolio integration."""
        self.config = config or self._default_config()
        
        # Core modules
        self.wallet_tracker: Optional[WalletTracker] = None
        self.strategy_mapper: Optional[StrategyMapper] = None
        self.ferris_rde: Optional[FerrisRDE] = None
        self.profit_allocator: Optional[ProfitCycleAllocator] = None
        self.fallback_logic: Optional[FallbackLogic] = None
        self.matrix_logic: Optional[MatrixMapLogic] = None
        self.glyph_vm: Optional[GlyphVM] = None
        
        # Integration state
        self.integration_state: Optional[IntegrationState] = None
        self.trade_signals: List[TradeSignal] = []
        self.max_signal_history = self.config.get('max_signal_history', 100)
        
        # Performance tracking
        self.total_signals = 0
        self.successful_trades = 0
        self.failed_trades = 0
        
        # State management
        self.last_sync = time.time()
        self.sync_interval = self.config.get('sync_interval', 60)  # 1 minute
        
        logger.info("🔗 Portfolio Integration initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'max_signal_history': 100,
            'sync_interval': 60,  # 1 minute
            'auto_sync_enabled': True,
            'strategy_integration_enabled': True,
            'ferris_integration_enabled': True,
            'profit_integration_enabled': True,
            'fallback_integration_enabled': True,
            'matrix_integration_enabled': True,
            'glyph_integration_enabled': True,
            'rebalance_threshold': 0.1,  # 10% threshold
            'cash_deployment_threshold': 0.5,  # 50% cash threshold
            'diversification_minimum': 2,  # Minimum assets
            'api_enabled': False,
            'exchanges': {
                'coinbase': {
                    'enabled': False,
                    'api_key': '',
                    'secret': '',
                    'passphrase': '',
                    'sandbox': True
                }
            }
        }
    
    def initialize_modules(self):
        """Initialize all Schwabot core modules."""
        try:
            # Initialize wallet tracker with API config
            wallet_config = {
                'api_enabled': self.config.get('api_enabled', False),
                'exchanges': self.config.get('exchanges', {}),
                'strategy_integration_enabled': True
            }
            self.wallet_tracker = WalletTracker(wallet_config)
            
            # Initialize other modules
            self.strategy_mapper = StrategyMapper()
            self.ferris_rde = FerrisRDE()
            self.profit_allocator = ProfitCycleAllocator()
            self.fallback_logic = FallbackLogic()
            self.matrix_logic = MatrixMapLogic()
            self.glyph_vm = GlyphVM()
            
            logger.info("✅ All Schwabot modules initialized")
            
        except Exception as e:
            logger.error(f"Error initializing modules: {e}")
    
    def sync_all_modules(self) -> bool:
        """
        Synchronize all modules with current portfolio state.
        
        Returns:
            True if sync was successful
        """
        try:
            if not self.wallet_tracker:
                logger.error("Wallet tracker not initialized")
                return False
            
            # Sync wallet with exchanges
            wallet_synced = self.wallet_tracker.sync_portfolio_with_exchanges()
            
            # Get portfolio summary
            portfolio_summary = self.wallet_tracker.get_portfolio_summary()
            
            # Generate strategy hash
            strategy_hash = self.wallet_tracker.generate_strategy_hash()
            
            # Update integration state
            self.integration_state = IntegrationState(
                timestamp=time.time(),
                wallet_synced=wallet_synced,
                strategy_active=True,
                ferris_phase=self.ferris_rde.current_phase.value if self.ferris_rde.current_phase else "unknown",
                portfolio_value=portfolio_summary['total_value'],
                cash_ratio=portfolio_summary['cash_balance'] / portfolio_summary['total_value'] if portfolio_summary['total_value'] > 0 else 0,
                pnl_ratio=portfolio_summary['total_pnl'] / portfolio_summary['total_value'] if portfolio_summary['total_value'] > 0 else 0,
                asset_diversity=len([a for a in portfolio_summary['asset_breakdown'].values() if a['value'] > 0]),
                strategy_hash=strategy_hash
            )
            
            # Inject into strategy mapper
            if self.strategy_mapper:
                self.wallet_tracker.inject_into_strategy_mapper(self.strategy_mapper)
            
            # Connect to Ferris RDE
            if self.ferris_rde:
                self.wallet_tracker.connect_to_ferris_rde(self.ferris_rde)
            
            # Update glyph visualizer
            if self.glyph_vm:
                self._update_glyph_visualizer()
            
            self.last_sync = time.time()
            logger.info("✅ All modules synchronized")
            return True
            
        except Exception as e:
            logger.error(f"Error syncing modules: {e}")
            return False
    
    def _update_glyph_visualizer(self):
        """Update glyph visualizer with current portfolio state."""
        try:
            if not self.integration_state:
                return
            
            # Add portfolio value glyph
            self.glyph_vm.add_glyph(
                "portfolio_value",
                GlyphType.PERFORMANCE,
                self.glyph_vm._determine_glyph_state(self.integration_state.portfolio_value),
                self.integration_state.portfolio_value
            )
            
            # Add cash ratio glyph
            self.glyph_vm.add_glyph(
                "cash_ratio",
                GlyphType.SYSTEM,
                self.glyph_vm._determine_glyph_state(self.integration_state.cash_ratio),
                self.integration_state.cash_ratio
            )
            
            # Add PNL ratio glyph
            self.glyph_vm.add_glyph(
                "pnl_ratio",
                GlyphType.PERFORMANCE,
                self.glyph_vm._determine_glyph_state(self.integration_state.pnl_ratio),
                self.integration_state.pnl_ratio
            )
            
            # Add asset diversity glyph
            self.glyph_vm.add_glyph(
                "asset_diversity",
                GlyphType.SYSTEM,
                self.glyph_vm._determine_glyph_state(self.integration_state.asset_diversity),
                self.integration_state.asset_diversity
            )
            
        except Exception as e:
            logger.error(f"Error updating glyph visualizer: {e}")
    
    def evaluate_portfolio_strategy(self) -> Optional[TradeSignal]:
        """
        Evaluate portfolio and generate trade signals.
        
        Returns:
            Trade signal if action is needed
        """
        try:
            if not self.integration_state:
                return None
            
            # Check for rebalancing needs
            if self.wallet_tracker.should_trigger_rebalance():
                return self._generate_rebalance_signal()
            
            # Check for cash deployment
            if self.integration_state.cash_ratio > self.config['cash_deployment_threshold']:
                return self._generate_cash_deployment_signal()
            
            # Check for strategy-based signals
            if self.strategy_mapper:
                return self._generate_strategy_signal()
            
            # Check for Ferris-based signals
            if self.ferris_rde:
                return self._generate_ferris_signal()
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating portfolio strategy: {e}")
            return None
    
    def _generate_rebalance_signal(self) -> TradeSignal:
        """Generate rebalancing trade signal."""
        try:
            suggestions = self.wallet_tracker.get_rebalance_suggestions()
            
            if not suggestions:
                return None
            
            # Get highest priority suggestion
            high_priority = [s for s in suggestions if s['priority'] == 'high']
            suggestion = high_priority[0] if high_priority else suggestions[0]
            
            # Determine action based on suggestion type
            if suggestion['type'] == 'deploy_cash':
                action = "buy"
                asset = self._select_best_asset_for_deployment()
                quantity = self._calculate_deployment_quantity()
            elif suggestion['type'] == 'diversify':
                action = "sell"
                asset = self._get_overweight_asset()
                quantity = self._calculate_diversification_quantity()
            else:
                action = "hold"
                asset = AssetType.USDC
                quantity = 0.0
            
            signal = TradeSignal(
                signal_id=f"rebalance_{int(time.time() * 1000)}",
                timestamp=time.time(),
                asset=asset,
                action=action,
                quantity=quantity,
                price=self.wallet_tracker._get_current_price(asset.value),
                confidence=0.8,
                source="rebalance",
                metadata={'suggestion': suggestion}
            )
            
            self._add_trade_signal(signal)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating rebalance signal: {e}")
            return None
    
    def _generate_cash_deployment_signal(self) -> TradeSignal:
        """Generate cash deployment signal."""
        try:
            asset = self._select_best_asset_for_deployment()
            quantity = self._calculate_deployment_quantity()
            
            signal = TradeSignal(
                signal_id=f"deploy_{int(time.time() * 1000)}",
                timestamp=time.time(),
                asset=asset,
                action="buy",
                quantity=quantity,
                price=self.wallet_tracker._get_current_price(asset.value),
                confidence=0.7,
                source="cash_deployment"
            )
            
            self._add_trade_signal(signal)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating cash deployment signal: {e}")
            return None
    
    def _generate_strategy_signal(self) -> Optional[TradeSignal]:
        """Generate strategy-based trade signal."""
        try:
            # Get current market data (simulated)
            market_data = {
                'price': 45000.0,  # BTC price
                'volume': 1000000.0,
                'volatility': 0.02,
                'timestamp': time.time()
            }
            
            # Get portfolio state
            portfolio_state = {
                'total_value': self.integration_state.portfolio_value,
                'cash_balance': self.integration_state.portfolio_value * self.integration_state.cash_ratio,
                'strategy_hash': self.integration_state.strategy_hash
            }
            
            # Select strategy
            strategy_config = self.strategy_mapper.select_strategy(market_data, portfolio_state)
            
            if not strategy_config:
                return None
            
            # Execute strategy
            strategy_result = self.strategy_mapper.execute_strategy(
                strategy_config, market_data, portfolio_state
            )
            
            if not strategy_result:
                return None
            
            # Convert to trade signal
            signal = TradeSignal(
                signal_id=f"strategy_{int(time.time() * 1000)}",
                timestamp=time.time(),
                asset=AssetType.BTC,  # Default to BTC
                action=strategy_result.signal_type,
                quantity=strategy_result.position_size,
                price=market_data['price'],
                confidence=strategy_result.confidence,
                source="strategy",
                metadata={'strategy_id': strategy_result.strategy_id}
            )
            
            self._add_trade_signal(signal)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating strategy signal: {e}")
            return None
    
    def _generate_ferris_signal(self) -> Optional[TradeSignal]:
        """Generate Ferris RDE-based trade signal."""
        try:
            # Get current market data
            market_data = {
                'price': 45000.0,
                'volume': 1000000.0,
                'volatility': 0.02,
                'timestamp': time.time()
            }
            
            # Generate Ferris signal
            ferris_signal = self.ferris_rde.generate_signal(market_data)
            
            if not ferris_signal:
                return None
            
            # Convert to trade signal
            signal = TradeSignal(
                signal_id=f"ferris_{int(time.time() * 1000)}",
                timestamp=time.time(),
                asset=AssetType.BTC,  # Default to BTC
                action=ferris_signal.signal_type,
                quantity=ferris_signal.strength * 0.1,  # Scale strength to quantity
                price=market_data['price'],
                confidence=ferris_signal.confidence,
                source="ferris",
                metadata={'phase': ferris_signal.phase.value}
            )
            
            self._add_trade_signal(signal)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating Ferris signal: {e}")
            return None
    
    def _select_best_asset_for_deployment(self) -> AssetType:
        """Select best asset for cash deployment."""
        # Simple logic - prefer BTC and ETH
        portfolio_summary = self.wallet_tracker.get_portfolio_summary()
        
        btc_value = portfolio_summary['asset_breakdown'].get('BTC', {}).get('value', 0)
        eth_value = portfolio_summary['asset_breakdown'].get('ETH', {}).get('value', 0)
        
        if btc_value < eth_value:
            return AssetType.BTC
        else:
            return AssetType.ETH
    
    def _calculate_deployment_quantity(self) -> float:
        """Calculate quantity for cash deployment."""
        portfolio_summary = self.wallet_tracker.get_portfolio_summary()
        cash_balance = portfolio_summary['cash_balance']
        
        # Deploy 20% of cash balance
        return cash_balance * 0.2
    
    def _get_overweight_asset(self) -> AssetType:
        """Get overweight asset for diversification."""
        portfolio_summary = self.wallet_tracker.get_portfolio_summary()
        
        for asset, data in portfolio_summary['asset_breakdown'].items():
            if data['percentage'] > 0.7:  # Over 70%
                return AssetType(asset)
        
        return AssetType.BTC  # Default
    
    def _calculate_diversification_quantity(self) -> float:
        """Calculate quantity for diversification."""
        # Reduce position by 10%
        return 0.1
    
    def _add_trade_signal(self, signal: TradeSignal):
        """Add trade signal to history."""
        self.trade_signals.append(signal)
        if len(self.trade_signals) > self.max_signal_history:
            self.trade_signals.pop(0)
        
        self.total_signals += 1
    
    def execute_trade_signal(self, signal: TradeSignal) -> bool:
        """
        Execute a trade signal.
        
        Args:
            signal: Trade signal to execute
            
        Returns:
            True if execution was successful
        """
        try:
            if signal.action == "hold":
                logger.info(f"Hold signal for {signal.asset.value}")
                return True
            
            # Create transaction
            transaction = self.wallet_tracker.add_transaction(
                asset=signal.asset,
                transaction_type=signal.action,
                quantity=signal.quantity,
                price=signal.price,
                fee=signal.quantity * signal.price * 0.001  # 0.1% fee
            )
            
            if transaction:
                self.successful_trades += 1
                logger.info(f"Executed {signal.action} {signal.quantity} {signal.asset.value} @ ${signal.price:.2f}")
                return True
            else:
                self.failed_trades += 1
                logger.error(f"Failed to execute trade signal: {signal.signal_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade signal: {e}")
            self.failed_trades += 1
            return False
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get integration summary."""
        return {
            "modules_initialized": all([
                self.wallet_tracker, self.strategy_mapper, self.ferris_rde,
                self.profit_allocator, self.fallback_logic, self.matrix_logic,
                self.glyph_vm
            ]),
            "last_sync": self.last_sync,
            "total_signals": self.total_signals,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "success_rate": self.successful_trades / self.total_signals if self.total_signals > 0 else 0.0,
            "integration_state": {
                "wallet_synced": self.integration_state.wallet_synced if self.integration_state else False,
                "strategy_active": self.integration_state.strategy_active if self.integration_state else False,
                "ferris_phase": self.integration_state.ferris_phase if self.integration_state else "unknown",
                "portfolio_value": self.integration_state.portfolio_value if self.integration_state else 0.0,
                "cash_ratio": self.integration_state.cash_ratio if self.integration_state else 0.0,
                "pnl_ratio": self.integration_state.pnl_ratio if self.integration_state else 0.0,
                "asset_diversity": self.integration_state.asset_diversity if self.integration_state else 0
            }
        }
    
    def get_recent_signals(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent trade signals."""
        recent_signals = self.trade_signals[-count:]
        return [
            {
                "signal_id": signal.signal_id,
                "timestamp": signal.timestamp,
                "asset": signal.asset.value,
                "action": signal.action,
                "quantity": signal.quantity,
                "price": signal.price,
                "confidence": signal.confidence,
                "source": signal.source
            }
            for signal in recent_signals
        ]
    
    def run_integration_cycle(self) -> bool:
        """
        Run a complete integration cycle.
        
        Returns:
            True if cycle was successful
        """
        try:
            # Sync all modules
            if not self.sync_all_modules():
                return False
            
            # Evaluate portfolio strategy
            signal = self.evaluate_portfolio_strategy()
            
            # Execute signal if confidence is high enough
            if signal and signal.confidence > 0.7:
                return self.execute_trade_signal(signal)
            
            return True
            
        except Exception as e:
            logger.error(f"Error running integration cycle: {e}")
            return False
    
    def start_continuous_integration(self, interval: Optional[float] = None):
        """
        Start continuous integration loop.
        
        Args:
            interval: Integration interval in seconds (default: sync_interval)
        """
        if interval is None:
            interval = self.sync_interval
        
        logger.info(f"Starting continuous integration (interval: {interval}s)")
        
        try:
            while True:
                self.run_integration_cycle()
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Continuous integration stopped by user")
        except Exception as e:
            logger.error(f"Error in continuous integration: {e}") 