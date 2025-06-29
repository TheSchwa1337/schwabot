#!/usr/bin/env python3
"""
Wallet Tracker Module
=====================

Tracks BTC, ETH, XRP, USDC, SOL positions with PNL + long-hold ledger
for Schwabot v0.05. Provides comprehensive portfolio management with
CCXT/Coinbase API integration and strategy system connectivity.
"""

import time
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np

# CCXT integration
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.warning("CCXT not available - API integration disabled")

logger = logging.getLogger(__name__)


class AssetType(Enum):
    """Asset type enumeration."""
    BTC = "BTC"
    ETH = "ETH"
    XRP = "XRP"
    USDC = "USDC"
    SOL = "SOL"


class PositionType(Enum):
    """Position type enumeration."""
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"


class ExchangeType(Enum):
    """Exchange type enumeration."""
    COINBASE = "coinbase"
    COINBASE_PRO = "coinbasepro"
    BINANCE = "binance"
    KRAKEN = "kraken"
    GEMINI = "gemini"


@dataclass
class Position:
    """Position data."""
    position_id: str
    asset: AssetType
    position_type: PositionType
    quantity: float
    entry_price: float
    current_price: float
    entry_time: float
    last_update: float
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transaction:
    """Transaction data."""
    transaction_id: str
    timestamp: float
    asset: AssetType
    transaction_type: str  # "buy", "sell", "transfer_in", "transfer_out"
    quantity: float
    price: float
    total_value: float
    fee: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot."""
    snapshot_id: str
    timestamp: float
    total_value: float
    total_pnl: float
    total_pnl_percentage: float
    positions: Dict[str, Position]
    cash_balance: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExchangeBalance:
    """Exchange balance data."""
    exchange: ExchangeType
    asset: AssetType
    free: float
    used: float
    total: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class WalletTracker:
    """
    Wallet Tracker for Schwabot v0.05.
    
    Tracks BTC, ETH, XRP, USDC, SOL positions with PNL + long-hold ledger
    for comprehensive portfolio management with CCXT/Coinbase API integration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the wallet tracker."""
        self.config = config or self._default_config()
        
        # Position management
        self.positions: Dict[str, Position] = {}
        self.asset_positions: Dict[AssetType, List[Position]] = {}
        
        # Transaction history
        self.transactions: List[Transaction] = []
        self.max_transaction_history = self.config.get('max_transaction_history', 1000)
        
        # Portfolio snapshots
        self.snapshots: List[PortfolioSnapshot] = []
        self.max_snapshot_history = self.config.get('max_snapshot_history', 100)
        
        # Performance tracking
        self.total_transactions = 0
        self.total_pnl = 0.0
        self.total_fees = 0.0
        
        # State management
        self.cash_balance = self.config.get('initial_cash_balance', 10000.0)
        self.last_update = time.time()
        
        # API integration
        self.exchanges: Dict[str, Any] = {}
        self.exchange_balances: Dict[str, ExchangeBalance] = {}
        self.api_enabled = self.config.get('api_enabled', False)
        
        # Strategy integration
        self.strategy_hash_registry = {}
        self.ferris_cycle_data = {}
        
        # Initialize components
        self._initialize_asset_tracking()
        self._initialize_api_connections()
        
        logger.info("💼 Wallet Tracker initialized with API integration")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'max_transaction_history': 1000,
            'max_snapshot_history': 100,
            'initial_cash_balance': 10000.0,
            'snapshot_interval': 3600,  # 1 hour
            'pnl_update_interval': 300,  # 5 minutes
            'supported_assets': ['BTC', 'ETH', 'XRP', 'USDC', 'SOL'],
            'default_fee_rate': 0.001,  # 0.1%
            'min_position_size': 10.0,  # $10 minimum
            'auto_snapshot_enabled': True,
            'api_enabled': False,
            'exchanges': {
                'coinbase': {
                    'enabled': False,
                    'api_key': '',
                    'secret': '',
                    'passphrase': '',
                    'sandbox': True
                },
                'binance': {
                    'enabled': False,
                    'api_key': '',
                    'secret': '',
                    'sandbox': True
                }
            },
            'sync_interval': 60,  # 1 minute
            'strategy_integration_enabled': True
        }
    
    def _initialize_asset_tracking(self):
        """Initialize asset position tracking."""
        for asset in AssetType:
            self.asset_positions[asset] = []
    
    def _initialize_api_connections(self):
        """Initialize API connections to exchanges."""
        if not CCXT_AVAILABLE or not self.api_enabled:
            logger.info("API integration disabled - using simulated data")
            return
        
        exchanges_config = self.config.get('exchanges', {})
        
        for exchange_name, exchange_config in exchanges_config.items():
            if not exchange_config.get('enabled', False):
                continue
            
            try:
                if exchange_name == 'coinbase':
                    exchange = ccxt.coinbasepro({
                        'apiKey': exchange_config['api_key'],
                        'secret': exchange_config['secret'],
                        'password': exchange_config.get('passphrase', ''),
                        'sandbox': exchange_config.get('sandbox', True)
                    })
                elif exchange_name == 'binance':
                    exchange = ccxt.binance({
                        'apiKey': exchange_config['api_key'],
                        'secret': exchange_config['secret'],
                        'sandbox': exchange_config.get('sandbox', True)
                    })
                else:
                    logger.warning(f"Unsupported exchange: {exchange_name}")
                    continue
                
                self.exchanges[exchange_name] = exchange
                logger.info(f"Connected to {exchange_name}")
                
            except Exception as e:
                logger.error(f"Failed to connect to {exchange_name}: {e}")
    
    def fetch_exchange_balances(self) -> Dict[str, Dict[str, float]]:
        """
        Fetch balances from all connected exchanges.
        
        Returns:
            Dictionary of exchange balances
        """
        balances = {}
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                raw_balance = exchange.fetch_balance()
                
                # Extract free balances for supported assets
                exchange_balances = {}
                for asset in AssetType:
                    asset_symbol = asset.value
                    if asset_symbol in raw_balance['total']:
                        free_amount = float(raw_balance['free'].get(asset_symbol, 0))
                        used_amount = float(raw_balance['used'].get(asset_symbol, 0))
                        total_amount = float(raw_balance['total'].get(asset_symbol, 0))
                        
                        if total_amount > 0:
                            exchange_balances[asset_symbol] = {
                                'free': free_amount,
                                'used': used_amount,
                                'total': total_amount
                            }
                            
                            # Store as ExchangeBalance object
                            balance_key = f"{exchange_name}_{asset_symbol}"
                            self.exchange_balances[balance_key] = ExchangeBalance(
                                exchange=ExchangeType(exchange_name),
                                asset=asset,
                                free=free_amount,
                                used=used_amount,
                                total=total_amount,
                                timestamp=time.time()
                            )
                
                balances[exchange_name] = exchange_balances
                logger.debug(f"Fetched balances from {exchange_name}: {exchange_balances}")
                
            except Exception as e:
                logger.error(f"Error fetching balances from {exchange_name}: {e}")
                balances[exchange_name] = {}
        
        return balances
    
    def sync_portfolio_with_exchanges(self) -> bool:
        """
        Synchronize portfolio with exchange balances.
        
        Returns:
            True if sync was successful
        """
        try:
            if not self.api_enabled:
                logger.info("API disabled - using simulated sync")
                return self._simulate_portfolio_sync()
            
            balances = self.fetch_exchange_balances()
            
            # Update positions based on exchange balances
            for exchange_name, exchange_balances in balances.items():
                for asset_symbol, balance_data in exchange_balances.items():
                    asset = AssetType(asset_symbol)
                    total_amount = balance_data['total']
                    
                    if total_amount > 0:
                        # Check if we have a position for this asset
                        existing_positions = self.asset_positions.get(asset, [])
                        
                        if not existing_positions:
                            # Create new position
                            current_price = self._get_current_price(asset_symbol)
                            self.add_position(
                                asset=asset,
                                position_type=PositionType.LONG,
                                quantity=total_amount,
                                entry_price=current_price
                            )
                        else:
                            # Update existing position
                            for position in existing_positions:
                                if position.quantity != total_amount:
                                    # Position size changed
                                    self._update_position_quantity(
                                        position.position_id, 
                                        total_amount
                                    )
            
            self.last_update = time.time()
            logger.info("Portfolio synchronized with exchanges")
            return True
            
        except Exception as e:
            logger.error(f"Error syncing portfolio: {e}")
            return False
    
    def _simulate_portfolio_sync(self) -> bool:
        """Simulate portfolio sync for testing."""
        # Simulate some positions
        simulated_balances = {
            'BTC': 0.05,
            'ETH': 0.5,
            'USDC': 5000.0,
            'SOL': 10.0,
            'XRP': 1000.0
        }
        
        for asset_symbol, amount in simulated_balances.items():
            if amount > 0:
                asset = AssetType(asset_symbol)
                current_price = self._get_simulated_price(asset_symbol)
                
                # Add or update position
                existing_positions = self.asset_positions.get(asset, [])
                if not existing_positions:
                    self.add_position(
                        asset=asset,
                        position_type=PositionType.LONG,
                        quantity=amount,
                        entry_price=current_price
                    )
        
        return True
    
    def _get_current_price(self, asset_symbol: str) -> float:
        """Get current price for an asset."""
        try:
            if not self.api_enabled:
                return self._get_simulated_price(asset_symbol)
            
            # Try to get price from first available exchange
            for exchange in self.exchanges.values():
                try:
                    ticker = exchange.fetch_ticker(f"{asset_symbol}/USDC")
                    return float(ticker['last'])
                except:
                    continue
            
            # Fallback to simulated price
            return self._get_simulated_price(asset_symbol)
            
        except Exception as e:
            logger.error(f"Error getting price for {asset_symbol}: {e}")
            return self._get_simulated_price(asset_symbol)
    
    def _get_simulated_price(self, asset_symbol: str) -> float:
        """Get simulated price for testing."""
        prices = {
            'BTC': 45000.0,
            'ETH': 3000.0,
            'USDC': 1.0,
            'SOL': 100.0,
            'XRP': 0.5
        }
        return prices.get(asset_symbol, 1.0)
    
    def _update_position_quantity(self, position_id: str, new_quantity: float):
        """Update position quantity."""
        if position_id in self.positions:
            position = self.positions[position_id]
            old_quantity = position.quantity
            position.quantity = new_quantity
            position.last_update = time.time()
            
            logger.info(f"Updated position {position_id}: {old_quantity} -> {new_quantity}")
    
    def generate_strategy_hash(self) -> str:
        """
        Generate strategy hash based on current portfolio state.
        
        Returns:
            Strategy hash string
        """
        try:
            # Get current holdings
            holdings = {}
            for asset in AssetType:
                positions = self.asset_positions.get(asset, [])
                total_quantity = sum(p.quantity for p in positions)
                holdings[asset.value] = total_quantity
            
            # Create hash string
            hash_data = {
                'holdings': holdings,
                'total_value': self.get_portfolio_summary()['total_value'],
                'timestamp': int(time.time())
            }
            
            hash_string = json.dumps(hash_data, sort_keys=True)
            return hashlib.sha256(hash_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating strategy hash: {e}")
            return ""
    
    def inject_into_strategy_mapper(self, strategy_mapper):
        """
        Inject portfolio data into strategy mapper.
        
        Args:
            strategy_mapper: Strategy mapper instance
        """
        try:
            if not self.config.get('strategy_integration_enabled', True):
                return
            
            # Get portfolio summary
            portfolio_summary = self.get_portfolio_summary()
            
            # Generate strategy hash
            strategy_hash = self.generate_strategy_hash()
            
            # Create portfolio state for strategy mapper
            portfolio_state = {
                'total_value': portfolio_summary['total_value'],
                'cash_balance': portfolio_summary['cash_balance'],
                'total_pnl': portfolio_summary['total_pnl'],
                'asset_breakdown': portfolio_summary['asset_breakdown'],
                'strategy_hash': strategy_hash,
                'last_update': self.last_update
            }
            
            # Store in strategy hash registry
            self.strategy_hash_registry[strategy_hash] = {
                'portfolio_state': portfolio_state,
                'timestamp': time.time(),
                'used_count': 0
            }
            
            logger.info(f"Injected portfolio state into strategy mapper (hash: {strategy_hash[:8]}...)")
            
        except Exception as e:
            logger.error(f"Error injecting into strategy mapper: {e}")
    
    def connect_to_ferris_rde(self, ferris_rde):
        """
        Connect wallet tracker to Ferris RDE.
        
        Args:
            ferris_rde: Ferris RDE instance
        """
        try:
            # Get current portfolio state
            portfolio_summary = self.get_portfolio_summary()
            
            # Create Ferris cycle data
            self.ferris_cycle_data = {
                'portfolio_value': portfolio_summary['total_value'],
                'cash_ratio': portfolio_summary['cash_balance'] / portfolio_summary['total_value'] if portfolio_summary['total_value'] > 0 else 0,
                'pnl_ratio': portfolio_summary['total_pnl'] / portfolio_summary['total_value'] if portfolio_summary['total_value'] > 0 else 0,
                'asset_diversity': len([a for a in portfolio_summary['asset_breakdown'].values() if a['value'] > 0]),
                'timestamp': time.time()
            }
            
            logger.info(f"Connected to Ferris RDE - Portfolio value: ${portfolio_summary['total_value']:.2f}")
            
        except Exception as e:
            logger.error(f"Error connecting to Ferris RDE: {e}")
    
    def get_ferris_cycle_data(self) -> Dict[str, Any]:
        """Get Ferris cycle data."""
        return self.ferris_cycle_data.copy()
    
    def should_trigger_rebalance(self) -> bool:
        """
        Check if portfolio should trigger rebalancing.
        
        Returns:
            True if rebalancing is needed
        """
        try:
            portfolio_summary = self.get_portfolio_summary()
            
            # Check cash ratio
            cash_ratio = portfolio_summary['cash_balance'] / portfolio_summary['total_value'] if portfolio_summary['total_value'] > 0 else 0
            
            # Check PNL threshold
            pnl_ratio = portfolio_summary['total_pnl'] / portfolio_summary['total_value'] if portfolio_summary['total_value'] > 0 else 0
            
            # Check asset diversity
            asset_count = len([a for a in portfolio_summary['asset_breakdown'].values() if a['value'] > 0])
            
            # Trigger conditions
            if cash_ratio > 0.8:  # Too much cash
                return True
            elif pnl_ratio < -0.1:  # Significant losses
                return True
            elif asset_count < 2:  # Not enough diversification
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking rebalance trigger: {e}")
            return False
    
    def get_rebalance_suggestions(self) -> List[Dict[str, Any]]:
        """
        Get portfolio rebalancing suggestions.
        
        Returns:
            List of rebalancing suggestions
        """
        try:
            suggestions = []
            portfolio_summary = self.get_portfolio_summary()
            
            # Check cash deployment
            cash_ratio = portfolio_summary['cash_balance'] / portfolio_summary['total_value'] if portfolio_summary['total_value'] > 0 else 0
            if cash_ratio > 0.5:
                suggestions.append({
                    'type': 'deploy_cash',
                    'reason': f'High cash ratio ({cash_ratio:.1%})',
                    'action': 'Consider deploying cash into assets',
                    'priority': 'high'
                })
            
            # Check asset concentration
            for asset, data in portfolio_summary['asset_breakdown'].items():
                if data['percentage'] > 0.7:  # Over 70% in one asset
                    suggestions.append({
                        'type': 'diversify',
                        'reason': f'High concentration in {asset} ({data["percentage"]:.1%})',
                        'action': f'Consider reducing {asset} position',
                        'priority': 'medium'
                    })
            
            # Check for losses
            if portfolio_summary['total_pnl'] < 0:
                suggestions.append({
                    'type': 'risk_management',
                    'reason': f'Portfolio showing losses (${portfolio_summary["total_pnl"]:.2f})',
                    'action': 'Review stop-loss levels and risk management',
                    'priority': 'high'
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting rebalance suggestions: {e}")
            return []
    
    def add_position(self, asset: AssetType, position_type: PositionType,
                    quantity: float, entry_price: float) -> Optional[Position]:
        """
        Add a new position.
        
        Args:
            asset: Asset type
            position_type: Position type (long/short/hold)
            quantity: Position quantity
            entry_price: Entry price
            
        Returns:
            Created position
        """
        try:
            position_id = f"{asset.value}_{position_type.value}_{int(time.time() * 1000)}"
            
            position = Position(
                position_id=position_id,
                asset=asset,
                position_type=position_type,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                entry_time=time.time(),
                last_update=time.time()
            )
            
            self.positions[position_id] = position
            self.asset_positions[asset].append(position)
            
            logger.info(f"Added position: {position_id} ({asset.value}, {quantity:.4f} @ ${entry_price:.2f})")
            return position
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return None
    
    def update_position_price(self, position_id: str, current_price: float) -> bool:
        """
        Update position current price and PNL.
        
        Args:
            position_id: Position ID
            current_price: Current market price
            
        Returns:
            True if update was successful
        """
        try:
            if position_id not in self.positions:
                logger.error(f"Position {position_id} not found")
                return False
            
            position = self.positions[position_id]
            position.current_price = current_price
            position.last_update = time.time()
            
            # Calculate PNL
            if position.position_type == PositionType.LONG:
                position.pnl = (current_price - position.entry_price) * position.quantity
            elif position.position_type == PositionType.SHORT:
                position.pnl = (position.entry_price - current_price) * position.quantity
            else:
                position.pnl = 0.0
            
            # Calculate PNL percentage
            if position.entry_price > 0:
                position.pnl_percentage = (position.pnl / (position.entry_price * position.quantity)) * 100
            else:
                position.pnl_percentage = 0.0
            
            logger.debug(f"Updated position {position_id}: PNL ${position.pnl:.2f} ({position.pnl_percentage:.2f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Error updating position price: {e}")
            return False
    
    def close_position(self, position_id: str, exit_price: float, 
                      exit_quantity: Optional[float] = None) -> Optional[Transaction]:
        """
        Close a position (partially or fully).
        
        Args:
            position_id: Position ID
            exit_price: Exit price
            exit_quantity: Quantity to close (None for full close)
            
        Returns:
            Transaction record
        """
        try:
            if position_id not in self.positions:
                logger.error(f"Position {position_id} not found")
                return None
            
            position = self.positions[position_id]
            
            # Determine exit quantity
            if exit_quantity is None:
                exit_quantity = position.quantity
            
            if exit_quantity > position.quantity:
                logger.error(f"Exit quantity {exit_quantity} exceeds position quantity {position.quantity}")
                return None
            
            # Calculate transaction details
            total_value = exit_quantity * exit_price
            fee = total_value * self.config['default_fee_rate']
            
            # Determine transaction type
            if position.position_type == PositionType.LONG:
                transaction_type = "sell"
            elif position.position_type == PositionType.SHORT:
                transaction_type = "buy"
            else:
                transaction_type = "transfer_out"
            
            # Create transaction
            transaction = Transaction(
                transaction_id=f"tx_{int(time.time() * 1000)}",
                timestamp=time.time(),
                asset=position.asset,
                transaction_type=transaction_type,
                quantity=exit_quantity,
                price=exit_price,
                total_value=total_value,
                fee=fee
            )
            
            # Update position
            if exit_quantity == position.quantity:
                # Full close
                self._remove_position(position_id)
            else:
                # Partial close
                position.quantity -= exit_quantity
                position.last_update = time.time()
            
            # Update cash balance
            if transaction_type == "sell":
                self.cash_balance += total_value - fee
            elif transaction_type == "buy":
                self.cash_balance -= total_value + fee
            
            # Add to transaction history
            self.transactions.append(transaction)
            if len(self.transactions) > self.max_transaction_history:
                self.transactions.pop(0)
            
            self.total_transactions += 1
            self.total_fees += fee
            
            logger.info(f"Closed position {position_id}: {exit_quantity:.4f} @ ${exit_price:.2f}")
            return transaction
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None
    
    def _remove_position(self, position_id: str):
        """Remove position from tracking."""
        if position_id in self.positions:
            position = self.positions[position_id]
            
            # Remove from asset positions
            if position.asset in self.asset_positions:
                self.asset_positions[position.asset] = [
                    p for p in self.asset_positions[position.asset] 
                    if p.position_id != position_id
                ]
            
            # Remove from main positions
            del self.positions[position_id]
    
    def add_transaction(self, asset: AssetType, transaction_type: str,
                       quantity: float, price: float, fee: float = 0.0) -> Transaction:
        """
        Add a transaction record.
        
        Args:
            asset: Asset type
            transaction_type: Transaction type
            quantity: Transaction quantity
            price: Transaction price
            fee: Transaction fee
            
        Returns:
            Created transaction
        """
        try:
            total_value = quantity * price
            
            transaction = Transaction(
                transaction_id=f"tx_{int(time.time() * 1000)}",
                timestamp=time.time(),
                asset=asset,
                transaction_type=transaction_type,
                quantity=quantity,
                price=price,
                total_value=total_value,
                fee=fee
            )
            
            # Update cash balance
            if transaction_type in ["buy", "transfer_out"]:
                self.cash_balance -= total_value + fee
            elif transaction_type in ["sell", "transfer_in"]:
                self.cash_balance += total_value - fee
            
            # Add to transaction history
            self.transactions.append(transaction)
            if len(self.transactions) > self.max_transaction_history:
                self.transactions.pop(0)
            
            self.total_transactions += 1
            self.total_fees += fee
            
            logger.info(f"Added transaction: {transaction_type} {quantity:.4f} {asset.value} @ ${price:.2f}")
            return transaction
            
        except Exception as e:
            logger.error(f"Error adding transaction: {e}")
            return self._create_default_transaction()
    
    def _create_default_transaction(self) -> Transaction:
        """Create default transaction."""
        return Transaction(
            transaction_id="default",
            timestamp=time.time(),
            asset=AssetType.USDC,
            transaction_type="transfer_in",
            quantity=0.0,
            price=1.0,
            total_value=0.0,
            fee=0.0
        )
    
    def create_snapshot(self) -> PortfolioSnapshot:
        """
        Create a portfolio snapshot.
        
        Returns:
            Portfolio snapshot
        """
        try:
            # Calculate total portfolio value
            total_value = self.cash_balance
            total_pnl = 0.0
            
            for position in self.positions.values():
                position_value = position.quantity * position.current_price
                total_value += position_value
                total_pnl += position.pnl
            
            # Calculate total PNL percentage
            total_invested = sum(p.quantity * p.entry_price for p in self.positions.values())
            total_pnl_percentage = (total_pnl / total_invested * 100) if total_invested > 0 else 0.0
            
            snapshot = PortfolioSnapshot(
                snapshot_id=f"snapshot_{int(time.time() * 1000)}",
                timestamp=time.time(),
                total_value=total_value,
                total_pnl=total_pnl,
                total_pnl_percentage=total_pnl_percentage,
                positions=self.positions.copy(),
                cash_balance=self.cash_balance
            )
            
            # Add to snapshot history
            self.snapshots.append(snapshot)
            if len(self.snapshots) > self.max_snapshot_history:
                self.snapshots.pop(0)
            
            self.last_update = time.time()
            
            logger.info(f"Created snapshot: Total value ${total_value:.2f}, PNL ${total_pnl:.2f}")
            return snapshot
            
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            return self._create_default_snapshot()
    
    def _create_default_snapshot(self) -> PortfolioSnapshot:
        """Create default snapshot."""
        return PortfolioSnapshot(
            snapshot_id="default",
            timestamp=time.time(),
            total_value=self.cash_balance,
            total_pnl=0.0,
            total_pnl_percentage=0.0,
            positions={},
            cash_balance=self.cash_balance
        )
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        try:
            # Calculate current metrics
            total_value = self.cash_balance
            total_pnl = 0.0
            asset_breakdown = {}
            
            for asset in AssetType:
                asset_positions = self.asset_positions[asset]
                asset_value = sum(p.quantity * p.current_price for p in asset_positions)
                asset_pnl = sum(p.pnl for p in asset_positions)
                
                total_value += asset_value
                total_pnl += asset_pnl
                
                if asset_value > 0:
                    asset_breakdown[asset.value] = {
                        "value": asset_value,
                        "percentage": (asset_value / total_value * 100) if total_value > 0 else 0.0,
                        "pnl": asset_pnl,
                        "positions_count": len(asset_positions)
                    }
            
            return {
                "total_value": total_value,
                "cash_balance": self.cash_balance,
                "total_pnl": total_pnl,
                "total_pnl_percentage": (total_pnl / (total_value - total_pnl) * 100) if (total_value - total_pnl) > 0 else 0.0,
                "asset_breakdown": asset_breakdown,
                "total_positions": len(self.positions),
                "total_transactions": self.total_transactions,
                "total_fees": self.total_fees,
                "last_update": self.last_update
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {
                "total_value": self.cash_balance,
                "cash_balance": self.cash_balance,
                "total_pnl": 0.0,
                "total_pnl_percentage": 0.0,
                "asset_breakdown": {},
                "total_positions": 0,
                "total_transactions": 0,
                "total_fees": 0.0,
                "last_update": self.last_update
            }
    
    def get_asset_positions(self, asset: AssetType) -> List[Dict[str, Any]]:
        """Get positions for a specific asset."""
        positions = self.asset_positions.get(asset, [])
        return [
            {
                "position_id": p.position_id,
                "position_type": p.position_type.value,
                "quantity": p.quantity,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "pnl": p.pnl,
                "pnl_percentage": p.pnl_percentage,
                "entry_time": p.entry_time,
                "last_update": p.last_update
            }
            for p in positions
        ]
    
    def get_recent_transactions(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent transactions."""
        recent_transactions = self.transactions[-count:]
        return [
            {
                "transaction_id": tx.transaction_id,
                "timestamp": tx.timestamp,
                "asset": tx.asset.value,
                "transaction_type": tx.transaction_type,
                "quantity": tx.quantity,
                "price": tx.price,
                "total_value": tx.total_value,
                "fee": tx.fee
            }
            for tx in recent_transactions
        ]
    
    def get_recent_snapshots(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent portfolio snapshots."""
        recent_snapshots = self.snapshots[-count:]
        return [
            {
                "snapshot_id": snap.snapshot_id,
                "timestamp": snap.timestamp,
                "total_value": snap.total_value,
                "total_pnl": snap.total_pnl,
                "total_pnl_percentage": snap.total_pnl_percentage,
                "cash_balance": snap.cash_balance,
                "positions_count": len(snap.positions)
            }
            for snap in recent_snapshots
        ]
    
    def get_pnl_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get PNL history for the specified number of days."""
        try:
            cutoff_time = time.time() - (days * 24 * 3600)
            relevant_snapshots = [
                snap for snap in self.snapshots 
                if snap.timestamp >= cutoff_time
            ]
            
            return [
                {
                    "timestamp": snap.timestamp,
                    "total_value": snap.total_value,
                    "total_pnl": snap.total_pnl,
                    "total_pnl_percentage": snap.total_pnl_percentage
                }
                for snap in relevant_snapshots
            ]
            
        except Exception as e:
            logger.error(f"Error getting PNL history: {e}")
            return []
    
    def export_portfolio_data(self, filepath: str) -> bool:
        """
        Export portfolio data to JSON file.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if export was successful
        """
        try:
            data = {
                "export_timestamp": time.time(),
                "portfolio_summary": self.get_portfolio_summary(),
                "positions": {
                    pos_id: {
                        "asset": pos.asset.value,
                        "position_type": pos.position_type.value,
                        "quantity": pos.quantity,
                        "entry_price": pos.entry_price,
                        "current_price": pos.current_price,
                        "pnl": pos.pnl,
                        "pnl_percentage": pos.pnl_percentage,
                        "entry_time": pos.entry_time
                    }
                    for pos_id, pos in self.positions.items()
                },
                "recent_transactions": self.get_recent_transactions(50),
                "recent_snapshots": self.get_recent_snapshots(10)
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported portfolio data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting portfolio data: {e}")
            return False
    
    def import_portfolio_data(self, filepath: str) -> bool:
        """
        Import portfolio data from JSON file.
        
        Args:
            filepath: Input file path
            
        Returns:
            True if import was successful
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Import positions
            for pos_id, pos_data in data.get("positions", {}).items():
                asset = AssetType(pos_data["asset"])
                position_type = PositionType(pos_data["position_type"])
                
                position = Position(
                    position_id=pos_id,
                    asset=asset,
                    position_type=position_type,
                    quantity=pos_data["quantity"],
                    entry_price=pos_data["entry_price"],
                    current_price=pos_data["current_price"],
                    entry_time=pos_data["entry_time"],
                    last_update=time.time(),
                    pnl=pos_data["pnl"],
                    pnl_percentage=pos_data["pnl_percentage"]
                )
                
                self.positions[pos_id] = position
                self.asset_positions[asset].append(position)
            
            logger.info(f"Imported portfolio data from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing portfolio data: {e}")
            return False 