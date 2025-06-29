"""
Trading Execution Engine - Order Execution and Position Management

This module implements the trading execution engine that handles order execution,
position management, risk control, and trade lifecycle management for Schwabot.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
import uuid
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Enumeration of order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderStatus(Enum):
    """Enumeration of order statuses."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class PositionSide(Enum):
    """Enumeration of position sides."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # "buy" or "sell"
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    limit_price: Optional[float]
    status: OrderStatus
    timestamp: datetime
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    notes: str = ""

@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    side: PositionSide
    quantity: float
    average_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None

@dataclass
class Trade:
    """Represents a completed trade."""
    trade_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    order_id: str
    commission: float
    pnl: float
    strategy: str

class TradingExecutionEngine:
    """
    Trading execution engine implementing order execution, position management,
    and risk control for Schwabot.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the trading execution engine.
        
        Args:
            config: Configuration dictionary for execution parameters
        """
        self.config = config or {}
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
        # Risk management parameters
        self.max_position_size = self.config.get('max_position_size', 0.1)  # 10% of portfolio
        self.max_daily_loss = self.config.get('max_daily_loss', 0.05)  # 5% daily loss limit
        self.max_drawdown = self.config.get('max_drawdown', 0.15)  # 15% max drawdown
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.02)  # 2% stop loss
        self.take_profit_pct = self.config.get('take_profit_pct', 0.04)  # 4% take profit
        
        # Portfolio tracking
        self.initial_capital = self.config.get('initial_capital', 100000.0)
        self.current_capital = self.initial_capital
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_capital = self.initial_capital
        self.current_drawdown = 0.0
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0.0
        
        # Market data cache
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        
        logger.info("Trading Execution Engine initialized")
    
    def place_order(self, symbol: str, order_type: OrderType, side: str, 
                   quantity: float, price: Optional[float] = None, 
                   stop_price: Optional[float] = None, 
                   limit_price: Optional[float] = None,
                   strategy: str = "default") -> str:
        """
        Place a new trading order.
        
        Args:
            symbol: Trading symbol
            order_type: Type of order
            side: "buy" or "sell"
            quantity: Order quantity
            price: Order price (for limit orders)
            stop_price: Stop price (for stop orders)
            limit_price: Limit price (for stop-limit orders)
            strategy: Strategy name
            
        Returns:
            Order ID
        """
        # Validate order parameters
        if not self._validate_order_parameters(symbol, side, quantity, price, stop_price, limit_price):
            raise ValueError("Invalid order parameters")
        
        # Check risk limits
        if not self._check_risk_limits(symbol, side, quantity, price):
            raise ValueError("Order exceeds risk limits")
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Create order
        order = Order(
            order_id=order_id,
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            limit_price=limit_price,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
            notes=f"Strategy: {strategy}"
        )
        
        # Store order
        self.orders[order_id] = order
        
        logger.info(f"Order placed: {order_id} - {side} {quantity} {symbol} at {price}")
        
        return order_id
    
    def execute_order(self, order_id: str, fill_price: float, 
                     fill_quantity: Optional[float] = None) -> bool:
        """
        Execute an order with given fill price and quantity.
        
        Args:
            order_id: Order ID to execute
            fill_price: Fill price
            fill_quantity: Fill quantity (defaults to full order quantity)
            
        Returns:
            True if execution successful, False otherwise
        """
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        if order.status != OrderStatus.PENDING:
            logger.error(f"Order {order_id} is not pending")
            return False
        
        # Determine fill quantity
        if fill_quantity is None:
            fill_quantity = order.quantity - order.filled_quantity
        
        # Calculate commission
        commission = self._calculate_commission(fill_quantity, fill_price)
        
        # Update order
        order.filled_quantity += fill_quantity
        order.average_fill_price = ((order.average_fill_price * (order.filled_quantity - fill_quantity)) + 
                                   (fill_price * fill_quantity)) / order.filled_quantity
        order.commission += commission
        
        # Update order status
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        # Create trade record
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            timestamp=datetime.now(),
            order_id=order_id,
            commission=commission,
            pnl=0.0,  # Will be calculated when position is closed
            strategy=order.notes.split(": ")[-1] if ": " in order.notes else "default"
        )
        
        self.trades.append(trade)
        
        # Update position
        self._update_position(order.symbol, order.side, fill_quantity, fill_price)
        
        # Update portfolio
        self._update_portfolio(trade)
        
        logger.info(f"Order executed: {order_id} - {fill_quantity} {order.symbol} at {fill_price}")
        
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        if order.status != OrderStatus.PENDING:
            logger.error(f"Order {order_id} is not pending")
            return False
        
        order.status = OrderStatus.CANCELLED
        
        logger.info(f"Order cancelled: {order_id}")
        
        return True
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get current position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position object or None if no position
        """
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """
        Get all current positions.
        
        Returns:
            Dictionary of all positions
        """
        return self.positions.copy()
    
    def update_position_prices(self, market_data: Dict[str, float]):
        """
        Update position prices with current market data.
        
        Args:
            market_data: Dictionary mapping symbols to current prices
        """
        for symbol, position in self.positions.items():
            if symbol in market_data:
                old_price = position.current_price
                new_price = market_data[symbol]
                position.current_price = new_price
                
                # Update unrealized P&L
                if position.side == PositionSide.LONG:
                    position.unrealized_pnl = (new_price - position.average_price) * position.quantity
                elif position.side == PositionSide.SHORT:
                    position.unrealized_pnl = (position.average_price - new_price) * position.quantity
                
                # Check stop loss and take profit
                self._check_stop_loss_take_profit(symbol, position, old_price, new_price)
    
    def close_position(self, symbol: str, quantity: Optional[float] = None, 
                      price: float = None) -> bool:
        """
        Close a position (partially or fully).
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to close (defaults to full position)
            price: Closing price
            
        Returns:
            True if position closed successfully, False otherwise
        """
        if symbol not in self.positions:
            logger.error(f"No position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        
        if quantity is None:
            quantity = position.quantity
        
        if quantity > position.quantity:
            logger.error(f"Close quantity {quantity} exceeds position quantity {position.quantity}")
            return False
        
        # Calculate P&L
        if position.side == PositionSide.LONG:
            pnl = (price - position.average_price) * quantity
        else:  # SHORT
            pnl = (position.average_price - price) * quantity
        
        # Calculate commission
        commission = self._calculate_commission(quantity, price)
        
        # Create trade record
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            symbol=symbol,
            side="sell" if position.side == PositionSide.LONG else "buy",
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
            order_id="",  # No specific order for position close
            commission=commission,
            pnl=pnl,
            strategy="position_close"
        )
        
        self.trades.append(trade)
        
        # Update position
        if quantity == position.quantity:
            # Full close
            del self.positions[symbol]
        else:
            # Partial close
            position.quantity -= quantity
            # Adjust average price for remaining position
            position.average_price = ((position.average_price * (position.quantity + quantity)) - 
                                    (price * quantity)) / position.quantity
        
        # Update portfolio
        self._update_portfolio(trade)
        
        logger.info(f"Position closed: {symbol} - {quantity} at {price}, P&L: {pnl}")
        
        return True
    
    def set_stop_loss(self, symbol: str, stop_price: float) -> bool:
        """
        Set stop loss for a position.
        
        Args:
            symbol: Trading symbol
            stop_price: Stop loss price
            
        Returns:
            True if stop loss set successfully, False otherwise
        """
        if symbol not in self.positions:
            logger.error(f"No position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        position.stop_loss = stop_price
        
        logger.info(f"Stop loss set for {symbol}: {stop_price}")
        
        return True
    
    def set_take_profit(self, symbol: str, take_profit_price: float) -> bool:
        """
        Set take profit for a position.
        
        Args:
            symbol: Trading symbol
            take_profit_price: Take profit price
            
        Returns:
            True if take profit set successfully, False otherwise
        """
        if symbol not in self.positions:
            logger.error(f"No position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        position.take_profit = take_profit_price
        
        logger.info(f"Take profit set for {symbol}: {take_profit_price}")
        
        return True
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary and performance metrics.
        
        Returns:
            Dictionary containing portfolio summary
        """
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        
        # Calculate performance metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        sharpe_ratio = self._calculate_sharpe_ratio()
        max_drawdown = self._calculate_max_drawdown()
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_pnl': self.total_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': total_realized_pnl,
            'total_return': total_return,
            'daily_pnl': self.daily_pnl,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'total_commission': self.total_commission,
            'position_count': len(self.positions),
            'pending_orders': len([o for o in self.orders.values() if o.status == OrderStatus.PENDING])
        }
    
    def _validate_order_parameters(self, symbol: str, side: str, quantity: float,
                                 price: Optional[float], stop_price: Optional[float],
                                 limit_price: Optional[float]) -> bool:
        """Validate order parameters."""
        if not symbol or quantity <= 0:
            return False
        
        if side not in ["buy", "sell"]:
            return False
        
        if price is not None and price <= 0:
            return False
        
        if stop_price is not None and stop_price <= 0:
            return False
        
        if limit_price is not None and limit_price <= 0:
            return False
        
        return True
    
    def _check_risk_limits(self, symbol: str, side: str, quantity: float, 
                          price: Optional[float]) -> bool:
        """Check if order exceeds risk limits."""
        # Check position size limit
        if price:
            position_value = quantity * price
            if position_value > self.current_capital * self.max_position_size:
                logger.warning(f"Order exceeds max position size limit")
                return False
        
        # Check daily loss limit
        if self.daily_pnl < -self.current_capital * self.max_daily_loss:
            logger.warning(f"Daily loss limit exceeded")
            return False
        
        # Check drawdown limit
        if self.current_drawdown > self.max_drawdown:
            logger.warning(f"Max drawdown limit exceeded")
            return False
        
        return True
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for a trade."""
        # Simple commission model: 0.1% of trade value
        trade_value = quantity * price
        commission_rate = 0.001  # 0.1%
        return trade_value * commission_rate
    
    def _update_position(self, symbol: str, side: str, quantity: float, price: float):
        """Update position after trade execution."""
        if symbol not in self.positions:
            # Create new position
            position_side = PositionSide.LONG if side == "buy" else PositionSide.SHORT
            self.positions[symbol] = Position(
                symbol=symbol,
                side=position_side,
                quantity=quantity,
                average_price=price,
                current_price=price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                timestamp=datetime.now()
            )
        else:
            # Update existing position
            position = self.positions[symbol]
            
            if (side == "buy" and position.side == PositionSide.LONG) or \
               (side == "sell" and position.side == PositionSide.SHORT):
                # Add to position
                total_quantity = position.quantity + quantity
                position.average_price = ((position.average_price * position.quantity) + 
                                        (price * quantity)) / total_quantity
                position.quantity = total_quantity
            else:
                # Reduce position
                if quantity >= position.quantity:
                    # Close position
                    del self.positions[symbol]
                else:
                    # Partial close
                    position.quantity -= quantity
    
    def _update_portfolio(self, trade: Trade):
        """Update portfolio after trade execution."""
        # Update capital
        trade_value = trade.quantity * trade.price
        if trade.side == "buy":
            self.current_capital -= trade_value
        else:  # sell
            self.current_capital += trade_value
        
        # Update P&L and commission
        self.total_pnl += trade.pnl
        self.total_commission += trade.commission
        self.daily_pnl += trade.pnl
        
        # Update trade counts
        self.total_trades += 1
        if trade.pnl > 0:
            self.winning_trades += 1
        elif trade.pnl < 0:
            self.losing_trades += 1
        
        # Update max capital and drawdown
        if self.current_capital > self.max_capital:
            self.max_capital = self.current_capital
        
        self.current_drawdown = (self.max_capital - self.current_capital) / self.max_capital
    
    def _check_stop_loss_take_profit(self, symbol: str, position: Position, 
                                   old_price: float, new_price: float):
        """Check and execute stop loss and take profit orders."""
        if position.stop_loss:
            if (position.side == PositionSide.LONG and new_price <= position.stop_loss) or \
               (position.side == PositionSide.SHORT and new_price >= position.stop_loss):
                logger.info(f"Stop loss triggered for {symbol} at {new_price}")
                self.close_position(symbol, price=new_price)
        
        if position.take_profit:
            if (position.side == PositionSide.LONG and new_price >= position.take_profit) or \
               (position.side == PositionSide.SHORT and new_price <= position.take_profit):
                logger.info(f"Take profit triggered for {symbol} at {new_price}")
                self.close_position(symbol, price=new_price)
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from trade history."""
        if len(self.trades) < 2:
            return 0.0
        
        # Calculate returns from trades
        returns = [trade.pnl for trade in self.trades]
        
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from trade history."""
        if not self.trades:
            return 0.0
        
        # Reconstruct portfolio value over time
        portfolio_values = [self.initial_capital]
        current_value = self.initial_capital
        
        for trade in self.trades:
            if trade.side == "buy":
                current_value -= trade.quantity * trade.price
            else:  # sell
                current_value += trade.quantity * trade.price
            
            current_value -= trade.commission
            portfolio_values.append(current_value)
        
        # Calculate drawdown
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def get_system_status(self) -> Dict:
        """Get current system status and statistics."""
        return {
            'total_orders': len(self.orders),
            'pending_orders': len([o for o in self.orders.values() if o.status == OrderStatus.PENDING]),
            'filled_orders': len([o for o in self.orders.values() if o.status == OrderStatus.FILLED]),
            'cancelled_orders': len([o for o in self.orders.values() if o.status == OrderStatus.CANCELLED]),
            'total_positions': len(self.positions),
            'total_trades': len(self.trades),
            'current_capital': self.current_capital,
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'risk_limits': {
                'max_position_size': self.max_position_size,
                'max_daily_loss': self.max_daily_loss,
                'max_drawdown': self.max_drawdown,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct
            }
        }

# Export main classes
__all__ = ['TradingExecutionEngine', 'OrderType', 'OrderStatus', 'PositionSide', 'Order', 'Position', 'Trade'] 