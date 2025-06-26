from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Risk Manager - Comprehensive Risk Management Engine for Schwabot
==============================================================

This module implements advanced risk management for Schwabot, including:
- Mathematical risk models (VaR, CVaR, Kelly, custom tensors)
- Position sizing and exposure limits
- Stop-loss and take-profit logic
- Real-time risk monitoring and alerts
- Integration hooks for the trading pipeline
- Logging and audit trail
"""

from core.unified_math_system import unified_math
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class RiskConfig:
    max_position_size: float = 0.1  # Fraction of portfolio
    max_drawdown: float = 0.2       # Max drawdown allowed (fraction)
    stop_loss_pct: float = 0.02     # Stop loss as percent
    take_profit_pct: float = 0.05   # Take profit as percent
    var_window: int = 100           # Window for VaR calculation
    cvar_alpha: float = 0.05        # CVaR confidence level
    kelly_fraction: float = 0.5     # Fraction of Kelly criterion to use
    risk_free_rate: float = 0.01    # Risk-free rate for Sharpe
    alert_threshold: float = 0.15   # Alert if risk exceeds this
    audit_log_path: str = "risk_audit.log"

@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    meta: Dict[str, Any] = field(default_factory=dict)

class RiskManager:
    pass
def __init__(self, config: Optional[RiskConfig] = None):
    self.config = config or RiskConfig()
    self.positions: Dict[str, Position] = {}
    self.pnl_history: List[float] = []
    self.lock = threading.Lock()
    self.audit_log = []
    self._load_audit_log()

def _load_audit_log(self):
    if os.path.exists(self.config.audit_log_path):
    try:
    pass
    with open(self.config.audit_log_path, 'r') as f:
    self.audit_log = json.load(f)
    except Exception as e:
    logger.warning(f"Failed to load audit log: {e}")

def _save_audit_log(self):
    try:
    pass
    with open(self.config.audit_log_path, 'w') as f:
    json.dump(self.audit_log, f, indent=2, default=str)
    except Exception as e:
    logger.warning(f"Failed to save audit log: {e}")

def log_event(self, event: str, details: Dict[str, Any]):
    entry = {
    "timestamp": datetime.now().isoformat(),
    "event": event,
    "details": details
    }
    self.audit_log.append(entry)
    self._save_audit_log()
    logger.info(f"Risk event: {event} | {details}")

def add_position(self, symbol: str, size: float, entry_price: float):
    with self.lock:
    if size > self.config.max_position_size:
    raise ValueError(f"Position size {size} exceeds max allowed {self.config.max_position_size}")
    stop_loss = entry_price * (1 - self.config.stop_loss_pct)
    take_profit = entry_price * (1 + self.config.take_profit_pct)
    pos = Position(
    symbol=symbol,
    size=size,
    entry_price=entry_price,
    entry_time=datetime.now(),
    stop_loss=stop_loss,
    take_profit=take_profit
    ]
    self.positions[symbol] = pos
    self.log_event("ADD_POSITION", pos.__dict__)

def remove_position(self, symbol: str):
    with self.lock:
    if symbol in self.positions:
    pos = self.positions.pop(symbol)
    self.log_event("REMOVE_POSITION", pos.__dict__)

def update_pnl(self, pnl: float):
    with self.lock:
    self.pnl_history.append(pnl)
    if len(self.pnl_history) > self.config.var_window:
    self.pnl_history = self.pnl_history[-self.config.var_window:]

def check_risk(self, symbol: str, current_price: float) -> Tuple[bool, str]:
    """Check if position should be closed due to risk limits."""
    with self.lock:
    pos = self.positions.get(symbol)
    if not pos:
    return False, "NO_POSITION"
    # Stop loss
    if current_price <= pos.stop_loss:
    self.log_event("STOP_LOSS_TRIGGERED", {"symbol": symbol, "price": current_price})
    return True, "STOP_LOSS"
    # Take profit
    if current_price >= pos.take_profit:
    self.log_event("TAKE_PROFIT_TRIGGERED", {"symbol": symbol, "price": current_price})
    return True, "TAKE_PROFIT"
    # Drawdown
    max_drawdown = self._calculate_drawdown()
    if max_drawdown > self.config.max_drawdown:
    self.log_event("DRAWDOWN_LIMIT_EXCEEDED", {"drawdown": max_drawdown})
    return True, "DRAWDOWN"
    return False, "OK"

def _calculate_drawdown(self) -> float:
    if not self.pnl_history:
    return 0.0
    peak = np.maximum.accumulate(self.pnl_history)
    drawdowns = (peak - self.pnl_history) / peak
    return float(unified_math.unified_math.max(drawdowns)) if len(drawdowns) > 0 else 0.0

def calculate_var(self) -> float:
    """Value at Risk (VaR) using historical simulation."""
    if len(self.pnl_history) < self.config.var_window:
    return 0.0
    var = -np.percentile(self.pnl_history, self.config.cvar_alpha * 100)
    self.log_event("VAR_CALCULATED", {"VaR": var})
    return var

def calculate_cvar(self) -> float:
    """Conditional Value at Risk (CVaR)."""
    if len(self.pnl_history) < self.config.var_window:
    return 0.0
    var = self.calculate_var()
    losses = [p for p in (self.pnl_history if p < -var]
    cvar = -unified_math.unified_math.mean(losses) for self.pnl_history if p < -var)
    cvar = -unified_math.unified_math.mean(losses) in ((self.pnl_history if p < -var)
    cvar = -unified_math.unified_math.mean(losses) for (self.pnl_history if p < -var)
    cvar = -unified_math.unified_math.mean(losses) in (((self.pnl_history if p < -var)
    cvar = -unified_math.unified_math.mean(losses) for ((self.pnl_history if p < -var)
    cvar = -unified_math.unified_math.mean(losses) in ((((self.pnl_history if p < -var)
    cvar = -unified_math.unified_math.mean(losses) for (((self.pnl_history if p < -var)
    cvar = -unified_math.unified_math.mean(losses) in (((((self.pnl_history if p < -var)
    cvar = -unified_math.unified_math.mean(losses) for ((((self.pnl_history if p < -var)
    cvar = -unified_math.unified_math.mean(losses) in (((((self.pnl_history if p < -var)
    cvar = -unified_math.unified_math.mean(losses) if losses else 0.0
    self.log_event("CVAR_CALCULATED", {"CVaR")))))))))): cvar})
    return cvar

def kelly_position_size(self, win_prob: float, win_loss_ratio: float) -> float:
    """Calculate position size using Kelly criterion."""
    kelly = (win_prob * (win_loss_ratio + 1) - 1) / win_loss_ratio
    kelly_size = unified_math.max(0.0, unified_math.min(self.config.kelly_fraction * kelly, self.config.max_position_size))
    self.log_event("KELLY_SIZE_CALCULATED", {"kelly_size": kelly_size})
    return kelly_size

def risk_alert(self) -> bool:
    var = self.calculate_var()
    if var > self.config.alert_threshold:
    self.log_event("RISK_ALERT", {"VaR": var})
    return True
    return False

def get_positions(self) -> List[Position]:
    with self.lock:
    return list(self.positions.values())

def get_audit_log(self) -> List[Dict[str, Any]:
    return self.audit_log

def reset(self):
    with self.lock:
    self.positions.clear()
    self.pnl_history.clear()
    self.audit_log.clear()
    self._save_audit_log()
    self.log_event("RESET", {})

# Example usage and test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rm = RiskManager()
    rm.add_position("BTCUSD", 0.05, 50000)
    for i in range(120):
    price = 50000 + np.random.normal(0, 100)
    pnl = price - 50000
    rm.update_pnl(pnl)
    close, reason = rm.check_risk("BTCUSD", price)
    if close:
    safe_print(f"Close position due to {reason} at price {price}")
    rm.remove_position("BTCUSD")
    break
    safe_print("Current positions:", rm.get_positions())
    safe_print("Audit log entries:", len(rm.get_audit_log()))