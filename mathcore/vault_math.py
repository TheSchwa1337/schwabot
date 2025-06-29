from math import log

def vault_growth(entry_price: float, hold_time: float, alpha: float = 1.02) -> float:
    """Vault grows with price change and time weighted by alpha curve."""
    return entry_price * (1 + alpha * log(hold_time + 1))

def vault_exit_trigger(current_price: float, entry_price: float, min_profit: float = 0.035) -> bool:
    """Return True if profit threshold is met for vault exit."""
    return (current_price - entry_price) / entry_price >= min_profit 