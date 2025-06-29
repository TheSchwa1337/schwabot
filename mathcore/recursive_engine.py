from .profit_vector import profit_percentage
from .hash_memory import is_hash_profitable
from .tick_cycle import is_exit_tick

def recursive_profit_validation(current_tick: int, hash_val: str, vault_price: float, market_price: float) -> bool:
    """Validate profit and tick for recursive exit logic."""
    profit = profit_percentage(vault_price, market_price)
    if is_hash_profitable(hash_val) and is_exit_tick(current_tick) and profit >= 0.035:
        return True
    return False 