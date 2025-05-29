import numpy as np
from typing import Callable, Any

def lotus_tick_hash_feed(N: int, echo_func: Callable[[int], float], profit_func: Callable[[int], float], shell_update_func: Callable[[np.ndarray], Any], ticks: int = 100) -> None:
    """Stream Lotus node tick/echo/profit values into the shell for a number of ticks."""
    for t in range(ticks):
        echo_vals = np.array([echo_func(n) for n in range(N)])
        profit_vals = np.array([profit_func(n) for n in range(N)])
        # Example: Combine echo and profit, or send separately
        shell_update_func(echo_vals + profit_vals) 