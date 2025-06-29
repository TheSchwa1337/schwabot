CYCLE_LENGTH = 16

def current_tick(timestamp: float, cycle_interval: float = 3600) -> int:
    """Return the current tick in the Ferris Wheel cycle."""
    return int(timestamp // cycle_interval) % CYCLE_LENGTH

def is_exit_tick(tick: int) -> bool:
    """Return True if tick is in the exit phase (12-16)."""
    return 12 <= tick <= 16 