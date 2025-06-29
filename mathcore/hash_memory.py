HASH_MEMORY = {}

def store_hash_profit(hash_val: str, profit: float) -> None:
    """Store the profit for a given hash value."""
    HASH_MEMORY[hash_val] = profit

def is_hash_profitable(hash_val: str, threshold: float = 0.025) -> bool:
    """Return True if the hash is associated with a profit above the threshold."""
    return HASH_MEMORY.get(hash_val, 0) >= threshold 