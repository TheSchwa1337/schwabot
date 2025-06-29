def is_ghost_entry(liquidity_gap: float, entropy_val: float, hash_similarity: float) -> bool:
    """Return True if all stealth entry conditions are met."""
    return (
        liquidity_gap > 0.5 and
        entropy_val > 0.8 and
        hash_similarity > 0.9
    ) 