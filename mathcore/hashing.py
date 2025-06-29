import hashlib

def price_hash(price: float, timestamp: float, tick_shift: float) -> str:
    """Creates a deterministic hash from price movement data."""
    data = f"{price:.8f}|{timestamp:.4f}|{tick_shift:.4f}"
    return hashlib.sha256(data.encode()).hexdigest()

def hash_similarity(h1: str, h2: str) -> float:
    """Returns a simple hamming distance as similarity score (0.0-1.0)."""
    return sum(a == b for a, b in zip(h1, h2)) / len(h1) 