import numpy as np
from typing import Tuple, List

def generate_lotus_nodes(N: int = 16, a: float = 0.5, b: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Generate spiral node coordinates for the Lotus mesh."""
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = a + b * np.arange(N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def node_checksum_magnitudes(N: int = 16) -> np.ndarray:
    """Mock checksum magnitudes for each node (replace with real logic as needed)."""
    # Example: Use a sinusoidal pattern or import your real checksum logic
    return np.abs(np.sin(np.linspace(0, 2 * np.pi, N)))

def print_node_spiral(N: int = 16) -> None:
    x, y = generate_lotus_nodes(N)
    M = node_checksum_magnitudes(N)
    print("Node Index | (x, y) Coordinate   | Checksum Magnitude (Mₙ)")
    print("-----------------------------------------------------------")
    for n in range(N):
        print(f"{n:2d}         | ({x[n]:6.2f}, {y[n]:6.2f}) | {M[n]:.2f}")

def generate_subharmonic_blooms(N: int = 16, K: int = 3, omega_0: float = np.pi/8, bloom_phase: float = np.pi/6) -> np.ndarray:
    """Generate subharmonic bloom values for each node (async riser logic)."""
    main_M = node_checksum_magnitudes(N)
    sub_buffers = np.zeros((N, K))
    for n in range(N):
        for j in range(K):
            phase = bloom_phase * j + np.random.uniform(0, 2 * np.pi)
            freq = omega_0 * (j + 1)
            sub_buffers[n][j] = main_M[n] * np.sin(freq * n + phase)
    return sub_buffers

def riser_bloom_step(sub_buffers: np.ndarray, tick: int) -> List[float]:
    """Async tick update for subharmonic bloom sequencer."""
    N, K = sub_buffers.shape
    bloom_outputs = []
    for n in range(N):
        for j in range(K):
            value = sub_buffers[n][j] * np.cos(tick + j)
            bloom_outputs.append(value)
    return bloom_outputs 