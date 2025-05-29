import numpy as np
from typing import Any

def mesh_to_shell_sync(lotus_echo: np.ndarray, mapping_matrix: np.ndarray, shell_buffers: np.ndarray) -> np.ndarray:
    """Update shell buffers from mesh node echoes using mapping matrix."""
    for m in range(shell_buffers.shape[0]):
        shell_buffers[m] += np.sum(mapping_matrix[:, m] * lotus_echo)
    return shell_buffers 