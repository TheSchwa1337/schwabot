import numpy as np
import yaml
from schwabot.init.omni_shell.lotus_omni_mesh import (
    generate_lotus_nodes, node_checksum_magnitudes, generate_subharmonic_blooms, riser_bloom_step
)
from schwabot.init.omni_shell.mesh_to_shell_sync import mesh_to_shell_sync

# Load mapping matrix from YAML
with open('schwabot/init/omni_shell/lotus_mesh_map.yaml', 'r') as f:
    mapping_matrix = np.array(yaml.safe_load(f)['mapping_matrix'])

N = mapping_matrix.shape[0]  # Number of Lotus nodes
M = mapping_matrix.shape[1]  # Number of Omni shell rings

# Initialize shell buffers
shell_buffers = np.zeros(M)

# Example echo and profit functions
main_M = node_checksum_magnitudes(N)
def echo_func(n: int) -> float:
    return main_M[n]
def profit_func(n: int) -> float:
    # Example: could be a function of node index or random
    return 0.1 * np.random.randn()

def shell_update(echo_plus_profit: np.ndarray) -> None:
    global shell_buffers
    shell_buffers = mesh_to_shell_sync(echo_plus_profit, mapping_matrix, shell_buffers)

if __name__ == "__main__":
    ticks = 50
    for t in range(ticks):
        echo_plus_profit = np.array([echo_func(n) + profit_func(n) for n in range(N)])
        shell_update(echo_plus_profit)
        if t % 10 == 0:
            print(f"Tick {t}: Shell Buffers: {shell_buffers}")
    print("Final Shell Buffers:", shell_buffers)
    # Optionally, regenerate the mapping diagram
    import schwabot.init.omni_shell.lotus_mesh_diagram 