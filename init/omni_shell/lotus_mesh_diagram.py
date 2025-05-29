import numpy as np
import matplotlib.pyplot as plt
import yaml

# Load mapping matrix from YAML
with open('schwabot/init/omni_shell/lotus_mesh_map.yaml', 'r') as f:
    mapping = yaml.safe_load(f)['mapping_matrix']

N = len(mapping)  # Number of Lotus nodes
M = len(mapping[0])  # Number of Omni shell rings

fig, ax = plt.subplots(figsize=(8, 6))

# Plot Lotus nodes (left)
lotus_y = np.linspace(0, 1, N)
ax.scatter([0]*N, lotus_y, s=100, c='gold', label='Lotus Nodes')
for i in range(N):
    ax.text(-0.03, lotus_y[i], f'L{i}', va='center', ha='right', fontsize=8)

# Plot Omni shell rings (right)
omni_y = np.linspace(0, 1, M)
ax.scatter([1]*M, omni_y, s=200, c='deepskyblue', label='Omni Shell Rings')
for j in range(M):
    ax.text(1.03, omni_y[j], f'R{j}', va='center', ha='left', fontsize=10)

# Draw connections
for i in range(N):
    for j in range(M):
        if mapping[i][j]:
            ax.plot([0, 1], [lotus_y[i], omni_y[j]], 'k-', lw=1, alpha=0.5)

ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.05, 1.05)
ax.axis('off')
ax.set_title('Lotus Mesh Node to Omni Shell Ring Mapping')
plt.tight_layout()
plt.savefig('schwabot/init/omni_shell/lotus_mesh_mapping_diagram.png')
plt.close(fig) 