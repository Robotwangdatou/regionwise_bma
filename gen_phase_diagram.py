"""
Generate Phase Diagram figure from real experiment data.
X-axis: log(n_k / M)
Y-axis: Delta AUC (Regionwise - Global)
Color: green if positive, red if negative
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load data
with open(r"C:\Users\robot\Desktop\M PAPER\figures\full_results.json") as f:
    data = json.load(f)

results = data["all_results"]

# Calculate per-dataset values
n_k = []
delta_auc = []
names = []

for r in results:
    nk = r["n"] / 2  # split into 2 regions
    d = r["d"]
    M = 5  # 5 models
    diff = r["rw_mean"] - r["global_mean"]
    n_k.append(nk)
    delta_auc.append(diff)
    names.append(r["dataset"])

n_k = np.array(n_k)
delta_auc = np.array(delta_auc)

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

# Color by sign
colors = ['#2ecc71' if d > 0.001 else '#e74c3c' if d < -0.001 else '#95a5a6' for d in delta_auc]

# Scatter plot
ax.scatter(n_k, delta_auc, c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

# Vertical line at n_k = 1000
ax.axvline(x=1000, color='black', linestyle='--', linewidth=1.5, label='Theoretical boundary ($n_k \approx 1000$)')

# Horizontal line at 0
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

# Labels and title
ax.set_xscale('log')
ax.set_xlabel('Local sample size per region $n_k$ (log scale)', fontsize=12)
ax.set_ylabel(r'$\Delta$AUC (Regionwise $-$ Global)', fontsize=12)
ax.set_title('Phase Transition of Regionwise BMA', fontsize=14, fontweight='bold')

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10, label='Regionwise improves'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='CEIWC dominates'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#95a5a6', markersize=10, label='No significant difference'),
    Line2D([0], [0], color='black', linestyle='--', label='Theoretical boundary'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Grid
ax.grid(True, alpha=0.3)

# Save
plt.tight_layout()
plt.savefig(r"C:\Users\robot\Desktop\M PAPER\figures\fig_phase_diagram.png", dpi=150, bbox_inches='tight')
print("Phase diagram saved to fig_phase_diagram.png")

# Also print summary stats
favorable = np.sum(delta_auc[n_k >= 1000] > 0)
challenging = np.sum(delta_auc[n_k < 1000] < 0)
print(f"Favorable regime (n_k >= 1000): {favorable}/{np.sum(n_k >= 1000)} datasets improve")
print(f"Challenging regime (n_k < 1000): {challenging}/{np.sum(n_k < 1000)} datasets degrade")
