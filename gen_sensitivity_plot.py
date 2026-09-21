"""Generate sensitivity plot from real data."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open(r"C:\Users\robot\Desktop\M PAPER\figures\sensitivity_results.json") as f:
    data = json.load(f)

eps_vals = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]

fig, ax = plt.subplots(figsize=(8, 5))

colors = {'creditcard': '#e74c3c', 'iris0': '#3498db'}
labels = {'creditcard': 'Creditcard (extreme imbalance)', 'iris0': 'Iris0 (small dataset)'}

for dname, vals in data.items():
    aucs = [vals[str(e)] for e in eps_vals]
    ax.plot(eps_vals, aucs, 'o-', color=colors.get(dname, '#2ecc71'), 
            label=labels.get(dname, dname), linewidth=2, markersize=8)

ax.axvspan(0.05, 0.20, alpha=0.1, color='green', label='Robust range [0.05, 0.20]')

ax.set_xlabel(r'Gate threshold $\epsilon_{min}$', fontsize=12)
ax.set_ylabel('AUC', fontsize=12)
ax.set_title(r'Sensitivity to Gate Threshold $\epsilon_{min}$', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r"C:\Users\robot\Desktop\M PAPER\figures\fig_sensitivity.png", dpi=150, bbox_inches='tight')
print("Sensitivity plot saved")
