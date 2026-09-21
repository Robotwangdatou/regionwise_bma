"""
MoE-CEIWC Simulation v2 (Corollary 7.1)
=========================================
Redesigned: LINEAR router (no hidden layer) to isolate the statistical
CEIWC effect from neural network memorization.

Setup:
- Global expert: linear classifier trained on ALL data
- Local expert: linear classifier trained on minority-region data only
- Router: LINEAR (logistic regression) deciding expert weight
- Minority region has a known signal that the local expert captures better
- With few minority samples, the router under-weights the local expert (CEIWC)
"""
import numpy as np
import json
import os
from sklearn.metrics import roc_auc_score

np.random.seed(42)

OUT_DIR = r"C:\Users\robot\Desktop\M PAPER\figures"
os.makedirs(OUT_DIR, exist_ok=True)

def make_data(n_min, n_maj=1000, d=3):
    """Two-class problem where minority class has a subregion with
    a known signal direction that the local expert should capture."""
    # Majority class: standard normal
    X_maj = np.random.randn(n_maj, d)
    y_maj = np.zeros(n_maj)
    
    # Minority class: centered at origin, but with a sub-region
    # where the true decision boundary is different
    X_min = np.random.randn(n_min, d) * 0.8
    y_min = np.ones(n_min)
    
    # The "local signal": in minority sub-region, feature 2 shifts
    # This is what the local expert should learn
    X_min[:, 1] += 0.5  # small shift
    
    X = np.vstack([X_maj, X_min])
    y = np.hstack([y_maj, y_min])
    
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

def train_logreg(X, y, epochs=5000, lr=0.1):
    """Train logistic regression from scratch."""
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        grad_w = X.T @ (p - y) / len(y)
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

def predict_logreg(X, w, b):
    z = X @ w + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

# ------------------------------------------------------------
# Run: vary n_min, measure router weight on local expert
# ------------------------------------------------------------
n_min_list = [15, 30, 50, 100, 200, 500, 1000]
results = []

print("=" * 70)
print("MoE-CEIWC Simulation v2 (Linear Router)")
print("=" * 70)

for n_min in n_min_list:
    # Generate data
    X, y = make_data(n_min)
    
    # Split train/test
    n = len(y)
    idx = np.random.permutation(n)
    n_train = int(0.7 * n)
    X_tr, X_te = X[idx[:n_train]], X[idx[n_train:]]
    y_tr, y_te = y[idx[:n_train]], y[idx[n_train:]]
    
    # Expert 1: global (trained on all training data)
    w_g, b_g = train_logreg(X_tr, y_tr)
    e1_tr = predict_logreg(X_tr, w_g, b_g)
    e1_te = predict_logreg(X_te, w_g, b_g)
    
    # Expert 2: local (trained on minority training samples only)
    min_tr = y_tr == 1
    if min_tr.sum() >= 10:
        w_l, b_l = train_logreg(X_tr[min_tr], y_tr[min_tr], epochs=3000)
    else:
        w_l, b_l = w_g.copy(), b_g
    e2_tr = predict_logreg(X_tr, w_l, b_l)
    e2_te = predict_logreg(X_te, w_l, b_l)
    
    # --- Linear router: g_local = sigmoid(x @ v + c) ---
    # Train to minimize mixture NLL
    # p = g * e2 + (1-g) * e1 = e1 + g*(e2-e1)
    v = np.zeros(X.shape[1])
    c = 0.0
    
    for epoch in range(5000):
        z = X_tr @ v + c
        g = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        p = g * e2_tr + (1 - g) * e1_tr
        p = np.clip(p, 1e-8, 1 - 1e-8)
        
        # d loss / d g = (p - y) * (e2 - e1) / n
        dg = ((p - y_tr) * (e2_tr - e1_tr) / len(y_tr))
        dz = dg * g * (1 - g)
        
        grad_v = X_tr.T @ dz
        grad_c = dz.mean()
        
        v -= 0.1 * grad_v
        c -= 0.1 * grad_c
    
    # Measure router weights on test minority samples
    g_te = 1.0 / (1.0 + np.exp(-(X_te @ v + c)))
    p_te = g_te * e2_te + (1 - g_te) * e1_te
    
    min_te = y_te == 1
    g_local_min = g_te[min_te].mean() if min_te.sum() > 0 else np.nan
    g_local_all = g_te.mean()
    
    try:
        auc_raw = roc_auc_score(y_te, p_te)
    except:
        auc_raw = 0.5
    
    # --- Gated router: enforce g >= 0.15 on minority region ---
    v_g = np.zeros(X.shape[1])
    c_g = 0.0
    for epoch in range(5000):
        z = X_tr @ v_g + c_g
        g = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        p = g * e2_tr + (1 - g) * e1_tr
        p = np.clip(p, 1e-8, 1 - 1e-8)
        dg = ((p - y_tr) * (e2_tr - e1_tr) / len(y_tr))
        # Add floor penalty: push g up on minority training points
        dg[min_tr] += 0.02 * (0.15 - g[min_tr]).clip(max=0)
        dz = dg * g * (1 - g)
        v_g -= 0.1 * X_tr.T @ dz
        c_g -= 0.1 * dz.mean()
    
    g_te_g = 1.0 / (1.0 + np.exp(-(X_te @ v_g + c_g)))
    p_te_g = g_te_g * e2_te + (1 - g_te_g) * e1_te
    g_local_min_g = g_te_g[min_te].mean() if min_te.sum() > 0 else np.nan
    try:
        auc_gated = roc_auc_score(y_te, p_te_g)
    except:
        auc_gated = 0.5
    
    result = {
        "n_min": n_min,
        "g_local_min_unconstrained": round(float(g_local_min), 4),
        "g_local_min_gated": round(float(g_local_min_g), 4),
        "auc_unconstrained": round(float(auc_raw), 4),
        "auc_gated": round(float(auc_gated), 4),
    }
    results.append(result)
    print(f"n_min={n_min:>5} | g_L(min)={g_local_min:.4f} -> {g_local_min_g:.4f} | "
          f"AUC={auc_raw:.4f} -> {auc_gated:.4f}")

# Save
with open(os.path.join(OUT_DIR, "moe_ceiwc_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("Summary table:")
print(f"{'n_min':>6} | {'g_L raw':>8} | {'g_L gate':>9} | {'AUC raw':>8} | {'AUC gate':>9}")
print("-" * 55)
for r in results:
    print(f"{r['n_min']:>6} | {r['g_local_min_unconstrained']:>8.4f} | "
          f"{r['g_local_min_gated']:>9.4f} | {r['auc_unconstrained']:>8.4f} | "
          f"{r['auc_gated']:>9.4f}")
print("=" * 70)
