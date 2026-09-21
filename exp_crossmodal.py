"""
Cross-modal CEIWC validation: CIFAR-10-LT with ResNet-18 features
Extract pretrained features, create long-tailed version, run BMA experiment.
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from scipy.linalg import inv
import os, json, warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
OUT_DIR = r"C:\Users\robot\Desktop\M PAPER\figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Step 1: Load CIFAR-10 and extract ResNet-18 features
# ============================================================
print("=" * 60)
print("CROSS-MODAL CEIWC: CIFAR-10-LT")
print("=" * 60)

print("\n[1/4] Loading CIFAR-10 and extracting ResNet-18 features...")

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Download CIFAR-10
train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Use pretrained ResNet-18, remove final FC layer
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Identity()  # 512-dim features
model.eval()

# Extract features (use subset for speed)
N_TRAIN = 8000  # subset
N_TEST = 2000

def extract_features(dataset, n_samples):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    features = []
    labels = []
    count = 0
    with torch.no_grad():
        for x, y in loader:
            feat = model(x)
            features.append(feat.numpy())
            labels.append(y.numpy())
            count += len(y)
            if count >= n_samples:
                break
    X = np.vstack(features)[:n_samples]
    y = np.concatenate(labels)[:n_samples]
    return X, y

print("  Extracting train features...")
X_tr_raw, y_tr_raw = extract_features(train_set, N_TRAIN)
print(f"  Train: {X_tr_raw.shape}")

print("  Extracting test features...")
X_te_raw, y_te_raw = extract_features(test_set, N_TEST)
print(f"  Test: {X_te_raw.shape}")

# ============================================================
# Step 2: Create long-tailed binary version
# ============================================================
print("\n[2/4] Creating long-tailed binary version...")

# Binary: class 0 (airplane) as minority, all others as majority
# Make it long-tailed by subsampling minority class
minority_class = 0
y_tr_binary = (y_tr_raw == minority_class).astype(int)
y_te_binary = (y_te_raw == minority_class).astype(int)

# Long-tail: keep only 10% of minority samples
n_minority_tr = int((y_tr_binary == 1).sum())
n_keep = max(50, n_minority_tr // 10)  # 10% long tail
min_idx = np.where(y_tr_binary == 1)[0]
maj_idx = np.where(y_tr_binary == 0)[0]

np.random.seed(SEED)
keep_min = np.random.choice(min_idx, n_keep, replace=False)
# Keep all majority
keep = np.concatenate([keep_min, maj_idx])
np.random.shuffle(keep)

X_tr = X_tr_raw[keep]
y_tr = y_tr_binary[keep]
X_te = X_te_raw
y_te = y_te_binary

n, d = X_tr.shape
ir = y_tr.sum() / (n - y_tr.sum())
print(f"  Long-tailed train: n={n}, d={d}, IR={ir:.4f}")
print(f"  Minority samples: {int(y_tr.sum())}")
print(f"  Test: n={len(y_te)}, minority={int(y_te.sum())}")

# PCA dimensionality reduction to d~50
print("  Applying PCA...")
pca = PCA(n_components=50, random_state=SEED)
X_tr = pca.fit_transform(X_tr)
X_te = pca.transform(X_te)
print(f"  After PCA: d={X_tr.shape[1]}")

# ============================================================
# Step 3: Run BMA experiment
# ============================================================
print("\n[3/4] Running BMA experiment...")

def min_var_w(Sigma, eps_min=None):
    M = Sigma.shape[0]
    tr = np.trace(Sigma) / max(M, 1)
    Sigma_reg = Sigma + 1e-6 * max(tr, 1e-10) * np.eye(M)
    Si = inv(Sigma_reg)
    w = Si @ np.ones(M)
    w = np.clip(w, 0, None)
    s = w.sum()
    w = w / s if s > 0 else np.ones(M) / M
    return w

models_list = [
    ("LR", LogisticRegression(max_iter=500, random_state=SEED, C=1.0)),
    ("RF", RandomForestClassifier(n_estimators=40, max_depth=6, random_state=SEED)),
    ("GB", GradientBoostingClassifier(n_estimators=25, max_depth=3, random_state=SEED)),
    ("SVM", SVC(probability=True, random_state=SEED, C=1.0)),
    ("kNN", KNeighborsClassifier(n_neighbors=7)),
]

N_SPLITS = 5
results = []

for split in range(N_SPLITS):
    seed = SEED + split
    # Use fixed test set, vary train split
    X_t, X_v, y_t, y_v = train_test_split(
        X_tr, y_tr, test_size=0.3, random_state=seed, stratify=y_tr)

    P_v_list, P_te_list, names = [], [], []
    for name, mdl in models_list:
        try:
            mdl.fit(X_t, y_t)
            P_v_list.append(mdl.predict_proba(X_v)[:, 1])
            P_te_list.append(mdl.predict_proba(X_te)[:, 1])
            names.append(name)
        except:
            pass

    if len(P_v_list) < 3:
        continue

    P_v = np.column_stack(P_v_list)
    P_te = np.column_stack(P_te_list)
    M = len(names)

    # Global BMA
    res = P_v - y_v[:, None]
    Sigma = np.cov(res, rowvar=False, bias=True)
    w_g = min_var_w(Sigma)
    auc_global = roc_auc_score(y_te, P_te @ w_g)

    # Single best
    aucs_v = [roc_auc_score(y_v, P_v[:, i]) for i in range(M)]
    best_i = np.argmax(aucs_v)
    auc_single = roc_auc_score(y_te, P_te[:, best_i])

    # Equal weight
    auc_equal = roc_auc_score(y_te, P_te.mean(axis=1))

    # Regionwise BMA (K=2, split by first PC)
    from sklearn.decomposition import PCA as PCA2
    pca2 = PCA2(n_components=1, random_state=seed)
    pca2.fit(X_t)
    val_pc = pca2.transform(X_v).flatten()
    te_pc = pca2.transform(X_te).flatten()
    median_pc = np.median(pca2.transform(X_t).flatten())

    rv = val_pc > median_pc
    rt = te_pc > median_pc
    pred_rw = np.zeros(len(y_te))
    for rmask, tmask in [(rv, rt), (~rv, ~rt)]:
        if rmask.sum() < 15 or tmask.sum() < 3:
            pred_rw[tmask] = P_te[tmask] @ w_g
            continue
        res_r = P_v[rmask] - y_v[rmask, None]
        Sigma_r = np.cov(res_r, rowvar=False, bias=True)
        w_r = min_var_w(Sigma_r)
        pred_rw[tmask] = P_te[tmask] @ w_r
    auc_rw = roc_auc_score(y_te, pred_rw)

    # CEIWC check: minority region weight
    # Find minority region (low PC = likely minority)
    minority_rmask = val_pc < median_pc
    res_min = P_v[minority_rmask] - y_v[minority_rmask, None]
    if minority_rmask.sum() >= 15:
        Sigma_min = np.cov(res_min, rowvar=False, bias=True)
        w_min = min_var_w(Sigma_min)
        w_minority = w_min[1] if M > 1 else w_min[0]  # RF weight in minority region
    else:
        w_minority = -1

    results.append({
        "split": split,
        "auc_global": auc_global,
        "auc_rw": auc_rw,
        "auc_single": auc_single,
        "auc_equal": auc_equal,
        "w_minority_region": float(w_minority),
    })
    print(f"  Split {split}: global={auc_global:.4f}, rw={auc_rw:.4f}, diff={auc_rw-auc_global:+.4f}, w_min={w_minority:.3f}")

# ============================================================
# Step 4: Summary
# ============================================================
print("\n[4/4] Summary")
avg_global = np.mean([r["auc_global"] for r in results])
avg_rw = np.mean([r["auc_rw"] for r in results])
avg_single = np.mean([r["auc_single"] for r in results])
avg_equal = np.mean([r["auc_equal"] for r in results])
avg_w_min = np.mean([r["w_minority_region"] for r in results])

print(f"\n  Mean AUC:")
print(f"    Single Best:  {avg_single:.4f}")
print(f"    Equal Weight:  {avg_equal:.4f}")
print(f"    Global BMA:    {avg_global:.4f}")
print(f"    Regionwise BMA:{avg_rw:.4f}")
print(f"    Diff (RW-G):   {avg_rw - avg_global:+.4f}")
print(f"  Minority-region avg weight: {avg_w_min:.3f}")

# Save
out = {
    "dataset": "CIFAR-10-LT (ResNet-18 features)",
    "n_train": int(n),
    "n_test": int(len(y_te)),
    "d_after_pca": int(X_tr.shape[1]),
    "imbalance_ratio": float(ir),
    "n_splits": N_SPLITS,
    "mean_auc_single": float(avg_single),
    "mean_auc_equal": float(avg_equal),
    "mean_auc_global": float(avg_global),
    "mean_auc_rw": float(avg_rw),
    "diff_rw_minus_global": float(avg_rw - avg_global),
    "mean_w_minority_region": float(avg_w_min),
    "per_split": results,
}

out_path = os.path.join(OUT_DIR, "crossmodal_cifar10.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\n  Results saved to {out_path}")
print("=" * 60)
