#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ==========================================================
# Geometry of Omission — Diagnostics & Transparency
# ==========================================================
# Author: Rebecca Whitworth (2025)
# Purpose: Extended diagnostics that accompany the main paper figures.
#          Regenerates Type II (correlated covariate) and Type III (latent network)
#          DGPs with fixed seeds, fits baseline models, and saves diagnostic plots.
# Output directory anchors to the repo root
# ==========================================================

# ==========================================================
# NOTE FOR REPLICATION / ECON READERS
# ==========================================================
# This diagnostic script uses *fixed seeds* rather than bootstrapped resampling and compiling.
# 
# In econometric convention, one might loop over draws or resample the DGP
# to show distributional robustness. Here, however, each DGP (Type II / III)
# is fully known and parameterized — the reconstruction pattern is deterministic
# conditional on ρ.  
#
# To run a Monte Carlo extension, simply vary the seed, e.g.:
#     for i in range(100): np.random.seed(42 + i); torch.manual_seed(42 + i)
# and store summary statistics of reconstruction rates R(ρ).
# Those Monte Carlo results can be plotted as confidence bands
# around the reconstruction curve R(ρ) or as histograms of recovered
# variance to illustrate the stability of the identification geometry.

# =========================
# Imports & setup
# =========================
import os
import numpy as np
import pandas as pd
from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve
)

import xgboost as xgb
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from scipy.stats import gaussian_kde
from scipy import stats
from scipy.special import erf

# Optional (for a cleaner correlation heatmap)
import seaborn as sns

# =========================
# Styling (match paper)
# =========================
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
mpl.rcParams.update({
    "text.usetex": False,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.titlesize": 11,
    "axes.edgecolor": "0.2",
    "axes.labelcolor": "0.1",
    "xtick.color": "0.2",
    "ytick.color": "0.2",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "figure.dpi": 300,
    "axes.titlepad": 6,
})

PALETTE = {
    "Logistic (no region)": "#082a54",   # dark blue
    "XGBoost (no region)":  "#e02b35",   # red
    "NeuralNet (BN)":       "#59a89c",   # teal
    "Diagnostic (region included)": "#a559aa",  # purple
    "True rate": "#cecece"
}
REGION_COLORS = ["#082a54", "#59a89c", "#a559aa"]  # three regions

# =========================
# Output path (diagnostics)
# =========================
base_dir = os.path.dirname(os.path.abspath(__file__))
export_dir = os.path.join(base_dir, "figures", "diagnostics")
os.makedirs(export_dir, exist_ok=True)

def savefig(name):
    if not name.lower().endswith(".png"):
        name += ".png"
    path = os.path.join(export_dir, name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"✅ saved {path}")

# =========================
# Shared helpers
# =========================
def cal_curve_quantile_shared(y_true, models_probs: dict, n_bins=10):
    """Shared-quantile reliability binning across models."""
    all_probs = np.concatenate(list(models_probs.values()))
    edges = np.quantile(all_probs, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    curves = {}
    for name, p in models_probs.items():
        inds = np.digitize(p, edges) - 1
        xs, ys = [], []
        for i in range(len(edges) - 1):
            m = inds == i
            if m.any():
                xs.append(p[m].mean())
                ys.append(y_true[m].mean())
        curves[name] = (np.array(xs), np.array(ys))
    return curves

def legend_below(ax, ncol=None, pad=-0.12):
    h, l = ax.get_legend_handles_labels()
    if ncol is None:
        ncol = max(2, len(l)//2)
    ax.figure.legend(h, l, loc="lower center",
                     bbox_to_anchor=(0.5, pad), ncol=ncol, frameon=False)
    if ax.get_legend():
        ax.legend_.remove()

# =========================
# Models
# =========================
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

def train_nn(X, y, epochs=25, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y[:, None], dtype=torch.float32)
    ds = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    m = MLP(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    loss = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for xb, yb in ds:
            opt.zero_grad(); L = loss(m(xb), yb); L.backward(); opt.step()
    return m

# =========================
# DGP — Type II (income shift; no region intercept)
# =========================
def simulate_typeII(seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    NOISE_SD = 1.2; INCOME_COEF = 0.00006
    income_shift = {1: 0.0, 2: 0.3, 3: 0.6}

    def make_region_income_only(region, n):
        income = np.random.lognormal(mean=10 + income_shift[region], sigma=0.5, size=n)
        dti  = np.random.beta(2, 5, size=n) * 2
        util = np.random.beta(2, 3, size=n)
        hist = np.random.uniform(0, 25, size=n)
        edu  = np.random.randint(0, 4, size=n)
        emp  = np.random.uniform(0, 20, size=n)
        y_star = (INCOME_COEF*income - 0.9*dti - 0.5*util + 0.04*hist + 0.1*edu + 0.03*emp
                  + np.random.normal(0, NOISE_SD, size=n))
        y = (y_star > 0).astype(int)
        return pd.DataFrame({"income":income,"dti":dti,"util":util,"hist":hist,
                             "edu":edu,"empyrs":emp,"region_id":region,"y":y})

    sizes = {1: 4000, 2: 3000, 3: 2000}
    df = pd.concat([make_region_income_only(r, sizes[r]) for r in (1,2,3)], ignore_index=True)
    return df

# =========================
# DGP — Type III (latent correlation via Gaussian copula)
# =========================
def gaussian_copula(marginals, corr, n):
    L = np.linalg.cholesky(corr)
    Z = np.random.randn(n, len(marginals)) @ L.T
    U = 0.5 * (1 + erf(Z / np.sqrt(2)))  # Φ(Z), vectorized
    X = np.column_stack([m(U[:, i]) for i, m in enumerate(marginals)])
    return X

def region_spec(r):
    if r == 1:
        inc  = lambda u: np.exp(9.6 + 0.55*stats.norm.ppf(u))
        util = lambda u: stats.beta(a=3, b=2).ppf(u)
        dti  = lambda u: 2*stats.beta(a=3, b=3).ppf(u)
    elif r == 2:
        inc  = lambda u: np.exp(10.0 + 0.50*stats.norm.ppf(u))
        util = lambda u: stats.beta(a=2.5, b=3).ppf(u)
        dti  = lambda u: 2*stats.beta(a=2.5, b=3).ppf(u)
    else:
        inc  = lambda u: np.exp(10.4 + 0.50*stats.norm.ppf(u))
        util = lambda u: stats.beta(a=2, b=4).ppf(u)
        dti  = lambda u: 2*stats.beta(a=2, b=4).ppf(u)
    hist   = lambda u: 25*u
    edu    = lambda u: (stats.randint(0,4).ppf(u)).astype(float)
    empyrs = lambda u: 20*u
    corr = np.array([
        [1.00, -0.35, -0.30,  0.25,  0.05,  0.30],
        [-0.35, 1.00,  0.25, -0.10,  0.05, -0.10],
        [-0.30, 0.25,  1.00, -0.10,  0.05, -0.05],
        [0.25, -0.10, -0.10,  1.00,  0.05,  0.10],
        [0.05,  0.05,  0.05,  0.05,  1.00,  0.05],
        [0.30, -0.10, -0.05,  0.10,  0.05,  1.00],
    ])
    return [inc, util, dti, hist, edu, empyrs], corr

def simulate_typeIII(seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    sizes = {1: 4000, 2: 3000, 3: 2000}
    parts = []
    for r in (1,2,3):
        marg, corr = region_spec(r)
        X = gaussian_copula(marg, corr, sizes[r])
        df = pd.DataFrame(X, columns=["income","util","dti","hist","edu","empyrs"])
        df["region_id"] = r
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)

    # outcome: same coefficients, NO region intercept
    NOISE_SD = 1.2; INCOME_COEF = 0.00006
    coefs = dict(income=INCOME_COEF, dti=-0.9, util=-0.5, hist=0.04, edu=0.10, empyrs=0.03)
    lin = (coefs["income"]*df["income"] + coefs["dti"]*df["dti"] + coefs["util"]*df["util"]
           + coefs["hist"]*df["hist"] + coefs["edu"]*df["edu"] + coefs["empyrs"]*df["empyrs"]
           + np.random.normal(0, NOISE_SD, size=len(df)))
    df["y"] = (lin > 0).astype(int)
    return df

# =========================
# Fit models (common)
# =========================
def fit_models(df, seed=42):
    train, test = train_test_split(df, test_size=0.3, stratify=df["region_id"], random_state=seed)
    feats = ["income","dti","util","hist","edu","empyrs"]
    scaler = StandardScaler().fit(train[feats])
    Xtr, Xte = scaler.transform(train[feats]), scaler.transform(test[feats])
    ytr, yte = train["y"].values, test["y"].values
    gre = test["region_id"].values

    # Logistic (no region)
    logit = LogisticRegression(max_iter=1000, random_state=seed).fit(Xtr, ytr)
    p_log = logit.predict_proba(Xte)[:,1]

    # XGBoost
    xgbm = xgb.XGBClassifier(max_depth=4, n_estimators=200, learning_rate=0.05,
                             subsample=0.8, random_state=seed, n_jobs=-1, eval_metric="logloss").fit(Xtr, ytr)
    p_xgb = xgbm.predict_proba(Xte)[:,1]

    # Neural net (BN)
    nnm = train_nn(Xtr, ytr, epochs=25, lr=1e-3, seed=seed)
    with torch.no_grad():
        p_nn = torch.sigmoid(nnm(torch.tensor(Xte, dtype=torch.float32))).numpy().ravel()

    # Diagnostic: add region one-hot
    Rtr = pd.get_dummies(train["region_id"], drop_first=True)
    Rte = pd.get_dummies(test["region_id"], drop_first=True).reindex(columns=Rtr.columns, fill_value=0)
    XtrD = np.column_stack([Xtr, Rtr.values]); XteD = np.column_stack([Xte, Rte.values])
    log_diag = LogisticRegression(max_iter=1000, random_state=seed).fit(XtrD, ytr)
    p_diag = log_diag.predict_proba(XteD)[:,1]

    models = OrderedDict([
        ("Logistic (no region)", p_log),
        ("XGBoost (no region)",  p_xgb),
        ("NeuralNet (BN)",       p_nn),
        ("Diagnostic (region included)", p_diag),
    ])

    return (Xtr, ytr, train["region_id"].values), (Xte, yte, gre), models, xgbm

# =========================
# Diagnostics — 1) ROC by region
# =========================
def plot_roc_by_region(y, g, models, name="diag_roc_by_region.png"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    regions = sorted(np.unique(g))
    for ax, r in zip(axes, regions):
        m = (g == r)
        for k, p in models.items():
            if len(np.unique(y[m])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y[m], p[m])
            aucv = roc_auc_score(y[m], p[m])
            ax.plot(fpr, tpr, lw=2, label=f"{k} (AUC={aucv:.3f})", color=PALETTE.get(k, "0.3"))
        ax.plot([0,1],[0,1], '--', color="0.6", lw=1)
        ax.set_title(f"Region {r}")
        ax.set_xlabel("FPR")
        if ax is axes[0]: ax.set_ylabel("TPR")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.grid(True, alpha=0.25)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncol=2, frameon=False)
    plt.tight_layout()
    savefig(name); plt.close(fig)

# =========================
# Diagnostics — 2) AUC heatmap (region × model)
# =========================
def plot_auc_matrix(y, g, models, name="diag_auc_matrix.png"):
    regions = sorted(np.unique(g))
    model_names = list(models.keys())
    M = np.zeros((len(regions), len(model_names)))
    for i, r in enumerate(regions):
        m = (g == r)
        for j, k in enumerate(model_names):
            if len(np.unique(y[m])) < 2:
                M[i, j] = np.nan
            else:
                M[i, j] = roc_auc_score(y[m], models[k][m])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=np.nanmin(M), vmax=np.nanmax(M))
    ax.set_xticks(range(len(model_names))); ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_yticks(range(len(regions)));    ax.set_yticklabels([f"Region {r}" for r in regions])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val < 0.7 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="AUC")
    ax.set_title("Per-Region AUC by Model")
    plt.tight_layout()
    savefig(name); plt.close(fig)

# =========================
# Diagnostics — 3) Residual histograms (ŷ − y)
# =========================
def plot_residual_histograms(y, models, name="diag_residuals_by_model.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.ravel()
    for ax, (k, p) in zip(axes, models.items()):
        resid = p - y  # continuous residual in [−1, 1]
        ax.hist(resid, bins=40, color=PALETTE.get(k, "0.4"), alpha=0.9, edgecolor="white")
        ax.set_title(k); ax.set_xlabel("ŷ − y"); ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    savefig(name); plt.close(fig)

# =========================
# Diagnostics — 4) Predicted probability histograms by region
# =========================
def plot_pred_hist_by_region(g, models, name="diag_pred_hist_by_region.png"):
    regions = sorted(np.unique(g))
    fig, axes = plt.subplots(len(regions), 1, figsize=(7, 7), sharex=True)
    for ax, r in zip(axes, regions):
        m = (g == r)
        for k, p in models.items():
            ax.hist(p[m], bins=40, density=True, histtype="step", lw=2,
                    label=k, color=PALETTE.get(k, "0.4"))
        ax.set_title(f"Region {r}")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Predicted Probability")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    plt.tight_layout()
    savefig(name); plt.close(fig)

# =========================
# Diagnostics — 5) Correlation heatmaps (Type III only)
# =========================
def plot_corr_heatmaps_typeIII(df, name="diag_corr_by_region_typeIII.png"):
    cols = ["income", "util", "dti", "hist", "edu", "empyrs"]
    regions = sorted(df["region_id"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True, sharey=True)
    for ax, r in zip(axes, regions):
        corr = df[df["region_id"] == r][cols].corr()
        sns.heatmap(corr, vmin=-1, vmax=1, cmap="coolwarm", center=0,
                    cbar=(r == regions[-1]), annot=False, ax=ax, square=True)
        ax.set_title(f"Region {r}")
    fig.suptitle("Within-Region Correlation Structure (Type III)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(name); plt.close(fig)

# =========================
# Diagnostics — 6) XGBoost feature importance
# =========================
def plot_xgb_feature_importance(xgb_model, feature_names, name="diag_xgb_feature_importance.png"):
    importance = xgb_model.feature_importances_
    idx = np.argsort(importance)[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(range(len(idx)), importance[idx], color="#6376f0")
    ax.set_xticks(range(len(idx))); ax.set_xticklabels([feature_names[i] for i in idx], rotation=20, ha="right")
    ax.set_ylabel("Gain"); ax.set_title("XGBoost Feature Importance")
    plt.tight_layout()
    savefig(name); plt.close(fig)

# =========================
# Diagnostics — 7) Calibration drift (shared-quantile)
# =========================
def plot_calibration_drift(y, models, name="diag_calibration_drift.png"):
    curves = cal_curve_quantile_shared(y, models, n_bins=10)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for k, (x, yy) in curves.items():
        ax.plot(x, yy, marker="o", lw=2.2, label=k, color=PALETTE.get(k, "0.4"))
    ax.plot([0,1],[0,1], "--", color="0.5", alpha=0.6)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Empirical Frequency")
    ax.set_title("Reliability (Calibration) — Shared Quantiles")
    legend_below(ax, ncol=2, pad=-0.10)
    plt.subplots_adjust(bottom=0.18)
    savefig(name); plt.close(fig)

# =========================
# Runner — Type II diagnostics
# =========================
def run_typeII_diagnostics():
    print("\n=== TYPE II (Correlated Covariate / Income shift) ===")
    df = simulate_typeII(seed=42)
    (Xtr, ytr, gtr), (Xte, yte, gte), models, xgbm = fit_models(df, seed=42)

    # quick text summary
    print("\nGlobal AUCs:")
    for k, p in models.items():
        print(f"  {k:28s} AUC={roc_auc_score(yte, p):.3f}")

    # per-region AUCs
    print("\nPer-Region AUCs:")
    for r in sorted(np.unique(gte)):
        m = (gte == r)
        for k, p in models.items():
            if len(np.unique(yte[m])) < 2:
                auc_r = np.nan
            else:
                auc_r = roc_auc_score(yte[m], p[m])
            print(f"  Region {r} — {k:28s}: AUC={auc_r:.3f}")

    # plots
    plot_roc_by_region(yte, gte, models, name="typeII_diag_roc_by_region.png")
    plot_auc_matrix(yte, gte, models, name="typeII_diag_auc_matrix.png")
    plot_residual_histograms(yte, models, name="typeII_diag_residuals_by_model.png")
    plot_pred_hist_by_region(gte, models, name="typeII_diag_pred_hist_by_region.png")
    plot_calibration_drift(yte, models, name="typeII_diag_calibration_drift.png")
    plot_xgb_feature_importance(xgbm, ["income","dti","util","hist","edu","empyrs"],
                                name="typeII_diag_xgb_feature_importance.png")

# =========================
# Runner — Type III diagnostics
# =========================
def run_typeIII_diagnostics():
    print("\n=== TYPE III (Latent Correlation Network / Gaussian copula) ===")
    df = simulate_typeIII(seed=42)
    (Xtr, ytr, gtr), (Xte, yte, gte), models, xgbm = fit_models(df, seed=42)

    # quick text summary
    print("\nGlobal AUCs:")
    for k, p in models.items():
        print(f"  {k:28s} AUC={roc_auc_score(yte, p):.3f}")

    # per-region AUCs
    print("\nPer-Region AUCs:")
    for r in sorted(np.unique(gte)):
        m = (gte == r)
        for k, p in models.items():
            if len(np.unique(yte[m])) < 2:
                auc_r = np.nan
            else:
                auc_r = roc_auc_score(yte[m], p[m])
            print(f"  Region {r} — {k:28s}: AUC={auc_r:.3f}")

    # plots
    plot_roc_by_region(yte, gte, models, name="typeIII_diag_roc_by_region.png")
    plot_auc_matrix(yte, gte, models, name="typeIII_diag_auc_matrix.png")
    plot_residual_histograms(yte, models, name="typeIII_diag_residuals_by_model.png")
    plot_pred_hist_by_region(gte, models, name="typeIII_diag_pred_hist_by_region.png")
    plot_calibration_drift(yte, models, name="typeIII_diag_calibration_drift.png")
    plot_xgb_feature_importance(xgbm, ["income","dti","util","hist","edu","empyrs"],
                                name="typeIII_diag_xgb_feature_importance.png")
    plot_corr_heatmaps_typeIII(df, name="typeIII_diag_corr_by_region.png")

# =========================
# Main
# =========================
if __name__ == "__main__":
    run_typeII_diagnostics()
    run_typeIII_diagnostics()
    print("\n✅ Diagnostics complete (see ~/Documents/Geometry_of_Omission/figures/diagnostics/)\n")


# In[ ]:




