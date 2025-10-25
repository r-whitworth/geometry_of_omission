#!/usr/bin/env python3
# ==========================================================
# Geometry of Omission — Reproduction Script (Figures 1–6)
# ==========================================================
# Author: Rebecca Whitworth (2025)
# Contact: rebeccawhitworth@gmail.com | github.com/r-whitworth
#
# Purpose:
#   Fully reproducible pipeline for:
#   "The Geometry of Omission: Type I, II, and III Identification in Correlated Data"
#
# What this script does (deterministic; fixed seeds):
#   • Generates synthetic datasets for Type I / II / III DGP regimes
#   • Trains Logistic, XGBoost, and a small BatchNorm MLP — each WITH and WITHOUT region
#   • Exports figures:
#       Fig 1  - DGP schematic (Type I/II/III)
#       Fig 2  - Group-wise Calibration (Type I)
#       Fig 3  - Reliability Curves (Type I)
#       Fig 4  - Type II predicted-probability distributions:
#                a) XGB no region  (legacy name kept)
#                b) XGB with region (legacy name kept)
#                c-f) Logistic/XGB/NN no/with region (added)
#       Fig 5  - Type III distributions (same layout & naming pattern as Fig 4)
#       Fig 6  - Reliability across regimes
#   • Writes CSVs:
#       reconstruction_Type_I.csv / reconstruction_Type_II.csv / reconstruction_Type_III.csv
#
# Notes:
#   • Figures & CSVs saved to repo-local output
#   • All random seeds fixed: NumPy, Torch, sklearn splits
#   • Figure titles are commented out for paper compile
# ==========================================================

# -----------------------------
# Imports
# -----------------------------
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scipy.stats import gaussian_kde
from scipy import stats
from scipy.special import erf

# -----------------------------
# Global style & constants
# -----------------------------
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams.update({
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.titlesize": 11,
    "axes.edgecolor": "0.2",
    "axes.labelcolor": "0.1",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 300,
})

TITLE_PAD_MAIN = 12
TITLE_PAD_SUB  = 8

PALETTE = {
    "Logistic (no region)": "#082a54",   # dark blue
    "XGBoost (no region)":  "#e02b35",   # red
    "NeuralNet (BN)":       "#59a89c",   # teal
    "Diagnostic (+region)": "#a559aa",   # purple
    "True rate":            "#cecece"    # gray
}
# Short-name palette for new code paths
PALETTE.update({
    "Logistic": PALETTE["Logistic (no region)"],
    "XGBoost": PALETTE["XGBoost (no region)"],
    "NeuralNet": PALETTE["NeuralNet (BN)"],
})
REGION_COLORS = ["#082a54", "#59a89c", "#a559aa"]

# -----------------------------
# Output directory
# -----------------------------
import pathlib

try:
    base_dir = pathlib.Path(__file__).resolve().parent
except NameError:
    # __file__ is not defined in notebooks or interactive sessions
    base_dir = pathlib.Path.cwd()

export_dir = base_dir / "figures" / "diagnostics"
export_dir.mkdir(parents=True, exist_ok=True)

def save_dgp(df, name):
    """Save a simulated DGP to CSV with deterministic name."""
    path = os.path.join(export_dir, f"dgp_{name}.csv")
    df.to_csv(path, index=False)
    print(f"✅ Saved {path}")

def savefig(name):
    if not name.lower().endswith(".png"):
        name += ".png"
    path = os.path.join(export_dir, name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"✅ saved {path}")

# -----------------------------
# Seeding for determinism
# -----------------------------
def seed_all(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

# -----------------------------
# Utilities
# -----------------------------
def shared_quantile_reliability(y_true, models_probs: dict, n_bins=10):
    all_probs = np.concatenate(list(models_probs.values()))
    edges = np.quantile(all_probs, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    curves = OrderedDict()
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

def reconstruction_ratios(y_true, models_no, models_with, regime_name):
    results = []
    for m in models_no.keys():
        auc_no   = roc_auc_score(y_true, models_no[m])
        auc_with = roc_auc_score(y_true, models_with[m])
        recon = (auc_no - 0.5) / (auc_with - 0.5) if (auc_with > 0.5) else np.nan
        results.append((m, auc_no, auc_with, recon))
    df = pd.DataFrame(results, columns=["Model", "AUC_no", "AUC_with", "Recon"])
    csv_path = os.path.join(export_dir, f"reconstruction_{regime_name.replace(' ', '_')}.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved {csv_path}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
    return df

# -----------------------------
# Models
# -----------------------------
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_nn_binary(X, y, epochs=25, lr=1e-3, seed=42):
    seed_all(seed)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y[:, None], dtype=torch.float32)
    ds = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    m = MLP(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    loss = nn.BCEWithLogitsLoss()
    m.train()
    for _ in range(epochs):
        for xb, yb in ds:
            opt.zero_grad()
            l = loss(m(xb), yb)
            l.backward()
            opt.step()
    m.eval()
    return m

# ==========================================================
# FIGURE 1 — DGP Geometry Schematic (Type I / II / III)
# ==========================================================
def figure_1_dgp_geometry(export_dir):
    seed_all(1)
    n = 300
    from scipy.stats import multivariate_normal

    Z1 = np.concatenate([np.full(n, -1), np.full(n, 1)])
    X1 = np.random.normal(0, 1, size=(2 * n,))
    Y1 = Z1 + np.random.normal(0, 0.5, size=(2 * n,))

    rho = 0.7
    Z2 = np.concatenate([np.full(n, -1), np.full(n, 1)])
    X2 = rho * Z2 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, size=(2 * n,))
    Y2 = 0.6 * X2 + np.random.normal(0, 0.4, size=(2 * n,))

    cov = np.array([[1.0, 0.8, 0.6],
                    [0.8, 1.0, 0.7],
                    [0.6, 0.7, 1.0]])
    data = multivariate_normal.rvs(mean=[0, 0, 0], cov=cov, size=2 * n, random_state=1)
    Z3 = np.sign(data[:, 0])
    X3 = data[:, 1]
    Y3 = data[:, 2] + 0.3 * Z3

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [
        (axes[0], X1, Y1, Z1, "Type I — Intercept Only\n(No Reconstruction)"),
        (axes[1], X2, Y2, Z2, "Type II — Correlated Covariate\n(Partial Reconstruction)"),
        (axes[2], X3, Y3, Z3, "Type III — Latent Correlation Network\n(Complete Reconstruction)"),
    ]
    colors = {-1: "#082a54", 1: "#a559aa"}

    for ax, X, Y, Z, title in panels:
        for zval, c in colors.items():
            mask = Z == zval
            ax.scatter(X[mask], Y[mask], color=c, s=18, alpha=0.8)
        ax.set_title(title, fontsize=10, pad=TITLE_PAD_SUB)
        ax.set_xlabel(r"$X$")
        ax.set_ylabel(r"$Y$")
        ax.axhline(0, color="0.8", lw=0.8)
        ax.axvline(0, color="0.8", lw=0.8)

    handles = [plt.Line2D([], [], color=c, marker="o", ls="", label=f"Group {i}")
               for i, c in zip(["A", "B"], colors.values())]
    fig.legend(handles, ["Group A", "Group B"], loc="lower center",
               ncol=2, frameon=False, fontsize=9)
    plt.subplots_adjust(top=0.85, bottom=0.20, wspace=0.35)
    savefig("fig_1_dgp_geometry")
    plt.close(fig)

def fit_bundle_both(df, seed=42):
    """
    Fit Logit, XGB, NN with AND without region using THE SAME train/test split.
    This is critical - we need to compare apples to apples.
    Returns models_no, models_with, and test set (Xte, yte, regions).
    """
    feats = ["income", "dti", "util", "hist", "edu", "empyrs"]
    
    # ONE split, stratified on outcome only (NOT on region!)
    train, test = train_test_split(df, test_size=0.3, stratify=df["y"], random_state=seed)
    
    # Scale features
    scaler = StandardScaler().fit(train[feats])
    Xtr_base = scaler.transform(train[feats])
    Xte_base = scaler.transform(test[feats])
    
    ytr = train["y"].values
    yte = test["y"].values
    gre = test["region_id"].values
    
    # ============================================
    # FIT WITHOUT REGION
    # ============================================
    Xtr_no = Xtr_base
    Xte_no = Xte_base
    
    # Logistic (no region)
    logit_no = LogisticRegression(max_iter=1000, random_state=seed).fit(Xtr_no, ytr)
    p_log_no = logit_no.predict_proba(Xte_no)[:, 1]
    
    # XGBoost (no region)
    xgb_no = xgb.XGBClassifier(
        max_depth=4, n_estimators=200, learning_rate=0.05,
        subsample=0.8, random_state=seed, eval_metric="logloss"
    ).fit(Xtr_no, ytr)
    p_xgb_no = xgb_no.predict_proba(Xte_no)[:, 1]
    
    # Neural Net (no region)
    nn_no = train_nn_binary(Xtr_no, ytr, epochs=25, lr=1e-3, seed=seed)
    with torch.no_grad():
        p_nn_no = torch.sigmoid(nn_no(torch.tensor(Xte_no, dtype=torch.float32))).numpy().ravel()
    
    models_no = OrderedDict([
        ("Logistic", p_log_no),
        ("XGBoost", p_xgb_no),
        ("NeuralNet", p_nn_no)
    ])
    
    # ============================================
    # FIT WITH REGION (add region dummies)
    # ============================================
    Rtr = pd.get_dummies(train["region_id"], drop_first=True)
    Rte = pd.get_dummies(test["region_id"], drop_first=True).reindex(columns=Rtr.columns, fill_value=0)
    
    Xtr_with = np.column_stack([Xtr_base, Rtr.values])
    Xte_with = np.column_stack([Xte_base, Rte.values])
    
    # Logistic (with region)
    logit_with = LogisticRegression(max_iter=1000, random_state=seed).fit(Xtr_with, ytr)
    p_log_with = logit_with.predict_proba(Xte_with)[:, 1]
    
    # XGBoost (with region)
    xgb_with = xgb.XGBClassifier(
        max_depth=4, n_estimators=200, learning_rate=0.05,
        subsample=0.8, random_state=seed, eval_metric="logloss"
    ).fit(Xtr_with, ytr)
    p_xgb_with = xgb_with.predict_proba(Xte_with)[:, 1]
    
    # Neural Net (with region)
    nn_with = train_nn_binary(Xtr_with, ytr, epochs=25, lr=1e-3, seed=seed)
    with torch.no_grad():
        p_nn_with = torch.sigmoid(nn_with(torch.tensor(Xte_with, dtype=torch.float32))).numpy().ravel()
    
    models_with = OrderedDict([
        ("Logistic", p_log_with),
        ("XGBoost", p_xgb_with),
        ("NeuralNet", p_nn_with)
    ])
    
    return models_no, models_with, (Xte_base, yte, gre)

# ==========================================================
# TYPE I — No signal at all (region unobservable anywhere)
# ==========================================================
def simulate_typeI(seed=42):
    seed_all(seed)
    regions = [-0.5, 0.0, 0.7]  # intercepts only
    sizes   = [4000, 3000, 2000]
    NOISE_SD    = 1.2
    INCOME_COEF = 0.00006
    
    def make_region(alpha, n):
        income = np.random.lognormal(10, 0.5, n)
        dti    = np.random.beta(2, 5, n) * 2
        util   = np.random.beta(2, 3, n)
        hist   = np.random.uniform(0, 25, n)
        edu    = np.random.randint(0, 4, n)
        empyrs = np.random.uniform(0, 20, n)
        y_star = (alpha + INCOME_COEF*income - 0.9*dti - 0.5*util
                  + 0.04*hist + 0.10*edu + 0.03*empyrs
                  + np.random.normal(0, NOISE_SD, n))
        y = (y_star > 0).astype(int)
        return pd.DataFrame({
            "income": income, "dti": dti, "util": util, "hist": hist,
            "edu": edu, "empyrs": empyrs, "region_id": alpha, "y": y  # <-- changed column name
        })
    
    parts = [make_region(a, n) for a, n in zip(regions, sizes)]
    df = pd.concat(parts, ignore_index=True)
    
    # Map intercepts to region IDs 1, 2, 3
    alpha_to_id = {-0.5: 1, 0.0: 2, 0.7: 3}
    df["region_id"] = df["region_id"].map(alpha_to_id)
    
    return df

# ==========================================================
# TYPE II — Single-path recovery via income only, tunable ρ
# ==========================================================
def simulate_typeII(seed=42, rho=0.3, n_by_region=(4000, 3000, 2000)):
    seed_all(seed)
    sizes = {r: n for r, n in zip((1, 2, 3), n_by_region)}
    region_code = {1: -1.0, 2: 0.0, 3: +1.0}

    # i.i.d. covariates unrelated to region
    def draw_other_X(n):
        dti    = np.random.beta(2, 5, n) * 2
        util   = np.random.beta(2, 3, n)
        hist   = np.random.uniform(0, 25, n)
        edu    = np.random.randint(0, 4, n)
        emp    = np.random.uniform(0, 20, n)
        return dti, util, hist, edu, emp

    mu_log_inc, sigma_log_inc = 10.0, 0.25
    INCOME_COEF, NOISE_SD = 0.00003, 1.5
    B = dict(dti=-0.8, util=-0.5, hist=0.04, edu=0.08, empyrs=0.03)
    
    REGION_INTERCEPT = 0.4  # <-- strength of direct region effect
    
    parts = []
    for r in (1, 2, 3):
        n = sizes[r]
        z_r = region_code[r]
        
        # income construction (keep your existing correlation structure)
        eps = np.random.normal(0, 1, n)
        income = np.exp(mu_log_inc + sigma_log_inc *
                (np.sqrt(1 - rho**2)*eps + rho*z_r)
                + np.random.normal(0, 0.15, n))
        
        dti, util, hist, edu, emp = draw_other_X(n)
        
        # NOW: add direct region effect to outcome
        lin = (INCOME_COEF*income + B["dti"]*dti + B["util"]*util
               + B["hist"]*hist + B["edu"]*edu + B["empyrs"]*emp
               + REGION_INTERCEPT * z_r  # <-- DIRECT EFFECT
               + np.random.normal(0, NOISE_SD, n))
        y = (lin > 0).astype(int)

        parts.append(pd.DataFrame({
            "income": income, "dti": dti, "util": util, "hist": hist,
            "edu": edu, "empyrs": emp, "region_id": r, "y": y
        }))

    df = pd.concat(parts, ignore_index=True)
    return df

# ==========================================================
# TYPE III — Latent correlation network, tunable ρ (copula)
# ==========================================================
def simulate_typeIII(seed=42, rho=0.5, n_by_region=(4000, 3000, 2000)):
    seed_all(seed)
    sizes = {r: n for r, n in zip((1, 2, 3), n_by_region)}
    region_code = {1: -1.0, 2: 0.0, 3: +1.0}  # <-- ADD THIS

    # base (ρ=1) correlation template; we'll scale off-diagonals by ρ
    base_corr = np.array([
        [ 1.00, -0.35, -0.30,  0.25,  0.05,  0.30],
        [-0.35,  1.00,  0.25, -0.10,  0.05, -0.10],
        [-0.30,  0.25,  1.00, -0.10,  0.05, -0.05],
        [ 0.25, -0.10, -0.10,  1.00,  0.05,  0.10],
        [ 0.05,  0.05,  0.05,  0.05,  1.00,  0.05],
        [ 0.30, -0.10, -0.05,  0.10,  0.05,  1.00],
    ])
    # scale off-diagonals by rho (keep ones on the diagonal)
    corr = np.eye(6) + rho * (base_corr - np.eye(6))

    def gcop(marginals, corr, n):
        L = np.linalg.cholesky(corr)
        Z = np.random.randn(n, len(marginals)) @ L.T
        U = 0.5 * (1 + erf(Z / np.sqrt(2)))  # Φ(Z)
        return np.column_stack([m(U[:, i]) for i, m in enumerate(marginals)])

    def spec_for_region(r):
        # region-specific income/util/dti marginals (means/shape differ)
        if r == 1:
            inc  = lambda u: np.exp(9.7 + 0.50*stats.norm.ppf(u))
            util = lambda u: stats.beta(a=3,   b=2).ppf(u)
            dti  = lambda u: 2*stats.beta(a=3, b=3).ppf(u)
        elif r == 2:
            inc  = lambda u: np.exp(10.0 + 0.50*stats.norm.ppf(u))
            util = lambda u: stats.beta(a=2.5, b=3).ppf(u)
            dti  = lambda u: 2*stats.beta(a=2.5,b=3).ppf(u)
        else:
            inc  = lambda u: np.exp(10.3 + 0.50*stats.norm.ppf(u))
            util = lambda u: stats.beta(a=2,   b=4).ppf(u)
            dti  = lambda u: 2*stats.beta(a=2, b=4).ppf(u)
        hist   = lambda u: 25*u
        edu    = lambda u: (stats.randint(0,4).ppf(u)).astype(float)
        empyrs = lambda u: 20*u
        return [inc, util, dti, hist, edu, empyrs]

    INCOME_COEF, NOISE_SD = 0.00008, 0.9
    B = dict(dti=-0.8, util=-0.5, hist=0.04, edu=0.08, empyrs=0.03)
    REGION_INTERCEPT = 0.4  # <-- ADD THIS: direct region effect

    parts = []
    for r in (1, 2, 3):
        n = sizes[r]
        z_r = region_code[r]  # <-- ADD THIS
        X = gcop(spec_for_region(r), corr, n)
        df_r = pd.DataFrame(X, columns=["income","dti","util","hist","edu","empyrs"])
        df_r["region_id"] = r
        lin = (INCOME_COEF*df_r["income"] + B["dti"]*df_r["dti"] + B["util"]*df_r["util"]
               + B["hist"]*df_r["hist"] + B["edu"]*df_r["edu"] + B["empyrs"]*df_r["empyrs"]
               + REGION_INTERCEPT * z_r
               + np.random.normal(0, NOISE_SD, n))
        df_r["y"] = (lin > 0).astype(int)
        parts.append(df_r)

    df = pd.concat(parts, ignore_index=True)
    return df

# ==========================================================
# Figure helpers — KDE panel for a single model
# ==========================================================
def plot_kde_by_region(ax, preds, regions, title):
    bins = np.linspace(0, 1, 200)
    for r, color in zip(sorted(np.unique(regions)), REGION_COLORS):
        vals = preds[regions == r]
        kde = gaussian_kde(vals)
        ax.plot(bins, kde(bins), color=color, lw=2.0, alpha=0.9, label=f"Region {int(r)}")
    ax.set_xlabel("Predicted Probability", fontsize=11, labelpad=8)
    ax.set_ylabel("Density", fontsize=11, labelpad=6)
    if title:  # only draw the title if a non-empty string is passed
        ax.set_title(title, fontsize=10, pad=TITLE_PAD_SUB)

# ==========================================================
# Figures 2 & 3 — Type I (Calibration, Bias, Reliability)
# ==========================================================
def figures_2_3_typeI(df, models_no, yte, gre):
    # Fig 2a: Group-wise calibration
    fig, ax = plt.subplots(figsize=(7, 4))
    true_rate = pd.DataFrame({"region": gre, "y": yte}).groupby("region")["y"].mean()
    pred_means = {
        name: pd.DataFrame({"region": gre, "pred": p}).groupby("region")["pred"].mean()
        for name, p in models_no.items()
    }
    cal_df = pd.concat(pred_means, axis=1)
    cal_df["True rate"] = true_rate
    colors = [PALETTE.get(c, "0.6") for c in cal_df.columns]
    cal_df.plot(kind="bar", ax=ax, color=colors, width=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Approval Rate"); ax.set_xlabel("Region")
    ax.legend(frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.30))
    savefig("fig_2a_groupwise_calibration")
    plt.close(fig)

    # Fig 2b: Group-wise prediction bias
    fig, ax = plt.subplots(figsize=(7, 4))
    resid_df = cal_df.drop(columns="True rate").subtract(cal_df["True rate"], axis=0)
    colors = [PALETTE.get(c, "0.6") for c in resid_df.columns]
    resid_df.plot(kind="bar", ax=ax, color=colors, width=0.8)
    ax.axhline(0, color="0.2", lw=1, linestyle="--")
    ax.set_ylabel("Prediction Bias (ŷ − y_true)"); ax.set_xlabel("Region")
    ax.legend(frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.30))
    savefig("fig_2b_groupwise_bias")
    plt.close(fig)

    # Fig 3: Reliability curves (shared quantiles)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    curves = shared_quantile_reliability(yte, models_no, n_bins=10)  # <- FIXED: use models_no
    line_styles = ["-", "--", "-."]
    for (name, (x, y)), ls in zip(curves.items(), line_styles):
        ax.plot(x, y, marker="o", lw=2.2, label=name,
                color=PALETTE.get(name, "0.3"), linestyle=ls)
    ax.plot([0, 1], [0, 1], "--", color="0.5", alpha=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Empirical Frequency")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    savefig("fig_3_reliability_quantile_shared")
    plt.close(fig)

    return yte, models_no  # <- Return what the old function returned
    
# ==========================================================
# Figure 4 — Type II (All six panels)
#   Keeps legacy names for XGB: 4a (no region), 4b (with region)
#   Adds 4c–4f for Logistic/NN no/with
# ==========================================================
def figure_4_typeII(models_no, models_with, g_test):
    # a) XGB (no region) — legacy name preserved
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_kde_by_region(ax, models_no["XGBoost"], g_test, "")
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=3)
    fig.subplots_adjust(top=0.88, bottom=0.32, left=0.12, right=0.97)
    savefig("fig_4a_typeII_XGB")
    plt.close(fig)

    # b) XGB (with region) — legacy name preserved
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_kde_by_region(ax, models_with["XGBoost"], g_test, "")
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=3)
    fig.subplots_adjust(top=0.88, bottom=0.32, left=0.12, right=0.97)
    savefig("fig_4b_typeII_diagnostic")
    plt.close(fig)

    # c–f) Logistic + NN, no/with region
    for tag, model_key, suffix in [
        ("c", "Logistic",   "Logistic_no_region"),
        ("d", "Logistic",   "Logistic_with_region"),
        ("e", "NeuralNet",  "NeuralNet_no_region"),
        ("f", "NeuralNet",  "NeuralNet_with_region"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4))
        preds = models_no[model_key] if "no_region" in suffix else models_with[model_key]
        title = suffix.replace("_", " ").title()
        plot_kde_by_region(ax, preds, g_test, "")
        ax.legend(frameon=False, fontsize=9, loc="upper center",
                  bbox_to_anchor=(0.5, -0.22), ncol=3)
        fig.subplots_adjust(top=0.88, bottom=0.32, left=0.12, right=0.97)
        savefig(f"fig_4{tag}_typeII_{suffix}")
        plt.close(fig)
        
# ==========================================================
# Figure 5 — Type III (All six panels; same naming convention)
# ==========================================================
def figure_5_typeIII(models_no, models_with, g_test):
    # a) XGB (no region) — legacy name preserved
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_kde_by_region(ax, models_no["XGBoost"], g_test, "")
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=3)
    fig.subplots_adjust(top=0.88, bottom=0.32, left=0.12, right=0.97)
    savefig("fig_5a_typeIII_XGB")
    plt.close(fig)

    # b) XGB (with region) — legacy name preserved
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_kde_by_region(ax, models_with["XGBoost"], g_test, "")
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=3)
    fig.subplots_adjust(top=0.88, bottom=0.32, left=0.12, right=0.97)
    savefig("fig_5b_typeIII_diagnostic")
    plt.close(fig)

    # c–f) Logistic + NN, no/with region
    for tag, model_key, suffix in [
        ("c", "Logistic",   "Logistic_no_region"),
        ("d", "Logistic",   "Logistic_with_region"),
        ("e", "NeuralNet",  "NeuralNet_no_region"),
        ("f", "NeuralNet",  "NeuralNet_with_region"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4))
        preds = models_no[model_key] if "no_region" in suffix else models_with[model_key]
        title = suffix.replace("_", " ").title()
        plot_kde_by_region(ax, preds, g_test, "")
        ax.legend(frameon=False, fontsize=9, loc="upper center",
                  bbox_to_anchor=(0.5, -0.22), ncol=3)
        fig.subplots_adjust(top=0.88, bottom=0.32, left=0.12, right=0.97)
        savefig(f"fig_5{tag}_typeIII_{suffix}")
        plt.close(fig)

# ==========================================================
# Figure 6 — Reliability Curves Across Regimes (Type I–III)
# ==========================================================
def figure_6_reliability_across_regimes(model_sets, export_dir=None):
    if export_dir is None:
        export_dir = globals().get("export_dir", ".")
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    line_styles = {"Type I": "-", "Type II": "--", "Type III": "-."}
    colors      = {"Type I": "#082a54", "Type II": "#e02b35", "Type III": "#59a89c"}
    for reg_name, (y_true, models) in model_sets.items():
        no_region_models = models  # already no-region sets passed from main
        combined = np.mean(np.column_stack(list(no_region_models.values())), axis=1)
        curves = shared_quantile_reliability(y_true, {reg_name: combined}, n_bins=10)
        for (_, (x, y)) in curves.items():
            ax.plot(x, y, lw=2.4, color=colors[reg_name],
                    linestyle=line_styles[reg_name], marker="o", label=reg_name)
    ax.plot([0, 1], [0, 1], "--", color="0.5", alpha=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Empirical Frequency")
    ax.legend(frameon=False, loc="lower right", ncol=1, title="Regime")
    savefig("fig_6_reliability_across_regimes")
    plt.close(fig)
    
# -----------------------------
# Correlation heatmaps (appendix)
# -----------------------------
def _corr_with_region(df):
    """
    Compute point-biserial style correlations between region dummies and features,
    plus the feature–feature correlation matrix. Returns a tidy matrix for heatmap.
    """
    feats = ["income","dti","util","hist","edu","empyrs"]
    # one-hots for region (drop_first=False so all appear)
    R = pd.get_dummies(df["region_id"], prefix="R", drop_first=False).astype(float)
    F = df[feats].astype(float)

    # feature-feature corr
    FF = F.corr()

    # region-feature correlations (Pearson on one-hots ~= difference in means scaled)
    RF = pd.DataFrame(index=R.columns, columns=feats, dtype=float)
    for rcol in R.columns:
        for f in feats:
            RF.loc[rcol, f] = np.corrcoef(R[rcol], F[f])[0, 1]

    # Build a single matrix to show both blocks
    # Order rows: Regions then Features
    # Order cols: Regions then Features
    RR = R.corr()
    top = pd.concat([RR, RF], axis=1)
    bottom = pd.concat([RF.T, FF], axis=1)
    M = pd.concat([top, bottom], axis=0)
    return M

def figure_heatmap_corr(df, title, outname):
    M = _corr_with_region(df)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(M.values, vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(M.shape[1])); ax.set_yticks(np.arange(M.shape[0]))
    ax.set_xticklabels(M.columns, rotation=90); ax.set_yticklabels(M.index)
    ax.set_title(title, pad=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", rotation=270, labelpad=12)
    plt.tight_layout()
    savefig(outname)
    plt.close(fig)

# ===================================================
# Save rho for fig 7
# ===================================================

def sweep_rho_for_typeII(rhos=(0.1,0.2,0.3,0.4,0.5), seed=42):
    rows = []
    for r in rhos:
        df = simulate_typeII(seed=seed, rho=r)
        m_no, m_with, (Xte, y, g) = fit_bundle_both(df, seed=seed)
        
        for m in m_no.keys():
            auc_no   = roc_auc_score(y, m_no[m])
            auc_with = roc_auc_score(y, m_with[m])
            R = (auc_no - 0.5) / (auc_with - 0.5) if (auc_with > 0.5) else np.nan
            rows.append({"rho": r, "model": m, "auc_no": auc_no, "auc_with": auc_with, "R": R})
    
    df_out = pd.DataFrame(rows)
    path = os.path.join(export_dir, "reconstruction_vs_rho_typeII.csv")
    df_out.to_csv(path, index=False)
    print(f"✅ Saved {path}")
    return df_out

# ==========================================================
# Production Pipe
# ==========================================================
def main():
    print("\n=== GEOMETRY OF OMISSION — REPRODUCTION RUN ===")
    print(f"• Using export directory: {export_dir}")
    os.makedirs(export_dir, exist_ok=True)

    # ------------------------------------------------------
    # 1. Figure 1 — DGP geometry schematic
    # ------------------------------------------------------
    print("• Generating Figure 1 (DGP geometry)…")
    figure_1_dgp_geometry(export_dir)

    # ------------------------------------------------------
    # 2–6. Type I–III simulations, fits, ratios, figures
    # ------------------------------------------------------
    # --- TYPE I ---
    df_typeI = simulate_typeI()
    save_dgp(df_typeI, "Type_I")
    mI_no, mI_with, (XteI, yI, gI) = fit_bundle_both(df_typeI)
    reconstruction_ratios(yI, mI_no, mI_with, "Type I")
    figures_2_3_typeI(df_typeI, mI_no, yI, gI)
    figure_heatmap_corr(df_typeI,  "Type I — Correlation Geometry",  "appendix_typeI_corr_heatmap")

    print("\n=== sanity check: Type I ===")
    print(df_typeI.groupby("region_id")[["income","y"]].mean())
    print("\ncorrelation with region_id:")
    print(df_typeI.corr(numeric_only=True)["region_id"].sort_values(ascending=False))
    
    # --- TYPE II ---
    df_typeII = simulate_typeII(rho=0.3)
    save_dgp(df_typeII, "Type_II")
    mII_no, mII_with, (XteII, yII, gII) = fit_bundle_both(df_typeII)
    reconstruction_ratios(yII, mII_no, mII_with, "Type II")
    figure_4_typeII(mII_no, mII_with, gII)
    figure_heatmap_corr(df_typeII, "Type II — Correlation Geometry", "appendix_typeII_corr_heatmap")

    print("\n=== sanity check: Type II geometry ===")
    print(df_typeII.groupby("region_id")[["income","dti","util","y"]].mean())
    print("\ncorrelation with region_id:")
    print(df_typeII.corr(numeric_only=True)["region_id"].sort_values(ascending=False))

    # --- TYPE III ---
    df_typeIII = simulate_typeIII(rho=0.5)
    save_dgp(df_typeIII, "Type_III")
    mIII_no, mIII_with, (XteIII, yIII, gIII) = fit_bundle_both(df_typeIII)
    reconstruction_ratios(yIII, mIII_no, mIII_with, "Type III")
    figure_5_typeIII(mIII_no, mIII_with, gIII)
    figure_heatmap_corr(df_typeIII,"Type III — Correlation Geometry","appendix_typeIII_corr_heatmap")

    print("\n=== sanity check: Type III ===")
    print(df_typeIII.groupby("region_id")[["income","y"]].mean())
    print("\ncorrelation with region_id:")
    print(df_typeIII.corr(numeric_only=True)["region_id"].sort_values(ascending=False))
    
    # ------------------------------------------------------
    # 7. Reliability across regimes (Type I–III summary)
    # ------------------------------------------------------
    print("• Building Figure 6 (reliability across regimes)…")
    y_sets = {
        "Type I": (yI, mI_no),
        "Type II": (yII, mII_no),
        "Type III": (yIII, mIII_no),
    }
    figure_6_reliability_across_regimes(model_sets=y_sets, export_dir=export_dir)
    
    print("\n✅ All figures (1–6) generated successfully.")
    print(f"   Output directory: {export_dir}\n")

    print("• Sweeping ρ for Type II reconstruction curves…")
    sweep_rho_for_typeII(rhos=np.linspace(0.1, 0.6, 6), seed=42)

# ----------------------------------------------------------
# STANDARD ENTRY POINT
# ----------------------------------------------------------
if __name__ == "__main__":
    main()
