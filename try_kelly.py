# Fix bug: we concatenated pnl and sharpe, but the covariance window used pnl only (correct).
# However, ensure mu_t shape matches Sigma (5,). Check shapes and re-run.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def kelly_var_allocator(df, lookback=64, z_alpha=2.33, var_target=250_000.0, cap_pct=0.50, shrink=0.10):
    pnl_cols = [c for c in df.columns if c.endswith("_pnl")]
    strategies = [c.replace("_pnl", "") for c in pnl_cols]
    assert all((s + "_sharpe") in df.columns for s in strategies), "Missing <strategy>_sharpe column(s)"
    
    pnl = df[[s + "_pnl" for s in strategies]].copy()
    sharpe = df[[s + "_sharpe" for s in strategies]].copy()
    
    sigma_d = pnl.rolling(lookback).std()
    mu_d = (sharpe / np.sqrt(252.0)) * sigma_d  # daily mean from annual Sharpe & daily vol
    
    weights = pd.DataFrame(0.0, index=df.index, columns=strategies)
    total_var = pd.Series(0.0, index=df.index, name="VaR_total")
    comp_var = pd.DataFrame(0.0, index=df.index, columns=strategies)
    pct_var = pd.DataFrame(0.0, index=df.index, columns=strategies)
    leftover_var = pd.Series(0.0, index=df.index, name="VaR_leftover")
    
    cap_abs = cap_pct * var_target
    
    for t in range(lookback, len(df)):
        pnl_win = pnl.iloc[t - lookback:t]
        if pnl_win.isna().any().any():
            continue
        # Σ is SxS
        Sigma = pnl_win.cov().values
        # shrinkage
        diag = np.diag(np.diag(Sigma))
        Sigma = (1 - shrink) * Sigma + shrink * diag
        
        mu_t = mu_d.iloc[t].values  # length S
        sharpe_t = sharpe.iloc[t].values
        
        if np.isnan(mu_t).any() or np.isnan(Sigma).any():
            continue
        
        try:
            q = np.linalg.solve(Sigma, mu_t)
        except np.linalg.LinAlgError:
            q = np.linalg.pinv(Sigma) @ mu_t
        
        q[sharpe_t <= 0] = 0.0
        
        if np.allclose(q, 0.0):
            leftover_var.iloc[t] = var_target
            continue
        
        d = q
        d_col = d.reshape(-1, 1)
        dSig = Sigma @ d_col
        quad = float(d_col.T @ dSig)
        if quad <= 0:
            leftover_var.iloc[t] = var_target
            continue
        
        VaR_unit = float(z_alpha * np.sqrt(quad))
        cVaR_unit_vec = z_alpha * (d * dSig.flatten()) / np.sqrt(quad)
        
        k_total = var_target / VaR_unit
        with np.errstate(divide='ignore', invalid='ignore'):
            k_caps = np.where(cVaR_unit_vec > 0, (cap_abs / cVaR_unit_vec), np.inf)
        k_final = min(k_total, np.min(k_caps))
        
        w = k_final * d
        w_col = w.reshape(-1, 1)
        wSig = Sigma @ w_col
        quad_w = float(w_col.T @ wSig)
        VaR_tot = float(z_alpha * np.sqrt(quad_w))
        cVaR_vec = z_alpha * (w * wSig.flatten()) / np.sqrt(quad_w) if VaR_tot > 0 else np.zeros_like(w)
        pcts = cVaR_vec / VaR_tot if VaR_tot > 0 else np.zeros_like(w)
        
        weights.iloc[t] = w
        total_var.iloc[t] = VaR_tot
        comp_var.iloc[t] = cVaR_vec
        pct_var.iloc[t] = pcts
        leftover_var.iloc[t] = max(0.0, var_target - VaR_tot)
    
    return {
        "weights_units": weights,
        "VaR_total": total_var,
        "cVaR": comp_var,
        "pct_VaR": pct_var,
        "VaR_leftover": leftover_var
    }

# Recreate synthetic demo
np.random.seed(7)
T = 400
idx = pd.bdate_range("2024-01-01", periods=T)
S = 5
names = [f"s{i+1}" for i in range(S)]
vols = np.array([450, 300, 220, 180, 150])
rho = 0.25
C = rho * np.ones((S, S)) + (1 - rho) * np.eye(S)
L = np.linalg.cholesky(C)
eps = np.random.normal(size=(T, S)) @ L.T
drift = np.zeros((T, S))
for i in range(S):
    regime = np.sin(np.linspace(0, 6, T) + i)
    drift[:, i] = 0.10 * vols[i] / 252 * np.sign(regime)
pnl = pd.DataFrame(eps * vols + drift, index=idx, columns=[f"{n}_pnl" for n in names])
win = 64
mu_d_roll = pnl.rolling(win).mean()
sd_d_roll = pnl.rolling(win).std()
sharpe_roll = (np.sqrt(252) * (mu_d_roll / sd_d_roll)).replace([np.inf, -np.inf], np.nan)
sharpe_roll.columns = [f"{n}_sharpe" for n in names]
df_in = pd.concat([pnl, sharpe_roll], axis=1)

# Run allocator
out = kelly_var_allocator(df_in, lookback=64, z_alpha=2.33, var_target=250_000.0, cap_pct=0.50, shrink=0.15)

weights = out["weights_units"]
VaR_total = out["VaR_total"]
cVaR = out["cVaR"]
pct_VaR = out["pct_VaR"]
VaR_leftover = out["VaR_leftover"]

last = weights.index[-1]
snapshot = pd.DataFrame({
    "weight_units": weights.loc[last],
    "cVaR_$": cVaR.loc[last],
    "%VaR": pct_VaR.loc[last]
}).sort_values("cVaR_$", ascending=False)

from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("Kelly+VaR allocation — last-day snapshot", snapshot.round(2))

fig, ax = plt.subplots()
ax.plot(VaR_total.index, VaR_total.values, label="Achieved VaR")
ax.plot(VaR_leftover.index, VaR_leftover.values, label="Unallocated VaR")
ax.axhline(250_000.0, linestyle="--")
ax.set_title("Portfolio VaR (achieved) vs target and leftover")
ax.set_ylabel("1-day VaR (USD), 99%")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.6)
plt.show()

out_df = pd.concat([weights.add_suffix("_w"), cVaR.add_suffix("_cVaR"), pct_VaR.add_suffix("_pct"), VaR_total.rename("VaR_total"), VaR_leftover.rename("VaR_leftover")], axis=1)
path = "/mnt/data/kelly_var_allocator_outputs.csv"
out_df.to_csv(path, index=True)

path