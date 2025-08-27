import numpy as np
import pandas as pd

def kelly_var_allocator_robust(
    df: pd.DataFrame,
    lookback: int = 64,
    z_alpha: float = 2.33,          # 1-day VaR Z-score (99% ≈ 2.33, 97.5% ≈ 1.96)
    var_target: float = 250_000.0,  # USD VaR budget
    cap_pct: float = 0.50,          # per-strategy VaR cap as fraction of total (e.g., 0.50 → 50%)
    shrink: float = 0.15,           # covariance shrink toward diagonal
    min_periods: int = 32,          # allow partial windows
    rebalance: str = "D"            # "D" daily or "W-FRI" weekly (Fridays only)
):
    """
    Kelly + VaR-constrained allocator.
    Expects df with columns for each strategy:
      '<name>_pnl'    : unit PnL series (daily $ PnL per 1 unit exposure)
      '<name>_sharpe' : rolling annualized Sharpe (e.g., 64-day)
    Returns dict with:
      - 'weights_units' : unit weights per strategy
      - 'VaR_total'     : total 1-day VaR ($) at confidence given by z_alpha
      - 'cVaR'          : component VaR by strategy ($, Euler allocation)
      - 'pct_VaR'       : % of total VaR by strategy
      - 'VaR_leftover'  : unallocated VaR (target - achieved, floored at 0)
      - 'debug_reason'  : text reason per date if allocation was skipped
    Notes:
      * No shorting when Sharpe <= 0  → those sleeves get 0.
      * If per-strategy VaR cap binds, remaining VaR stays unallocated.
      * Uses simple shrinkage for covariance stability.
      * Interprets VaR under Normal assumption: VaR = z * sqrt(w' Σ w).
    """
    # ---- detect strategies and validate columns
    pnl_cols = [c for c in df.columns if c.endswith("_pnl")]
    if not pnl_cols:
        raise ValueError("No '*_pnl' columns found in df.")
    strategies = [c[:-4] for c in pnl_cols]  # strip "_pnl"
    missing = [s for s in strategies if f"{s}_sharpe" not in df.columns]
    if missing:
        raise ValueError(f"Missing Sharpe columns: {missing} (expected '<name>_sharpe')")

    pnl = df[[f"{s}_pnl" for s in strategies]].copy()
    sharpe = df[[f"{s}_sharpe" for s in strategies]].copy()

    # Clean inputs
    pnl = pnl.replace([np.inf, -np.inf], np.nan)
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan).fillna(0.0).ffill()

    # Rolling daily vol of unit PnL ($/day per unit)
    sigma_d = pnl.rolling(lookback, min_periods=min_periods).std()

    # Daily expected PnL from annual Sharpe & daily vol:
    #   S_ann = sqrt(252) * (mu_d / sigma_d)  =>  mu_d = (S_ann / sqrt(252)) * sigma_d
    mu_d = (sharpe / np.sqrt(252.0)) * sigma_d

    idx = df.index
    cap_abs = cap_pct * var_target

    weights = pd.DataFrame(0.0, index=idx, columns=strategies)
    VaR_total = pd.Series(np.nan, index=idx, name="VaR_total")
    cVaR = pd.DataFrame(0.0, index=idx, columns=strategies)
    pct_VaR = pd.DataFrame(0.0, index=idx, columns=strategies)
    VaR_leftover = pd.Series(np.nan, index=idx, name="VaR_leftover")
    reason = pd.Series("", index=idx, name="skip_reason")

    def _should_rebalance(ts):
        if rebalance == "D":
            return True
        return ts.weekday() == 4  # Friday

    for t, ts in enumerate(idx):
        # Warmup
        if t < lookback:
            reason.iloc[t] = "warmup"
            continue
        if not _should_rebalance(ts):
            # carry forward last known values
            weights.iloc[t] = weights.iloc[t-1]
            VaR_total.iloc[t] = VaR_total.iloc[t-1]
            cVaR.iloc[t] = cVaR.iloc[t-1]
            pct_VaR.iloc[t] = pct_VaR.iloc[t-1]
            VaR_leftover.iloc[t] = VaR_leftover.iloc[t-1]
            reason.iloc[t] = "no_rebalance"
            continue

        # Window for cov
        win = pnl.iloc[t - lookback:t].dropna(how="any")
        if len(win) < min_periods:
            reason.iloc[t] = "not_enough_clean_rows"
            continue

        Sigma = win.cov().values
        if not np.isfinite(Sigma).all():
            reason.iloc[t] = "bad_cov"
            continue

        # Shrink toward diagonal
        D = np.diag(np.diag(Sigma))
        Sigma = (1 - shrink) * Sigma + shrink * D

        mu_t = mu_d.iloc[t].values
        sh_t = sharpe.iloc[t].values
        if not np.isfinite(mu_t).all():
            reason.iloc[t] = "bad_mu"
            continue

        # Kelly direction q = Σ^{-1} μ
        try:
            q = np.linalg.solve(Sigma, mu_t)
        except np.linalg.LinAlgError:
            q = np.linalg.pinv(Sigma) @ mu_t

        # No shorting for Sharpe <= 0
        q[sh_t <= 0] = 0.0
        if np.allclose(q, 0.0):
            VaR_total.iloc[t] = 0.0
            VaR_leftover.iloc[t] = var_target
            reason.iloc[t] = "all_sharpe<=0"
            continue

        d = q  # direction (units)
        d_col = d.reshape(-1, 1)
        dSig = Sigma @ d_col
        quad = float(d_col.T @ dSig)
        if not np.isfinite(quad) or quad <= 0:
            reason.iloc[t] = "nonpos_quad"
            continue

        # Unit-scale VaR and component VaR
        VaR_unit = float(z_alpha * np.sqrt(quad))
        # Euler component VaR at unit scale: cVaR_i = z * d_i * (Σ d)_i / sqrt(d' Σ d)
        cVaR_unit_vec = z_alpha * (d * dSig.flatten()) / np.sqrt(quad)

        # Scale to target VaR, enforce per-strategy caps
        k_total = var_target / VaR_unit
        with np.errstate(divide='ignore', invalid='ignore'):
            k_caps = np.where(cVaR_unit_vec > 0, cap_abs / cVaR_unit_vec, np.inf)
        k_final = float(np.nanmin([k_total, np.nanmin(k_caps)]))
        if not np.isfinite(k_final) or k_final <= 0:
            reason.iloc[t] = "no_scaling_possible"
            continue

        w = k_final * d
        w_col = w.reshape(-1, 1)
        wSig = Sigma @ w_col
        quad_w = float(w_col.T @ wSig)
        if not np.isfinite(quad_w) or quad_w <= 0:
            reason.iloc[t] = "nonpos_quad_scaled"
            continue

        VaR_tot = float(z_alpha * np.sqrt(quad_w))
        cVaR_vec = z_alpha * (w * wSig.flatten()) / np.sqrt(quad_w)
        pcts = cVaR_vec / VaR_tot if VaR_tot > 0 else np.zeros_like(w)

        # store
        weights.iloc[t] = w
        VaR_total.iloc[t] = VaR_tot
        cVaR.iloc[t] = cVaR_vec
        pct_VaR.iloc[t] = pcts
        VaR_leftover.iloc[t] = max(0.0, var_target - VaR_tot)
        reason.iloc[t] = ""

    return {
        "weights_units": weights,
        "VaR_total": VaR_total,
        "cVaR": cVaR,
        "pct_VaR": pct_VaR,
        "VaR_leftover": VaR_leftover,
        "debug_reason": reason
    }


# ---------- Example usage ----------
if __name__ == "__main__":
    # Build synthetic 5-strategy dataset
    np.random.seed(7)
    T = 400
    idx = pd.bdate_range("2024-01-01", periods=T)
    S = 5
    names = [f"s{i+1}" for i in range(S)]
    vols = np.array([450, 300, 220, 180, 150])  # daily $ stdev per unit
    rho = 0.25
    C = rho * np.ones((S, S)) + (1 - rho) * np.eye(S)
    L = np.linalg.cholesky(C)
    eps = np.random.normal(size=(T, S)) @ L.T

    drift = np.zeros((T, S))
    for i in range(S):
        regime = np.sign(np.sin(np.linspace(0, 6, T) + 0.8 * i))
        drift[:, i] = 0.10 * vols[i] / 252 * regime

    pnl = pd.DataFrame(eps * vols + drift, index=idx, columns=[f"{n}_pnl" for n in names])

    # 64d annualized Sharpe
    win = 64
    mu_d = pnl.rolling(win, min_periods=32).mean()
    sd_d = pnl.rolling(win, min_periods=32).std()
    sharpe = (np.sqrt(252) * (mu_d / sd_d)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sharpe.columns = [f"{n}_sharpe" for n in names]

    df_in = pd.concat([pnl, sharpe], axis=1)

    # Run allocator (weekly rebalancing shows changes clearly)
    out = kelly_var_allocator_robust(df_in, lookback=64, z_alpha=2.33,
                                     var_target=250_000.0, cap_pct=0.50,
                                     shrink=0.15, min_periods=32, rebalance="W-FRI")

    # Print last-day snapshot
    last = df_in.index[-1]
    snap = pd.DataFrame({
        "weight_units": out["weights_units"].loc[last],
        "cVaR_$": out["cVaR"].loc[last],
        "%VaR": out["pct_VaR"].loc[last]
    }).sort_values("cVaR_$", ascending=False)
    print("Last-day snapshot:\n", snap.round(2))

    # Save outputs
    out_df = pd.concat([
        out["weights_units"].add_suffix("_w"),
        out["cVaR"].add_suffix("_cVaR"),
        out["pct_VaR"].add_suffix("_pct"),
        out["VaR_total"].rename("VaR_total"),
        out["VaR_leftover"].rename("VaR_leftover"),
        out["debug_reason"]
    ], axis=1)
    out_df.to_csv("kelly_var_allocator_outputs.csv", index=True)
    print("Saved: kelly_var_allocator_outputs.csv")
