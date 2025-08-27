import numpy as np
import pandas as pd
from numpy.linalg import eig

# ---------- core: multi-lag max-autocorr weights ----------
def max_autocorr_multilag_weights(
    X_window: pd.DataFrame,
    lags: int = 5,
    lag_weights: np.ndarray | None = None,
    shrink: float = 0.10,
    standardize: bool = True,
):
    """
    Solve  argmax_w   (w' C_agg w) / (w' C0 w),
    where C_agg = sum_{ℓ=1..L} ω_ℓ * Cov(x_t, x_{t-ℓ})  and  C0 = Cov(x_t).
    Returns w normalized so that w' C0 w = 1 (unit variance in-window).
    
    X_window : T×N DataFrame of *changes* (returns or dollar diffs) over the lookback
    lags     : number of positive lags included
    lag_weights : array of length `lags` (default: exponential decay)
    shrink   : diagonal shrinkage (0..1) for stability
    standardize : if True, z-score each column within the window (robust to scale)
    """
    W = X_window.copy()
    # (optional) per-window standardization for scale invariance
    if standardize:
        W = (W - W.mean()) / W.std(ddof=1).replace(0.0, np.nan)
        W = W.dropna(how="any")
    else:
        W = W - W.mean()

    T, N = W.shape
    if T < lags + 2 or N == 0:
        return pd.Series(np.zeros(N), index=X_window.columns)

    if lag_weights is None:
        # Exponential decay toward longer lags (tune tau if you wish)
        tau = max(2, lags/2)
        lag_weights = np.exp(-(np.arange(1, lags+1))/tau)
        lag_weights = lag_weights / lag_weights.sum()

    Xv = W.values
    # C0: contemporaneous covariance
    C0 = (Xv.T @ Xv) / (T - 1)
    # shrink toward diagonal
    C0 = (1 - shrink) * C0 + shrink * np.diag(np.diag(C0))

    # Aggregate cross-covariances over lags
    Cagg = np.zeros_like(C0)
    for L, wL in zip(range(1, lags+1), lag_weights):
        Xf, Xb = Xv[L:], Xv[:-L]           # align t with t-L
        Cℓ = (Xf.T @ Xb) / (T - L)
        # symmetrize & shrink for numerical stability
        Cℓ = 0.5 * (Cℓ + Cℓ.T)
        Cℓ = (1 - shrink) * Cℓ + shrink * np.diag(np.diag(Cℓ))
        Cagg += wL * Cℓ

    # Generalized eigenproblem: Cagg w = λ C0 w  → eig(C0^{-1} Cagg)
    A = np.linalg.pinv(C0) @ Cagg
    vals, vecs = eig(A)
    k = np.argmax(np.real(vals))
    w = np.real(vecs[:, k])

    # Normalize to unit variance under C0: w' C0 w = 1
    var_y = float(w.T @ C0 @ w)
    if var_y > 1e-12:
        w = w / np.sqrt(var_y)
    else:
        w = np.zeros_like(w)

    return pd.Series(w, index=X_window.columns)

# ---------- walk-forward wrapper (weekly rebalancing by default) ----------
def wf_max_autocorr_multilag_series(
    X: pd.DataFrame,
    lookback: int = 80,
    lags: int = 5,
    lag_weights: np.ndarray | None = None,
    shrink: float = 0.10,
    standardize: bool = True,
    rebalance: str = "W-FRI",  # "D" for daily; "W-FRI" for weekly
):
    """
    Walk-forward series using multi-lag persistence weights.
    X must be *changes* (returns or dollar diffs). Output y is the 1-step-ahead OOS change.
    """
    X = X.dropna(how="all")
    idx = X.index

    # rebalancing dates
    if rebalance == "D":
        rebal_dates = idx
    else:
        rebal_dates = X.resample(rebalance).last().index.intersection(idx)

    weights = pd.DataFrame(np.nan, index=idx, columns=X.columns)
    y = pd.Series(0.0, index=idx)

    for t in rebal_dates:
        i = idx.get_loc(t)
        if i < lookback or i >= len(idx) - 1:
            continue
        win = X.iloc[i - lookback : i].dropna(how="any")
        if len(win) < max(lookback // 2, lags + 2):
            continue

        w_t = max_autocorr_multilag_weights(
            win, lags=lags, lag_weights=lag_weights,
            shrink=shrink, standardize=standardize
        )
        weights.iloc[i] = w_t.values

        # 1-step-ahead out-of-sample *change* (return or dollar diff)
        y.iloc[i + 1] = float(X.iloc[i + 1].values @ w_t.values)

    # carry last weights forward for readability (optional)
    weights = weights.ffill()
    return weights, y