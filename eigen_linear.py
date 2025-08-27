import numpy as np
import pandas as pd
from numpy.linalg import eig
from typing import Optional

# ----------------------------
# Utilities
# ----------------------------
def _shrink(M, shrink=0.10):
    """Shrink covariance matrix toward diagonal."""
    D = np.diag(np.diag(M))
    return (1.0 - shrink) * M + shrink * D

def _hp_cov(X: pd.DataFrame, ma: int = 15, shrink: float = 0.10):
    """Estimate covariance of high-frequency component via MA filter."""
    L = X.rolling(ma, min_periods=ma//2).mean()
    H = X - L      # high-pass residual
    H = H.dropna()
    if len(H) < ma: 
        return np.eye(X.shape[1])  # fallback
    Z = (H - H.mean()).values
    S = (Z.T @ Z) / (len(H) - 1)
    return _shrink(S, shrink=shrink)

# ----------------------------
# Core: multi-lag + HF penalty
# ----------------------------
def max_autocorr_multilag_snr_weights(
    X_window: pd.DataFrame,
    lags: int = 5,
    lag_weights: Optional[np.ndarray] = None,
    shrink: float = 0.10,
    standardize: bool = True,
    hp_ma: int = 15,
    eta: float = 0.5,   # penalty strength for HF variance
):
    """
    Solve   max_w   (w' C_ml w) / (w' (C0 + eta*Σ_H) w)
    where:
      C_ml = sum_l ω_l * Cov(x_t, x_{t-l})   (multi-lag persistence)
      C0   = contemporaneous covariance
      Σ_H  = covariance of high-frequency residual (I - MA_k)x
    Returns weights normalized so that w' C0 w = 1.
    """
    W = X_window.copy()
    if standardize:
        W = (W - W.mean()) / W.std(ddof=1).replace(0.0, np.nan)
        W = W.dropna(how="any")
    else:
        W = W - W.mean()

    T, N = W.shape
    if T < lags + 2 or N == 0:
        return pd.Series(np.zeros(N), index=X_window.columns)

    if lag_weights is None:
        tau = max(2, lags/2.0)
        lag_weights = np.exp(-(np.arange(1, lags+1)) / tau)
        lag_weights = lag_weights / lag_weights.sum()

    Xv = W.values
    C0 = _shrink((Xv.T @ Xv) / (T - 1), shrink=shrink)

    # Multi-lag numerator
    C_ml = np.zeros_like(C0)
    for L, wL in zip(range(1, lags+1), lag_weights):
        Xf, Xb = Xv[L:], Xv[:-L]
        C = (Xf.T @ Xb) / (T - L)
        C = 0.5 * (C + C.T)   # symmetrize
        C_ml += wL * _shrink(C, shrink=shrink)

    # High-frequency covariance penalty
    Σ_H = _hp_cov(W, ma=hp_ma, shrink=shrink)

    # Generalized eigenproblem: C_ml w = λ (C0 + eta*Σ_H) w
    A = 0.5 * (C_ml + C_ml.T)
    B = 0.5 * (C0 + eta * Σ_H + (C0 + eta * Σ_H).T)

    Atil = np.linalg.pinv(B) @ A
    vals, vecs = eig(Atil)
    k = int(np.argmax(np.real(vals)))
    w = np.real(vecs[:, k])

    # Normalize so that w' C0 w = 1
    var_y = float(w.T @ C0 @ w)
    if var_y > 1e-12:
        w = w / np.sqrt(var_y)
    else:
        w = np.zeros_like(w)

    return pd.Series(w, index=W.columns)

# ----------------------------
# Walk-forward wrapper
# ----------------------------
def wf_max_autocorr_multilag_snr_series(
    X: pd.DataFrame,
    lookback: int = 80,
    lags: int = 5,
    lag_weights: Optional[np.ndarray] = None,
    shrink: float = 0.10,
    standardize: bool = True,
    hp_ma: int = 15,
    eta: float = 0.5,
    rebalance: str = "W-FRI",   # "D" for daily, "W-FRI" for weekly
):
    """
    Walk-forward multi-lag autocorr basket with high-frequency penalty.
    X must be *changes* (returns or dollar diffs).
    Returns:
      weights : DataFrame of weights over time
      y       : Series of out-of-sample basket changes
    """
    X = X.dropna(how="all")
    idx = X.index
    rebal_dates = idx if rebalance == "D" else X.resample(rebalance).last().index.intersection(idx)

    weights = pd.DataFrame(np.nan, index=idx, columns=X.columns)
    y = pd.Series(0.0, index=idx)

    for t in rebal_dates:
        i = idx.get_loc(t)
        if i < lookback or i >= len(idx) - 1:
            continue
        win = X.iloc[i - lookback : i].dropna(how="any")
        if len(win) < max(lookback // 2, lags + 2):
            continue

        w_t = max_autocorr_multilag_snr_weights(
            win, lags=lags, lag_weights=lag_weights, shrink=shrink,
            standardize=standardize, hp_ma=hp_ma, eta=eta
        )
        weights.iloc[i] = w_t.values
        y.iloc[i + 1] = float(X.iloc[i + 1].values @ w_t.values)

    return weights.ffill(), y