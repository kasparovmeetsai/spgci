import numpy as np
import pandas as pd

def _shrink(M, shrink=0.1):
    """Shrink covariance matrix toward diagonal."""
    D = np.diag(np.diag(M))
    return (1.0 - shrink) * M + shrink * D

def _hp_returns(X: pd.DataFrame, ma: int = 15):
    """High-pass filter returns via moving average residuals."""
    L = X.rolling(ma, min_periods=ma//2).mean()
    H = X - L
    return H.dropna()

def _min_noise_weights(X_window: pd.DataFrame, hp_ma=15, shrink=0.1):
    """
    Solve   min_w  w' Σ_H w   s.t. 1'w = 1
    where Σ_H = cov of high-frequency residuals (returns - MA).
    """
    H = _hp_returns(X_window, ma=hp_ma)
    if len(H) < hp_ma or X_window.shape[1] == 0:
        return pd.Series(np.zeros(X_window.shape[1]), index=X_window.columns)
    Z = (H - H.mean()).values
    Σ_H = (Z.T @ Z) / (len(H) - 1)
    Σ_H = _shrink(Σ_H, shrink)

    # closed form: w = Σ^-1 1 / (1' Σ^-1 1)
    invΣ1 = np.linalg.pinv(Σ_H) @ np.ones((Σ_H.shape[0],1))
    w = invΣ1 / (np.ones((1,Σ_H.shape[0])) @ invΣ1)
    return pd.Series(w.flatten(), index=X_window.columns)

def wf_min_noise_basket(X: pd.DataFrame, lookback=80, hp_ma=15, shrink=0.1, rebalance="W-FRI"):
    """
    Walk-forward minimal-noise basket of flat-price returns.
    
    X : DataFrame of returns (or daily dollar diffs) for flat prices.
    lookback : estimation window length
    hp_ma : MA length used for high-pass filter
    shrink : covariance shrinkage
    rebalance : 'D' for daily or 'W-FRI' for weekly rebalancing
    
    Returns:
      weights : DataFrame of basket weights over time
      y       : Series of OOS basket returns (walk-forward)
    """
    X = X.dropna(how="all")
    idx = X.index
    rebal_dates = idx if rebalance=="D" else X.resample(rebalance).last().index.intersection(idx)

    weights = pd.DataFrame(np.nan, index=idx, columns=X.columns)
    y = pd.Series(0.0, index=idx)

    for t in rebal_dates:
        i = idx.get_loc(t)
        if i < lookback or i >= len(idx)-1:
            continue
        win = X.iloc[i-lookback:i].dropna()
        if len(win) < lookback//2:
            continue
        w_t = _min_noise_weights(win, hp_ma=hp_ma, shrink=shrink)
        weights.iloc[i] = w_t.values
        # OOS basket return
        y.iloc[i+1] = float(X.iloc[i+1].values @ w_t.values)

    return weights.ffill(), y