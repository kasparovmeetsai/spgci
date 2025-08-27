import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy.linalg import eig

def wf_max_autocorr_series(returns_df, lookback=80, shrink=0.10, rebalance="W-FRI"):
    """
    Walk-forward: each rebalance date, compute w maximizing lag-1 autocorr of y_t = w'x_t
    by solving C1 w = λ C0 w on last 'lookback' days; then apply to next period.
    """
    X = returns_df.copy()
    X = X.dropna(how="all")
    rebal_dates = X.resample(rebalance).last().index.intersection(X.index)
    weights = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    y_ret = pd.Series(0.0, index=X.index)

    for t in rebal_dates:
        i = X.index.get_loc(t)
        if i < lookback or i >= len(X)-1: 
            continue
        W = X.iloc[i-lookback:i]
        Wc = W - W.mean()
        if len(Wc) < lookback: 
            continue

        # C0 = Cov(x_t), C1 = Cov(x_t, x_{t-1}) over window
        Xf, Xb = Wc.values[1:], Wc.values[:-1]
        C0 = (Wc.values.T @ Wc.values) / (len(Wc)-1)
        C1 = (Xf.T @ Xb) / (len(Wc)-1)

        # shrink & symmetrize for stability
        C0 = (1-shrink)*C0 + shrink*np.diag(np.diag(C0))
        C1s = 0.5*(C1 + C1.T)
        C1s = (1-shrink)*C1s + shrink*np.diag(np.diag(C1s))

        # generalized eigen: C1s w = λ C0 w  → eig(C0^{-1} C1s)
        A = np.linalg.pinv(C0) @ C1s
        vals, vecs = eig(A)
        w = np.real(vecs[:, np.argmax(np.real(vals))])
        # normalize to unit variance in-window: w' C0 w = 1
        var_y = float(w.T @ C0 @ w)
        if var_y <= 1e-12: 
            continue
        w = w / np.sqrt(var_y)

        weights.iloc[i] = w
        # apply to next day (strictly OOS)
        y_ret.iloc[i+1] = float(X.iloc[i+1].values @ w)

    # forward-fill weights between rebalances (for plotting)
    weights = weights.ffill()
    return weights, y_ret

# ---------- DEMO (synthetic noisy series with weak persistent driver) ----------
np.random.seed(10)
T, N = 600, 6
dates = pd.bdate_range("2020-01-01", periods=T)

# Hidden AR(1) driver + idiosyncratic noise
phi = 0.30
z = np.zeros(T); eps = np.random.normal(scale=0.8, size=T)
for t in range(1, T):
    z[t] = phi*z[t-1] + eps[t]
alphas = np.array([1.0, 0.8, 0.6, 0.5, -0.2, -0.6])
noise  = np.random.normal(scale=[1.2,1.0,0.9,0.9,0.8,1.1], size=(T,N))
returns = pd.DataFrame(z.reshape(-1,1)@alphas.reshape(1,-1) + noise, index=dates,
                       columns=[f"S{i+1}" for i in range(N)])

# Walk-forward lag-1 autocorr optimizer (weekly rebal)
W, y = wf_max_autocorr_series(returns, lookback=80, shrink=0.10, rebalance="W-FRI")

# Benchmarks and diagnostics
y_eq = returns.mean(axis=1)
cum_opt = (1 + y.fillna(0)/40).cumprod()
cum_eq  = (1 + y_eq/40).cumprod()

def rolling_lag1(series, win=120):
    a = series.fillna(0).values
    out = np.full(len(series), np.nan)
    for t in range(win, len(series)):
        b = a[t-win:t]; b = b - b.mean(); sd = b.std()
        out[t] = np.corrcoef(b[1:], b[:-1])[0,1] if sd>1e-8 else np.nan
    return pd.Series(out, index=series.index)

rho_opt = rolling_lag1(y,    win=120)
rho_eq  = rolling_lag1(y_eq, win=120)

# ---------- PLOTS (saved as PNGs) ----------
plt.figure(figsize=(9,3.6))
plt.plot(W.index, W.values)
plt.title("Walk-forward max-autocorr weights (weekly rebal, lookback=80)")
plt.ylabel("Weight"); plt.grid(True, ls="--", alpha=0.5)
plt.tight_layout(); plt.savefig("weights_wf.png"); plt.close()

plt.figure(figsize=(9,3.6))
plt.plot(rho_opt.index, rho_opt.values, label="Optimized")
plt.plot(rho_eq.index,  rho_eq.values,  label="Equal-weight")
plt.title("Rolling lag-1 autocorrelation of returns (win=120)")
plt.ylabel("lag-1 ρ"); plt.legend(); plt.grid(True, ls="--", alpha=0.5)
plt.tight_layout(); plt.savefig("rho_wf.png"); plt.close()

plt.figure(figsize=(9,3.6))
plt.plot(cum_opt.index, cum_opt.values, label="Optimized")
plt.plot(cum_eq.index,  cum_eq.values,  label="Equal-weight")
plt.title("Cumulative series (scaled, OOS, weekly rebal)")
plt.ylabel("Cumulative value"); plt.legend(); plt.grid(True, ls="--", alpha=0.5)
plt.tight_layout(); plt.savefig("cum_wf.png"); plt.close()

print("Saved: weights_wf.png, rho_wf.png, cum_wf.png")