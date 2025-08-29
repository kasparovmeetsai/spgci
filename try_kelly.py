import numpy as np
import pandas as pd

def rolling_win_rate_and_rr(returns: pd.DataFrame, window: int = 64):
    """
    Compute rolling win rate p and risk-reward r = avg_win / |avg_loss| for each strategy.
    returns: DataFrame of daily PnL or returns per strategy (aligned columns).
    """
    # Win rate: rolling mean of 1[ret>0]
    p = returns.rolling(window).apply(lambda x: np.mean(x > 0), raw=True)

    # Average win and |avg loss|
    def avg_win(x):
        w = x[x > 0]
        return w.mean() if w.size else np.nan

    def avg_loss_abs(x):
        l = x[x < 0]
        return abs(l.mean()) if l.size else np.nan

    avg_w = returns.rolling(window).apply(avg_win, raw=False)
    avg_l = returns.rolling(window).apply(avg_loss_abs, raw=False)

    # Risk-reward r (guard against 0/NaN)
    r = avg_w / avg_l
    r = r.replace([np.inf, -np.inf], np.nan)

    return p, r

def half_kelly_from_p_r(p: pd.Series, r: pd.Series):
    """
    Vectorized half-Kelly per date for one strategy given aligned p, r Series.
    Kelly for Bernoulli with payoff ratio r: f* = p - (1-p)/r.
    """
    f_star = p - (1 - p) / r
    f_star = f_star.replace([np.inf, -np.inf], np.nan)
    f_half = 0.5 * f_star
    # clip negatives and NaNs to 0 (long-only / no bet)
    return f_half.clip(lower=0).fillna(0)

def allocate_half_kelly(
    sharpe_df: pd.DataFrame,
    window: int = 64,
    returns_df: pd.DataFrame | None = None,
):
    """
    Build long-only, no-leverage weights from half-Kelly logic.

    Inputs
    ------
    sharpe_df : DataFrame of rolling Sharpe per strategy (your `kelly_portfolio`)
    window    : lookback window used to compute p & r if returns_df is provided
    returns_df: OPTIONAL DataFrame of daily strategy returns/PnL (same columns as sharpe_df)

    Output
    ------
    weights : DataFrame of weights per date (sum across strategies ≤ 1.0)
    """
    sharpe_df = sharpe_df.copy()
    sharpe_df = sharpe_df.reindex(sorted(sharpe_df.columns), axis=1)

    if returns_df is not None:
        # Align and compute p, r per strategy
        returns_df = returns_df.reindex(columns=sharpe_df.columns).loc[sharpe_df.index]
        p_win, rr = rolling_win_rate_and_rr(returns_df, window=window)

        # Half-Kelly per strategy
        kelly_frac = pd.DataFrame(index=sharpe_df.index, columns=sharpe_df.columns, dtype=float)
        for col in sharpe_df.columns:
            kelly_frac[col] = half_kelly_from_p_r(p_win[col], rr[col])

        # Optional Sharpe gate: zero out if Sharpe ≤ 0 (don’t bet on losers)
        mask_pos_sharpe = (sharpe_df > 0).astype(float)
        raw_scores = (kelly_frac * mask_pos_sharpe).fillna(0.0)

    else:
        # Fallback: use positive Sharpe as risk-reward proxy (no win-rate info)
        raw_scores = sharpe_df.clip(lower=0).fillna(0.0)
        # Half-Kelly flavor: compress aggressiveness (divide by 2)
        raw_scores = 0.5 * raw_scores

    # Normalize to sum ≤ 1.0 each day (no leverage)
    row_sums = raw_scores.sum(axis=1)
    # Where row sum > 0, scale to exactly 1.0; otherwise keep zeros (sit in cash)
    scale = pd.Series(np.where(row_sums > 0, 1.0 / row_sums, 0.0), index=row_sums.index)
    weights = (raw_scores.T * scale).T
    weights = weights.fillna(0.0).clip(lower=0.0)

    # Numerical tidying
    weights[weights.abs() < 1e-12] = 0.0

    return weights
    
    
    
# Compute Sharpe slope (first difference)
sharpe_slope = sharpe_df.diff()

# Parameters for blending level + slope
alpha, beta = 1.0, 0.5   # tune these
score_combo = alpha * sharpe_df + beta * sharpe_slope

# Gating logic:
# If Sharpe < 0 but slope > 0 → allow partial allocation (e.g. 25%)
# If Sharpe > 0 and slope > 0 → full allocation
# If Sharpe > 0 and slope < 0 → reduce allocation
gate = pd.DataFrame(0.0, index=sharpe_df.index, columns=sharpe_df.columns)

gate[(sharpe_df < 0) & (sharpe_slope > 0)] = 0.25
gate[(sharpe_df > 0) & (sharpe_slope > 0)] = 1.0
gate[(sharpe_df > 0) & (sharpe_slope < 0)] = 0.5

# Raw Kelly scores, adjusted by gate and blended score
raw_scores = (kelly_frac * (1 + score_combo.clip(lower=0)) * gate).fillna(0.0)