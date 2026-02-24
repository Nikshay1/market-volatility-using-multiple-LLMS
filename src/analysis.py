"""
Statistical analysis and volatility modeling for Agentic Dissonance v2.

Implements:
- 5-day forward realized volatility calculation
- Baseline GARCH(1,1) model
- Variance proxy model with lagged disagreement effects
- AIC/BIC/RMSE/MAE comparison
- Visualization of results
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from typing import Dict, Tuple, Optional, List
from arch import arch_model

from . import config

# Suppress arch warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


def load_disagreement_signals(path: str = None) -> pd.DataFrame:
    """
    Load disagreement signals from CSV.
    
    Returns:
        DataFrame with disagreement signals
    """
    path = path or config.DISAGREEMENT_SIGNALS_PATH
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Disagreement signals not found at {path}. "
            "Run backtest first."
        )
    
    df = pd.read_csv(path, parse_dates=['date'])
    return df


def load_market_data(path: str = None) -> pd.DataFrame:
    """
    Load market data from CSV.
    
    Returns:
        DataFrame with market data
    """
    path = path or config.RAW_MARKET_DATA_PATH
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Market data not found at {path}. "
            "Run backtest first."
        )
    
    df = pd.read_csv(path, parse_dates=['Date'])
    return df


def calculate_forward_volatility(
    market_df: pd.DataFrame,
    window: int = None,
    ticker: str = None
) -> pd.DataFrame:
    """
    Calculate forward realized volatility.
    
    RV_{t+1:t+window} = sqrt(sum(r²_{t+1} to r²_{t+window}))
    
    Args:
        market_df: DataFrame with market data including Log_Return
        window: Forward window size (default from config)
        ticker: Optional ticker filter
        
    Returns:
        DataFrame with forward volatility column added
    """
    window = window or config.FORWARD_VOLATILITY_WINDOW
    
    df = market_df.copy()
    
    if ticker and 'Ticker' in df.columns:
        df = df[df['Ticker'] == ticker]
    
    # Sort by date
    df = df.sort_values('Date')
    
    if 'Ticker' in df.columns:
        # Calculate per ticker
        def calc_fwd_vol(group):
            returns = group['Log_Return'].values
            fwd_vol = []
            for i in range(len(returns)):
                if i + window < len(returns):
                    future_returns = returns[i+1:i+1+window]
                    rv = np.sqrt(np.sum(future_returns ** 2))
                    fwd_vol.append(rv)
                else:
                    fwd_vol.append(np.nan)
            group['Forward_Volatility'] = fwd_vol
            return group
        
        df = df.groupby('Ticker', group_keys=False).apply(calc_fwd_vol, include_groups=False)
    else:
        # Single ticker
        returns = df['Log_Return'].values
        fwd_vol = []
        for i in range(len(returns)):
            if i + window < len(returns):
                future_returns = returns[i+1:i+1+window]
                rv = np.sqrt(np.sum(future_returns ** 2))
                fwd_vol.append(rv)
            else:
                fwd_vol.append(np.nan)
        df['Forward_Volatility'] = fwd_vol
    
    return df


def merge_data(
    disagreement_df: pd.DataFrame,
    market_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge disagreement signals with market data.
    Aligns D_t with σ_{t+1:t+5}.
    
    Args:
        disagreement_df: DataFrame with disagreement signals
        market_df: DataFrame with market data and forward volatility
        
    Returns:
        Merged DataFrame
    """
    # Standardize date column names
    dis_df = disagreement_df.copy()
    mkt_df = market_df.copy()
    
    dis_df['date'] = pd.to_datetime(dis_df['date'])
    mkt_df['date'] = pd.to_datetime(mkt_df['Date'])
    
    # Determine merge columns
    if 'ticker' in dis_df.columns and 'Ticker' in mkt_df.columns:
        merge_cols = ['date', 'ticker']
        mkt_df = mkt_df.rename(columns={'Ticker': 'ticker'})
    else:
        merge_cols = ['date']
    
    # Select relevant columns from market data
    mkt_cols = ['date', 'Log_Return', 'Forward_Volatility', 'Daily_Volatility', 'Close']
    if 'ticker' in merge_cols:
        mkt_cols.append('ticker')
    
    mkt_subset = mkt_df[mkt_cols].copy()
    
    # Merge
    merged = pd.merge(dis_df, mkt_subset, on=merge_cols, how='inner')
    
    # CHANGE: Calculate Delta Disagreement (Rate of Change)
    # The raw level of disagreement matters less than the change in disagreement.
    # A sudden spike in agent conflict is a stronger predictor of future volatility.
    merged = merged.sort_values('date')
    merged['D_conf_change'] = merged['disagreement_conf'].diff().fillna(0)
    
    # Drop rows with missing forward volatility
    merged = merged.dropna(subset=['Forward_Volatility'])
    
    return merged


def run_correlation_analysis(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate correlation between disagreement metrics and forward volatility.
    
    Args:
        df: Merged DataFrame with disagreement and volatility
        
    Returns:
        Dictionary with correlation results
    """
    results = {}
    
    target = 'Forward_Volatility'
    metrics = ['disagreement_conf', 'mean_score', 'avg_confidence', 
               'score_fundamental', 'score_sentiment', 'score_technical', 'score_macro']
    
    for metric in metrics:
        if metric in df.columns:
            valid = df[[metric, target]].dropna()
            if len(valid) > 2:
                corr, p_value = stats.pearsonr(valid[metric], valid[target])
                results[f'corr_{metric}'] = corr
                results[f'pval_{metric}'] = p_value
    
    # Spearman correlation for disagreement
    if 'disagreement_conf' in df.columns:
        valid = df[['disagreement_conf', target]].dropna()
        if len(valid) > 2:
            spearman_corr, spearman_p = stats.spearmanr(valid['disagreement_conf'], valid[target])
            results['spearman_disagreement'] = spearman_corr
            results['spearman_pval'] = spearman_p
    
    return results


def fit_garch_baseline(
    returns: np.ndarray,
    train_size: float = None
) -> Dict[str, any]:
    """
    Fit baseline GARCH(1,1) model.
    
    Args:
        returns: Array of log returns
        train_size: Fraction for training (default from config)
        
    Returns:
        Dictionary with model, results, and metrics
    """
    train_size = train_size or config.TRAIN_TEST_SPLIT
    
    # Scale returns to percentage
    returns_pct = returns * 100
    
    # Check minimum data requirement
    if len(returns_pct) < 50:
        print(f"GARCH baseline: Insufficient data ({len(returns_pct)} points, need 50+)")
        return None
    
    # Train/test split
    n = len(returns_pct)
    train_n = int(n * train_size)
    
    if train_n < 30:
        print(f"GARCH baseline: Training set too small ({train_n} points)")
        return None
    
    train_returns = returns_pct[:train_n]
    test_returns = returns_pct[train_n:]
    
    try:
        # Fit once on training segment for parameter reporting
        model = arch_model(train_returns, vol='Garch', p=1, q=1, rescale=False)
        results = model.fit(disp='off', show_warning=False)

        # One-step-ahead rolling forecasts over test segment
        rolling_vol_forecasts = []
        for i in range(len(test_returns)):
            history = returns_pct[:train_n + i]
            roll_model = arch_model(history, vol='Garch', p=1, q=1, rescale=False)
            roll_results = roll_model.fit(disp='off', show_warning=False)
            variance_fcst = roll_results.forecast(horizon=1, reindex=False).variance.values[-1, 0]
            rolling_vol_forecasts.append(np.sqrt(max(variance_fcst, 1e-10)))

        realized_vol_test = np.abs(test_returns) / 100
        predicted_vol_test = np.array(rolling_vol_forecasts) / 100
        realized_var_test = (test_returns / 100) ** 2
        predicted_var_test = np.maximum(predicted_vol_test ** 2, 1e-12)

        rmse = np.sqrt(np.mean((realized_vol_test - predicted_vol_test) ** 2))
        mae = np.mean(np.abs(realized_vol_test - predicted_vol_test))
        qlike = np.mean(np.log(predicted_var_test) + (realized_var_test / predicted_var_test))

        return {
            'model': model,
            'results': results,
            'aic': results.aic,
            'bic': results.bic,
            'rmse': rmse,
            'mae': mae,
            'qlike': qlike,
            'train_vol': results.conditional_volatility,
            'test_realized_vol': realized_vol_test,
            'test_realized_var': realized_var_test,
            'test_pred_vol': predicted_vol_test,
            'test_pred_var': predicted_var_test,
            'params': dict(results.params),
            'train_n': train_n
        }
        
    except Exception as e:
        print(f"GARCH baseline fitting failed: {e}")
        return None


def fit_garch_x(
    returns: np.ndarray,
    exog: np.ndarray,
    train_size: float = None
) -> Dict[str, any]:
    """
    Fit a variance-impact alternative using lagged disagreement in a volatility proxy regression.

    This is NOT a true GARCH-X specification in `arch` (which places exogenous terms in
    the mean equation). Instead, it directly tests variance impact with:

    RV_t = c + ϕ1*RV_{t-1} + ϕ5*mean(RV_{t-5:t-1})
             + γ1*D_{t-1} + γ5*mean(D_{t-5:t-1}) + u_t

    where RV_t = r_t^2 (returns scaled to percent). Lag structure is explicit:
    - D_{t-1}: one-day lag of disagreement
    - mean(D_{t-5:t-1}): rolling 5-day lag aggregate (excluding day t)
    
    Args:
        returns: Array of log returns
        exog: Array of exogenous variable (disagreement)
        train_size: Fraction for training
        
    Returns:
        Dictionary with model, results, and metrics
    """
    train_size = train_size or config.TRAIN_TEST_SPLIT
    
    # Scale returns to percentage and build realized variance proxy
    returns_pct = returns * 100
    rv = returns_pct ** 2
    
    # Check minimum data requirement
    if len(returns_pct) < 50:
        print(f"Variance proxy model: Insufficient data ({len(returns_pct)} points, need 50+)")
        return None
    
    # Train/test split
    n = len(returns_pct)
    train_n = int(n * train_size)
    
    if train_n < 30:
        print(f"Variance proxy model: Training set too small ({train_n} points)")
        return None

    train_rv = rv[:train_n]
    train_dis = exog[:train_n]

    model_df = pd.DataFrame({
        'rv': train_rv,
        'rv_l1': pd.Series(train_rv).shift(1),
        'rv_l5_mean': pd.Series(train_rv).shift(1).rolling(window=5).mean(),
        'd_l1': pd.Series(train_dis).shift(1),
        'd_l5_mean': pd.Series(train_dis).shift(1).rolling(window=5).mean(),
    }).dropna()

    if len(model_df) < 30:
        print(f"Variance proxy model: Post-lag sample too small ({len(model_df)} points)")
        return None
    
    try:
        X = sm.add_constant(model_df[['rv_l1', 'rv_l5_mean', 'd_l1', 'd_l5_mean']])
        results = sm.OLS(model_df['rv'], X).fit()

        # One-step-ahead rolling forecasts over test segment using expanding window
        rolling_var_forecasts = []
        for i in range(train_n, n):
            hist_rv = rv[:i]
            hist_dis = exog[:i]
            hist_df = pd.DataFrame({
                'rv': hist_rv,
                'rv_l1': pd.Series(hist_rv).shift(1),
                'rv_l5_mean': pd.Series(hist_rv).shift(1).rolling(window=5).mean(),
                'd_l1': pd.Series(hist_dis).shift(1),
                'd_l5_mean': pd.Series(hist_dis).shift(1).rolling(window=5).mean(),
            }).dropna()

            if len(hist_df) < 30:
                rolling_var_forecasts.append(np.nan)
                continue

            roll_X = sm.add_constant(hist_df[['rv_l1', 'rv_l5_mean', 'd_l1', 'd_l5_mean']])
            roll_results = sm.OLS(hist_df['rv'], roll_X).fit()

            feature_row = pd.DataFrame([{
                'rv_l1': rv[i - 1],
                'rv_l5_mean': np.mean(rv[max(0, i - 5):i]),
                'd_l1': exog[i - 1],
                'd_l5_mean': np.mean(exog[max(0, i - 5):i])
            }])
            feature_row = sm.add_constant(feature_row, has_constant='add')
            rolling_var_forecasts.append(roll_results.predict(feature_row).iloc[0])

        predicted_var_test = np.maximum(np.array(rolling_var_forecasts), 1e-12)
        realized_var_test = rv[train_n:] / (100 ** 2)
        predicted_var_test = predicted_var_test / (100 ** 2)

        realized_vol_test = np.sqrt(realized_var_test)
        predicted_vol_test = np.sqrt(predicted_var_test)

        rmse = np.sqrt(np.mean((realized_vol_test - predicted_vol_test) ** 2))
        mae = np.mean(np.abs(realized_vol_test - predicted_vol_test))
        qlike = np.mean(np.log(predicted_var_test) + (realized_var_test / predicted_var_test))
        
        d_l1_coef = results.params.get('d_l1')
        d_l1_pval = results.pvalues.get('d_l1')
        d_l5_coef = results.params.get('d_l5_mean')
        d_l5_pval = results.pvalues.get('d_l5_mean')
        
        return {
            'model': 'variance_proxy_ols',
            'results': results,
            'aic': results.aic,
            'bic': results.bic,
            'rmse': rmse,
            'mae': mae,
            'qlike': qlike,
            'train_vol': np.sqrt(np.maximum(results.fittedvalues.values, 1e-10)),
            'test_realized_vol': realized_vol_test,
            'test_realized_var': realized_var_test,
            'test_pred_vol': predicted_vol_test,
            'test_pred_var': predicted_var_test,
            'params': dict(results.params),
            'd_l1_coef': d_l1_coef,
            'd_l1_pval': d_l1_pval,
            'd_l5_coef': d_l5_coef,
            'd_l5_pval': d_l5_pval,
            'train_n': train_n
        }
        
    except Exception as e:
        print(f"Variance proxy model fitting failed: {e}")
        return None


def diebold_mariano_test(loss_baseline: np.ndarray, loss_model: np.ndarray) -> Dict[str, float]:
    """Diebold-Mariano test for equal predictive accuracy (1-step forecasts)."""
    valid = np.isfinite(loss_baseline) & np.isfinite(loss_model)
    d = np.asarray(loss_baseline[valid] - loss_model[valid])

    if len(d) < 5:
        return {'dm_stat': np.nan, 'p_value': np.nan, 'mean_loss_diff': np.nan, 'n_obs': len(d)}

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    dm_stat = mean_d / np.sqrt(var_d / len(d)) if var_d > 0 else np.nan
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=len(d) - 1)) if np.isfinite(dm_stat) else np.nan

    return {
        'dm_stat': dm_stat,
        'p_value': p_value,
        'mean_loss_diff': mean_d,
        'n_obs': len(d)
    }


def create_visualization(
    df: pd.DataFrame,
    correlation_results: Dict,
    garch_baseline: Dict,
    variance_model: Dict,
    save_path: str = None
) -> None:
    """
    Create 3-panel dashboard visualization.
    
    Layout:
    - Top: Time Series - Blue (Disagreement) vs Orange (Next Day Volatility)
    - Bottom Left: Scatter - Disagreement vs Volatility (Goal: Positive Slope)
    - Bottom Right: Boxplots - Agent Score Distributions (Goal: Not flatlining at 0)
    
    Args:
        df: Merged DataFrame
        correlation_results: Correlation analysis results
        garch_baseline: Baseline GARCH results
        variance_model: Variance model results
        save_path: Path to save the figure
    """
    save_path = save_path or config.RESULTS_PLOT_PATH
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Agentic Dissonance - 3-Panel Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid: 2 rows - top full width, bottom split in 2
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.25)
    
    # ================================================================
    # PANEL 1 (TOP): Time Series - Blue Leads Orange?
    # ================================================================
    ax1 = fig.add_subplot(gs[0, :])  # Full width
    if 'date' in df.columns:
        df_sorted = df.sort_values('date').copy()
        
        # Normalize both series to 0-1 for overlay
        d_conf = df_sorted['disagreement_conf']
        fwd_vol = df_sorted['Forward_Volatility']
        
        d_conf_norm = (d_conf - d_conf.min()) / (d_conf.max() - d_conf.min() + 1e-8)
        fwd_vol_norm = (fwd_vol - fwd_vol.min()) / (fwd_vol.max() - fwd_vol.min() + 1e-8)
        
        # Plot both on same axis (normalized)
        ax1.plot(df_sorted['date'], d_conf_norm, 
                color='steelblue', label='Agent Disagreement (BLUE)', 
                linewidth=2, alpha=0.9)
        ax1.plot(df_sorted['date'], fwd_vol_norm, 
                color='darkorange', label='Next Day Volatility (ORANGE)', 
                linewidth=2, alpha=0.9)
        
        # Fill between to show spikes
        ax1.fill_between(df_sorted['date'], 0, d_conf_norm, alpha=0.2, color='steelblue')
        ax1.fill_between(df_sorted['date'], 0, fwd_vol_norm, alpha=0.2, color='darkorange')
        
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Normalized Magnitude (0-1)', fontsize=11)
        ax1.set_title('TIME SERIES: Agent Disagreement vs Next Day Volatility\n(Goal: Blue spikes BEFORE Orange)', 
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)
    
    # ================================================================
    # PANEL 2 (BOTTOM LEFT): Scatter - Positive Slope?
    # ================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    if 'disagreement_conf' in df.columns and 'Forward_Volatility' in df.columns:
        valid = df[['disagreement_conf', 'Forward_Volatility']].dropna()
        
        # Scatter plot
        ax2.scatter(valid['disagreement_conf'], valid['Forward_Volatility'], 
                   alpha=0.6, c='steelblue', s=50, edgecolors='white', linewidth=0.5)
        
        # Add RED trend line
        if len(valid) > 2:
            z = np.polyfit(valid['disagreement_conf'], valid['Forward_Volatility'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid['disagreement_conf'].min(), valid['disagreement_conf'].max(), 100)
            ax2.plot(x_line, p(x_line), 'r-', linewidth=3, alpha=0.9, label=f'Trend (slope={z[0]:.3f})')
            
            # Add slope annotation with verdict
            slope_positive = z[0] > 0
            verdict_text = "[+] POSITIVE SLOPE" if slope_positive else "[-] NEGATIVE SLOPE"
            verdict_color = 'green' if slope_positive else 'red'
            ax2.annotate(verdict_text, xy=(0.02, 0.95), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', color=verdict_color)
        
        corr = correlation_results.get('corr_disagreement_conf', 0)
        
        ax2.set_xlabel('Agent Disagreement', fontsize=11)
        ax2.set_ylabel('Next Day Volatility', fontsize=11)
        ax2.set_title(f'SCATTER: Disagreement vs Volatility (r={corr:.3f})\n(Goal: Positive Slope)', 
                     fontsize=12, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
    
    # ================================================================
    # PANEL 3 (BOTTOM RIGHT): Agent Score Boxplots - Not Flatlining?
    # ================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    score_cols = ['score_sentiment', 'score_technical', 'score_macro']
    available_cols = [c for c in score_cols if c in df.columns]
    if available_cols:
        data_to_plot = [df[col].dropna() for col in available_cols]
        labels = [col.replace('score_', '').capitalize() for col in available_cols]
        
        bp = ax3.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, widths=0.6)
        colors = ['coral', 'forestgreen', 'goldenrod']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=2)
        ax3.axhline(y=-1, color='red', linestyle=':', alpha=0.3)
        ax3.axhline(y=1, color='green', linestyle=':', alpha=0.3)
        
        # Check if any agent is flatlining (std < 0.1)
        flatline_check = []
        for i, col in enumerate(available_cols):
            std = df[col].std()
            if std < 0.1:
                flatline_check.append(labels[i])
        
        if flatline_check:
            verdict_text = f"[!] FLATLINE: {', '.join(flatline_check)}"
            verdict_color = 'red'
        else:
            verdict_text = "[OK] ALL AGENTS ACTIVE"
            verdict_color = 'green'
        
        ax3.annotate(verdict_text, xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=11, fontweight='bold', color=verdict_color)
        
        ax3.set_ylabel('Score (-1 to +1)')
        ax3.set_title('BOXPLOTS: Agent Score Distributions\n(Goal: Not flatlining at 0.0)', 
                     fontsize=12, fontweight='bold')
        ax3.set_ylim(-1.3, 1.3)
        ax3.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to {save_path}")
    
    plt.close()


def create_disagreement_figure(
    df: pd.DataFrame,
    correlation_results: Dict,
    save_path: str = None
) -> None:
    """
    Create Figure 1: Agent Disagreement vs. Future Volatility (standalone).
    
    Args:
        df: Merged DataFrame with disagreement and volatility data
        correlation_results: Correlation analysis results
        save_path: Path to save the figure (default: output/fig1_disagreement.png)
    """
    save_path = save_path or os.path.join(config.OUTPUT_DIR, "fig1_disagreement.png")
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if 'disagreement_conf' in df.columns and 'Forward_Volatility' in df.columns:
        valid = df[['disagreement_conf', 'Forward_Volatility']].dropna()
        
        # Scatter plot with alpha=0.6 for density
        ax.scatter(valid['disagreement_conf'], valid['Forward_Volatility'], 
                  alpha=0.6, c='steelblue', s=50, edgecolors='white', linewidth=0.5)
        
        # Add Linear Regression Trendline (Red)
        slope = 0
        if len(valid) > 2:
            z = np.polyfit(valid['disagreement_conf'], valid['Forward_Volatility'], 1)
            slope = z[0]
            p = np.poly1d(z)
            x_line = np.linspace(valid['disagreement_conf'].min(), valid['disagreement_conf'].max(), 100)
            ax.plot(x_line, p(x_line), 'r-', linewidth=2.5, label='Linear Regression')
        
        # Get correlation
        corr = correlation_results.get('corr_disagreement_conf', 0)
        
        # Axis labels
        ax.set_xlabel('D_conf (Agent Disagreement/Std Dev)', fontsize=12)
        ax.set_ylabel('Forward_Volatility (Next-Day Realized Volatility)', fontsize=12)
        ax.set_title('Agent Disagreement vs. Future Volatility', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Annotation: Text box showing correlation coefficient and slope
        ax.text(0.05, 0.95, f'r = {corr:.2f}\nslope = {slope:.4f}', transform=ax.transAxes, fontsize=14,
               verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Figure 1 saved to {save_path}")
    
    plt.close()


def create_mean_score_figure(
    df: pd.DataFrame,
    correlation_results: Dict,
    save_path: str = None
) -> None:
    """
    Create Figure 2: Mean Agent Score vs. Future Volatility (standalone).
    
    Args:
        df: Merged DataFrame with mean_score and volatility data
        correlation_results: Correlation analysis results
        save_path: Path to save the figure (default: output/fig2_mean_score.png)
    """
    save_path = save_path or os.path.join(config.OUTPUT_DIR, "fig2_mean_score.png")
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if 'mean_score' in df.columns and 'Forward_Volatility' in df.columns:
        valid = df[['mean_score', 'Forward_Volatility']].dropna()
        
        # Scatter plot with alpha=0.6 for density
        ax.scatter(valid['mean_score'], valid['Forward_Volatility'], 
                  alpha=0.6, c='steelblue', s=50, edgecolors='white', linewidth=0.5)
        
        # Add Linear Regression Trendline (Red)
        if len(valid) > 2:
            z = np.polyfit(valid['mean_score'], valid['Forward_Volatility'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid['mean_score'].min(), valid['mean_score'].max(), 100)
            ax.plot(x_line, p(x_line), 'r-', linewidth=2.5, label='Linear Regression')
        
        # Get correlation
        corr = correlation_results.get('corr_mean_score', 0)
        pval = correlation_results.get('pval_mean_score', 1)
        
        # Format p-value
        if pval < 0.001:
            pval_str = "p < 0.001"
        else:
            pval_str = f"p = {pval:.3f}"
        
        # Axis labels
        ax.set_xlabel('Mean_Score (Average of Sentiment, Macro, Technical)', fontsize=12)
        ax.set_ylabel('Forward_Volatility', fontsize=12)
        ax.set_title('Mean Agent Score vs. Future Volatility', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # X-axis: Bearish (-1.0) on left, Bullish (+1.0) on right (standard convention)
        # No inversion needed - default axis ordering is correct
        
        # Add labels for Bearish/Bullish
        ax.text(0.02, 0.02, '← Bearish', transform=ax.transAxes, fontsize=10, 
               verticalalignment='bottom', fontstyle='italic', color='red')
        ax.text(0.98, 0.02, 'Bullish →', transform=ax.transAxes, fontsize=10, 
               verticalalignment='bottom', horizontalalignment='right', fontstyle='italic', color='green')
        
        # Annotation: Text box showing correlation coefficient and p-value
        ax.text(0.05, 0.95, f'r = {corr:.3f}\n({pval_str})', transform=ax.transAxes, fontsize=14,
               verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Figure 2 saved to {save_path}")
    
    plt.close()


def create_timeline_figure(
    df: pd.DataFrame,
    save_path: str = None
) -> None:
    """
    Create Figure 3: 2019 Panic Timeline - Agent Sentiment vs. NVDA Price.
    
    Dual-axis chart showing Mean Score bars (red/green) and NVDA Close Price.
    
    Args:
        df: Merged DataFrame with mean_score, Close, and date data
        save_path: Path to save the figure (default: output/fig3_timeline.png)
    """
    save_path = save_path or os.path.join(config.OUTPUT_DIR, "fig3_timeline.png")
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    if 'mean_score' in df.columns and 'date' in df.columns:
        df_sorted = df.sort_values('date').copy()
        
        # Ensure date is datetime
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])
        
        # --- Left Y-Axis: Mean Score Bars ---
        colors = ['red' if score < 0 else 'green' for score in df_sorted['mean_score']]
        
        ax1.bar(df_sorted['date'], df_sorted['mean_score'], 
               color=colors, alpha=0.7, width=1.5)
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Mean_Score (Agent Sentiment)', fontsize=12, color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
        
        # Set y-limits to center the zero line
        max_abs = max(abs(df_sorted['mean_score'].min()), abs(df_sorted['mean_score'].max())) * 1.2
        ax1.set_ylim(-max_abs, max_abs)
        
        # --- Right Y-Axis: NVDA Close Price ---
        ax2 = ax1.twinx()
        
        if 'Close' in df_sorted.columns:
            ax2.plot(df_sorted['date'], df_sorted['Close'], 
                    color='black', linewidth=2, label='NVDA Close Price')
            ax2.set_ylabel('NVDA Close Price ($)', fontsize=12, color='black')
            ax2.tick_params(axis='y', labelcolor='black')
            ax2.legend(loc='upper right', fontsize=10)
            
            # Highlight major drops where bar is deep red (score < -0.3)
            panic_zones = df_sorted[df_sorted['mean_score'] < -0.3]
            for _, row in panic_zones.iterrows():
                ax1.axvspan(row['date'] - pd.Timedelta(days=1), 
                           row['date'] + pd.Timedelta(days=1),
                           alpha=0.15, color='red', zorder=0)
        
        # Title and formatting
        ax1.set_title('2019 Panic Timeline: Agent Sentiment vs. NVDA Price', 
                     fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Add legend for bar colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Bullish (Score > 0)'),
            Patch(facecolor='red', alpha=0.7, label='Bearish (Score < 0)'),
            Patch(facecolor='red', alpha=0.15, label='Panic Zone (Score < -0.3)')
        ]
        ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Figure 3 saved to {save_path}")
    
    plt.close()


def create_mean_score_vs_realized_volatility_timeseries(
    df: pd.DataFrame,
    save_path: str = None
) -> None:
    """
    Create Time-Series Plot: Mean Score (μt) vs Realized Volatility (σRV t+1).
    
    Dual-axis chart showing:
    - Left Y-Axis: Mean Score (μt) as a line
    - Right Y-Axis: Forward/Realized Volatility (σRV t+1) as a line
    
    Args:
        df: Merged DataFrame with mean_score, Forward_Volatility, and date data
        save_path: Path to save the figure (default: output/mean_score_vs_realized_volatility_timeseries.png)
    """
    save_path = save_path or os.path.join(config.OUTPUT_DIR, "mean_score_vs_realized_volatility_timeseries.png")
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    if 'mean_score' in df.columns and 'Forward_Volatility' in df.columns and 'date' in df.columns:
        df_sorted = df.sort_values('date').copy()
        
        # Ensure date is datetime
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])
        
        # --- Left Y-Axis: Mean Score (μt) ---
        color_mean = 'steelblue'
        ax1.plot(df_sorted['date'], df_sorted['mean_score'], 
                color=color_mean, linewidth=2, label='μt (Mean Score)', marker='o', markersize=3)
        ax1.fill_between(df_sorted['date'], 0, df_sorted['mean_score'], alpha=0.2, color=color_mean)
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('μt (Mean Score)', fontsize=12, color=color_mean)
        ax1.tick_params(axis='y', labelcolor=color_mean)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
        # --- Right Y-Axis: Realized Volatility (σRV t+1) ---
        ax2 = ax1.twinx()
        color_vol = 'darkorange'
        
        ax2.plot(df_sorted['date'], df_sorted['Forward_Volatility'], 
                color=color_vol, linewidth=2, label='σRV t+1 (Realized Volatility)', marker='s', markersize=3)
        ax2.fill_between(df_sorted['date'], 0, df_sorted['Forward_Volatility'], alpha=0.2, color=color_vol)
        
        ax2.set_ylabel('σRV t+1 (Realized Volatility)', fontsize=12, color=color_vol)
        ax2.tick_params(axis='y', labelcolor=color_vol)
        
        # Title
        ax1.set_title('Time-Series: Mean Score (μt) vs Realized Volatility (σRV t+1)', 
                     fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
        
        # Calculate and display correlation
        valid = df_sorted[['mean_score', 'Forward_Volatility']].dropna()
        if len(valid) > 2:
            corr = valid['mean_score'].corr(valid['Forward_Volatility'])
            ax1.text(0.02, 0.98, f'Correlation: r = {corr:.3f}', transform=ax1.transAxes, fontsize=11,
                    verticalalignment='top', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Mean Score vs Realized Volatility Timeseries saved to {save_path}")
    
    plt.close()


def create_topology_figure(save_path: str = None) -> None:
    """
    Create Debate Topology Diagram: Blind & Battle Protocol / Agent Interaction Graph.
    
    Visualizes the 2-round debate protocol:
    - Round 1 (BLIND VOTE): Agents analyze independently (no interaction)
    - Round 2 (BATTLE): Each agent critiques opposing views
    
    Args:
        save_path: Path to save the figure (default: output/topology.png)
    """
    from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
    
    save_path = save_path or os.path.join(config.OUTPUT_DIR, "topology.png")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    colors = {
        'sentiment': '#FF6B6B',    # Coral red
        'technical': '#4ECDC4',    # Teal
        'macro': '#FFE66D',        # Yellow
        'fundamental': '#95E1D3',  # Mint green
        'aggregator': '#A8E6CF',   # Light green
        'data': '#DDA0DD',         # Plum
        'round1': '#E8F4FD',       # Light blue
        'round2': '#FFF3E0',       # Light orange
    }
    
    # Title
    ax.text(7, 9.5, 'Blind & Battle Protocol: Agent Interaction Topology', 
            fontsize=16, fontweight='bold', ha='center', va='center')
    
    # === ROUND 1: BLIND VOTE (Left side) ===
    # Background box for Round 1
    round1_box = Rectangle((0.5, 2), 5.5, 6.5, fill=True, 
                                 facecolor=colors['round1'], edgecolor='steelblue', 
                                 linewidth=2, alpha=0.5, zorder=0)
    ax.add_patch(round1_box)
    ax.text(3.25, 8.2, 'ROUND 1: BLIND VOTE', fontsize=12, fontweight='bold', 
            ha='center', color='steelblue')
    ax.text(3.25, 7.8, '(Independent Analysis)', fontsize=9, ha='center', 
            color='steelblue', style='italic')
    
    # Data Sources (top of Round 1)
    data_sources = [
        ('News\nHeadlines', 1.5, 7),
        ('Technical\nIndicators', 3.25, 7),
        ('Macro\nData', 5, 7),
    ]
    for label, x, y in data_sources:
        circle = Circle((x, y), 0.5, fill=True, facecolor=colors['data'], 
                            edgecolor='purple', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y, label, fontsize=7, ha='center', va='center', fontweight='bold')
    
    # Agents in Round 1 (isolated boxes)
    agents_r1 = [
        ('Sentiment\nAgent', 1.5, 5, colors['sentiment']),
        ('Technical\nAgent', 3.25, 5, colors['technical']),
        ('Macro\nAgent', 5, 5, colors['macro']),
    ]
    for label, x, y, color in agents_r1:
        rect = FancyBboxPatch((x-0.7, y-0.5), 1.4, 1, boxstyle="round,pad=0.05", 
                                   facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, label, fontsize=8, ha='center', va='center', fontweight='bold')
        
        # Arrow from data to agent
        ax.annotate('', xy=(x, y+0.5), xytext=(x, 6.5),
                   arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))
    
    # Fundamental Agent (uses external data)
    rect = FancyBboxPatch((2.55, 2.8), 1.4, 1, boxstyle="round,pad=0.05", 
                               facecolor=colors['fundamental'], edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(3.25, 3.3, 'Fundamental\nAgent', fontsize=8, ha='center', va='center', fontweight='bold')
    ax.text(3.25, 2.5, '(SEC + FRED)', fontsize=7, ha='center', style='italic', color='gray')
    
    # Arrows from agents to "Beliefs R1" box
    beliefs_r1 = FancyBboxPatch((2.25, 4.2), 2, 0.5, boxstyle="round,pad=0.02", 
                                     facecolor='white', edgecolor='steelblue', linewidth=2)
    ax.add_patch(beliefs_r1)
    ax.text(3.25, 4.45, 'Initial Beliefs', fontsize=8, ha='center', va='center', 
            fontweight='bold', color='steelblue')
    
    # === ROUND 2: BATTLE (Right side) ===
    # Background box for Round 2
    round2_box = Rectangle((6.5, 2), 7, 6.5, fill=True, 
                                 facecolor=colors['round2'], edgecolor='darkorange', 
                                 linewidth=2, alpha=0.5, zorder=0)
    ax.add_patch(round2_box)
    ax.text(10, 8.2, 'ROUND 2: BATTLE', fontsize=12, fontweight='bold', 
            ha='center', color='darkorange')
    ax.text(10, 7.8, '(Critique Opposing Views)', fontsize=9, ha='center', 
            color='darkorange', style='italic')
    
    # Agents in Round 2 with critique arrows
    agents_r2 = [
        ('Sentiment', 7.5, 6, colors['sentiment']),
        ('Technical', 10, 6, colors['technical']),
        ('Macro', 12.5, 6, colors['macro']),
    ]
    for label, x, y, color in agents_r2:
        rect = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8, boxstyle="round,pad=0.05", 
                                   facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, label, fontsize=8, ha='center', va='center', fontweight='bold')
    
    # Fundamental in Round 2
    rect = FancyBboxPatch((9.4, 4), 1.2, 0.8, boxstyle="round,pad=0.05", 
                               facecolor=colors['fundamental'], edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(10, 4.4, 'Fundamental', fontsize=8, ha='center', va='center', fontweight='bold')
    
    # Battle arrows (bidirectional critique)
    # Curved arrows between agents showing debate
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patches as mpatches
    
    # Draw debate arrows
    debate_pairs = [
        ((7.5, 5.5), (10, 5.5), 'red'),   # Sentiment <-> Technical
        ((10, 5.5), (12.5, 5.5), 'red'),  # Technical <-> Macro
        ((7.5, 5.5), (10, 4.8), 'red'),   # Sentiment <-> Fundamental
        ((12.5, 5.5), (10, 4.8), 'red'),  # Macro <-> Fundamental
    ]
    
    for start, end, color in debate_pairs:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='<->', color=color, lw=1.5,
                                 connectionstyle='arc3,rad=0.2'))
    
    # Arrow from Round 1 to Round 2
    ax.annotate('', xy=(6.5, 5), xytext=(5.5, 5),
               arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
    ax.text(6, 5.3, 'Share\nOpposing\nViews', fontsize=7, ha='center', va='bottom')
    
    # === AGGREGATOR (Bottom) ===
    agg_rect = FancyBboxPatch((8.5, 2.3), 3, 1, boxstyle="round,pad=0.1", 
                                   facecolor=colors['aggregator'], edgecolor='darkgreen', 
                                   linewidth=2)
    ax.add_patch(agg_rect)
    ax.text(10, 2.8, 'AGGREGATOR', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(10, 2.5, 'Compute Disagreement Dconf', fontsize=8, ha='center', style='italic')
    
    # Arrows from agents to aggregator
    ax.annotate('', xy=(8.5, 2.8), xytext=(7.5, 5.5),
               arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))
    ax.annotate('', xy=(10, 3.3), xytext=(10, 4),
               arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))
    ax.annotate('', xy=(11.5, 2.8), xytext=(12.5, 5.5),
               arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))
    
    # Output
    output_rect = FancyBboxPatch((8.5, 0.5), 3, 0.8, boxstyle="round,pad=0.05", 
                                      facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(output_rect)
    ax.text(10, 0.9, 'Dconf, μt → Volatility Forecast', fontsize=9, ha='center', 
            va='center', fontweight='bold')
    
    ax.annotate('', xy=(10, 1.3), xytext=(10, 2.3),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Legend
    legend_items = [
        (colors['data'], 'Data Sources'),
        (colors['sentiment'], 'Belief Agents'),
        (colors['aggregator'], 'Aggregator'),
        ('red', 'Debate/Critique'),
    ]
    for i, (color, label) in enumerate(legend_items):
        rect = Rectangle((0.7, 1.3 - i*0.4), 0.3, 0.25, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(1.1, 1.42 - i*0.4, label, fontsize=8, va='center')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Topology diagram saved to {save_path}")
    
    plt.close()


def create_residuals_figure(
    garch_baseline: Dict,
    variance_model: Dict,
    save_path: str = None
) -> None:
    """
    Create residual diagnostics: baseline GARCH vs disagreement-variance proxy.
    
    Four-panel plot showing:
    - Top Left: Baseline GARCH standardized residuals
    - Top Right: Variance proxy residuals
    - Bottom Left: Residual histograms comparison
    - Bottom Right: Q-Q plots comparison
    
    Args:
        garch_baseline: Baseline GARCH results dictionary
        variance_model: Variance model results dictionary
        save_path: Path to save the figure (default: output/residuals.png)
    """
    save_path = save_path or os.path.join(config.OUTPUT_DIR, "residuals.png")
    
    # Check if models are available
    if garch_baseline is None or variance_model is None:
        print("Warning: GARCH models not available, skipping residuals plot")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Residual Diagnostics: Baseline GARCH vs Variance Proxy', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Get standardized residuals from both models
    try:
        baseline_resid = np.asarray(garch_baseline['results'].std_resid)
        vm_results = variance_model['results']
        vm_raw_resid = np.asarray(vm_results.resid)
        vm_resid_std = np.std(vm_raw_resid) + 1e-10
        garchx_resid = vm_raw_resid / vm_resid_std
    except (KeyError, AttributeError) as e:
        print(f"Warning: Could not extract residuals: {e}")
        plt.close()
        return
    
    # === Panel 1 (Top Left): Baseline GARCH Residuals Time Series ===
    ax1 = axes[0, 0]
    ax1.plot(baseline_resid, color='steelblue', alpha=0.7, linewidth=0.8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.axhline(y=2, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_title('Baseline GARCH(1,1) Standardized Residuals', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Observation', fontsize=10)
    ax1.set_ylabel('Standardized Residual', fontsize=10)
    ax1.set_ylim(-5, 5)
    
    # Add stats annotation
    baseline_std = np.std(baseline_resid)
    baseline_kurt = ((baseline_resid - np.mean(baseline_resid))**4).mean() / baseline_std**4
    ax1.text(0.02, 0.98, f'Std: {baseline_std:.3f}\nKurtosis: {baseline_kurt:.2f}', 
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # === Panel 2 (Top Right): Variance proxy residuals time series ===
    ax2 = axes[0, 1]
    ax2.plot(garchx_resid, color='darkorange', alpha=0.7, linewidth=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=2, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_title('Variance Proxy Standardized Residuals', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Observation', fontsize=10)
    ax2.set_ylabel('Standardized Residual', fontsize=10)
    ax2.set_ylim(-5, 5)
    
    # Add stats annotation
    garchx_std = np.std(garchx_resid)
    garchx_kurt = ((garchx_resid - np.mean(garchx_resid))**4).mean() / garchx_std**4
    ax2.text(0.02, 0.98, f'Std: {garchx_std:.3f}\nKurtosis: {garchx_kurt:.2f}', 
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # === Panel 3 (Bottom Left): Histogram Comparison ===
    ax3 = axes[1, 0]
    bins = np.linspace(-4, 4, 40)
    ax3.hist(baseline_resid, bins=bins, alpha=0.5, color='steelblue', 
             label=f'Baseline (σ={baseline_std:.2f})', density=True, edgecolor='white')
    ax3.hist(garchx_resid, bins=bins, alpha=0.5, color='darkorange', 
             label=f'Variance Proxy (σ={garchx_std:.2f})', density=True, edgecolor='white')
    
    # Add normal distribution overlay
    x = np.linspace(-4, 4, 100)
    from scipy.stats import norm
    ax3.plot(x, norm.pdf(x), 'k--', linewidth=2, label='N(0,1)')
    
    ax3.set_title('Residual Distribution Comparison', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Standardized Residual', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_xlim(-4, 4)
    
    # === Panel 4 (Bottom Right): Q-Q Plot Comparison ===
    ax4 = axes[1, 1]
    from scipy.stats import probplot
    
    # Q-Q for baseline
    (osm_b, osr_b), (slope_b, intercept_b, r_b) = probplot(baseline_resid, dist="norm")
    ax4.scatter(osm_b, osr_b, alpha=0.5, s=20, color='steelblue', label='Baseline')
    
    # Q-Q for variance proxy model
    (osm_x, osr_x), (slope_x, intercept_x, r_x) = probplot(garchx_resid, dist="norm")
    ax4.scatter(osm_x, osr_x, alpha=0.5, s=20, color='darkorange', label='Variance Proxy')
    
    # Reference line
    xlim = ax4.get_xlim()
    ax4.plot(xlim, xlim, 'k--', linewidth=2, label='Normal')
    
    ax4.set_title('Q-Q Plot: Residuals vs Normal Distribution', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Theoretical Quantiles', fontsize=10)
    ax4.set_ylabel('Sample Quantiles', fontsize=10)
    ax4.legend(loc='lower right', fontsize=9)
    
    # Add R² annotation
    ax4.text(0.02, 0.98, f'Baseline R²: {r_b**2:.4f}\nVariance Proxy R²: {r_x**2:.4f}', 
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Residuals diagnostic plot saved to {save_path}")
    
    plt.close()


def run_full_analysis(verbose: bool = True) -> Dict[str, any]:
    """
    Run the complete analysis pipeline.
    
    Args:
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with all analysis results
    """
    if verbose:
        print("="*60)
        print("AGENTIC DISSONANCE V2 - ANALYSIS")
        print("="*60)
    
    # Load data
    if verbose:
        print("\n1. Loading data...")
    
    disagreement_df = load_disagreement_signals()
    market_df = load_market_data()
    
    if verbose:
        print(f"   Disagreement signals: {len(disagreement_df)} rows")
        print(f"   Market data: {len(market_df)} rows")
    
    # Calculate forward volatility
    if verbose:
        print("\n2. Calculating forward volatility...")
    
    market_df = calculate_forward_volatility(market_df)
    
    # Merge data
    if verbose:
        print("\n3. Merging data...")
    
    merged_df = merge_data(disagreement_df, market_df)
    
    if verbose:
        print(f"   Merged dataset: {len(merged_df)} rows")
    
    if len(merged_df) < 10:
        print("WARNING: Very few data points for analysis. Results may be unreliable.")
    
    # Correlation analysis
    if verbose:
        print("\n4. Correlation analysis...")
    
    correlation_results = run_correlation_analysis(merged_df)
    
    if verbose:
        corr = correlation_results.get('corr_disagreement_conf', 0)
        pval = correlation_results.get('pval_disagreement_conf', 1)
        print(f"   Disagreement ↔ Forward Vol: r={corr:.4f} (p={pval:.4f})")
    
    # Fit GARCH models using DAILY DATA
    # Now that debates run daily, merged_df contains daily observations
    if verbose:
        print("\n5. Fitting GARCH models...")
    
    # Get daily returns and disagreement from merged_df (now daily aligned)
    merged_df = merged_df.sort_values('date')
    
    daily_returns = merged_df['Log_Return'].values
    daily_disagreement = merged_df['disagreement_conf'].values
    
    if verbose:
        print(f"   Using {len(daily_returns)} DAILY returns for GARCH fitting")
        print(f"   (Clean alignment: daily debates → daily volatility)")
    
    # Fit baseline GARCH on daily returns
    if verbose:
        print("   Fitting baseline GARCH(1,1)...")
    garch_baseline = fit_garch_baseline(daily_returns)
    
    # Fit disagreement-driven variance model
    if verbose:
        print("   Fitting disagreement-variance proxy model...")
    
    if len(daily_disagreement) > 50:
        variance_model = fit_garch_x(daily_returns, daily_disagreement)
    else:
        if verbose:
            print("   Warning: Insufficient data for variance proxy fitting")
        variance_model = None
    
    dm_test = None
    if garch_baseline and variance_model:
        baseline_sq_err = (garch_baseline['test_realized_vol'] - garch_baseline['test_pred_vol']) ** 2
        variance_sq_err = (variance_model['test_realized_vol'] - variance_model['test_pred_vol']) ** 2
        dm_test = diebold_mariano_test(baseline_sq_err, variance_sq_err)

        test_dates = merged_df['date'].iloc[garch_baseline['train_n']:].reset_index(drop=True)
        forecast_table = pd.DataFrame({
            'date': test_dates,
            'realized_vol': garch_baseline['test_realized_vol'],
            'garch_pred_vol': garch_baseline['test_pred_vol'],
            'variance_pred_vol': variance_model['test_pred_vol'],
            'garch_sq_error': baseline_sq_err,
            'variance_sq_error': variance_sq_err
        })
        forecast_output_path = os.path.join(config.OUTPUT_DIR, 'test_forecasts.csv')
        forecast_table.to_csv(forecast_output_path, index=False)

        if verbose:
            print(f"   Saved out-of-sample forecast table to {forecast_output_path}")
    
    # Create visualization
    if verbose:
        print("\n7. Creating visualization...")
    
    create_visualization(merged_df, correlation_results, garch_baseline, variance_model)
    
    # Create standalone disagreement figure (fig1_disagreement.png)
    create_disagreement_figure(merged_df, correlation_results)
    
    # Create standalone mean score figure (fig2_mean_score.png)
    create_mean_score_figure(merged_df, correlation_results)
    
    # Create timeline figure (fig3_timeline.png)
    create_timeline_figure(merged_df)
    
    # Create mean score vs realized volatility time-series plot
    create_mean_score_vs_realized_volatility_timeseries(merged_df)
    
    # Create topology diagram (protocol visualization)
    create_topology_figure()
    
    # Create GARCH residuals diagnostic plot
    create_residuals_figure(garch_baseline, variance_model)
    
    # Summary statistics
    if verbose:
        print("\n8. Summary Statistics:")
        print("-" * 50)
        print(f"   Mean disagreement: {merged_df['disagreement_conf'].mean():.4f}")
        print(f"   Std disagreement: {merged_df['disagreement_conf'].std():.4f}")
        print(f"   Mean forward vol: {merged_df['Forward_Volatility'].mean():.4f}")
        print(f"   Mean confidence: {merged_df['avg_confidence'].mean():.4f}")
    
    results = {
        'merged_data': merged_df,
        'correlation': correlation_results,
        'garch_baseline': garch_baseline,
        'variance_model': variance_model,
        'dm_test': dm_test,
        'disagreement_improves_model': (
            variance_model is not None and 
            garch_baseline is not None and
            variance_model['rmse'] < garch_baseline['rmse']
        )
    }
    
    if verbose:
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        if results['disagreement_improves_model']:
            print("✓ Disagreement signal IMPROVES volatility forecasting!")
        else:
            print("✗ Disagreement signal does not improve model (or fitting failed)")
    
    return results


def print_executive_summary(results: Dict) -> None:
    """Print out-of-sample predictive proof for the volatility models."""
    print("\n" + "="*80)
    print("                EXECUTIVE SUMMARY (OUT-OF-SAMPLE PROOF)")
    print("="*80)

    garch = results.get('garch_baseline')
    variance_model = results.get('variance_model') or results.get('garch_x')
    dm_test = results.get('dm_test') or {}

    if not garch or not variance_model:
        print("  FATAL ERROR: Models failed to converge.")
        return

    rmse_imp = (garch['rmse'] - variance_model['rmse']) / garch['rmse'] * 100
    mae_imp = (garch['mae'] - variance_model['mae']) / garch['mae'] * 100
    qlike_imp = (garch['qlike'] - variance_model['qlike']) / garch['qlike'] * 100

    print("\n1. OUT-OF-SAMPLE FORECAST ACCURACY")
    print(f"    Baseline GARCH    -> RMSE: {garch['rmse']:.6f}, MAE: {garch['mae']:.6f}, QLIKE: {garch['qlike']:.6f}")
    print(f"    Disagreement Model-> RMSE: {variance_model['rmse']:.6f}, MAE: {variance_model['mae']:.6f}, QLIKE: {variance_model['qlike']:.6f}")

    print("\n2. RELATIVE IMPROVEMENT (Disagreement vs Baseline)")
    print(f"    RMSE improvement:  {rmse_imp:+.2f}%")
    print(f"    MAE improvement:   {mae_imp:+.2f}%")
    print(f"    QLIKE improvement: {qlike_imp:+.2f}%")

    print("\n3. DIEBOLD-MARIANO TEST (Squared Error Differential)")
    print(f"    DM statistic: {dm_test.get('dm_stat', np.nan):.4f}")
    print(f"    p-value:      {dm_test.get('p_value', np.nan):.4f}")
    print(f"    test obs:     {dm_test.get('n_obs', 0)}")

    if np.isfinite(dm_test.get('p_value', np.nan)) and dm_test.get('p_value', 1.0) < 0.05 and rmse_imp > 0:
        print("\nPROOF VERDICT: PASS - disagreement model improves predictive accuracy out-of-sample.")
    elif rmse_imp > 0:
        print("\nPROOF VERDICT: MIXED - lower out-of-sample errors, but no strong DM significance.")
    else:
        print("\nPROOF VERDICT: FAIL - baseline remains stronger out-of-sample.")

    print("="*80 + "\n")


def print_detailed_report(results: Dict) -> None:
    """
    Print a detailed analysis report.
    
    Args:
        results: Dictionary with analysis results
    """
    print("\n" + "="*60)
    print("DETAILED ANALYSIS REPORT")
    print("="*60)
    
    df = results.get('merged_data')
    if df is not None:
        print(f"\nDataset: {len(df)} observations")
        
        if 'ticker' in df.columns:
            tickers = df['ticker'].unique()
            print(f"Tickers: {', '.join(tickers)}")
        
        date_range = f"{df['date'].min()} to {df['date'].max()}"
        print(f"Date range: {date_range}")
    
    print("\nCorrelation Results:")
    corr = results.get('correlation', {})
    for key, value in corr.items():
        if key.startswith('corr_'):
            metric = key.replace('corr_', '')
            pval = corr.get(f'pval_{metric}', 1.0)
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"  {metric}: r={value:.4f} (p={pval:.4f}) {sig}")
    
    baseline = results.get('garch_baseline')
    variance_model = results.get('variance_model') or results.get('garch_x')
    
    if baseline and variance_model:
        print("\nGARCH Model Parameters:")
        print("\n  GARCH(1,1):")
        for param, value in baseline['params'].items():
            print(f"    {param}: {value:.4f}")
        
        print("\n  Disagreement Variance Proxy:")
        for param, value in variance_model['params'].items():
            print(f"    {param}: {value:.4f}")
        
        print("\n  Model Improvement Metrics (Out-of-Sample):")
        rmse_improvement = (baseline['rmse'] - variance_model['rmse']) / baseline['rmse'] * 100
        mae_improvement = (baseline['mae'] - variance_model['mae']) / baseline['mae'] * 100
        qlike_improvement = (baseline['qlike'] - variance_model['qlike']) / baseline['qlike'] * 100

        print(f"    GARCH RMSE:                  {baseline['rmse']:.6f}")
        print(f"    Disagreement Variance RMSE:  {variance_model['rmse']:.6f}")
        print(f"    RMSE improvement:            {rmse_improvement:.2f}%")
        print(f"    GARCH MAE:                   {baseline['mae']:.6f}")
        print(f"    Disagreement Variance MAE:   {variance_model['mae']:.6f}")
        print(f"    MAE improvement:             {mae_improvement:.2f}%")
        print(f"    GARCH QLIKE:                 {baseline['qlike']:.6f}")
        print(f"    Disagreement Variance QLIKE: {variance_model['qlike']:.6f}")
        print(f"    QLIKE improvement:           {qlike_improvement:.2f}%")


if __name__ == "__main__":
    results = run_full_analysis(verbose=True)
    print_executive_summary(results)
    print_detailed_report(results)
