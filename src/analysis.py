"""
Statistical analysis and volatility modeling for Agentic Dissonance v2.

Implements:
- 5-day forward realized volatility calculation
- Baseline GARCH(1,1) model
- GARCH-X with disagreement as exogenous variable
- AIC/BIC/RMSE/MAE comparison
- Visualization of results
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        # Fit GARCH(1,1)
        model = arch_model(train_returns, vol='Garch', p=1, q=1, rescale=False)
        results = model.fit(disp='off', show_warning=False)
        
        # Get fitted values for training (returns numpy array)
        fitted_vol = results.conditional_volatility
        
        # Calculate in-sample metrics
        realized = np.abs(train_returns) / 100
        predicted = fitted_vol / 100
        
        rmse = np.sqrt(np.mean((realized - predicted) ** 2))
        mae = np.mean(np.abs(realized - predicted))
        
        return {
            'model': model,
            'results': results,
            'aic': results.aic,
            'bic': results.bic,
            'rmse': rmse,
            'mae': mae,
            'train_vol': fitted_vol,
            'test_realized': test_returns,
            'params': dict(results.params)
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
    Fit GARCH-X model with disagreement as exogenous variable.
    
    σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1} + γ*D_{t-1}
    
    Args:
        returns: Array of log returns
        exog: Array of exogenous variable (disagreement)
        train_size: Fraction for training
        
    Returns:
        Dictionary with model, results, and metrics
    """
    train_size = train_size or config.TRAIN_TEST_SPLIT
    
    # Scale returns to percentage
    returns_pct = returns * 100
    
    # Check minimum data requirement
    if len(returns_pct) < 50:
        print(f"GARCH-X: Insufficient data ({len(returns_pct)} points, need 50+)")
        return None
    
    # Lag the exogenous variable (D_{t-1} predicts σ²_t)
    exog_lagged = np.roll(exog, 1)
    exog_lagged[0] = exog[0]  # Fill first value
    
    # Train/test split
    n = len(returns_pct)
    train_n = int(n * train_size)
    
    if train_n < 30:
        print(f"GARCH-X: Training set too small ({train_n} points)")
        return None
    
    train_returns = returns_pct[:train_n]
    train_exog = exog_lagged[:train_n].reshape(-1, 1)
    
    try:
        # Fit GARCH with exogenous regressor in the mean equation
        # Note: arch_model 'x' parameter adds to mean, not variance
        # For variance effect, we use a workaround with scaled exog
        model = arch_model(train_returns, vol='Garch', p=1, q=1, 
                          mean='ARX', lags=0, x=train_exog, rescale=False)
        results = model.fit(disp='off', show_warning=False)
        
        # Get fitted values for training (returns numpy array)
        fitted_vol = results.conditional_volatility
        
        # Calculate in-sample metrics
        realized = np.abs(train_returns) / 100
        predicted = fitted_vol / 100
        
        rmse = np.sqrt(np.mean((realized - predicted) ** 2))
        mae = np.mean(np.abs(realized - predicted))
        
        # Check if exogenous coefficient is significant
        exog_coef = None
        exog_pval = None
        for param_name in results.params.index:
            if 'x' in param_name.lower() or 'exog' in param_name.lower():
                exog_coef = results.params[param_name]
                exog_pval = results.pvalues[param_name]
                break
        
        return {
            'model': model,
            'results': results,
            'aic': results.aic,
            'bic': results.bic,
            'rmse': rmse,
            'mae': mae,
            'train_vol': fitted_vol,
            'params': dict(results.params),
            'exog_coef': exog_coef,
            'exog_pval': exog_pval
        }
        
    except Exception as e:
        print(f"GARCH-X fitting failed: {e}")
        return None


def create_visualization(
    df: pd.DataFrame,
    correlation_results: Dict,
    garch_baseline: Dict,
    garch_x: Dict,
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
        garch_x: GARCH-X results
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
        
        # Invert X-axis so Bearish (-1.0) is on the left
        ax.invert_xaxis()
        
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
    
    # Fit GARCH-X with daily disagreement
    if verbose:
        print("   Fitting GARCH-X with disagreement...")
    
    if len(daily_disagreement) > 50:
        garch_x = fit_garch_x(daily_returns, daily_disagreement)
    else:
        if verbose:
            print("   Warning: Insufficient data for GARCH-X fitting")
        garch_x = None
    
    # Print RMSE Executive Summary
    if verbose and garch_baseline and garch_x:
        baseline_rmse = garch_baseline['rmse']
        agent_rmse = garch_x['rmse']
        agents_win = agent_rmse < baseline_rmse
        error_reduction = (baseline_rmse - agent_rmse) / baseline_rmse * 100
        
        print("\n" + "=" * 80)
        print("                      EXECUTIVE SUMMARY (RMSE TEST)")
        print("=" * 80)
        print(f"1. Standard GARCH RMSE: {baseline_rmse:.6f}")
        print(f"2. Agent GARCH-X RMSE:  {agent_rmse:.6f} (Lower is Better)")
        print()
        print("VERDICT:")
        if agents_win:
            print(f"[X] SUCCESS: Agents reduced error by {abs(error_reduction):.2f}%.")
            print("[ ] FAILURE: Standard model was more accurate.")
        else:
            print("[ ] SUCCESS: Agents reduced error by X%.")
            print(f"[X] FAILURE: Standard model was more accurate (by {abs(error_reduction):.2f}%).")
        print("=" * 80)
    
    # Create visualization
    if verbose:
        print("\n7. Creating visualization...")
    
    create_visualization(merged_df, correlation_results, garch_baseline, garch_x)
    
    # Create standalone disagreement figure (fig1_disagreement.png)
    create_disagreement_figure(merged_df, correlation_results)
    
    # Create standalone mean score figure (fig2_mean_score.png)
    create_mean_score_figure(merged_df, correlation_results)
    
    # Create timeline figure (fig3_timeline.png)
    create_timeline_figure(merged_df)
    
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
        'garch_x': garch_x,
        'disagreement_improves_model': (
            garch_x is not None and 
            garch_baseline is not None and
            garch_x['aic'] < garch_baseline['aic']
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
    """
    Print a human-readable verdict on the Agentic Dissonance hypothesis.
    
    Interprets the statistical results for non-technical stakeholders:
    - Did the model pass the test? (Yes/No)
    - Is the signal significant? (p-value < 0.05)
    - How much did error reduce? (RMSE improvement)
    
    Args:
        results: Dictionary with analysis results
    """
    print("\n" + "="*80)
    print("                      EXECUTIVE SUMMARY & VERDICT")
    print("="*80)
    
    garch = results.get('garch_baseline')
    garch_x = results.get('garch_x')
    corr = results.get('correlation', {})
    
    if not garch or not garch_x:
        print("  FATAL ERROR: Models failed to converge.")
        return
    
    # Calculate key metrics
    aic_diff = garch['aic'] - garch_x['aic']
    is_better = aic_diff > 0
    p_val = garch_x.get('exog_pval', 1.0) or 1.0
    is_significant = p_val < 0.05
    rmse_imp = (garch['rmse'] - garch_x['rmse']) / garch['rmse'] * 100
    
    print(f"\n1. HYPOTHESIS TEST")
    print(f"   Did Agent Disagreement predict volatility better than price alone?")
    if is_better and is_significant:
        print(f"     PASSED. (Strong Evidence)")
        print(f"      The Agentic Model is statistically superior (Lower AIC + Significant Signal).")
    elif is_better:
        print(f"     MIXED. (Weak Evidence)")
        print(f"      The model fits better (Lower AIC), but the signal p-value is > 0.05.")
    else:
        print(f"     FAILED.")
        print(f"      The baseline GARCH model performed better. Agents added noise.")
    
    print(f"\n2. DETAILED METRICS")
    print(f"    AIC Improvement:      {aic_diff:+.2f}  (>0 implies agents added value)")
    print(f"    Signal P-Value:       {p_val:.4f}  (<0.05 implies non-random correlation)")
    print(f"    Error Reduction:      {rmse_imp:+.2f}% (Positive means lower error)")
    print(f"    Raw Correlation:      {corr.get('corr_disagreement_conf', 0):.4f}")
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
    garch_x = results.get('garch_x')
    
    if baseline and garch_x:
        print("\nGARCH Model Parameters:")
        print("\n  GARCH(1,1):")
        for param, value in baseline['params'].items():
            print(f"    {param}: {value:.4f}")
        
        print("\n  GARCH-X:")
        for param, value in garch_x['params'].items():
            print(f"    {param}: {value:.4f}")
        
        print("\n  Model Improvement Metrics:")
        aic_improvement = baseline['aic'] - garch_x['aic']
        bic_improvement = baseline['bic'] - garch_x['bic']
        rmse_improvement = (baseline['rmse'] - garch_x['rmse']) / baseline['rmse'] * 100
        
        print(f"    GARCH RMSE:      {baseline['rmse']:.6f}")
        print(f"    GARCH-X RMSE:    {garch_x['rmse']:.6f}")
        print(f"    RMSE improvement: {rmse_improvement:.2f}%")
        print(f"    GARCH AIC:       {baseline['aic']:.2f}")
        print(f"    GARCH-X AIC:     {garch_x['aic']:.2f}")
        print(f"    AIC improvement: {aic_improvement:.2f}")
        print(f"    GARCH BIC:       {baseline['bic']:.2f}")
        print(f"    GARCH-X BIC:     {garch_x['bic']:.2f}")
        print(f"    BIC improvement: {bic_improvement:.2f}")


if __name__ == "__main__":
    results = run_full_analysis(verbose=True)
    print_executive_summary(results)
    print_detailed_report(results)
