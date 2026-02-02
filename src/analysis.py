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
        
        df = df.groupby('Ticker', group_keys=False).apply(calc_fwd_vol)
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
    
    # Train/test split
    n = len(returns_pct)
    train_n = int(n * train_size)
    
    train_returns = returns_pct[:train_n]
    test_returns = returns_pct[train_n:]
    
    try:
        # Fit GARCH(1,1)
        model = arch_model(train_returns, vol='Garch', p=1, q=1, rescale=False)
        results = model.fit(disp='off', show_warning=False)
        
        # Forecast on test set
        forecasts = results.forecast(horizon=1, start=0, reindex=False)
        
        # Get fitted values for training
        fitted_vol = results.conditional_volatility
        
        # Forecast for test set
        test_forecasts = []
        for i in range(len(test_returns)):
            fc = results.forecast(horizon=1, start=train_n + i, reindex=False)
            test_forecasts.append(np.sqrt(fc.variance.values[-1, 0]))
        
        test_forecasts = np.array(test_forecasts)
        
        # Calculate test metrics (convert back to decimal)
        realized = np.abs(test_returns) / 100
        predicted = test_forecasts / 100
        
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
            'test_forecasts': test_forecasts,
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
    
    # Lag the exogenous variable (D_{t-1} predicts σ²_t)
    exog_lagged = np.roll(exog, 1)
    exog_lagged[0] = exog[0]  # Fill first value
    
    # Train/test split
    n = len(returns_pct)
    train_n = int(n * train_size)
    
    train_returns = returns_pct[:train_n]
    train_exog = exog_lagged[:train_n].reshape(-1, 1)
    test_returns = returns_pct[train_n:]
    test_exog = exog_lagged[train_n:].reshape(-1, 1)
    
    try:
        # Fit GARCH-X (using exogenous in variance)
        model = arch_model(train_returns, vol='Garch', p=1, q=1, 
                          x=train_exog, rescale=False)
        results = model.fit(disp='off', show_warning=False)
        
        # Get fitted values for training
        fitted_vol = results.conditional_volatility
        
        # Forecast for test set
        test_forecasts = []
        for i in range(len(test_returns)):
            fc = results.forecast(horizon=1, start=train_n + i, 
                                 x=test_exog[i:i+1], reindex=False)
            test_forecasts.append(np.sqrt(fc.variance.values[-1, 0]))
        
        test_forecasts = np.array(test_forecasts)
        
        # Calculate test metrics
        realized = np.abs(test_returns) / 100
        predicted = test_forecasts / 100
        
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
            'test_forecasts': test_forecasts,
            'test_realized': test_returns,
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
    Create comprehensive visualization of analysis results.
    
    Layout:
    - Top Left: "Slope of Truth" - Scatter with trendline
    - Top Right: "Crystal Ball" - Time series with echo effect
    - Bottom: "Pulse Check" - Agent score distributions
    
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
    
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('Agentic Dissonance v2 - Experiment Results', fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid: 2 rows, top row has 2 plots, bottom row has 3 plots
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.2, 1], hspace=0.35, wspace=0.3)
    
    # ================================================================
    # 1. TOP LEFT: "Slope of Truth" - Scatter with Trendline
    # ================================================================
    ax1 = fig.add_subplot(gs[0, 0:2])
    if 'disagreement_conf' in df.columns and 'Forward_Volatility' in df.columns:
        valid = df[['disagreement_conf', 'Forward_Volatility']].dropna()
        
        # Scatter plot
        ax1.scatter(valid['disagreement_conf'], valid['Forward_Volatility'], 
                   alpha=0.6, c='steelblue', s=50, edgecolors='white', linewidth=0.5)
        
        # Add RED trend line (the "Slope of Truth")
        if len(valid) > 2:
            z = np.polyfit(valid['disagreement_conf'], valid['Forward_Volatility'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid['disagreement_conf'].min(), valid['disagreement_conf'].max(), 100)
            ax1.plot(x_line, p(x_line), 'r-', linewidth=3, alpha=0.9, label=f'Trend (slope={z[0]:.3f})')
            
            # Add slope annotation
            slope_direction = "UPWARD ↗" if z[0] > 0 else "DOWNWARD ↘" if z[0] < 0 else "FLAT →"
            ax1.annotate(f'Slope: {slope_direction}', xy=(0.02, 0.95), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', 
                        color='green' if z[0] > 0 else 'red')
        
        corr = correlation_results.get('corr_disagreement_conf', 0)
        pval = correlation_results.get('pval_disagreement_conf', 1)
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        
        ax1.set_xlabel('Agent Disagreement (D_conf)', fontsize=11)
        ax1.set_ylabel('5-Day Forward Volatility', fontsize=11)
        ax1.set_title(f'"Slope of Truth": Disagreement vs Future Volatility (r={corr:.3f}{sig})', 
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
    
    # ================================================================
    # 2. TOP RIGHT: Model Performance Summary
    # ================================================================
    ax_summary = fig.add_subplot(gs[0, 2])
    ax_summary.axis('off')
    
    if garch_baseline and garch_x:
        aic_diff = garch_baseline['aic'] - garch_x['aic']
        rmse_imp = (garch_baseline['rmse'] - garch_x['rmse']) / garch_baseline['rmse'] * 100
        p_val = garch_x.get('exog_pval', 1.0) or 1.0
        
        verdict = "PASSED ✓" if aic_diff > 0 and p_val < 0.05 else "MIXED ~" if aic_diff > 0 else "FAILED ✗"
        verdict_color = 'green' if "PASSED" in verdict else 'orange' if "MIXED" in verdict else 'red'
        
        summary_text = f"""
EXPERIMENT VERDICT
━━━━━━━━━━━━━━━━━━

Result: {verdict}

KEY METRICS:
• AIC Improvement: {aic_diff:+.1f}
• RMSE Reduction: {rmse_imp:+.2f}%
• Signal P-Value: {p_val:.4f}

INTERPRETATION:
{"Agents ADD predictive value!" if aic_diff > 0 else "Baseline model is better."}
{"Signal is statistically significant." if p_val < 0.05 else "Signal may be noise."}
"""
        ax_summary.text(0.1, 0.95, summary_text, transform=ax_summary.transAxes, 
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ================================================================
    # 3. MIDDLE: "Crystal Ball" - Time Series with Echo Effect
    # ================================================================
    ax2 = fig.add_subplot(gs[1, :])
    if 'date' in df.columns:
        df_sorted = df.sort_values('date').copy()
        
        # Normalize both series to 0-1 for overlay
        d_conf = df_sorted['disagreement_conf']
        fwd_vol = df_sorted['Forward_Volatility']
        
        d_conf_norm = (d_conf - d_conf.min()) / (d_conf.max() - d_conf.min() + 1e-8)
        fwd_vol_norm = (fwd_vol - fwd_vol.min()) / (fwd_vol.max() - fwd_vol.min() + 1e-8)
        
        # Plot both on same axis (normalized)
        ax2.plot(df_sorted['date'], d_conf_norm, 
                color='steelblue', label='Agent Disagreement (D_conf)', 
                linewidth=2, alpha=0.9)
        ax2.plot(df_sorted['date'], fwd_vol_norm, 
                color='coral', label='Future Volatility (5-day ahead)', 
                linewidth=2, alpha=0.9)
        
        # Fill between to show spikes
        ax2.fill_between(df_sorted['date'], 0, d_conf_norm, alpha=0.2, color='steelblue')
        ax2.fill_between(df_sorted['date'], 0, fwd_vol_norm, alpha=0.2, color='coral')
        
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Normalized Magnitude (0-1)', fontsize=11)
        ax2.set_title('"Crystal Ball": Time Series - Disagreement vs Volatility (Look for Echo Effect: Blue leads Orange)', 
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
    
    # ================================================================
    # 4. BOTTOM LEFT: Agent Score Distribution (Pulse Check)
    # ================================================================
    ax5 = fig.add_subplot(gs[2, 0])
    score_cols = ['score_sentiment', 'score_technical', 'score_macro']
    available_cols = [c for c in score_cols if c in df.columns]
    if available_cols:
        data_to_plot = [df[col].dropna() for col in available_cols]
        labels = [col.replace('score_', '').capitalize() for col in available_cols]
        
        bp = ax5.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
        colors = ['coral', 'forestgreen', 'goldenrod']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=2)
        ax5.axhline(y=-1, color='red', linestyle=':', alpha=0.3)
        ax5.axhline(y=1, color='green', linestyle=':', alpha=0.3)
        ax5.set_ylabel('Score (-1 to +1)')
        ax5.set_title('"Pulse Check": Agent Score Distributions', fontsize=11, fontweight='bold')
        ax5.set_ylim(-1.2, 1.2)
        ax5.grid(True, alpha=0.3)
    
    # ================================================================
    # 5. BOTTOM CENTER: Agent Confidence Distribution
    # ================================================================
    ax6 = fig.add_subplot(gs[2, 1])
    conf_cols = ['confidence_sentiment', 'confidence_technical', 'confidence_macro']
    available_conf = [c for c in conf_cols if c in df.columns]
    if available_conf:
        data_to_plot = [df[col].dropna() for col in available_conf]
        labels = [col.replace('confidence_', '').capitalize() for col in available_conf]
        
        bp = ax6.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
        colors = ['coral', 'forestgreen', 'goldenrod']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax6.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2)
        ax6.set_ylabel('Confidence (0 to 1)')
        ax6.set_title('Agent Confidence Levels', fontsize=11, fontweight='bold')
        ax6.set_ylim(-0.1, 1.1)
        ax6.grid(True, alpha=0.3)
    
    # ================================================================
    # 6. BOTTOM RIGHT: Model Comparison AIC/BIC
    # ================================================================
    ax3 = fig.add_subplot(gs[2, 2])
    if garch_baseline and garch_x:
        models = ['GARCH(1,1)\n(Baseline)', 'GARCH-X\n(+Disagreement)']
        aic_values = [garch_baseline['aic'], garch_x['aic']]
        
        x = np.arange(len(models))
        bars = ax3.bar(x, aic_values, color=['gray', 'steelblue'], width=0.5, edgecolor='black')
        
        # Highlight winner
        winner_idx = 0 if aic_values[0] < aic_values[1] else 1
        bars[winner_idx].set_color('green')
        bars[winner_idx].set_alpha(0.8)
        
        ax3.set_ylabel('AIC (lower is better)')
        ax3.set_title('Model Comparison', fontsize=11, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to {save_path}")
    
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
    
    # Fit GARCH models
    if verbose:
        print("\n5. Fitting GARCH models...")
    
    returns = merged_df['Log_Return'].values
    disagreement = merged_df['disagreement_conf'].values
    
    if verbose:
        print("   Fitting baseline GARCH(1,1)...")
    garch_baseline = fit_garch_baseline(returns)
    
    if verbose:
        print("   Fitting GARCH-X with disagreement...")
    garch_x = fit_garch_x(returns, disagreement)
    
    # Print model comparison
    if verbose and garch_baseline and garch_x:
        print("\n6. Model Comparison:")
        print("-" * 50)
        print(f"{'Metric':<15} {'GARCH(1,1)':<15} {'GARCH-X':<15} {'Better':<10}")
        print("-" * 50)
        
        aic_better = "GARCH-X" if garch_x['aic'] < garch_baseline['aic'] else "GARCH"
        bic_better = "GARCH-X" if garch_x['bic'] < garch_baseline['bic'] else "GARCH"
        rmse_better = "GARCH-X" if garch_x['rmse'] < garch_baseline['rmse'] else "GARCH"
        mae_better = "GARCH-X" if garch_x['mae'] < garch_baseline['mae'] else "GARCH"
        
        print(f"{'AIC':<15} {garch_baseline['aic']:<15.2f} {garch_x['aic']:<15.2f} {aic_better:<10}")
        print(f"{'BIC':<15} {garch_baseline['bic']:<15.2f} {garch_x['bic']:<15.2f} {bic_better:<10}")
        print(f"{'RMSE':<15} {garch_baseline['rmse']:<15.6f} {garch_x['rmse']:<15.6f} {rmse_better:<10}")
        print(f"{'MAE':<15} {garch_baseline['mae']:<15.6f} {garch_x['mae']:<15.6f} {mae_better:<10}")
        
        if garch_x.get('exog_coef') is not None:
            print(f"\n   GARCH-X exogenous coefficient: {garch_x['exog_coef']:.4f}")
            print(f"   GARCH-X exogenous p-value: {garch_x['exog_pval']:.4f}")
            sig = "***" if garch_x['exog_pval'] < 0.01 else "**" if garch_x['exog_pval'] < 0.05 else "*" if garch_x['exog_pval'] < 0.1 else ""
            print(f"   Significance: {sig}")
    
    # Create visualization
    if verbose:
        print("\n7. Creating visualization...")
    
    create_visualization(merged_df, correlation_results, garch_baseline, garch_x)
    
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
        
        print(f"    AIC improvement: {aic_improvement:.2f}")
        print(f"    BIC improvement: {bic_improvement:.2f}")
        print(f"    RMSE improvement: {rmse_improvement:.2f}%")


if __name__ == "__main__":
    results = run_full_analysis(verbose=True)
    print_executive_summary(results)
    print_detailed_report(results)
