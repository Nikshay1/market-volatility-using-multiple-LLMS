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
    
    Args:
        df: Merged DataFrame
        correlation_results: Correlation analysis results
        garch_baseline: Baseline GARCH results
        garch_x: GARCH-X results
        save_path: Path to save the figure
    """
    save_path = save_path or config.RESULTS_PLOT_PATH
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Agentic Dissonance v2 - Analysis Results', fontsize=14, fontweight='bold')
    
    # 1. Disagreement vs Forward Volatility scatter
    ax1 = axes[0, 0]
    if 'disagreement_conf' in df.columns and 'Forward_Volatility' in df.columns:
        valid = df[['disagreement_conf', 'Forward_Volatility']].dropna()
        ax1.scatter(valid['disagreement_conf'], valid['Forward_Volatility'], 
                   alpha=0.6, c='steelblue', s=30)
        
        # Add trend line
        if len(valid) > 2:
            z = np.polyfit(valid['disagreement_conf'], valid['Forward_Volatility'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid['disagreement_conf'].min(), valid['disagreement_conf'].max(), 100)
            ax1.plot(x_line, p(x_line), 'r--', alpha=0.8, label='Trend')
        
        corr = correlation_results.get('corr_disagreement_conf', 0)
        ax1.set_xlabel('Disagreement (D_conf)')
        ax1.set_ylabel('5-Day Forward Volatility')
        ax1.set_title(f'Disagreement vs Future Volatility (r={corr:.3f})')
        ax1.legend()
    
    # 2. Time series of disagreement and volatility
    ax2 = axes[0, 1]
    if 'date' in df.columns:
        df_sorted = df.sort_values('date')
        ax2_twin = ax2.twinx()
        
        ax2.plot(df_sorted['date'], df_sorted['disagreement_conf'], 
                color='steelblue', label='Disagreement', alpha=0.8)
        ax2_twin.plot(df_sorted['date'], df_sorted['Forward_Volatility'], 
                     color='coral', label='Fwd Vol', alpha=0.8)
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Disagreement', color='steelblue')
        ax2_twin.set_ylabel('Forward Volatility', color='coral')
        ax2.set_title('Time Series: Disagreement & Volatility')
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. Model comparison - AIC/BIC
    ax3 = axes[1, 0]
    if garch_baseline and garch_x:
        models = ['GARCH(1,1)', 'GARCH-X']
        aic_values = [garch_baseline['aic'], garch_x['aic']]
        bic_values = [garch_baseline['bic'], garch_x['bic']]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, aic_values, width, label='AIC', color='steelblue')
        bars2 = ax3.bar(x + width/2, bic_values, width, label='BIC', color='coral')
        
        ax3.set_ylabel('Information Criterion')
        ax3.set_title('Model Comparison: AIC & BIC (lower is better)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models)
        ax3.legend()
        
        # Add value labels
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # 4. Model comparison - RMSE/MAE
    ax4 = axes[1, 1]
    if garch_baseline and garch_x:
        models = ['GARCH(1,1)', 'GARCH-X']
        rmse_values = [garch_baseline['rmse'], garch_x['rmse']]
        mae_values = [garch_baseline['mae'], garch_x['mae']]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, rmse_values, width, label='RMSE', color='forestgreen')
        bars2 = ax4.bar(x + width/2, mae_values, width, label='MAE', color='goldenrod')
        
        ax4.set_ylabel('Error')
        ax4.set_title('Model Comparison: RMSE & MAE (lower is better)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models)
        ax4.legend()
        
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax4.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # 5. Agent scores distribution
    ax5 = axes[2, 0]
    score_cols = ['score_fundamental', 'score_sentiment', 'score_technical', 'score_macro']
    available_cols = [c for c in score_cols if c in df.columns]
    if available_cols:
        data_to_plot = [df[col].dropna() for col in available_cols]
        labels = [col.replace('score_', '').capitalize() for col in available_cols]
        
        bp = ax5.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['steelblue', 'coral', 'forestgreen', 'goldenrod']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax5.set_ylabel('Score')
        ax5.set_title('Agent Score Distributions')
    
    # 6. Confidence distribution
    ax6 = axes[2, 1]
    conf_cols = ['confidence_fundamental', 'confidence_sentiment', 
                 'confidence_technical', 'confidence_macro']
    available_conf = [c for c in conf_cols if c in df.columns]
    if available_conf:
        data_to_plot = [df[col].dropna() for col in available_conf]
        labels = [col.replace('confidence_', '').capitalize() for col in available_conf]
        
        bp = ax6.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['steelblue', 'coral', 'forestgreen', 'goldenrod']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax6.set_ylabel('Confidence')
        ax6.set_title('Agent Confidence Distributions')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
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
    print_detailed_report(results)
