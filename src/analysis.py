"""
Statistical analysis and visualization of disagreement vs volatility.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from typing import Dict, Tuple, Optional

from . import config


def load_disagreement_signals() -> pd.DataFrame:
    """
    Load disagreement signals from CSV.
    
    Returns:
        DataFrame with disagreement signals
    """
    if not os.path.exists(config.DISAGREEMENT_SIGNALS_PATH):
        raise FileNotFoundError(
            f"Disagreement signals not found at {config.DISAGREEMENT_SIGNALS_PATH}. "
            "Please run backtest.py first."
        )
    
    df = pd.read_csv(config.DISAGREEMENT_SIGNALS_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_market_data() -> pd.DataFrame:
    """
    Load market data from CSV.
    
    Returns:
        DataFrame with market data
    """
    if not os.path.exists(config.RAW_MARKET_DATA_PATH):
        raise FileNotFoundError(
            f"Market data not found at {config.RAW_MARKET_DATA_PATH}. "
            "Please run data_loader.py first."
        )
    
    df = pd.read_csv(config.RAW_MARKET_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def calculate_forward_volatility(
    market_df: pd.DataFrame,
    window: int = None
) -> pd.DataFrame:
    """
    Calculate forward realized volatility.
    
    σ_{t+1:t+window} = std(returns from t+1 to t+window)
    
    Args:
        market_df: DataFrame with market data including Log_Return
        window: Forward window size (default from config)
        
    Returns:
        DataFrame with forward volatility column added
    """
    window = window or config.FORWARD_VOLATILITY_WINDOW
    
    # Calculate rolling standard deviation of returns
    # Shift by -1 to get FORWARD volatility (next N days)
    market_df = market_df.copy()
    
    # Calculate forward volatility
    # For each day t, we want std of returns from t+1 to t+window
    forward_vol = []
    returns = market_df['Log_Return'].values
    
    for i in range(len(returns)):
        if i + window < len(returns):
            # Get returns from i+1 to i+window (inclusive)
            future_returns = returns[i+1:i+1+window]
            vol = np.std(future_returns) * np.sqrt(252)  # Annualized
            forward_vol.append(vol)
        else:
            forward_vol.append(np.nan)
    
    market_df['Forward_Volatility'] = forward_vol
    
    return market_df


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
    # Ensure date columns are datetime
    disagreement_df = disagreement_df.copy()
    market_df = market_df.copy()
    
    disagreement_df['date'] = pd.to_datetime(disagreement_df['date'])
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    
    # Merge on date
    merged = pd.merge(
        disagreement_df,
        market_df[['Date', 'Log_Return', 'Forward_Volatility']],
        left_on='date',
        right_on='Date',
        how='inner'
    )
    
    # Drop rows with missing forward volatility
    merged = merged.dropna(subset=['Forward_Volatility'])
    
    return merged


def run_correlation_analysis(
    df: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate correlation between disagreement metrics and forward volatility.
    
    Args:
        df: Merged DataFrame with disagreement and volatility
        
    Returns:
        Dictionary with correlation results
    """
    results = {}
    
    # Scalar disagreement vs forward volatility
    corr_scalar, pval_scalar = stats.pearsonr(
        df['disagreement_scalar'],
        df['Forward_Volatility']
    )
    results['corr_scalar'] = corr_scalar
    results['pval_scalar'] = pval_scalar
    
    # Semantic divergence vs forward volatility
    corr_semantic, pval_semantic = stats.pearsonr(
        df['disagreement_semantic'],
        df['Forward_Volatility']
    )
    results['corr_semantic'] = corr_semantic
    results['pval_semantic'] = pval_semantic
    
    # Spearman correlations (rank-based)
    spearman_scalar, spearman_pval_scalar = stats.spearmanr(
        df['disagreement_scalar'],
        df['Forward_Volatility']
    )
    results['spearman_scalar'] = spearman_scalar
    results['spearman_pval_scalar'] = spearman_pval_scalar
    
    spearman_semantic, spearman_pval_semantic = stats.spearmanr(
        df['disagreement_semantic'],
        df['Forward_Volatility']
    )
    results['spearman_semantic'] = spearman_semantic
    results['spearman_pval_semantic'] = spearman_pval_semantic
    
    return results


def run_ols_regression(
    df: pd.DataFrame,
    dependent_var: str = 'Forward_Volatility',
    independent_vars: list = None
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run OLS regression of forward volatility on disagreement metrics.
    
    Args:
        df: Merged DataFrame
        dependent_var: Name of dependent variable
        independent_vars: List of independent variable names
        
    Returns:
        OLS regression results
    """
    if independent_vars is None:
        independent_vars = ['disagreement_scalar', 'disagreement_semantic']
    
    # Prepare data
    y = df[dependent_var].values
    X = df[independent_vars].values
    X = sm.add_constant(X)  # Add intercept
    
    # Fit OLS
    model = sm.OLS(y, X)
    results = model.fit()
    
    return results


def run_garch_x_analysis(
    df: pd.DataFrame,
    use_exog: bool = True
) -> Optional[Dict]:
    """
    Fit GARCH-X model with disagreement as exogenous variable.
    
    σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1} + γ*D_{t-1}
    
    Args:
        df: Merged DataFrame with returns and disagreement
        use_exog: Whether to include exogenous variables
        
    Returns:
        Dictionary with GARCH results, or None if fitting fails
    """
    try:
        from arch import arch_model
    except ImportError:
        print("arch library not available. Skipping GARCH-X analysis.")
        return None
    
    try:
        # Prepare return series (scaled by 100 for numerical stability)
        returns = df['Log_Return'].values * 100
        
        if use_exog:
            # Use combined disagreement as exogenous variable
            # Lag the disagreement by 1 to predict next period volatility
            exog = df[['disagreement_scalar', 'disagreement_semantic']].values[:-1]
            returns = returns[1:]  # Align with lagged exog
            
            # Fit GARCH(1,1)-X model
            model = arch_model(
                returns,
                vol='Garch',
                p=1,
                q=1,
                x=exog,
                mean='Constant'
            )
        else:
            # Standard GARCH(1,1) for comparison
            model = arch_model(
                returns,
                vol='Garch',
                p=1,
                q=1,
                mean='Constant'
            )
        
        # Fit model
        results = model.fit(disp='off')
        
        return {
            'model': 'GARCH-X' if use_exog else 'GARCH',
            'aic': results.aic,
            'bic': results.bic,
            'log_likelihood': results.loglikelihood,
            'params': results.params.to_dict(),
            'pvalues': results.pvalues.to_dict(),
            'summary': str(results.summary())
        }
        
    except Exception as e:
        print(f"GARCH fitting error: {e}")
        return None


def create_visualization(
    df: pd.DataFrame,
    correlation_results: Dict,
    ols_results,
    save_path: str = None
) -> None:
    """
    Create visualization of disagreement vs volatility analysis.
    
    Args:
        df: Merged DataFrame
        correlation_results: Dictionary with correlation metrics
        ols_results: OLS regression results
        save_path: Path to save the figure
    """
    save_path = save_path or config.RESULTS_PLOT_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Agentic Dissonance: Disagreement vs Forward Volatility Analysis', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Scalar disagreement vs forward volatility
    ax1 = axes[0, 0]
    ax1.scatter(df['disagreement_scalar'], df['Forward_Volatility'], 
                alpha=0.5, edgecolors='none', s=50)
    
    # Add regression line
    z = np.polyfit(df['disagreement_scalar'], df['Forward_Volatility'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['disagreement_scalar'].min(), df['disagreement_scalar'].max(), 100)
    ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
    
    ax1.set_xlabel('Scalar Disagreement (std of scores)')
    ax1.set_ylabel('Forward 5-Day Volatility (annualized)')
    ax1.set_title(f'Scalar Disagreement\nr={correlation_results["corr_scalar"]:.3f}, '
                  f'p={correlation_results["pval_scalar"]:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Semantic divergence vs forward volatility
    ax2 = axes[0, 1]
    ax2.scatter(df['disagreement_semantic'], df['Forward_Volatility'], 
                alpha=0.5, edgecolors='none', s=50, color='green')
    
    z = np.polyfit(df['disagreement_semantic'], df['Forward_Volatility'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['disagreement_semantic'].min(), df['disagreement_semantic'].max(), 100)
    ax2.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
    
    ax2.set_xlabel('Semantic Divergence (1 - cosine similarity)')
    ax2.set_ylabel('Forward 5-Day Volatility (annualized)')
    ax2.set_title(f'Semantic Divergence\nr={correlation_results["corr_semantic"]:.3f}, '
                  f'p={correlation_results["pval_semantic"]:.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time series of disagreement and volatility
    ax3 = axes[1, 0]
    
    # Normalize both series for comparison
    scalar_norm = (df['disagreement_scalar'] - df['disagreement_scalar'].mean()) / df['disagreement_scalar'].std()
    vol_norm = (df['Forward_Volatility'] - df['Forward_Volatility'].mean()) / df['Forward_Volatility'].std()
    
    ax3.plot(df['date'], scalar_norm, label='Disagreement (normalized)', alpha=0.7)
    ax3.plot(df['date'], vol_norm, label='Forward Vol (normalized)', alpha=0.7)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Normalized Value')
    ax3.set_title('Time Series: Disagreement vs Forward Volatility')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 4: Summary statistics box
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary text
    summary_text = f"""
ANALYSIS SUMMARY
{'='*40}

Sample Size: {len(df)} trading days

CORRELATION ANALYSIS
---------------------
Pearson (Scalar):    r = {correlation_results['corr_scalar']:.4f}  (p = {correlation_results['pval_scalar']:.4f})
Pearson (Semantic):  r = {correlation_results['corr_semantic']:.4f}  (p = {correlation_results['pval_semantic']:.4f})
Spearman (Scalar):   ρ = {correlation_results['spearman_scalar']:.4f}  (p = {correlation_results['spearman_pval_scalar']:.4f})
Spearman (Semantic): ρ = {correlation_results['spearman_semantic']:.4f}  (p = {correlation_results['spearman_pval_semantic']:.4f})

OLS REGRESSION
--------------
R² = {ols_results.rsquared:.4f}
Adjusted R² = {ols_results.rsquared_adj:.4f}
F-statistic = {ols_results.fvalue:.4f}
F p-value = {ols_results.f_pvalue:.4f}

INTERPRETATION
--------------
{"✓ Significant relationship found" if correlation_results['pval_scalar'] < 0.05 or correlation_results['pval_semantic'] < 0.05 else "✗ No significant relationship found"}
between agent disagreement and forward volatility.
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    
    plt.close()


def run_full_analysis(verbose: bool = True) -> Dict:
    """
    Run the complete analysis pipeline.
    
    Args:
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with all analysis results
    """
    print("=" * 60)
    print("ANALYSIS: Disagreement vs Forward Volatility")
    print("=" * 60)
    
    # Load data
    if verbose:
        print("\n1. Loading data...")
    
    disagreement_df = load_disagreement_signals()
    market_df = load_market_data()
    
    if verbose:
        print(f"   Disagreement signals: {len(disagreement_df)} days")
        print(f"   Market data: {len(market_df)} days")
    
    # Calculate forward volatility
    if verbose:
        print("\n2. Calculating forward volatility...")
    
    market_df = calculate_forward_volatility(market_df)
    
    # Merge data
    if verbose:
        print("\n3. Merging datasets...")
    
    merged_df = merge_data(disagreement_df, market_df)
    
    if verbose:
        print(f"   Merged dataset: {len(merged_df)} days")
    
    if len(merged_df) < 10:
        raise ValueError("Not enough data for analysis. Need at least 10 observations.")
    
    # Run correlation analysis
    if verbose:
        print("\n4. Running correlation analysis...")
    
    corr_results = run_correlation_analysis(merged_df)
    
    if verbose:
        print(f"   Scalar Disagreement vs Vol: r={corr_results['corr_scalar']:.4f} (p={corr_results['pval_scalar']:.4f})")
        print(f"   Semantic Divergence vs Vol: r={corr_results['corr_semantic']:.4f} (p={corr_results['pval_semantic']:.4f})")
    
    # Run OLS regression
    if verbose:
        print("\n5. Running OLS regression...")
    
    ols_results = run_ols_regression(merged_df)
    
    if verbose:
        print(f"   R² = {ols_results.rsquared:.4f}")
        print(f"   Adjusted R² = {ols_results.rsquared_adj:.4f}")
    
    # Run GARCH-X (optional)
    if verbose:
        print("\n6. Running GARCH-X analysis...")
    
    garch_results = run_garch_x_analysis(merged_df, use_exog=True)
    garch_baseline = run_garch_x_analysis(merged_df, use_exog=False)
    
    if garch_results and garch_baseline:
        if verbose:
            print(f"   GARCH-X AIC: {garch_results['aic']:.2f}")
            print(f"   GARCH   AIC: {garch_baseline['aic']:.2f}")
            print(f"   Improvement: {garch_baseline['aic'] - garch_results['aic']:.2f}")
    
    # Create visualization
    if verbose:
        print("\n7. Creating visualization...")
    
    create_visualization(merged_df, corr_results, ols_results)
    
    # Compile results
    results = {
        'sample_size': len(merged_df),
        'correlation': corr_results,
        'ols': {
            'r_squared': ols_results.rsquared,
            'adj_r_squared': ols_results.rsquared_adj,
            'f_statistic': ols_results.fvalue,
            'f_pvalue': ols_results.f_pvalue,
            'coefficients': dict(zip(
                ['const', 'disagreement_scalar', 'disagreement_semantic'],
                ols_results.params
            )),
            'pvalues': dict(zip(
                ['const', 'disagreement_scalar', 'disagreement_semantic'],
                ols_results.pvalues
            ))
        },
        'garch': garch_results,
        'garch_baseline': garch_baseline
    }
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {config.RESULTS_PLOT_PATH}")
    print("=" * 60)
    
    return results


def print_detailed_report(results: Dict) -> None:
    """
    Print a detailed analysis report.
    
    Args:
        results: Dictionary with analysis results
    """
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS REPORT: Agentic Dissonance")
    print("=" * 70)
    
    print(f"\nSample Size: {results['sample_size']} trading days")
    
    print("\n" + "-" * 40)
    print("CORRELATION ANALYSIS")
    print("-" * 40)
    
    corr = results['correlation']
    print(f"\nPearson Correlation:")
    print(f"  Scalar Disagreement:   r = {corr['corr_scalar']:+.4f}  (p = {corr['pval_scalar']:.4f})")
    print(f"  Semantic Divergence:   r = {corr['corr_semantic']:+.4f}  (p = {corr['pval_semantic']:.4f})")
    
    print(f"\nSpearman Correlation:")
    print(f"  Scalar Disagreement:   ρ = {corr['spearman_scalar']:+.4f}  (p = {corr['spearman_pval_scalar']:.4f})")
    print(f"  Semantic Divergence:   ρ = {corr['spearman_semantic']:+.4f}  (p = {corr['spearman_pval_semantic']:.4f})")
    
    print("\n" + "-" * 40)
    print("OLS REGRESSION")
    print("-" * 40)
    
    ols = results['ols']
    print(f"\nModel: Forward_Vol ~ Disagreement_Scalar + Disagreement_Semantic")
    print(f"\nR²:           {ols['r_squared']:.4f}")
    print(f"Adjusted R²:  {ols['adj_r_squared']:.4f}")
    print(f"F-statistic:  {ols['f_statistic']:.4f}")
    print(f"F p-value:    {ols['f_pvalue']:.4f}")
    
    print("\nCoefficients:")
    for name, coef in ols['coefficients'].items():
        pval = ols['pvalues'].get(name, 0)
        sig = '*' if pval < 0.05 else ''
        print(f"  {name:25s}: {coef:+.6f}  (p = {pval:.4f}) {sig}")
    
    if results.get('garch') and results.get('garch_baseline'):
        print("\n" + "-" * 40)
        print("GARCH-X ANALYSIS")
        print("-" * 40)
        
        garch = results['garch']
        baseline = results['garch_baseline']
        
        print(f"\nModel Comparison:")
        print(f"  GARCH(1,1)   AIC: {baseline['aic']:.2f}")
        print(f"  GARCH-X      AIC: {garch['aic']:.2f}")
        print(f"  Improvement:      {baseline['aic'] - garch['aic']:.2f}")
        
        print("\nGARCH-X Parameters:")
        for name, param in garch['params'].items():
            pval = garch['pvalues'].get(name, 0)
            sig = '*' if pval < 0.05 else ''
            print(f"  {name:20s}: {param:+.6f}  (p = {pval:.4f}) {sig}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    # Interpret results
    scalar_sig = corr['pval_scalar'] < 0.05
    semantic_sig = corr['pval_semantic'] < 0.05
    
    if scalar_sig or semantic_sig:
        print("\n✓ SIGNIFICANT RELATIONSHIP FOUND")
        if scalar_sig:
            direction = "positive" if corr['corr_scalar'] > 0 else "negative"
            print(f"\n  Scalar disagreement shows a {direction} relationship with")
            print(f"  forward volatility (r = {corr['corr_scalar']:.4f}, p = {corr['pval_scalar']:.4f}).")
        if semantic_sig:
            direction = "positive" if corr['corr_semantic'] > 0 else "negative"
            print(f"\n  Semantic divergence shows a {direction} relationship with")
            print(f"  forward volatility (r = {corr['corr_semantic']:.4f}, p = {corr['pval_semantic']:.4f}).")
    else:
        print("\n✗ NO SIGNIFICANT RELATIONSHIP FOUND")
        print("\n  Neither disagreement metric shows a statistically significant")
        print("  relationship with forward realized volatility at α = 0.05.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    results = run_full_analysis(verbose=True)
    print_detailed_report(results)
