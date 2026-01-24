"""
Backtest runner for Agentic Dissonance v2.

Executes daily debates across the date range with:
- Multi-asset support
- LLM output caching
- Resume capability
- No look-ahead bias
"""

import os
import time
import pandas as pd
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from tqdm import tqdm

from . import config
from .data_loader import (
    fetch_market_data, fetch_multi_ticker_data, load_market_data,
    get_market_context_for_date, format_context_for_agent, fetch_news,
    get_trading_dates
)
from .debate_engine import DebateRoom
from .disagreement import calculate_all_metrics


def get_last_processed_date(ticker: str = None) -> Optional[str]:
    """
    Get the last processed date from existing results.
    
    Args:
        ticker: Optional ticker filter
        
    Returns:
        Last processed date string or None
    """
    if not os.path.exists(config.DISAGREEMENT_SIGNALS_PATH):
        return None
    
    try:
        df = pd.read_csv(config.DISAGREEMENT_SIGNALS_PATH)
        if ticker and 'ticker' in df.columns:
            df = df[df['ticker'] == ticker]
        if df.empty:
            return None
        return df['date'].max()
    except Exception:
        return None


def run_backtest(
    tickers: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    verbose: bool = True,
    save_intermediate: bool = True,
    resume_from_date: str = None,
    num_rounds: int = None,
    test_mode: bool = False,
    test_days: int = 5
) -> pd.DataFrame:
    """
    Run the full backtest, executing debates for each trading day.
    
    Args:
        tickers: List of ticker symbols (default from config)
        start_date: Start date
        end_date: End date
        verbose: Whether to print progress
        save_intermediate: Whether to save after each day
        resume_from_date: Date to resume from (YYYY-MM-DD)
        num_rounds: Number of debate rounds (default from config)
        test_mode: If True, only process test_days
        test_days: Number of days to process in test mode
        
    Returns:
        DataFrame with disagreement signals
    """
    tickers = tickers or config.TICKER_LIST
    start_date = start_date or config.START_DATE
    end_date = end_date or config.END_DATE
    num_rounds = num_rounds or config.DEBATE_ROUNDS
    
    # If single ticker passed as string, convert to list
    if isinstance(tickers, str):
        tickers = [tickers]
    
    # Load or fetch market data
    if verbose:
        print(f"Loading market data for {len(tickers)} ticker(s)...")
    
    try:
        market_df = load_market_data()
        # Check if all tickers are present
        if 'Ticker' in market_df.columns:
            missing = set(tickers) - set(market_df['Ticker'].unique())
            if missing:
                print(f"Fetching missing tickers: {missing}")
                market_df = fetch_multi_ticker_data(tickers)
    except FileNotFoundError:
        market_df = fetch_multi_ticker_data(tickers)
    
    # Load existing results if resuming
    existing_results = []
    if resume_from_date:
        if os.path.exists(config.DISAGREEMENT_SIGNALS_PATH):
            existing_df = pd.read_csv(config.DISAGREEMENT_SIGNALS_PATH)
            # Filter to keep results before resume date
            existing_df = existing_df[existing_df['date'] < resume_from_date]
            existing_results = existing_df.to_dict('records')
            if verbose:
                print(f"Resuming from {resume_from_date}, keeping {len(existing_results)} existing results")
    
    results = existing_results.copy()
    
    # Process each ticker
    for ticker in tickers:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing {ticker}")
            print(f"{'='*60}")
        
        # Get trading dates for this ticker
        trading_dates = get_trading_dates(market_df, ticker, start_date, end_date)
        
        if resume_from_date:
            resume_dt = pd.to_datetime(resume_from_date)
            trading_dates = [d for d in trading_dates if d >= resume_dt]
        
        if test_mode:
            trading_dates = trading_dates[:test_days]
            if verbose:
                print(f"Test mode: processing {len(trading_dates)} days")
        
        if verbose:
            print(f"Processing {len(trading_dates)} trading days")
        
        # Initialize debate room for this ticker
        room = DebateRoom(num_rounds=num_rounds, ticker=ticker)
        
        # Process each day
        for date in tqdm(trading_dates, desc=f"{ticker}", disable=not verbose):
            try:
                # Get market context (no look-ahead: only uses data <= date)
                market_context = get_market_context_for_date(market_df, date, ticker)
                
                if market_context.get('error'):
                    if verbose:
                        print(f"  Skipping {date}: {market_context['error']}")
                    continue
                
                # Fetch news (also respects no look-ahead)
                news = fetch_news(ticker, date)
                
                # Format context
                context_str = format_context_for_agent(market_context, news, ticker)
                
                # Run debate
                debate_result = room.run_daily_debate(context_str, date, verbose=False)
                
                # Build result row
                row = {
                    "date": debate_result["date"],
                    "ticker": ticker,
                    "disagreement_conf": debate_result["metrics"]["disagreement_conf"],
                    "mean_score": debate_result["metrics"]["mean_score"],
                    "avg_confidence": debate_result["metrics"]["avg_confidence"],
                    "num_rounds": num_rounds
                }
                
                # Add individual agent data
                signal = debate_result.get("disagreement_signal", {})
                for key in ["score_fundamental", "score_sentiment", "score_technical", "score_macro",
                           "confidence_fundamental", "confidence_sentiment", "confidence_technical", "confidence_macro"]:
                    row[key] = signal.get(key, 0.0)
                
                results.append(row)
                
                # Save intermediate results
                if save_intermediate:
                    _save_results(results)
                
            except Exception as e:
                print(f"Error processing {ticker} on {date}: {e}")
                continue
    
    # Save final results
    df = _save_results(results)
    
    if verbose:
        print(f"\nBacktest complete!")
        print(f"Processed {len(results)} days across {len(tickers)} ticker(s)")
        print(f"Results saved to {config.DISAGREEMENT_SIGNALS_PATH}")
    
    return df


def _save_results(results: List[Dict]) -> pd.DataFrame:
    """
    Save results to CSV.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        DataFrame of results
    """
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Sort by ticker and date
    if 'ticker' in df.columns and 'date' in df.columns:
        df = df.sort_values(['ticker', 'date'])
    elif 'date' in df.columns:
        df = df.sort_values('date')
    
    df.to_csv(config.DISAGREEMENT_SIGNALS_PATH, index=False)
    return df


def run_quick_test(
    ticker: str = None,
    num_days: int = 3,
    num_rounds: int = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run a quick test with limited days for validation.
    
    Args:
        ticker: Ticker to test (default: first from config)
        num_days: Number of days to process
        num_rounds: Debate rounds (default from config)
        verbose: Whether to print progress
        
    Returns:
        DataFrame with results
    """
    ticker = ticker or config.TICKER_LIST[0]
    
    if verbose:
        print(f"Running quick test: {ticker}, {num_days} days, {num_rounds or config.DEBATE_ROUNDS} rounds")
    
    return run_backtest(
        tickers=[ticker],
        verbose=verbose,
        save_intermediate=True,
        num_rounds=num_rounds,
        test_mode=True,
        test_days=num_days
    )


def run_rounds_comparison(
    ticker: str = None,
    num_days: int = 5,
    rounds_list: List[int] = None,
    verbose: bool = True
) -> Dict[int, pd.DataFrame]:
    """
    Compare different debate round configurations.
    
    Args:
        ticker: Ticker to test
        num_days: Days to process per configuration
        rounds_list: List of round counts to test
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping round count to results DataFrame
    """
    ticker = ticker or config.TICKER_LIST[0]
    rounds_list = rounds_list or [2, 3, 4]
    
    results = {}
    
    for num_rounds in rounds_list:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing {num_rounds} debate rounds")
            print(f"{'='*60}")
        
        start_time = time.time()
        
        df = run_backtest(
            tickers=[ticker],
            num_rounds=num_rounds,
            test_mode=True,
            test_days=num_days,
            verbose=False,
            save_intermediate=False
        )
        
        elapsed = time.time() - start_time
        
        results[num_rounds] = {
            "data": df,
            "avg_disagreement": df["disagreement_conf"].mean() if len(df) > 0 else 0,
            "avg_confidence": df["avg_confidence"].mean() if len(df) > 0 else 0,
            "runtime_seconds": elapsed
        }
        
        if verbose:
            print(f"  Avg Disagreement: {results[num_rounds]['avg_disagreement']:.4f}")
            print(f"  Avg Confidence: {results[num_rounds]['avg_confidence']:.4f}")
            print(f"  Runtime: {elapsed:.2f}s")
    
    # Print comparison table
    if verbose:
        print(f"\n{'='*60}")
        print("ROUNDS COMPARISON")
        print(f"{'='*60}")
        print(f"{'Rounds':<10} {'Disagreement':<15} {'Confidence':<12} {'Runtime':<10}")
        print("-" * 50)
        for num_rounds, data in results.items():
            print(f"{num_rounds:<10} {data['avg_disagreement']:<15.4f} "
                  f"{data['avg_confidence']:<12.4f} {data['runtime_seconds']:<10.2f}s")
    
    return results


def main():
    """Command-line interface for backtest."""
    parser = argparse.ArgumentParser(
        description="Run Agentic Dissonance v2 backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.backtest --test --days 3
  python -m src.backtest --resume 2024-06-01
  python -m src.backtest --ticker AAPL --rounds 2
  python -m src.backtest --compare-rounds
"""
    )
    
    parser.add_argument("--test", action="store_true", 
                        help="Run quick test mode")
    parser.add_argument("--days", type=int, default=5, 
                        help="Number of days in test mode")
    parser.add_argument("--resume", type=str, metavar="YYYY-MM-DD",
                        help="Resume from specified date")
    parser.add_argument("--ticker", type=str, 
                        help="Single ticker to process")
    parser.add_argument("--rounds", type=int, 
                        help="Number of debate rounds")
    parser.add_argument("--compare-rounds", action="store_true",
                        help="Compare 2/3/4 round configurations")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if args.compare_rounds:
        results = run_rounds_comparison(
            ticker=args.ticker,
            num_days=args.days,
            verbose=verbose
        )
    elif args.test:
        df = run_quick_test(
            ticker=args.ticker,
            num_days=args.days,
            num_rounds=args.rounds,
            verbose=verbose
        )
        if verbose:
            print("\nTest Results:")
            print(df)
    else:
        tickers = [args.ticker] if args.ticker else None
        df = run_backtest(
            tickers=tickers,
            resume_from_date=args.resume,
            num_rounds=args.rounds,
            verbose=verbose
        )
        if verbose:
            print("\nSample Results:")
            print(df.head(10))


if __name__ == "__main__":
    main()
