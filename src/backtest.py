"""
Backtest runner for daily debate execution across the date range.
"""

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from tqdm import tqdm

from . import config
from .data_loader import (
    fetch_market_data, load_market_data,
    get_market_context_for_date, format_context_for_agent, fetch_news
)
from .debate_engine import DebateRoom


def get_trading_dates(market_df: pd.DataFrame) -> List[datetime]:
    """
    Get list of trading dates from market data.
    
    Args:
        market_df: DataFrame with market data
        
    Returns:
        List of datetime objects for trading days
    """
    dates = pd.to_datetime(market_df['Date']).dt.to_pydatetime()
    return list(dates)


def run_backtest(
    start_date: str = None,
    end_date: str = None,
    ticker: str = None,
    verbose: bool = True,
    save_intermediate: bool = True,
    resume_from_date: str = None
) -> pd.DataFrame:
    """
    Run the full backtest, executing debates for each trading day.
    
    Args:
        start_date: Start date (default from config)
        end_date: End date (default from config)
        ticker: Ticker symbol (default from config)
        verbose: Whether to print progress
        save_intermediate: Save results after each day
        resume_from_date: Resume from a specific date (for interrupted runs)
        
    Returns:
        DataFrame with disagreement signals
    """
    # Use defaults from config
    start_date = start_date or config.START_DATE
    end_date = end_date or config.END_DATE
    ticker = ticker or config.TICKER
    
    print(f"=" * 60)
    print(f"BACKTEST: {ticker}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Debate Rounds: {config.DEBATE_ROUNDS}")
    print(f"=" * 60)
    
    # Load or fetch market data
    try:
        market_df = load_market_data()
        print(f"Loaded existing market data: {len(market_df)} days")
    except FileNotFoundError:
        print("Fetching market data...")
        market_df = fetch_market_data(ticker, start_date, end_date)
    
    # Get trading dates
    trading_dates = get_trading_dates(market_df)
    print(f"Trading days to process: {len(trading_dates)}")
    
    # Initialize debate room
    debate_room = DebateRoom()
    
    # Load existing results if resuming
    results = []
    processed_dates = set()
    
    if resume_from_date and os.path.exists(config.DISAGREEMENT_SIGNALS_PATH):
        existing_df = pd.read_csv(config.DISAGREEMENT_SIGNALS_PATH)
        existing_df['date'] = pd.to_datetime(existing_df['date'])
        
        resume_date = datetime.strptime(resume_from_date, "%Y-%m-%d")
        
        # Keep results before resume date
        for _, row in existing_df.iterrows():
            if row['date'] < resume_date:
                results.append(row.to_dict())
                processed_dates.add(row['date'].strftime("%Y-%m-%d"))
        
        print(f"Resuming from {resume_from_date}. Loaded {len(results)} previous results.")
    
    # Process each trading day
    for date in tqdm(trading_dates, desc="Processing days", disable=not verbose):
        date_str = date.strftime("%Y-%m-%d")
        
        # Skip if already processed
        if date_str in processed_dates:
            continue
        
        try:
            # Get market context up to (and including) this date
            market_context = get_market_context_for_date(market_df, date)
            
            if 'error' in market_context:
                print(f"\nSkipping {date_str}: {market_context['error']}")
                continue
            
            # Fetch news (respecting no look-ahead)
            news_headlines = fetch_news(ticker, date)
            
            # Format context for agents
            context_str = format_context_for_agent(market_context, news_headlines, ticker)
            
            # Run debate
            debate_results = debate_room.run_daily_debate(
                context_str, date, verbose=False
            )
            
            # Extract results
            row = {
                'date': date_str,
                'disagreement_scalar': debate_results['metrics']['disagreement_scalar'],
                'disagreement_semantic': debate_results['metrics']['disagreement_semantic'],
                'score_fundamental': debate_results['metrics']['score_fundamental'],
                'score_sentiment': debate_results['metrics']['score_sentiment'],
            }
            results.append(row)
            processed_dates.add(date_str)
            
            # Save intermediate results
            if save_intermediate and len(results) % 5 == 0:
                _save_results(results)
                if verbose:
                    tqdm.write(f"Checkpoint saved: {len(results)} days processed")
            
            # Rate limiting between days
            time.sleep(config.RATE_LIMIT_DELAY * 2)
            
        except Exception as e:
            print(f"\nError processing {date_str}: {e}")
            # Save what we have before continuing
            if save_intermediate and results:
                _save_results(results)
            # Wait longer before retrying
            time.sleep(config.RATE_LIMIT_DELAY * 5)
            continue
    
    # Final save
    results_df = _save_results(results)
    
    print(f"\n{'=' * 60}")
    print(f"BACKTEST COMPLETE")
    print(f"Total days processed: {len(results)}")
    print(f"Results saved to: {config.DISAGREEMENT_SIGNALS_PATH}")
    print(f"{'=' * 60}")
    
    return results_df


def _save_results(results: List[Dict]) -> pd.DataFrame:
    """
    Save results to CSV.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        DataFrame of results
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    df = pd.DataFrame(results)
    df = df.sort_values('date')
    df.to_csv(config.DISAGREEMENT_SIGNALS_PATH, index=False)
    
    return df


def run_quick_test(num_days: int = 5) -> pd.DataFrame:
    """
    Run a quick test with limited days for validation.
    
    Args:
        num_days: Number of days to process
        
    Returns:
        DataFrame with results
    """
    print(f"Running quick test with {num_days} days...")
    
    # Load market data
    try:
        market_df = load_market_data()
    except FileNotFoundError:
        market_df = fetch_market_data()
    
    # Get first N trading dates
    trading_dates = get_trading_dates(market_df)[:num_days]
    
    # Initialize debate room
    debate_room = DebateRoom()
    
    results = []
    
    for date in tqdm(trading_dates, desc="Test run"):
        date_str = date.strftime("%Y-%m-%d")
        
        try:
            market_context = get_market_context_for_date(market_df, date)
            news_headlines = fetch_news(config.TICKER, date)
            context_str = format_context_for_agent(market_context, news_headlines)
            
            debate_results = debate_room.run_daily_debate(context_str, date, verbose=True)
            
            results.append({
                'date': date_str,
                'disagreement_scalar': debate_results['metrics']['disagreement_scalar'],
                'disagreement_semantic': debate_results['metrics']['disagreement_semantic'],
                'score_fundamental': debate_results['metrics']['score_fundamental'],
                'score_sentiment': debate_results['metrics']['score_sentiment'],
            })
            
            time.sleep(config.RATE_LIMIT_DELAY * 2)
            
        except Exception as e:
            print(f"Error on {date_str}: {e}")
            continue
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run debate backtest")
    parser.add_argument("--test", action="store_true", help="Run quick test mode")
    parser.add_argument("--days", type=int, default=5, help="Days for test mode")
    parser.add_argument("--resume", type=str, help="Resume from date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    if args.test:
        df = run_quick_test(args.days)
        print("\nTest Results:")
        print(df)
    else:
        df = run_backtest(resume_from_date=args.resume)
        print("\nSample Results:")
        print(df.head(10))
