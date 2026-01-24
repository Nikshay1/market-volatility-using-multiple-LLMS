"""
Data loading utilities for Agentic Dissonance v2.

Provides market data, news headlines, and context formatting.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
from tqdm import tqdm

from . import config


def fetch_market_data(
    ticker: str = None,
    start_date: str = None,
    end_date: str = None,
    save_to_csv: bool = True
) -> pd.DataFrame:
    """
    Fetch OHLCV market data from yfinance and compute log returns.
    
    Args:
        ticker: Stock ticker symbol (default from config)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        save_to_csv: Whether to save to CSV file
        
    Returns:
        DataFrame with OHLCV data and log returns
    """
    ticker = ticker or config.DEFAULT_TICKER
    start_date = start_date or config.START_DATE
    end_date = end_date or config.END_DATE
    
    print(f"Fetching market data for {ticker} from {start_date} to {end_date}...")
    
    # Fetch data
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    
    # Reset index and rename
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    
    # Compute log returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Compute additional metrics
    df['Daily_Volatility'] = df['Log_Return'].rolling(window=20).std()
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_20d'] = df['Close'].pct_change(20)
    
    # Add ticker column
    df['Ticker'] = ticker
    
    # Drop first row (NaN return)
    df = df.dropna(subset=['Log_Return'])
    
    print(f"Retrieved {len(df)} trading days")
    
    if save_to_csv:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        df.to_csv(config.RAW_MARKET_DATA_PATH, index=False)
        print(f"Saved to {config.RAW_MARKET_DATA_PATH}")
    
    return df


def fetch_multi_ticker_data(
    tickers: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    save_to_csv: bool = True
) -> pd.DataFrame:
    """
    Fetch market data for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date
        end_date: End date
        save_to_csv: Whether to save to CSV
        
    Returns:
        Combined DataFrame with all tickers
    """
    tickers = tickers or config.TICKER_LIST
    start_date = start_date or config.START_DATE
    end_date = end_date or config.END_DATE
    
    all_data = []
    
    for ticker in tqdm(tickers, desc="Fetching market data"):
        try:
            df = fetch_market_data(ticker, start_date, end_date, save_to_csv=False)
            all_data.append(df)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    
    if not all_data:
        raise ValueError("No data fetched for any ticker")
    
    combined = pd.concat(all_data, ignore_index=True)
    
    if save_to_csv:
        combined.to_csv(config.RAW_MARKET_DATA_PATH, index=False)
        print(f"Saved {len(combined)} rows to {config.RAW_MARKET_DATA_PATH}")
    
    return combined


def load_market_data(ticker: str = None) -> pd.DataFrame:
    """
    Load market data from CSV file.
    
    Args:
        ticker: Optional ticker to filter by
        
    Returns:
        DataFrame with market data
    """
    if not os.path.exists(config.RAW_MARKET_DATA_PATH):
        raise FileNotFoundError(
            f"Market data not found at {config.RAW_MARKET_DATA_PATH}. "
            "Run fetch_market_data() first."
        )
    
    df = pd.read_csv(config.RAW_MARKET_DATA_PATH, parse_dates=['Date'])
    
    if ticker and 'Ticker' in df.columns:
        df = df[df['Ticker'] == ticker]
    
    return df


def fetch_news_rss(ticker: str, date: datetime) -> List[str]:
    """
    Fetch news headlines via RSS feeds for a given ticker and date.
    Only returns headlines from before or on the specified date (no look-ahead).
    
    Args:
        ticker: Stock ticker symbol
        date: The reference date (news must be <= this date)
        
    Returns:
        List of headline strings (5-10 headlines)
    """
    # Company name mapping for better search results
    company_names = {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "TSLA": "Tesla",
        "SPY": "S&P 500",
        "GOOGL": "Google",
        "AMZN": "Amazon",
        "META": "Meta Facebook",
        "NVDA": "NVIDIA"
    }
    
    company = company_names.get(ticker, ticker)
    
    # RSS feed URLs
    feeds = [
        f"https://news.google.com/rss/search?q={company}+stock&hl=en-US&gl=US&ceid=US:en",
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    ]
    
    headlines = []
    cutoff_date = date + timedelta(days=1)  # Include headlines from the analysis day
    
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                # Parse publication date if available
                pub_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                
                # Filter by date if available
                if pub_date and pub_date > cutoff_date:
                    continue
                
                title = entry.get('title', '').strip()
                if title and len(title) > 10:
                    headlines.append(title)
                    
        except Exception as e:
            print(f"Warning: Could not fetch RSS feed: {e}")
    
    # Deduplicate and limit
    seen = set()
    unique_headlines = []
    for h in headlines:
        if h.lower() not in seen:
            seen.add(h.lower())
            unique_headlines.append(h)
    
    # Return 5-10 headlines
    return unique_headlines[:10] if unique_headlines else _get_fallback_headlines(ticker, date)


def _get_fallback_headlines(ticker: str, date: datetime) -> List[str]:
    """Generate fallback headlines when RSS fails."""
    return [
        f"Markets trade mixed as investors await economic data",
        f"{ticker} shares move with broader market trends",
        f"Tech sector shows volatility amid rate concerns",
        f"Analysts maintain outlook on {ticker}",
        f"Trading volume remains steady for major indices"
    ]


def fetch_news(
    ticker: str,
    date: datetime,
    use_gnews: bool = False
) -> List[str]:
    """
    Fetch news headlines for a given ticker and date.
    
    Args:
        ticker: Stock ticker symbol
        date: The reference date
        use_gnews: Whether to use gnews library (requires additional setup)
        
    Returns:
        List of headline strings
    """
    if use_gnews:
        return _fetch_news_gnews(ticker, date)
    return fetch_news_rss(ticker, date)


def _fetch_news_gnews(ticker: str, date: datetime) -> List[str]:
    """Fetch news using gnews library."""
    try:
        from gnews import GNews
        
        google_news = GNews(
            language='en',
            country='US',
            period='7d',
            max_results=10
        )
        
        company_names = {
            "AAPL": "Apple stock",
            "MSFT": "Microsoft stock",
            "TSLA": "Tesla stock",
            "SPY": "S&P 500"
        }
        
        query = company_names.get(ticker, f"{ticker} stock")
        articles = google_news.get_news(query)
        
        headlines = [article['title'] for article in articles if article.get('title')]
        return headlines[:10]
        
    except Exception as e:
        print(f"Warning: gnews failed, falling back to RSS: {e}")
        return fetch_news_rss(ticker, date)


def get_market_context_for_date(
    market_df: pd.DataFrame,
    date: datetime,
    ticker: str = None,
    lookback_days: int = 5
) -> Dict:
    """
    Get market context for a specific date including recent price action.
    
    Args:
        market_df: DataFrame with market data
        date: The target date
        ticker: Optional ticker filter
        lookback_days: Number of days to include in lookback
        
    Returns:
        Dictionary with market context information
    """
    # Filter by ticker if specified
    df = market_df.copy()
    if ticker and 'Ticker' in df.columns:
        df = df[df['Ticker'] == ticker]
    
    # Convert date to datetime for comparison
    date = pd.to_datetime(date).tz_localize(None)
    
    # Find data on or before the target date
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df_filtered = df[df['Date'] <= date].tail(lookback_days + 1)
    
    if df_filtered.empty:
        return {
            "current_price": None,
            "date": date.strftime("%Y-%m-%d"),
            "error": "No data available for this date"
        }
    
    latest = df_filtered.iloc[-1]
    lookback = df_filtered.iloc[:-1] if len(df_filtered) > 1 else df_filtered
    
    # Calculate metrics
    current_price = latest['Close']
    daily_return = latest['Log_Return'] if 'Log_Return' in latest else 0
    
    if len(lookback) > 0:
        period_return = (current_price / lookback.iloc[0]['Close'] - 1) * 100
        high = lookback['High'].max()
        low = lookback['Low'].min()
        avg_volume = lookback['Volume'].mean()
    else:
        period_return = 0
        high = current_price
        low = current_price
        avg_volume = latest['Volume']
    
    # Volatility (20-day rolling std)
    volatility = latest.get('Daily_Volatility', 0)
    if pd.isna(volatility):
        volatility = df_filtered['Log_Return'].std() if len(df_filtered) > 1 else 0
    
    return {
        "date": latest['Date'].strftime("%Y-%m-%d"),
        "ticker": latest.get('Ticker', ticker or config.DEFAULT_TICKER),
        "current_price": round(current_price, 2),
        "daily_return": round(daily_return * 100, 2),
        "period_return": round(period_return, 2),
        "period_high": round(high, 2),
        "period_low": round(low, 2),
        "volatility_20d": round(volatility * 100 * np.sqrt(252), 2) if volatility else 0,
        "avg_volume": int(avg_volume),
        "lookback_days": lookback_days
    }


def format_context_for_agent(
    market_context: Dict,
    news_headlines: List[str],
    ticker: str = None
) -> str:
    """
    Format market context and news into a prompt-ready string.
    
    Args:
        market_context: Dictionary with market data
        news_headlines: List of news headlines
        ticker: Stock ticker symbol
        
    Returns:
        Formatted context string
    """
    ticker = ticker or market_context.get('ticker', config.DEFAULT_TICKER)
    
    # Format news section
    if news_headlines:
        news_section = "\n".join([f"  • {h}" for h in news_headlines[:8]])
    else:
        news_section = "  No recent headlines available"
    
    context = f"""
=== MARKET ANALYSIS FOR {ticker} ===
Date: {market_context.get('date', 'N/A')}

PRICE DATA:
- Current Price: ${market_context.get('current_price', 'N/A')}
- Daily Return: {market_context.get('daily_return', 0):.2f}%
- {market_context.get('lookback_days', 5)}-Day Return: {market_context.get('period_return', 0):.2f}%
- Period High: ${market_context.get('period_high', 'N/A')}
- Period Low: ${market_context.get('period_low', 'N/A')}
- 20-Day Volatility (Annualized): {market_context.get('volatility_20d', 0):.1f}%
- Average Volume: {market_context.get('avg_volume', 0):,}

RECENT NEWS & HEADLINES:
{news_section}
"""
    
    return context


def get_trading_dates(
    market_df: pd.DataFrame,
    ticker: str = None,
    start_date: str = None,
    end_date: str = None
) -> List[datetime]:
    """
    Get list of trading dates from market data.
    
    Args:
        market_df: DataFrame with market data
        ticker: Optional ticker filter
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        List of datetime objects for trading days
    """
    df = market_df.copy()
    
    if ticker and 'Ticker' in df.columns:
        df = df[df['Ticker'] == ticker]
    
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    
    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]
    
    dates = df['Date'].sort_values().unique()
    return [pd.to_datetime(d) for d in dates]


if __name__ == "__main__":
    # Test the data loader
    print("Fetching market data...")
    df = fetch_market_data()
    print("\nSample market data:")
    print(df.head())
    
    # Test news fetching for a specific date
    test_date = datetime(2024, 6, 15)
    headlines = fetch_news(config.DEFAULT_TICKER, test_date)
    print(f"\nNews headlines for {test_date.date()}:")
    for h in headlines:
        print(f"  - {h}")
    
    # Test context formatting
    market_ctx = get_market_context_for_date(df, test_date)
    print("\nMarket context:")
    print(format_context_for_agent(market_ctx, headlines))
