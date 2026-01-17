"""
Data loading utilities for market data and news headlines.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from tqdm import tqdm

from . import config


def fetch_market_data(
    ticker: str = config.TICKER,
    start_date: str = config.START_DATE,
    end_date: str = config.END_DATE,
    save_to_csv: bool = True
) -> pd.DataFrame:
    """
    Fetch OHLCV market data from yfinance and compute log returns.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        save_to_csv: Whether to save the data to CSV
        
    Returns:
        DataFrame with OHLCV data and log returns
    """
    print(f"Fetching market data for {ticker} from {start_date} to {end_date}...")
    
    # Fetch data from yfinance
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data retrieved for {ticker}")
    
    # Reset index to make Date a column
    df = df.reset_index()
    
    # Ensure Date column is datetime and remove timezone
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    
    # Compute log returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Drop the first row with NaN log return
    df = df.dropna(subset=['Log_Return'])
    
    # Select and rename columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Log_Return']]
    
    # Save to CSV
    if save_to_csv:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        df.to_csv(config.RAW_MARKET_DATA_PATH, index=False)
        print(f"Market data saved to {config.RAW_MARKET_DATA_PATH}")
    
    print(f"Retrieved {len(df)} trading days of data")
    return df


def load_market_data() -> pd.DataFrame:
    """
    Load market data from CSV file.
    
    Returns:
        DataFrame with market data
    """
    if not os.path.exists(config.RAW_MARKET_DATA_PATH):
        raise FileNotFoundError(
            f"Market data file not found at {config.RAW_MARKET_DATA_PATH}. "
            "Please run fetch_market_data() first."
        )
    
    df = pd.read_csv(config.RAW_MARKET_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
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
    headlines = []
    
    # RSS feed sources for financial news
    rss_feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/reuters/companyNews",
    ]
    
    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:  # Limit per feed
                # Check if entry has a published date
                pub_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    pub_date = datetime(*entry.updated_parsed[:6])
                
                # If we can't determine date, skip (strict no look-ahead)
                if pub_date is None:
                    continue
                    
                # Only include if published before or on the reference date
                if pub_date.date() <= date.date():
                    title = entry.title if hasattr(entry, 'title') else ""
                    if title and ticker.upper() in title.upper():
                        headlines.append(title)
                    elif title and len(headlines) < 10:
                        # Include general business news if we don't have enough
                        headlines.append(title)
                        
        except Exception as e:
            # Silently continue if a feed fails
            continue
    
    # Deduplicate and limit to 10 headlines
    headlines = list(dict.fromkeys(headlines))[:10]
    
    return headlines


def fetch_news(
    ticker: str,
    date: datetime,
    use_gnews: bool = False
) -> List[str]:
    """
    Fetch news headlines for a given ticker and date.
    Uses RSS feeds by default; can use gnews if specified.
    
    Args:
        ticker: Stock ticker symbol
        date: The reference date
        use_gnews: Whether to use gnews library (requires additional setup)
        
    Returns:
        List of headline strings
    """
    if use_gnews:
        return _fetch_news_gnews(ticker, date)
    else:
        return fetch_news_rss(ticker, date)


def _fetch_news_gnews(ticker: str, date: datetime) -> List[str]:
    """
    Fetch news using gnews library.
    
    Args:
        ticker: Stock ticker symbol
        date: The reference date
        
    Returns:
        List of headline strings
    """
    try:
        from gnews import GNews
        
        # Initialize GNews with date constraints
        google_news = GNews(
            language='en',
            country='US',
            max_results=10,
            start_date=(date - timedelta(days=1)).date(),
            end_date=date.date()
        )
        
        # Search for news about the ticker
        articles = google_news.get_news(ticker)
        
        headlines = [article['title'] for article in articles if article.get('title')]
        return headlines[:10]
        
    except ImportError:
        print("gnews not installed. Falling back to RSS feeds.")
        return fetch_news_rss(ticker, date)
    except Exception as e:
        print(f"Error fetching news from gnews: {e}. Falling back to RSS feeds.")
        return fetch_news_rss(ticker, date)


def get_market_context_for_date(
    market_df: pd.DataFrame,
    date: datetime,
    lookback_days: int = 5
) -> Dict:
    """
    Get market context for a specific date including recent price action.
    
    Args:
        market_df: DataFrame with market data
        date: The target date
        lookback_days: Number of days to include in lookback
        
    Returns:
        Dictionary with market context information
    """
    # Filter data up to and including the date
    mask = market_df['Date'] <= date
    available_data = market_df[mask].tail(lookback_days + 1)
    
    if available_data.empty:
        return {
            "date": date.strftime("%Y-%m-%d"),
            "error": "No market data available for this date"
        }
    
    latest = available_data.iloc[-1]
    
    # Calculate various metrics
    context = {
        "date": date.strftime("%Y-%m-%d"),
        "current_price": round(latest['Close'], 2),
        "daily_return_pct": round(latest['Log_Return'] * 100, 2),
        "volume": int(latest['Volume']),
    }
    
    # Add lookback statistics if we have enough data
    if len(available_data) > 1:
        returns = available_data['Log_Return'].values
        context["avg_return_5d"] = round(np.mean(returns) * 100, 2)
        context["volatility_5d"] = round(np.std(returns) * 100, 2)
        context["price_change_5d_pct"] = round(
            (available_data.iloc[-1]['Close'] / available_data.iloc[0]['Close'] - 1) * 100, 2
        )
        context["max_price_5d"] = round(available_data['High'].max(), 2)
        context["min_price_5d"] = round(available_data['Low'].min(), 2)
    
    return context


def format_context_for_agent(
    market_context: Dict,
    news_headlines: List[str],
    ticker: str = config.TICKER
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
    context_parts = [
        f"=== Market Analysis for {ticker} on {market_context.get('date', 'N/A')} ===",
        "",
        "MARKET DATA:",
        f"- Current Price: ${market_context.get('current_price', 'N/A')}",
        f"- Daily Return: {market_context.get('daily_return_pct', 'N/A')}%",
        f"- Trading Volume: {market_context.get('volume', 'N/A'):,}" if market_context.get('volume') else "- Trading Volume: N/A",
    ]
    
    # Add 5-day lookback if available
    if 'avg_return_5d' in market_context:
        context_parts.extend([
            "",
            "5-DAY LOOKBACK:",
            f"- Average Daily Return: {market_context.get('avg_return_5d', 'N/A')}%",
            f"- Volatility (Std Dev): {market_context.get('volatility_5d', 'N/A')}%",
            f"- Cumulative Price Change: {market_context.get('price_change_5d_pct', 'N/A')}%",
            f"- 5-Day High: ${market_context.get('max_price_5d', 'N/A')}",
            f"- 5-Day Low: ${market_context.get('min_price_5d', 'N/A')}",
        ])
    
    # Add news headlines
    context_parts.extend([
        "",
        "RECENT NEWS HEADLINES:",
    ])
    
    if news_headlines:
        for i, headline in enumerate(news_headlines[:10], 1):
            context_parts.append(f"{i}. {headline}")
    else:
        context_parts.append("- No recent headlines available")
    
    return "\n".join(context_parts)


if __name__ == "__main__":
    # Test the data loader
    df = fetch_market_data()
    print("\nSample market data:")
    print(df.head())
    
    # Test news fetching for a specific date
    test_date = datetime(2024, 6, 15)
    headlines = fetch_news(config.TICKER, test_date)
    print(f"\nNews headlines for {test_date.date()}:")
    for h in headlines:
        print(f"  - {h}")
