"""
Data loading utilities for Agentic Dissonance v2.

Provides market data, news headlines, and context formatting.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Tuple
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
    use_gnews: bool = False,
    return_metadata: bool = False
) -> Union[List[str], Tuple[List[str], Dict[str, Union[int, str, bool, Dict[str, int]]]]]:
    """
    Fetch historical news from local CSV to avoid look-ahead bias.
    
    THE "TIME MACHINE" FIX:
    Live RSS feeds only show today's news. This function looks up a local file
    (data/historical_news.csv) and filters for headlines that appeared on or
    before the backtest date.
    
    Dataset assumptions:
    - historical_news.csv must exist and include Date/Ticker/Headline columns.
    - Lookback window is configurable (3-7 trading days recommended).
    - Headline deduplication removes exact and near-duplicate variants.
    
    Args:
        ticker: Stock ticker symbol
        date: The reference date (Friday - news must be <= this date)
        use_gnews: Ignored - kept for compatibility
        return_metadata: If True, returns (headlines, metadata)
        
    Returns:
        List of headline strings, or (headlines, metadata) if return_metadata=True
    """
    lookback_days = _validated_news_lookback_days(config.NEWS_LOOKBACK_DAYS)
    min_headlines = max(1, config.NEWS_MIN_HEADLINES_PER_DAY)
    similarity_threshold = config.NEWS_DEDUP_SIMILARITY_THRESHOLD

    df = load_historical_news(strict=True)

    target_date = pd.to_datetime(date).date()
    start_date = target_date - timedelta(days=lookback_days + 3)

    mask = (
        (df['Ticker'] == ticker)
        & (df['Date'].dt.date <= target_date)
        & (df['Date'].dt.date >= start_date)
    )
    scoped = df[mask].copy().sort_values('Date', ascending=False)
    if scoped.empty:
        metadata = {
            "is_low_information": True,
            "headline_count": 0,
            "source_counts": {},
            "date_counts": {},
            "lookback_days": lookback_days,
            "threshold": min_headlines,
            "ticker": ticker,
            "date": target_date.isoformat()
        }
        _log_headline_coverage(ticker, target_date, metadata)
        return ([], metadata) if return_metadata else []

    recent_dates = sorted(scoped['Date'].dt.date.unique(), reverse=True)[:lookback_days]
    scoped = scoped[scoped['Date'].dt.date.isin(recent_dates)].copy()

    deduped_rows = _deduplicate_news_rows(scoped, similarity_threshold)
    headlines = [row['Headline'] for row in deduped_rows]

    day_counts = {}
    for row in deduped_rows:
        day_key = row['Date'].date().isoformat()
        day_counts[day_key] = day_counts.get(day_key, 0) + 1

    source_counts: Dict[str, int] = {}
    for row in deduped_rows:
        source = str(row.get('Source', 'unknown')).strip() or 'unknown'
        source_counts[source] = source_counts.get(source, 0) + 1

    low_info_days = [d for d, c in day_counts.items() if c < min_headlines]
    metadata = {
        "is_low_information": bool(low_info_days),
        "headline_count": len(headlines),
        "source_counts": source_counts,
        "date_counts": day_counts,
        "low_information_days": low_info_days,
        "lookback_days": lookback_days,
        "threshold": min_headlines,
        "ticker": ticker,
        "date": target_date.isoformat()
    }

    _log_headline_coverage(ticker, target_date, metadata)

    if return_metadata:
        return headlines, metadata
    return headlines


def load_historical_news(strict: bool = True) -> pd.DataFrame:
    """Load and validate historical news dataset."""
    csv_path = os.path.join(config.DATA_DIR, "historical_news.csv")
    if not os.path.exists(csv_path):
        message = (
            f"Required historical news file is missing: {csv_path}. "
            "Run scripts/import_kaggle.py before research backtests."
        )
        if strict:
            raise FileNotFoundError(message)
        print(f"Warning: {message}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    required_cols = {"Date", "Ticker", "Headline"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            "historical_news.csv is missing required columns: "
            f"{sorted(missing_cols)}"
        )

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if df['Date'].isna().any():
        raise ValueError("historical_news.csv contains invalid Date values.")

    df['Ticker'] = df['Ticker'].astype(str).str.upper().str.strip()
    df['Headline'] = df['Headline'].astype(str).str.strip()
    if 'Source' not in df.columns:
        df['Source'] = 'unknown'
    else:
        df['Source'] = df['Source'].fillna('unknown').astype(str).str.strip().replace('', 'unknown')

    return df


def _validated_news_lookback_days(lookback_days: int) -> int:
    """Validate lookback days for historical news context windows."""
    if 3 <= lookback_days <= 7:
        return lookback_days
    raise ValueError(
        f"NEWS_LOOKBACK_DAYS must be between 3 and 7 (received {lookback_days})."
    )


def _deduplicate_news_rows(rows: pd.DataFrame, similarity_threshold: float) -> List[Dict]:
    """Deduplicate by exact match and near-duplicate semantic string similarity."""
    deduped: List[Dict] = []
    seen_exact = set()

    for _, row in rows.iterrows():
        headline = str(row['Headline']).strip()
        normalized = headline.lower()
        if not normalized or normalized in seen_exact:
            continue

        is_similar = any(
            SequenceMatcher(None, normalized, existing['Headline'].lower()).ratio() >= similarity_threshold
            for existing in deduped
        )
        if is_similar:
            continue

        seen_exact.add(normalized)
        deduped.append({
            "Date": row['Date'],
            "Ticker": row['Ticker'],
            "Headline": headline,
            "Source": row.get('Source', 'unknown')
        })

    return deduped


def _log_headline_coverage(ticker: str, date: datetime.date, metadata: Dict[str, Union[int, str, bool, Dict[str, int]]]) -> None:
    """Append headline coverage stats for methods appendix."""
    path = config.HEADLINE_COVERAGE_STATS_PATH
    rows = []
    date_counts = metadata.get('date_counts', {}) if isinstance(metadata, dict) else {}
    if date_counts:
        for news_date, count in date_counts.items():
            rows.append({
                "asof_date": date.isoformat(),
                "ticker": ticker,
                "news_date": news_date,
                "headline_count": count,
                "is_low_information": metadata.get('is_low_information', False)
            })
    else:
        rows.append({
            "asof_date": date.isoformat(),
            "ticker": ticker,
            "news_date": "",
            "headline_count": 0,
            "is_low_information": True
        })

    coverage_df = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    coverage_df.to_csv(path, mode='a', header=write_header, index=False)


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
    ticker: str = None,
    news_metadata: Optional[Dict] = None
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

    news_metadata = news_metadata or {}
    source_counts = news_metadata.get('source_counts', {})
    date_counts = news_metadata.get('date_counts', {})
    source_summary = ", ".join([f"{k}:{v}" for k, v in sorted(source_counts.items())]) or "N/A"
    date_summary = ", ".join([f"{k}:{v}" for k, v in sorted(date_counts.items())]) or "N/A"
    
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
- Headline Count: {news_metadata.get('headline_count', len(news_headlines))}
- Source Coverage: {source_summary}
- Date Coverage: {date_summary}
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
