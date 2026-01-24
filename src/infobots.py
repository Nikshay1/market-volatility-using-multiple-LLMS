"""
Data agents (Infobots) that provide structured data to belief agents.

These agents do NOT produce scores - they only inject contextual data.
"""

import os
import yfinance as yf
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

from . import config

# Conditional import for FRED API
try:
    from fredapi import Fred
except ImportError:
    Fred = None


class FundamentalInfobot:
    """
    Provides fundamental financial data for a ticker.
    
    Data includes:
    - P/E ratio
    - Revenue growth
    - Debt/Equity ratio
    - Profit margin
    """
    
    def __init__(self, ticker: str = None):
        """
        Initialize the Fundamental Infobot.
        
        Args:
            ticker: Stock ticker symbol
        """
        self.ticker = ticker or config.DEFAULT_TICKER
        self._cache: Dict[str, Any] = {}
    
    def fetch_data(self, date: datetime = None) -> Dict[str, Any]:
        """
        Fetch fundamental data for the ticker.
        
        Args:
            date: Reference date (for caching key)
            
        Returns:
            Dictionary with fundamental metrics
        """
        cache_key = f"{self.ticker}_{date.strftime('%Y-%m-%d') if date else 'latest'}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            
            # Extract fundamental metrics
            data = {
                "pe_ratio": info.get("trailingPE") or info.get("forwardPE") or "N/A",
                "revenue_growth": info.get("revenueGrowth") or "N/A",
                "debt_to_equity": info.get("debtToEquity") or "N/A",
                "profit_margin": info.get("profitMargins") or "N/A",
                "forward_eps": info.get("forwardEps") or "N/A",
                "book_value": info.get("bookValue") or "N/A",
                "market_cap": info.get("marketCap") or "N/A",
                "sector": info.get("sector") or "N/A",
                "industry": info.get("industry") or "N/A"
            }
            
            # Format numeric values
            data = self._format_metrics(data)
            
            self._cache[cache_key] = data
            return data
            
        except Exception as e:
            print(f"Warning: Could not fetch fundamental data for {self.ticker}: {e}")
            return self._get_mock_data()
    
    def _format_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format numeric metrics for display."""
        formatted = {}
        for key, value in data.items():
            if isinstance(value, float):
                if key in ["pe_ratio", "debt_to_equity"]:
                    formatted[key] = f"{value:.2f}"
                elif key in ["revenue_growth", "profit_margin"]:
                    formatted[key] = f"{value * 100:.1f}%"
                elif key == "market_cap":
                    if value >= 1e12:
                        formatted[key] = f"${value/1e12:.2f}T"
                    elif value >= 1e9:
                        formatted[key] = f"${value/1e9:.2f}B"
                    else:
                        formatted[key] = f"${value/1e6:.2f}M"
                else:
                    formatted[key] = f"{value:.2f}"
            else:
                formatted[key] = str(value)
        return formatted
    
    def _get_mock_data(self) -> Dict[str, Any]:
        """Return mock data when API fails."""
        return {
            "pe_ratio": "25.0",
            "revenue_growth": "10.0%",
            "debt_to_equity": "1.5",
            "profit_margin": "15.0%",
            "forward_eps": "5.00",
            "book_value": "50.00",
            "market_cap": "$2.5T",
            "sector": "Technology",
            "industry": "Consumer Electronics"
        }
    
    def format_for_context(self, date: datetime = None) -> str:
        """
        Format fundamental data as a context string for agents.
        
        Args:
            date: Reference date
            
        Returns:
            Formatted context string
        """
        data = self.fetch_data(date)
        
        return f"""
FUNDAMENTAL DATA ({self.ticker}):
- P/E Ratio: {data['pe_ratio']}
- Revenue Growth: {data['revenue_growth']}
- Debt/Equity: {data['debt_to_equity']}
- Profit Margin: {data['profit_margin']}
- Forward EPS: {data['forward_eps']}
- Book Value: {data['book_value']}
- Market Cap: {data['market_cap']}
- Sector: {data['sector']}
- Industry: {data['industry']}
"""


class MacroInfobot:
    """
    Provides macroeconomic data and indicators from FRED API.
    
    Data includes:
    - Interest rates (Fed Funds Rate)
    - Inflation (CPI)
    - Treasury yields (10Y, 2Y)
    - Unemployment rate
    - GDP growth
    
    Requires FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
    Falls back to mock data if API key not set.
    """
    
    def __init__(self):
        """Initialize the Macro Infobot with FRED client."""
        self._cache: Dict[str, Any] = {}
        self._fred_client = None
        self._fred_available = False
        
        # Initialize FRED client if API key is available
        if config.FRED_API_KEY and Fred is not None:
            try:
                self._fred_client = Fred(api_key=config.FRED_API_KEY)
                self._fred_available = True
                print("FRED API initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize FRED API: {e}")
        elif Fred is None:
            print("Warning: fredapi not installed. Run: pip install fredapi")
        else:
            print("Warning: FRED_API_KEY not set. Using mock macro data.")
    
    def fetch_data(self, date: datetime = None) -> Dict[str, Any]:
        """
        Fetch macroeconomic data from FRED.
        
        Args:
            date: Reference date
            
        Returns:
            Dictionary with macro indicators
        """
        cache_key = date.strftime('%Y-%m-%d') if date else 'latest'
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if self._fred_available:
            data = self._fetch_from_fred(date)
        else:
            data = self._get_mock_data_for_date(date)
        
        self._cache[cache_key] = data
        return data
    
    def _fetch_from_fred(self, date: datetime = None) -> Dict[str, Any]:
        """
        Fetch real data from FRED API.
        
        Args:
            date: Reference date (fetches most recent data on or before this date)
            
        Returns:
            Dictionary with macro indicators
        """
        if date is None:
            date = datetime.now()
        
        # Calculate date range for fetching (look back up to 90 days for latest data)
        end_date = date
        start_date = date - timedelta(days=90)
        
        try:
            # Fetch Fed Funds Rate
            fed_rate = self._get_latest_fred_value(
                config.FRED_SERIES["fed_funds_rate"], 
                start_date, end_date
            )
            
            # Fetch 10-Year Treasury
            treasury_10y = self._get_latest_fred_value(
                config.FRED_SERIES["treasury_10y"],
                start_date, end_date
            )
            
            # Fetch 2-Year Treasury
            treasury_2y = self._get_latest_fred_value(
                config.FRED_SERIES["treasury_2y"],
                start_date, end_date
            )
            
            # Calculate yield curve
            if treasury_10y is not None and treasury_2y is not None:
                yield_curve = treasury_10y - treasury_2y
            else:
                yield_curve = None
            
            # Fetch Unemployment
            unemployment = self._get_latest_fred_value(
                config.FRED_SERIES["unemployment"],
                start_date, end_date
            )
            
            # Fetch CPI and calculate YoY inflation
            inflation = self._calculate_cpi_yoy(date)
            
            # Determine policy stance
            policy_stance = self._determine_policy_stance(fed_rate, yield_curve)
            
            return {
                "fed_funds_rate": f"{fed_rate:.2f}%" if fed_rate else "N/A",
                "inflation_cpi": f"{inflation:.1f}%" if inflation else "N/A",
                "treasury_10y": f"{treasury_10y:.2f}%" if treasury_10y else "N/A",
                "treasury_2y": f"{treasury_2y:.2f}%" if treasury_2y else "N/A",
                "yield_curve_10y2y": f"{yield_curve:+.2f}%" if yield_curve else "N/A",
                "yield_curve_status": "Inverted" if yield_curve and yield_curve < 0 else "Normal",
                "unemployment": f"{unemployment:.1f}%" if unemployment else "N/A",
                "policy_stance": policy_stance,
                "data_source": "FRED"
            }
            
        except Exception as e:
            print(f"Warning: Error fetching FRED data: {e}")
            return self._get_mock_data_for_date(date)
    
    def _get_latest_fred_value(
        self, 
        series_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[float]:
        """Get the most recent value for a FRED series."""
        try:
            series = self._fred_client.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            if series is not None and len(series) > 0:
                # Get the last non-NaN value
                valid = series.dropna()
                if len(valid) > 0:
                    return float(valid.iloc[-1])
        except Exception as e:
            print(f"Warning: Could not fetch {series_id}: {e}")
        return None
    
    def _calculate_cpi_yoy(self, date: datetime) -> Optional[float]:
        """Calculate year-over-year CPI inflation."""
        try:
            end_date = date
            start_date = date - timedelta(days=400)  # Need ~13 months of data
            
            series = self._fred_client.get_series(
                config.FRED_SERIES["cpi"],
                observation_start=start_date,
                observation_end=end_date
            )
            
            if series is not None and len(series) > 12:
                current = series.iloc[-1]
                year_ago = series.iloc[-13]  # Approximately 12 months ago
                yoy_change = ((current - year_ago) / year_ago) * 100
                return float(yoy_change)
        except Exception as e:
            print(f"Warning: Could not calculate CPI YoY: {e}")
        return None
    
    def _determine_policy_stance(
        self, 
        fed_rate: Optional[float], 
        yield_curve: Optional[float]
    ) -> str:
        """Determine Fed policy stance based on indicators."""
        if fed_rate is None:
            return "Unknown"
        
        if fed_rate > 5.0:
            stance = "Restrictive - Fed maintaining high rates"
        elif fed_rate > 4.0:
            stance = "Moderately Restrictive - Elevated rates"
        elif fed_rate > 2.5:
            stance = "Neutral - Rates near equilibrium"
        elif fed_rate > 1.0:
            stance = "Accommodative - Below-normal rates"
        else:
            stance = "Very Accommodative - Near-zero rates"
        
        if yield_curve is not None and yield_curve < 0:
            stance += " (Inverted yield curve signals recession risk)"
        
        return stance
    
    def _get_mock_data_for_date(self, date: datetime = None) -> Dict[str, Any]:
        """
        Get mock macro data for a specific date.
        Uses realistic estimates for 2024 data.
        """
        if date is None:
            date = datetime.now()
        
        month = date.month
        
        # Fed Funds Rate trajectory in 2024
        if month <= 3:
            fed_rate = 5.33
        elif month <= 6:
            fed_rate = 5.33
        elif month <= 9:
            fed_rate = 5.25
        else:
            fed_rate = 4.75
        
        # Inflation (CPI YoY)
        if month <= 3:
            inflation = 3.4
        elif month <= 6:
            inflation = 3.3
        elif month <= 9:
            inflation = 2.9
        else:
            inflation = 2.4
        
        # Treasury yields
        treasury_10y = 4.2 + (month - 6) * 0.05
        treasury_2y = 4.5 + (month - 6) * 0.03
        yield_curve = treasury_10y - treasury_2y
        
        policy_stance = self._determine_policy_stance(fed_rate, yield_curve)
        
        return {
            "fed_funds_rate": f"{fed_rate:.2f}%",
            "inflation_cpi": f"{inflation:.1f}%",
            "treasury_10y": f"{treasury_10y:.2f}%",
            "treasury_2y": f"{treasury_2y:.2f}%",
            "yield_curve_10y2y": f"{yield_curve:+.2f}%",
            "yield_curve_status": "Inverted" if yield_curve < 0 else "Normal",
            "unemployment": "4.0%",
            "policy_stance": policy_stance,
            "data_source": "Mock (FRED_API_KEY not set)"
        }
    
    def format_for_context(self, date: datetime = None) -> str:
        """
        Format macro data as a context string for agents.
        
        Args:
            date: Reference date
            
        Returns:
            Formatted context string
        """
        data = self.fetch_data(date)
        
        return f"""
MACROECONOMIC DATA:
- Fed Funds Rate: {data['fed_funds_rate']}
- Inflation (CPI YoY): {data['inflation_cpi']}
- 10-Year Treasury: {data.get('treasury_10y', 'N/A')}
- 2-Year Treasury: {data.get('treasury_2y', 'N/A')}
- Yield Curve (10Y-2Y): {data['yield_curve_10y2y']} ({data['yield_curve_status']})
- Unemployment: {data['unemployment']}
- Policy Stance: {data['policy_stance']}
- Data Source: {data.get('data_source', 'Unknown')}
"""


def create_infobots(ticker: str = None) -> tuple:
    """
    Factory function to create both infobots.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Tuple of (FundamentalInfobot, MacroInfobot)
    """
    return FundamentalInfobot(ticker), MacroInfobot()


if __name__ == "__main__":
    # Test the infobots
    from datetime import datetime
    
    print("Testing FundamentalInfobot...")
    fund_bot = FundamentalInfobot("AAPL")
    print(fund_bot.format_for_context(datetime(2024, 6, 15)))
    
    print("\nTesting MacroInfobot...")
    macro_bot = MacroInfobot()
    print(macro_bot.format_for_context(datetime(2024, 6, 15)))
    
    print("\nTesting factory function...")
    f, m = create_infobots("MSFT")
    print(f"Created infobots for {f.ticker}")
