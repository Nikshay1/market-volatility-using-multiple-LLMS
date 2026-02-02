"""
Data agents (Infobots) that provide structured data to belief agents.

These agents do NOT produce scores - they only inject contextual data.

MacroInfobot fetches historical data from yfinance:
- VIX (^VIX): Market Fear
- 10 Year Yield (^TNX): Interest Rate expectations
- Crude Oil (CL=F): Inflation proxy
- Dollar Index (DX-Y.NYB): Global liquidity proxy
"""

import yfinance as yf
from typing import Dict, Any
from datetime import datetime, timedelta
import pandas as pd

from . import config


class FundamentalInfobot:
    """
    DISABLED: FundamentalInfobot to prevent Look-Ahead Bias.
    
    Using yfinance.info to fetch current P/E ratios and revenue growth
    for historical dates would use 2026 data to predict 2018 volatility,
    which invalidates the entire experiment.
    """
    
    def __init__(self, ticker: str = None):
        self.ticker = ticker or config.DEFAULT_TICKER
        self._cache: Dict[str, Any] = {}
    
    def format_for_context(self, date: datetime = None) -> str:
        """Returns empty string - disabled to prevent Look-Ahead Bias."""
        return ""


class MacroInfobot:
    """
    Provides macroeconomic data using yfinance.
    
    Fetches historical closing prices for:
    - VIX (^VIX): Market Fear index
    - 10 Year Yield (^TNX): Interest Rate expectations
    - Crude Oil (CL=F): Inflation/Energy proxy
    - Dollar Index (DX-Y.NYB): Global liquidity proxy
    """
    
    def __init__(self):
        """Initialize the Macro Infobot."""
        self._cache: Dict[str, Any] = {}
    
    def fetch_data(self, date: datetime = None) -> Dict[str, Any]:
        """
        Fetch macro data using yfinance for VIX, TNX, Oil, Dollar.
        
        Args:
            date: Reference date
            
        Returns:
            Dictionary with macro indicators
        """
        if date is None:
            date = datetime.now()
            
        cache_key = date.strftime('%Y-%m-%d')
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            end_date = date
            start_date = date - timedelta(days=5)
            
            # Fetch 4 key daily proxies from yfinance
            tickers = ["^VIX", "^TNX", "CL=F", "DX-Y.NYB"]
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
            
            if not data.empty:
                latest = data.iloc[-1]
                
                result = {
                    "vix": f"{latest.get('^VIX', 20.0):.2f}" if pd.notna(latest.get('^VIX')) else "N/A",
                    "treasury_10y": f"{latest.get('^TNX', 4.0):.2f}%" if pd.notna(latest.get('^TNX')) else "N/A",
                    "oil": f"${latest.get('CL=F', 70.0):.2f}" if pd.notna(latest.get('CL=F')) else "N/A",
                    "dollar": f"{latest.get('DX-Y.NYB', 100.0):.2f}" if pd.notna(latest.get('DX-Y.NYB')) else "N/A",
                    "data_source": "yfinance"
                }
                self._cache[cache_key] = result
                return result
                
        except Exception as e:
            print(f"Warning: Could not fetch yfinance macro data: {e}")
        
        # Fallback to default values if yfinance fails
        return {
            "vix": "20.00",
            "treasury_10y": "4.00%",
            "oil": "$70.00",
            "dollar": "100.00",
            "data_source": "fallback"
        }
    
    def format_for_context(self, date: datetime = None) -> str:
        """
        Format macro data as a context string for agents.
        
        Args:
            date: Reference date
            
        Returns:
            Formatted context string with VIX, TNX, Oil, Dollar
        """
        data = self.fetch_data(date)
        
        return f"""
MACRO DATA:
- VIX (Fear Index): {data.get('vix', 'N/A')}
- 10Y Yield: {data.get('treasury_10y', 'N/A')}
- Oil: {data.get('oil', 'N/A')}
- DXY (Dollar): {data.get('dollar', 'N/A')}
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
    from datetime import datetime
    
    print("Testing MacroInfobot with yfinance...")
    macro_bot = MacroInfobot()
    print(macro_bot.format_for_context(datetime(2019, 8, 15)))
    
    print("\nTesting FundamentalInfobot (should be empty)...")
    fund_bot = FundamentalInfobot("MSFT")
    result = fund_bot.format_for_context(datetime(2019, 8, 15))
    print(f"FundamentalInfobot output: '{result}' (empty = correct)")
