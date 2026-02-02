"""
Kaggle Data Import Script for Market Volatility Project.

Converts the "Daily Financial News for 6000+ Stocks" dataset from Kaggle
into the format expected by the data_loader.

Usage:
    1. Download "analyst_ratings_processed.csv" from Kaggle:
       https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
    
    2. Place the file in the project root directory
    
    3. Run: python scripts/import_kaggle.py
    
    4. The script will create data/historical_news.csv
"""

import pandas as pd
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_FILE = "analyst_ratings_processed.csv"  # User must download this from Kaggle
OUTPUT_FILE = os.path.join("data", "historical_news.csv")
TARGET_TICKER = "AAPL"


def convert_dataset():
    """
    Convert Kaggle dataset to the format expected by data_loader.
    
    Expected input columns: stock, title, date
    Output columns: Date, Ticker, Headline
    """
    print(f"Reading {INPUT_FILE}...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"\nERROR: {INPUT_FILE} not found!")
        print("\nTo use historical news data:")
        print("1. Download from: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests")
        print("2. Place 'analyst_ratings_processed.csv' in the project root")
        print("3. Run this script again")
        return False
    
    try:
        df = pd.read_csv(INPUT_FILE)
        
        print(f"Total rows in source: {len(df)}")
        print(f"Available tickers: {df['stock'].nunique()}")
        
        print(f"\nFiltering for {TARGET_TICKER}...")
        df = df[df['stock'] == TARGET_TICKER].copy()
        
        if len(df) == 0:
            print(f"ERROR: No data found for {TARGET_TICKER}")
            print(f"Available tickers sample: {df['stock'].unique()[:10]}")
            return False
        
        # Rename columns to expected format
        df = df.rename(columns={'title': 'Headline', 'date': 'Date', 'stock': 'Ticker'})
        
        # Ensure UTC handling for timestamps like "2020-05-12 13:42:00"
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True).dt.strftime('%Y-%m-%d')
        
        # Remove any rows with invalid dates
        df = df.dropna(subset=['Date'])
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Create output directory if needed
        os.makedirs("data", exist_ok=True)
        
        # Save to CSV
        df[['Date', 'Ticker', 'Headline']].to_csv(OUTPUT_FILE, index=False)
        
        print(f"\nSuccess!")
        print(f"  Saved {len(df)} headlines for {TARGET_TICKER}")
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"  Output file: {OUTPUT_FILE}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def show_sample():
    """Show a sample of the converted data."""
    if os.path.exists(OUTPUT_FILE):
        print(f"\nSample data from {OUTPUT_FILE}:")
        df = pd.read_csv(OUTPUT_FILE)
        print(df.head(10).to_string())


if __name__ == "__main__":
    print("="*60)
    print("KAGGLE DATA IMPORT SCRIPT")
    print("="*60)
    print(f"Target ticker: {TARGET_TICKER}")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print("="*60 + "\n")
    
    success = convert_dataset()
    
    if success:
        show_sample()
    
    print("\nDone!")
