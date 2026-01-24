"""
Configuration settings for Agentic Dissonance v2.

Confidence-weighted multi-agent disagreement framework for volatility modeling.
"""

import os
from typing import Optional, List


# ============================================================
# LLM Backend Configuration
# ============================================================

# Options: "groq" or "ollama"
LLM_BACKEND: str = os.environ.get("LLM_BACKEND", "ollama")


# ============================================================
# Ollama Configuration (Local LLM)
# ============================================================

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "mistral")
OLLAMA_TEMPERATURE: float = 0.0  # Deterministic outputs
OLLAMA_MAX_TOKENS: int = 1024


# ============================================================
# Groq API Configuration (Cloud LLM)
# ============================================================

def get_groq_api_key() -> str:
    """Get GROQ API key from environment variable."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Please set it with: export GROQ_API_KEY='your-api-key'"
        )
    return api_key


GROQ_API_KEY: Optional[str] = os.environ.get("GROQ_API_KEY")
GROQ_MODEL: str = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE: float = 0.0  # Deterministic outputs
GROQ_MAX_TOKENS: int = 1024


# ============================================================
# FRED API Configuration (Macroeconomic Data)
# ============================================================

# Get free API key from: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY: Optional[str] = os.environ.get("FRED_API_KEY")

# FRED series IDs for macro indicators
FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",      # Effective Federal Funds Rate
    "cpi": "CPIAUCSL",                  # Consumer Price Index
    "treasury_10y": "DGS10",            # 10-Year Treasury
    "treasury_2y": "DGS2",              # 2-Year Treasury
    "unemployment": "UNRATE",            # Unemployment Rate
    "gdp": "GDP"                         # Gross Domestic Product
}


# ============================================================
# Market Data Configuration
# ============================================================

# Multi-asset support
TICKER_LIST: List[str] = ["AAPL", "MSFT", "TSLA", "SPY"]
DEFAULT_TICKER: str = "AAPL"  # Default for single-ticker operations

# Date range for analysis
START_DATE: str = "2024-01-01"
END_DATE: str = "2024-12-31"


# ============================================================
# Debate Configuration
# ============================================================

DEBATE_ROUNDS: int = 2  # 2 rounds for optimal disagreement signal
NUM_BELIEF_AGENTS: int = 4  # Fundamental, Sentiment, Technical, Macro


# ============================================================
# Embedding Model Configuration
# ============================================================

EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# File Paths
# ============================================================

# Get the project root directory
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR: str = os.path.join(PROJECT_ROOT, "output")
CACHE_DIR: str = os.path.join(DATA_DIR, "cache")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Data files
RAW_MARKET_DATA_PATH: str = os.path.join(DATA_DIR, "raw_market_data.csv")
DISAGREEMENT_SIGNALS_PATH: str = os.path.join(DATA_DIR, "disagreement_signals.csv")

# Output files
RESULTS_PLOT_PATH: str = os.path.join(OUTPUT_DIR, "results.png")


# ============================================================
# Rate Limiting Configuration
# ============================================================

API_RETRY_ATTEMPTS: int = 3
API_RETRY_DELAY: float = 2.0  # seconds
RATE_LIMIT_DELAY: float = 1.0  # seconds between API calls


# ============================================================
# Analysis Configuration
# ============================================================

FORWARD_VOLATILITY_WINDOW: int = 5  # 5-day forward realized volatility
TRAIN_TEST_SPLIT: float = 0.7  # 70% train, 30% test


# ============================================================
# Caching Configuration
# ============================================================

ENABLE_LLM_CACHE: bool = True  # Cache LLM outputs to disk
CACHE_EXPIRY_DAYS: int = 30  # Cache validity period


# ============================================================
# Agent Descriptions (for prompts)
# ============================================================

AGENT_SCORE_MEANINGS = {
    "fundamental": "long-term valuation and fundamental risk",
    "sentiment": "short-term crowd psychology and news sentiment",
    "technical": "trend direction and momentum signals",
    "macro": "macroeconomic risk and policy conditions"
}
