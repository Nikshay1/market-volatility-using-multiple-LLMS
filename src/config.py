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
OLLAMA_TEMPERATURE: float = 0.7  # Increased to force heterogeneity
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
GROQ_MODEL: str = "llama-3.1-8b-instant"
GROQ_TEMPERATURE: float = 0.7  # Increased to force heterogeneity
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
# CHANGE: Switch to NVDA (NVIDIA) for analysis
TICKER_LIST: List[str] = ["NVDA"]
DEFAULT_TICKER: str = "NVDA"  # Default for single-ticker operations

# Date range for analysis
# Covers "Funding Secured", Model 3 Ramp, COVID Crash
START_DATE: str = "2017-06-01"
END_DATE: str = "2020-06-10"


# ============================================================
# Debate Configuration
# ============================================================

DEBATE_ROUNDS: int = 2  # Optimal for signal vs noise (Round 1 = individual bias, Round 2 = social friction)
NUM_BELIEF_AGENTS: int = 3  # Reduced (Fundamental Agent removed to prevent look-ahead bias)


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
