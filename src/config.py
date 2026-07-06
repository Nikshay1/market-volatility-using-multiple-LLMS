"""
Configuration settings for Agentic Dissonance v2.

Confidence-weighted multi-agent disagreement framework for volatility modeling.
"""

import os
import re
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
OLLAMA_TOP_P: float = float(os.environ.get("OLLAMA_TOP_P", "0.9"))
OLLAMA_MAX_TOKENS: int = 1024


# ============================================================
# Groq API Configuration (Cloud LLM)
# ============================================================

def get_groq_api_keys() -> List[str]:
    """Get one or more Groq API keys from environment variables."""
    raw_values = [
        os.environ.get("GROQ_API_KEY", ""),
        os.environ.get("GROQ_API_KEYS", ""),
    ]
    keys: List[str] = []
    for raw_value in raw_values:
        for key in re.split(r"[,;\s]+", raw_value.strip()):
            if key and key not in keys:
                keys.append(key)
    return keys


_GROQ_API_KEY_INDEX: int = 0


def get_groq_api_key() -> str:
    """Get the active Groq API key from environment variables."""
    keys = get_groq_api_keys()
    if not keys:
        raise ValueError(
            "GROQ_API_KEY or GROQ_API_KEYS environment variable is not set. "
            "Please set one before using the Groq backend."
        )
    return keys[_GROQ_API_KEY_INDEX % len(keys)]


def rotate_groq_api_key() -> bool:
    """Rotate to the next Groq API key, if multiple keys are configured."""
    global _GROQ_API_KEY_INDEX
    keys = get_groq_api_keys()
    if len(keys) <= 1:
        return False
    _GROQ_API_KEY_INDEX = (_GROQ_API_KEY_INDEX + 1) % len(keys)
    return True


def groq_api_key_count() -> int:
    """Return the number of configured Groq API keys."""
    return len(get_groq_api_keys())


GROQ_API_KEY: Optional[str] = os.environ.get("GROQ_API_KEY")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_TEMPERATURE: float = 0.7  # Increased to force heterogeneity
GROQ_TOP_P: float = float(os.environ.get("GROQ_TOP_P", "0.9"))
GROQ_MAX_TOKENS: int = 1024


# ============================================================
# Per-Agent Model Controls
# ============================================================

SENTIMENT_MODEL: str = os.environ.get("SENTIMENT_MODEL", "")
SENTIMENT_TEMPERATURE: float = float(os.environ.get("SENTIMENT_TEMPERATURE", "0.85"))
SENTIMENT_TOP_P: float = float(os.environ.get("SENTIMENT_TOP_P", "0.95"))

TECHNICAL_MODEL: str = os.environ.get("TECHNICAL_MODEL", "")
TECHNICAL_TEMPERATURE: float = float(os.environ.get("TECHNICAL_TEMPERATURE", "0.55"))
TECHNICAL_TOP_P: float = float(os.environ.get("TECHNICAL_TOP_P", "0.8"))

MACRO_MODEL: str = os.environ.get("MACRO_MODEL", "")
MACRO_TEMPERATURE: float = float(os.environ.get("MACRO_TEMPERATURE", "0.65"))
MACRO_TOP_P: float = float(os.environ.get("MACRO_TOP_P", "0.85"))

# Calibration controls
CORRELATION_THRESHOLD: float = float(os.environ.get("CORRELATION_THRESHOLD", "0.8"))
CORRELATION_PERSISTENCE_DAYS: int = int(os.environ.get("CORRELATION_PERSISTENCE_DAYS", "5"))
CALIBRATION_TEMPERATURE_STEP: float = float(os.environ.get("CALIBRATION_TEMPERATURE_STEP", "0.1"))


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

# Multi-asset support: cross-sector panel for robustness analysis
TICKER_LIST: List[str] = ["NVDA"]
DEFAULT_TICKER: str = "NVDA"  # Default for single-ticker operations

# Date range for panel/regime analysis
START_DATE: str = "2018-01-01"
END_DATE: str = "2020-06-10"

# Regime buckets for subperiod robustness checks
REGIME_WINDOWS = {
    "pre_covid": ("2018-01-01", "2020-02-29"),
    "covid_shock": ("2020-03-01", "2020-12-31"),
    "post_covid": ("2021-01-01", "2021-12-31"),
    "rate_hike_era": ("2022-01-01", "2024-12-31")
}


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

DATA_DIR: str = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "output"))
CACHE_DIR: str = os.environ.get("CACHE_DIR", os.path.join(DATA_DIR, "cache"))

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Data files
RAW_MARKET_DATA_PATH: str = os.path.join(DATA_DIR, "raw_market_data.csv")
DISAGREEMENT_SIGNALS_PATH: str = os.environ.get(
    "DISAGREEMENT_SIGNALS_PATH",
    os.path.join(DATA_DIR, "disagreement_signals.csv")
)

# Output files
RESULTS_PLOT_PATH: str = os.path.join(OUTPUT_DIR, "results.png")
HEADLINE_COVERAGE_STATS_PATH: str = os.path.join(OUTPUT_DIR, "headline_coverage_stats.csv")
DIVERSITY_REPORT_PATH: str = os.path.join(OUTPUT_DIR, "diversity_report.json")
ROBUSTNESS_SUMMARY_PATH: str = os.path.join(OUTPUT_DIR, "robustness_summary.md")


# ============================================================
# Rate Limiting Configuration
# ============================================================

API_RETRY_ATTEMPTS: int = 3
API_RETRY_DELAY: float = float(os.environ.get("API_RETRY_DELAY", "2.0"))
RATE_LIMIT_DELAY: float = float(os.environ.get("RATE_LIMIT_DELAY", "1.0"))  # seconds between API calls
BACKTEST_DAY_DELAY: float = float(os.environ.get("BACKTEST_DAY_DELAY", "12.0"))


# ============================================================
# Analysis Configuration
# ============================================================

FORWARD_VOLATILITY_WINDOW: int = 5  # 5-day forward realized volatility
TRAIN_TEST_SPLIT: float = 0.7  # 70% train, 30% test
ROBUSTNESS_ROLLING_WINDOW_DAYS: int = 60
ROBUSTNESS_MIN_PASS_WINDOWS: int = 3
SIGNIFICANCE_ALPHA: float = 0.05

# Historical news requirements for backtests
NEWS_LOOKBACK_DAYS: int = int(os.environ.get("NEWS_LOOKBACK_DAYS", "5"))
NEWS_MIN_HEADLINES_PER_DAY: int = int(os.environ.get("NEWS_MIN_HEADLINES_PER_DAY", "1"))
NEWS_DEDUP_SIMILARITY_THRESHOLD: float = float(
    os.environ.get("NEWS_DEDUP_SIMILARITY_THRESHOLD", "0.9")
)


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
