"""
Configuration settings for the Agentic Dissonance framework.
"""

import os
from typing import Optional


# ============================================================
# API Configuration
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
# Market Data Configuration
# ============================================================

TICKER: str = "AAPL"
START_DATE: str = "2024-01-01"  # 5 days of data (~5 trading days)
END_DATE: str = "2024-01-30"    # Reduced to minimize API calls


# ============================================================
# Debate Configuration
# ============================================================

DEBATE_ROUNDS: int = 3  # Number of debate rounds


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

RAW_MARKET_DATA_PATH: str = os.path.join(DATA_DIR, "raw_market_data.csv")
DISAGREEMENT_SIGNALS_PATH: str = os.path.join(DATA_DIR, "disagreement_signals.csv")
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
