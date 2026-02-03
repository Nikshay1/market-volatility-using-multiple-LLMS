# Agentic Dissonance v2: Multi-Agent Volatility Forecasting Framework

> **Research Framework for Measuring LLM Agent Disagreement as a Market Volatility Signal**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Hypothesis](#core-hypothesis)
3. [System Architecture](#system-architecture)
4. [The Blind & Battle Protocol](#the-blind--battle-protocol)
5. [Belief Agents Deep Dive](#belief-agents-deep-dive)
6. [Data Sources & Infobots](#data-sources--infobots)
7. [Disagreement Metrics](#disagreement-metrics)
8. [GARCH-X Volatility Modeling](#garch-x-volatility-modeling)
9. [Output Visualizations](#output-visualizations)
10. [Installation & Setup](#installation--setup)
11. [Command Reference](#command-reference)
12. [File Structure](#file-structure)
13. [Configuration Options](#configuration-options)
14. [Troubleshooting](#troubleshooting)

---

## Project Overview

**Agentic Dissonance** is a research framework that uses multiple LLM-powered agents to forecast market volatility. The core innovation: instead of using LLM outputs directly as trading signals, we measure the **disagreement** between agents as a proxy for market uncertainty.

### Key Insight

When heterogeneous AI agents—each viewing the market through a different analytical lens—cannot reach consensus, it signals:

- **High Disagreement → Uncertain market conditions → Higher expected volatility**
- **Low Disagreement → Clear market consensus → Lower expected volatility**

This framework implements the **Multi-Agent Debate protocol** (Du et al., 2023) adapted for financial volatility prediction.

---

## Core Hypothesis

```
H₁: Agent disagreement (D_conf) positively correlates with next-day realized volatility (σᴿⱽ)

σᴿⱽ_{t+1} = f(D_conf_t) + ε

Where:
- D_conf = Confidence-weighted variance of agent scores
- σᴿⱽ = 5-day forward realized volatility (annualized)
```

The framework tests this hypothesis by:

1. Running daily debates between 3 heterogeneous LLM agents
2. Computing disagreement signals from their outputs
3. Adding disagreement as an exogenous variable to GARCH models
4. Comparing GARCH-X (with D_conf) vs baseline GARCH(1,1)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AGENTIC DISSONANCE v2                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │    DATA      │    │   INFOBOTS   │    │   MARKET     │                   │
│  │    LOADER    │    │              │    │    DATA      │                   │
│  │              │    │ • MacroBot   │    │              │                   │
│  │ • yfinance   │────│ • VIX/TNX    │────│ • OHLCV      │                   │
│  │ • News RSS   │    │ • Oil/DXY    │    │ • Returns    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│          │                  │                   │                            │
│          └──────────────────┴───────────────────┘                            │
│                             │                                                │
│                    ┌────────▼────────┐                                       │
│                    │  CONTEXT STRING │                                       │
│                    │   (formatted)   │                                       │
│                    └────────┬────────┘                                       │
│                             │                                                │
│           ┌─────────────────┼─────────────────┐                              │
│           ▼                 ▼                 ▼                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │  SENTIMENT   │  │  TECHNICAL   │  │    MACRO     │                       │
│  │    AGENT     │  │    AGENT     │  │    AGENT     │                       │
│  │              │  │              │  │              │                       │
│  │ • News       │  │ • Price      │  │ • VIX        │                       │
│  │ • Headlines  │  │ • Momentum   │  │ • Rates      │                       │
│  │ • Fear/Greed │  │ • Trends     │  │ • Risk-On/Off│                       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                       │
│         │                 │                 │                                │
│         ▼                 ▼                 ▼                                │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │              DEBATE ENGINE (Blind & Battle)          │                   │
│  │                                                       │                   │
│  │  Round 1: BLIND VOTE (agents analyze in isolation)   │                   │
│  │  Round 2: BATTLE (agents critique opposing views)    │                   │
│  └───────────────────────────┬──────────────────────────┘                   │
│                              │                                               │
│                     ┌────────▼────────┐                                      │
│                     │   AGGREGATOR    │                                      │
│                     │                 │                                      │
│                     │ • Mean Score μt │                                      │
│                     │ • D_conf        │                                      │
│                     │ • Confidence    │                                      │
│                     └────────┬────────┘                                      │
│                              │                                               │
│                     ┌────────▼────────┐                                      │
│                     │    ANALYSIS     │                                      │
│                     │                 │                                      │
│                     │ • Correlation   │                                      │
│                     │ • GARCH(1,1)    │                                      │
│                     │ • GARCH-X       │                                      │
│                     │ • Visualization │                                      │
│                     └─────────────────┘                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Blind & Battle Protocol

The **Blind & Battle** protocol is specifically designed to prevent "herding"—where agents converge to the same opinion because they see group consensus.

### Protocol Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    BLIND & BATTLE PROTOCOL                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ╔════════════════════════════════════════════════════════════╗ │
│  ║ ROUND 1: BLIND VOTE                                        ║ │
│  ╠════════════════════════════════════════════════════════════╣ │
│  ║                                                            ║ │
│  ║  Each agent receives:                                      ║ │
│  ║  • Market data (OHLCV, returns)                           ║ │
│  ║  • News headlines (last 24 hours)                         ║ │
│  ║  • Macro indicators (VIX, yields, oil)                    ║ │
│  ║                                                            ║ │
│  ║  Each agent DOES NOT receive:                              ║ │
│  ║  • Other agents' scores                                    ║ │
│  ║  • Group mean or consensus                                 ║ │
│  ║                                                            ║ │
│  ║  OUTPUT: Initial independent beliefs                       ║ │
│  ╚════════════════════════════════════════════════════════════╝ │
│                          ↓                                       │
│  ╔════════════════════════════════════════════════════════════╗ │
│  ║ ROUND 2: BATTLE MODE                                       ║ │
│  ╠════════════════════════════════════════════════════════════╣ │
│  ║                                                            ║ │
│  ║  Each agent receives:                                      ║ │
│  ║  • Their OWN Round 1 response                              ║ │
│  ║  • The MOST OPPOSING agent's argument only                 ║ │
│  ║    (Found by: max |scoreᵢ - scoreⱼ|)                       ║ │
│  ║                                                            ║ │
│  ║  Each agent DOES NOT receive:                              ║ │
│  ║  • Group mean (prevents herding to center)                 ║ │
│  ║  • All other agents' responses                             ║ │
│  ║                                                            ║ │
│  ║  INSTRUCTION: "Critique this opposing view and             ║ │
│  ║               defend your position"                        ║ │
│  ║                                                            ║ │
│  ║  OUTPUT: Final refined beliefs (with disagreement signal)  ║ │
│  ╚════════════════════════════════════════════════════════════╝ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Design?

| Design Choice | Problem It Solves | Result |
|---------------|-------------------|--------|
| **No mean shown** | Agents converge to avoid conflict | Preserves disagreement |
| **Opposing argument only** | Information overload | Focused critique |
| **"Defend your position"** | Agents simply accept critique | Maintains diversity |
| **2 rounds optimal** | Too many rounds = convergence | Signal-to-noise balance |

---

## Belief Agents Deep Dive

### Agent Output Format

All agents output a structured JSON response:

```json
{
  "score": 0.45,
  "confidence": 0.78,
  "reasoning": "Brief explanation of the analysis..."
}
```

| Field | Range | Meaning |
|-------|-------|---------|
| `score` | [-1, +1] | Market direction belief (-1 = very bearish, +1 = very bullish) |
| `confidence` | [0, 1] | How confident the agent is in their assessment |
| `reasoning` | String | Explanation of the logic behind the score |

---

### 1. Sentiment Agent

**Purpose**: Analyzes short-term crowd psychology and news sentiment.

**System Prompt**:
```
You are a high-frequency news sentiment analyst specializing in 
the Semiconductor sector.

TASK: Analyze the provided news headlines from the LAST 24 HOURS.
OBJECTIVE: Determine if the overnight news cycle will trigger 
immediate volatility for NVDA's trading session TODAY.
```

**Input Data**:
- News headlines (last 24 hours from RSS feeds)
- Market narratives and themes
- Fear/greed indicators

**Scoring Logic**:
| Score | Trigger |
|-------|---------|
| -1.0 to -0.5 | "Crypto Winter" headlines, analyst downgrades, inventory oversupply |
| 0.0 | Routine announcements, product recaps, unrelated news |
| +0.5 to +1.0 | "Crypto Recovery" signs, data center orders, analyst upgrades |

**Unique Perspective**: Focuses on **immediate momentum**—what is the psychological state of the market *right now*?

---

### 2. Technical Agent

**Purpose**: Analyzes price action, trends, and momentum signals.

**System Prompt**:
```
You are a swing trading technical analyst focused on daily momentum.

TASK: Analyze the price action of the PREVIOUS TRADING DAY.
OBJECTIVE: Predict volatility for the UPCOMING SESSION based on 
yesterday's close.
```

**Input Data**:
- OHLCV data (Open, High, Low, Close, Volume)
- 5-day price returns
- 20-day realized volatility
- Daily range and candle patterns

**Scoring Logic**:
| Score | Pattern |
|-------|---------|
| -1.0 to -0.5 | Bearish Engulfing, closing near lows, resistance rejection |
| 0.0 | Doji (indecision), inside day, tight range consolidation |
| +0.5 to +1.0 | Bullish Engulfing, closing near highs, gap up on volume |

**Unique Perspective**: Focuses on **continuation vs. mean reversion**—does yesterday's pattern predict follow-through?

---

### 3. Macro Agent

**Purpose**: Analyzes macroeconomic conditions and risk sentiment.

**System Prompt**:
```
You are a macro-risk analyst monitoring daily liquidity and 
sector rotation.

TASK: Analyze the DAILY % CHANGE in the following proxies:
1. VIX (Market Fear): >5% spike = Risk Off
2. SOXX (Semiconductor ETF): Leading or lagging?
3. BTC (Bitcoin): GPU mining demand proxy for 2019
```

**Input Data**:
- VIX (^VIX) - Fear Index
- 10-Year Treasury Yield (^TNX)
- Crude Oil (CL=F) - Inflation/energy proxy
- Dollar Index (DX-Y.NYB) - Liquidity proxy

**Scoring Logic**:
| Score | Condition |
|-------|-----------|
| -1.0 to -0.5 | VIX spiking >5%, Bitcoin crashing, broad sell-off (Risk-Off) |
| 0.0 | Mixed signals, VIX flat, Tech flat (Neutral) |
| +0.5 to +1.0 | VIX collapsing, Bitcoin rallying, rotation into Tech (Risk-On) |

**Unique Perspective**: Focuses on **cross-asset signals**—is broad risk sentiment supporting or pressuring equities?

---

## Data Sources & Infobots

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  EXTERNAL APIs                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ yfinance                                                  │   │
│  │ • Market OHLCV data (NVDA, etc.)                         │   │
│  │ • VIX, Treasury yields, Oil, Dollar Index                │   │
│  │ • Historical prices with no look-ahead bias              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ News RSS Feeds                                            │   │
│  │ • Google News RSS: "/rss/search?q={ticker}"              │   │
│  │ • Yahoo Finance RSS                                       │   │
│  │ • Fallback: Pre-formatted headlines from local CSV       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  INFOBOTS (Data Injection Agents)                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ MacroInfobot                                              │   │
│  │ • Fetches VIX, TNX, Oil, Dollar from yfinance            │   │
│  │ • Formats as context string for agents                   │   │
│  │ • Caches results to avoid redundant API calls            │   │
│  │                                                           │   │
│  │ Output format:                                            │   │
│  │ MACRO DATA:                                               │   │
│  │ - VIX (Fear Index): 18.45                                │   │
│  │ - 10Y Yield: 4.25%                                       │   │
│  │ - Oil: $75.30                                            │   │
│  │ - DXY (Dollar): 102.50                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ FundamentalInfobot (DISABLED)                             │   │
│  │                                                           │   │
│  │ ⚠️ Disabled to prevent look-ahead bias!                   │   │
│  │                                                           │   │
│  │ Using yfinance.info to fetch current P/E ratios for      │   │
│  │ historical dates would use 2024 data to predict 2019     │   │
│  │ volatility, which invalidates the experiment.            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Look-Ahead Bias Prevention

The framework is carefully designed to prevent future information from leaking into predictions:

| Data Type | Safety Mechanism |
|-----------|------------------|
| Market data | Only data `<= current_date` is used |
| News headlines | Filtered to `published_date <= current_date` |
| Macro indicators | Fetched for historical date, not current |
| Fundamentals | **Disabled entirely** (yfinance.info is always "current") |

---

## Disagreement Metrics

### Primary Metric: D_conf (Confidence-Weighted Variance)

This is the main signal used for volatility prediction:

```
D_conf = Σ(cᵢ × (sᵢ - μ)²) / Σ(cᵢ)

Where:
  cᵢ = Confidence of agent i
  sᵢ = Score of agent i
  μ  = Confidence-weighted mean score
```

**Interpretation**:
- **High D_conf**: Agents with high confidence disagree strongly → Market confusion → Higher volatility expected
- **Low D_conf**: Agents agree or low-confidence outliers → Clear consensus → Lower volatility expected

### Secondary Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| `mean_score` | μ = Σ(cᵢ × sᵢ) / Σ(cᵢ) | Overall market sentiment direction |
| `avg_confidence` | Mean of all confidence values | How certain agents are overall |
| `semantic_divergence` | 1 - cosine_similarity(embeddings) | Divergence in reasoning (NLP-based) |

### Example Calculation

```
Agent Outputs:
  Sentiment:  score = +0.6, confidence = 0.8
  Technical:  score = -0.3, confidence = 0.7
  Macro:      score = +0.2, confidence = 0.9

Step 1: Confidence-weighted mean
  μ = (0.8×0.6 + 0.7×(-0.3) + 0.9×0.2) / (0.8 + 0.7 + 0.9)
  μ = (0.48 - 0.21 + 0.18) / 2.4 = 0.1875

Step 2: Confidence-weighted variance
  D_conf = [0.8×(0.6-0.1875)² + 0.7×(-0.3-0.1875)² + 0.9×(0.2-0.1875)²] / 2.4
  D_conf = [0.136 + 0.166 + 0.0001] / 2.4 = 0.126

Result: Moderate disagreement (D_conf ≈ 0.13)
```

---

## GARCH-X Volatility Modeling

### Model Specification

**Baseline GARCH(1,1)**:
```
σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}
```

**GARCH-X with Disagreement**:
```
Mean Equation: r_t = μ + γ × D_conf_{t-1} + ε_t
Variance: σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}

Where:
  γ = Exogenous coefficient (captures D_conf impact)
  D_conf_{t-1} = Lagged disagreement (prediction, not contemporaneous)
```

### Model Comparison Metrics

| Metric | Meaning | Better Model Has |
|--------|---------|------------------|
| **AIC** | Akaike Information Criterion | Lower value |
| **BIC** | Bayesian Information Criterion | Lower value |
| **RMSE** | Root Mean Square Error | Lower value |
| **MAE** | Mean Absolute Error | Lower value |
| **p-value** | Significance of γ (exogenous coef) | < 0.05 |

### Expected Results

```
============================================================
              EXECUTIVE SUMMARY (RMSE TEST)
============================================================
1. Standard GARCH RMSE: 0.012345
2. Agent GARCH-X RMSE:  0.012234 (Lower is Better)

VERDICT:
[X] SUCCESS: Agents reduced error by 0.90%.
[ ] FAILURE: Standard model was more accurate.
============================================================
```

---

## Output Visualizations

The analysis module generates multiple visualization files:

| File | Description |
|------|-------------|
| `output/results.png` | 3-panel dashboard (time series, scatter, boxplots) |
| `output/fig1_disagreement.png` | Scatter: D_conf vs Forward Volatility |
| `output/fig2_mean_score.png` | Scatter: Mean Score vs Forward Volatility |
| `output/fig3_timeline.png` | Timeline: Mean Score bars + Price overlay |
| `output/mean_score_vs_realized_volatility_timeseries.png` | Dual-axis time series: μt vs σᴿⱽ |
| `output/topology.png` | Blind & Battle protocol diagram |
| `output/residuals.png` | GARCH residual diagnostics (4-panel) |

### Key Visualizations Explained

**1. Disagreement vs Forward Volatility (fig1_disagreement.png)**
- X-axis: D_conf (agent disagreement)
- Y-axis: 5-day forward realized volatility
- Red trendline: Linear regression
- Goal: Positive slope indicates disagreement predicts volatility

**2. Mean Score Timeline (fig3_timeline.png)**
- Left axis: Mean score bars (green = bullish, red = bearish)
- Right axis: Stock price line
- Panic zones: Highlighted when score < -0.3

**3. GARCH Residuals (residuals.png)**
- Top row: Standardized residuals for both models
- Bottom left: Histogram comparing residual distributions
- Bottom right: Q-Q plot vs normal distribution

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- Ollama (for local LLM) or Groq API key (for cloud)

### Step-by-Step Installation

```powershell
# 1. Clone the repository
git clone https://github.com/Nikshay1/market-volatility-using-multiple-LLMS.git
cd market-volatility-using-multiple-LLMS

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama (if using local LLM)
ollama serve
# In another terminal: ollama pull mistral

# 5. (Optional) Set API keys
$env:FRED_API_KEY = "your-fred-api-key"
$env:GROQ_API_KEY = "your-groq-api-key"  # If using cloud
```

### LLM Backend Options

**Option 1: Ollama (Local) - Recommended**
```powershell
# Start Ollama server
ollama serve

# Pull a model
ollama pull mistral

# The system will automatically use Ollama
```

**Option 2: Groq (Cloud)**
```powershell
# Set environment variable
$env:LLM_BACKEND = "groq"
$env:GROQ_API_KEY = "your-api-key"

# Run any command as normal
python -m src.backtest --test --days 3
```

---

## Command Reference

### Backtest Commands

```powershell
# Quick test (3 days)
python -m src.backtest --test --days 3

# Full backtest (all configured tickers/dates)
python -m src.backtest

# Single ticker
python -m src.backtest --ticker NVDA

# Resume interrupted backtest
python -m src.backtest --resume 2019-06-01

# Compare debate round configurations
python -m src.backtest --compare-rounds

# Custom rounds
python -m src.backtest --rounds 3
```

### Analysis Commands

```powershell
# Run full analysis pipeline
python -m src.analysis

# This will:
# 1. Load disagreement signals from data/disagreement_signals.csv
# 2. Load market data from data/raw_market_data.csv
# 3. Compute forward volatility
# 4. Run correlation analysis
# 5. Fit GARCH and GARCH-X models
# 6. Generate all visualizations in output/
```

### Individual Module Tests

```powershell
# Test debate engine
python -m src.debate_engine

# Test data loader
python -m src.data_loader

# Test agents
python -m src.agents

# Test disagreement metrics
python -m src.disagreement

# Test infobots
python -m src.infobots
```

---

## File Structure

```
market-volatility-using-multiple-LLMS/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # All configuration settings
│   ├── agents.py             # Belief agents (Sentiment, Technical, Macro)
│   ├── aggregator.py         # Statistics computation, opposing argument formatting
│   ├── analysis.py           # GARCH modeling, correlations, visualization
│   ├── backtest.py           # Main backtest loop with resume capability
│   ├── data_loader.py        # Market data, news fetching, context formatting
│   ├── debate_engine.py      # Blind & Battle protocol implementation
│   ├── disagreement.py       # Disagreement metric calculations
│   └── infobots.py           # Data injection agents (MacroInfobot)
├── data/
│   ├── raw_market_data.csv   # Fetched OHLCV data
│   ├── disagreement_signals.csv  # Backtest results
│   └── cache/                # Cached LLM responses
├── output/
│   ├── results.png           # Main dashboard
│   ├── fig1_disagreement.png # Scatter plot
│   ├── fig2_mean_score.png   # Mean score scatter
│   ├── fig3_timeline.png     # Timeline chart
│   ├── mean_score_vs_realized_volatility_timeseries.png
│   ├── topology.png          # Protocol diagram
│   └── residuals.png         # GARCH diagnostics
├── requirements.txt          # Python dependencies
├── commands.md               # Quick command reference
├── locallyLLM.md            # Ollama setup guide
├── fredapi_usage.md         # FRED API documentation
└── Final_README.md          # This file
```

---

## Configuration Options

All settings are in `src/config.py`:

### LLM Configuration

```python
# Backend selection
LLM_BACKEND = "ollama"  # or "groq"

# Ollama settings
OLLAMA_MODEL = "mistral"
OLLAMA_TEMPERATURE = 0.7
OLLAMA_MAX_TOKENS = 1024

# Groq settings
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_TEMPERATURE = 0.7
```

### Market Configuration

```python
# Analysis period
START_DATE = "2019-01-01"
END_DATE = "2020-01-01"

# Ticker(s) to analyze
TICKER_LIST = ["NVDA"]
DEFAULT_TICKER = "NVDA"
```

### Debate Configuration

```python
DEBATE_ROUNDS = 2  # Optimal for signal preservation
NUM_BELIEF_AGENTS = 3  # Sentiment, Technical, Macro
```

### Analysis Configuration

```python
FORWARD_VOLATILITY_WINDOW = 5  # 5-day forward volatility
TRAIN_TEST_SPLIT = 0.7  # 70% train, 30% test
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: disagreement_signals.csv` | Run `python -m src.backtest` first |
| `FileNotFoundError: raw_market_data.csv` | Run `python -m src.backtest` first |
| `Ollama connection refused` | Run `ollama serve` in a separate terminal |
| `Rate limit errors (Groq)` | Switch to Ollama or wait 60 seconds |
| `GARCH fitting failed` | Need minimum 50 data points |
| `All agents have same score` | Increase `OLLAMA_TEMPERATURE` in config |
| `Empty visualizations` | Ensure 20+ data points in backtest results |

### Common Warnings (Safe to Ignore)

```
Warning: Could not fetch yfinance macro data
  → Falls back to default values, does not affect results

Warning: Using fallback headlines
  → RSS feeds may be rate-limited, uses cached headlines
```

---

## Research References

- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). **Improving Factuality and Reasoning in Language Models through Multiagent Debate**. arXiv:2305.14325.

- Bollerslev, T. (1986). **Generalized Autoregressive Conditional Heteroskedasticity**. Journal of Econometrics, 31(3), 307-327.

---

## License

MIT License - See LICENSE file for details.

---

## Author

Built for research purposes. Not financial advice.
