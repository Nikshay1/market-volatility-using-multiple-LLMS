# Commands Reference

Complete command reference for Agentic Dissonance v2.

---

## Prerequisites

```powershell
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set FRED API key for real macroeconomic data
# Get free key: https://fred.stlouisfed.org/docs/api/api_key.html
$env:FRED_API_KEY = "your-fred-api-key"

# 4. (Optional) Set Groq API key if using cloud backend
$env:GROQ_API_KEY = "your-groq-api-key"
```

> **Note**: Without `FRED_API_KEY`, the macro agent uses simulated data.

---

## Backtest Commands

### Quick Test

```powershell
python -m src.backtest --test --days 3
```

Runs a quick validation on 3 days for the default ticker (AAPL).

**Output**: Prints debate results to console, saves to `data/disagreement_signals.csv`

---

### Full Backtest (All Tickers)

```powershell
python -m src.backtest
```

Runs the full backtest for all configured tickers: **AAPL, MSFT, TSLA, SPY**

**Date Range**: 2024-01-01 to 2024-12-31 (configurable in `src/config.py`)

**Output**: `data/disagreement_signals.csv` with columns:
- `date`, `ticker`
- `disagreement_conf`, `mean_score`, `avg_confidence`
- `score_fundamental`, `score_sentiment`, `score_technical`, `score_macro`
- `confidence_fundamental`, `confidence_sentiment`, `confidence_technical`, `confidence_macro`

---

### Single Ticker Backtest

```powershell
python -m src.backtest --ticker AAPL
```

Runs backtest for a single ticker only.

---

### Resume from Date

```powershell
python -m src.backtest --resume 2024-06-01
```

Continues a previously interrupted backtest from the specified date.

---

### Compare Debate Rounds

```powershell
python -m src.backtest --compare-rounds
```

Compares 2, 3, and 4 debate round configurations on a sample of days.

**Output**: Table comparing disagreement, confidence, and runtime for each configuration.

---

### CLI Arguments Summary

| Argument | Description | Example |
|----------|-------------|---------|
| `--test` | Enable test mode | `--test --days 5` |
| `--days N` | Number of days in test mode | `--days 10` |
| `--ticker SYM` | Single ticker to process | `--ticker TSLA` |
| `--resume DATE` | Resume from YYYY-MM-DD | `--resume 2024-06-01` |
| `--rounds N` | Number of debate rounds | `--rounds 2` |
| `--compare-rounds` | Compare 2/3/4 round configs | |
| `--quiet` | Suppress progress output | |

---

## Analysis Commands

### Run Full Analysis

```powershell
python -m src.analysis
```

Performs complete statistical analysis:

1. **Loads data** from CSV files
2. **Computes 5-day forward realized volatility**
3. **Runs correlation analysis** (Pearson + Spearman)
4. **Fits GARCH(1,1) baseline model**
5. **Fits GARCH-X with D_conf** as exogenous variable
6. **Compares models**: AIC, BIC, RMSE, MAE
7. **Generates visualization**: `output/results.png`

**Output**:
```
============================================================
AGENTIC DISSONANCE V2 - ANALYSIS
============================================================

Model Comparison:
-------------------------------------------------
Metric          GARCH(1,1)      GARCH-X         Better
-------------------------------------------------
AIC             1245.32         1238.45         GARCH-X
BIC             1256.78         1253.21         GARCH-X
RMSE            0.012345        0.011234        GARCH-X
MAE             0.009876        0.008765        GARCH-X

GARCH-X exogenous coefficient: 0.0234
GARCH-X exogenous p-value: 0.0156 **
```

---

## Visualization Output

The analysis generates a 6-panel visualization:

| Panel | Content |
|-------|---------|
| Top-Left | Disagreement vs Forward Volatility scatter |
| Top-Right | Time series: Disagreement & Volatility |
| Mid-Left | Model comparison: AIC & BIC |
| Mid-Right | Model comparison: RMSE & MAE |
| Bottom-Left | Agent score distributions |
| Bottom-Right | Agent confidence distributions |

Saved to: `output/results.png`

---

## Workflow

```
┌─────────────────────────────────────────────────┐
│                Complete Workflow                 │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. Quick Test                                   │
│     python -m src.backtest --test --days 3      │
│              ↓                                   │
│  2. Full Backtest                               │
│     python -m src.backtest                      │
│              ↓                                   │
│  3. Analysis                                    │
│     python -m src.analysis                      │
│              ↓                                   │
│  4. Review Results                              │
│     output/results.png                          │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## Estimated Runtime

### With Ollama (Local)

| Days | Tickers | Approximate Time |
|------|---------|------------------|
| 3 | 1 | ~5 min |
| 10 | 1 | ~20 min |
| 250 | 1 | ~8 hours |
| 250 | 4 | ~32 hours |

### With Groq (Cloud)

| Days | Rate Limit Impact |
|------|-------------------|
| 5 | No issues |
| 20+ | May hit rate limits |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `ollama` | LLM backend: `ollama` or `groq` |
| `GROQ_API_KEY` | - | Required for Groq backend |
| `FRED_API_KEY` | - | FRED API key for real macro data |
| `OLLAMA_MODEL` | `mistral` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: disagreement_signals.csv` | Run backtest first |
| `FileNotFoundError: raw_market_data.csv` | Run backtest first |
| `Ollama connection refused` | Run `ollama serve` |
| Rate limit errors | Switch to Ollama or wait |
| Low GARCH improvement | Normal - not all periods show signal |
| Empty visualization | Ensure sufficient data points (20+) |

---

## Advanced Usage

### Modify Date Range

Edit `src/config.py`:
```python
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"
```

### Modify Ticker List

Edit `src/config.py`:
```python
TICKER_LIST = ["AAPL", "GOOGL", "AMZN", "NVDA"]
```

### Change Debate Rounds

```powershell
python -m src.backtest --rounds 3
```

Or edit `src/config.py`:
```python
DEBATE_ROUNDS = 2  # Default: 2 rounds
```
