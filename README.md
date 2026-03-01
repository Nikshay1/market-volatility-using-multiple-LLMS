# Agentic Dissonance v3: Multi-Agent Disagreement for Volatility Forecasting

> **Research framework for testing whether disagreement among heterogeneous LLM agents improves volatility prediction out-of-sample.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

1. [What This Project Proves](#what-this-project-proves)
2. [Core Hypothesis](#core-hypothesis)
3. [What Changed in v3 (Research-Critical)](#what-changed-in-v3-research-critical)
4. [System Architecture](#system-architecture)
5. [Data Design and Anti-Leakage Rules](#data-design-and-anti-leakage-rules)
6. [Agent and Debate Protocol](#agent-and-debate-protocol)
7. [Disagreement Signal](#disagreement-signal)
8. [Modeling and Evaluation](#modeling-and-evaluation)
9. [Experiment Design for Professor-Grade Evidence](#experiment-design-for-professor-grade-evidence)
10. [Repository Structure](#repository-structure)
11. [Setup](#setup)
12. [Command Reference](#command-reference)
13. [Outputs and Artifacts](#outputs-and-artifacts)
14. [Interpreting Results](#interpreting-results)
15. [Known Limitations](#known-limitations)
16. [Roadmap](#roadmap)

---

## What This Project Proves

This project is not a trading bot.

It is a **research testbed** for one specific scientific claim:

> If specialized AI agents disagree more today, then tomorrow’s market volatility should be higher.

The key deliverable is therefore **out-of-sample forecasting evidence**, not just model fitting or pretty plots.

---

## Core Hypothesis

\[
\sigma^{RV}_{t+1:t+h} = f(D_t, \text{market history}) + \varepsilon_t
\]

Where:

- \(D_t\): confidence-weighted disagreement among belief agents at date \(t\)
- \(\sigma^{RV}_{t+1:t+h}\): future realized volatility over horizon \(h\)

**Research success criterion:**

1. A disagreement-augmented volatility model beats a strong baseline **out-of-sample**.
2. Improvement is robust across assets and market regimes.
3. Improvement is statistically tested (not anecdotal).

---

## What Changed in v3 (Research-Critical)

The framework has been re-focused around five fundamental improvements required for academically defensible results.

### 1) Volatility model specification is aligned with the hypothesis

Previously, disagreement risk could be inserted in ways that blur mean vs variance interpretation.

**Now:** model design explicitly treats disagreement as a **volatility-relevant feature**, with transparent lagging and specification reporting.

---

### 2) Evaluation is out-of-sample first

In-sample fit is no longer considered proof.

**Now:**

- rolling/recursive test forecasts are primary,
- metrics are computed on held-out windows,
- model comparison focuses on OOS RMSE/MAE/QLIKE,
- optional forecast-comparison significance testing is recommended.

---

### 3) News context is quality-controlled

Weak or generic headlines can destroy signal quality.

**Now:**

- historical news coverage is treated as a required dataset,
- lookback windows are configurable,
- coverage diagnostics are tracked,
- low-information days can be flagged/excluded in primary analysis.

---

### 4) Agent heterogeneity is structural (not cosmetic)

Prompt differences alone are often insufficient.

**Now:** agent diversity is treated as a measurable property:

- differentiated roles and feature focus,
- independent scoring behavior monitored over time,
- disagreement decomposition available for diagnostics.

---

### 5) Research scope is expanded for robustness

Single ticker / single year evidence is fragile.

**Now:** experiments are designed for multi-ticker, multi-regime validation to support generalizable conclusions.

---

## System Architecture

```text
Market Data + Historical News + Macro Proxies
                |
                v
      Context Builder (date-safe)
                |
                v
     Multi-Agent Debate (Blind -> Critique)
                |
                v
  Disagreement Signal Construction (D_conf)
                |
                v
 Volatility Modeling (Baseline vs Disagreement-Augmented)
                |
                v
Out-of-Sample Evaluation + Robustness Diagnostics
```

---

## Data Design and Anti-Leakage Rules

To keep the study academically valid:

- **No look-ahead in market context**: each date uses only data available up to that date.
- **No look-ahead in news**: only headlines published on/before decision time.
- **No future fundamentals leakage**: avoid current snapshot fundamentals for historical dates unless point-in-time data is available.
- **Explicit lagging**: disagreement at \(t\) predicts future volatility, not contemporaneous volatility.

If any data source cannot satisfy these constraints, it should be disabled or clearly labeled as exploratory only.

---

## Agent and Debate Protocol

The project uses a two-stage protocol to preserve meaningful disagreement:

1. **Blind Round**: each agent predicts independently from shared date-safe context.
2. **Critique Round**: each agent is shown an opposing argument and may revise with justification.

Agents output:

- `score` in [-1, 1]
- `confidence` in [0, 1]
- `reasoning` (auditable text)

This creates an interpretable social signal rather than a simple average opinion.

---

## Disagreement Signal

Primary signal:

- **Confidence-weighted variance** of agent scores (`disagreement_conf`)

Useful companion diagnostics:

- mean score,
- average confidence,
- pairwise disagreement/correlation,
- disagreement change (first difference).

---

## Modeling and Evaluation

### Baseline

- Standard volatility model on returns only.

### Disagreement-augmented model

- Baseline + disagreement-derived feature(s), with explicit lag structure.

### Evaluation protocol (recommended)

- train/test split by time,
- rolling one-step-ahead forecasts in test period,
- compare OOS metrics:
  - RMSE
  - MAE
  - QLIKE
- optionally add Diebold–Mariano or bootstrap difference tests.

### Publication-safe interpretation

A model is considered better only if improvements are:

1. OOS,
2. repeated across assets/regimes,
3. statistically non-random.

---

## Experiment Design for Professor-Grade Evidence

Use this minimal matrix:

- **Assets**: multiple equities + index proxy (e.g., NVDA, AAPL, MSFT, SPY)
- **Regimes**: at least 3 distinct periods (calm, crisis, recovery/rate cycle)
- **Frequencies**: daily signal construction with consistent forecast horizon
- **Outputs**:
  1. per-asset OOS table,
  2. regime-wise performance table,
  3. aggregate win-rate summary,
  4. failure-case analysis.

If disagreement helps only one ticker/period, report that honestly as a conditional finding.

---

## Repository Structure

```text
market-volatility-using-multiple-LLMS/
├── README.md
├── requirements.txt
├── scripts/
│   └── import_kaggle.py
└── src/
    ├── agents.py          # Belief agents (sentiment/technical/macro)
    ├── aggregator.py      # Disagreement aggregation + critique formatting
    ├── analysis.py        # Volatility modeling + evaluation + visualization
    ├── backtest.py        # Daily debate backtest runner
    ├── config.py          # Config (tickers, dates, model settings, paths)
    ├── data_loader.py     # Market/news loading + context formatting
    ├── debate_engine.py   # Blind & critique orchestration
    ├── disagreement.py    # Disagreement metrics
    └── infobots.py        # Structured macro/fundamental context injectors
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

```bash
# Choose one backend
export LLM_BACKEND=ollama
# or
export LLM_BACKEND=groq

# If using Groq
export GROQ_API_KEY="your_key"

# Optional
export FRED_API_KEY="your_fred_key"
```

---

## Command Reference

### 1) Quick backtest smoke test

```bash
python -m src.backtest --test --days 3
```

### 2) Full backtest

```bash
python -m src.backtest
```

### 3) Resume backtest from a specific date

If the backtest is interrupted (e.g., API errors, connection loss, or manual stop), you can resume from any date without losing previous progress. All results before the resume date are preserved.

```bash
python -m src.backtest --resume 2018-03-15
```

This keeps all existing data in `data/disagreement_signals.csv` up to `2018-03-14` and restarts processing from `2018-03-15` onward.

You can also combine `--resume` with other flags:

```bash
# Resume for a specific ticker with custom debate rounds
python -m src.backtest --resume 2019-06-01 --ticker NVDA --rounds 2
```

### 4) Run analysis

```bash
python -m src.analysis
```

### 5) Compare debate round counts

```bash
python -m src.backtest --compare-rounds --days 5
```

---

## Outputs and Artifacts

Typical artifacts:

- `data/raw_market_data.csv`
- `data/disagreement_signals.csv`
- figures under `output/`

For reproducible research reports, also export:

- OOS forecast series by date,
- per-model metric tables,
- regime-sliced summaries.

---

## Interpreting Results

### Strong positive result

- OOS disagreement model beats baseline in most assets/regimes,
- significance tests support non-random improvement,
- effect direction aligns with hypothesis (higher disagreement -> higher future vol).

### Mixed result

- improvement appears only in stress regimes or specific sectors.

### Negative result

- baseline remains stronger; disagreement behaves as noise under current setup.

A negative result is still publishable if methodology is rigorous.

---

## Known Limitations

- LLM output instability across model versions.
- News source quality/coverage dependency.
- Potential sensitivity to prompt wording and debate structure.
- Forecast gains may be regime-conditional rather than universal.

---

## Roadmap

- Add formal forecast comparison tests and confidence intervals by default.
- Add automated data quality gates (coverage, missingness, lag validation).
- Add richer disagreement decomposition and explainability dashboard.
- Add standardized experiment manifests for exact reproducibility.

---

## Citation / Research Reporting Note

If you use this framework in an academic submission, include:

1. data coverage table,
2. anti-leakage methodology,
3. OOS protocol details,
4. robustness checks,
5. all negative/neutral findings (not only wins).

That transparency is what turns a demo into research evidence.
