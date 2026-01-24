# FRED API Usage Guide

How the Agentic Dissonance framework fetches macroeconomic data from FRED.

---

## How Data Fetching Works

### Per-Day Fetching (Current Implementation)

The system fetches FRED data **per trading day**. For each day in your backtest, the `MacroInfobot.fetch_data(date)` method:

1. Checks local cache for that date
2. If not cached, makes **5 separate API calls** to FRED:
   - `FEDFUNDS` - Fed Funds Rate
   - `DGS10` - 10-Year Treasury Yield
   - `DGS2` - 2-Year Treasury Yield  
   - `UNRATE` - Unemployment Rate
   - `CPIAUCSL` - Consumer Price Index (for YoY inflation calculation)

```
Each day → 5 FRED API requests (if not cached)
```

### Lookback Window

For each date, the system fetches data with a **90-day lookback window**:

```python
end_date = date
start_date = date - timedelta(days=90)
```

This ensures it gets the most recent available value for indicators that update monthly (like CPI and unemployment).

---

## Rate Limit Analysis

### FRED API Limits

| Limit Type | Value |
|------------|-------|
| Requests per minute | **120** |
| Daily limit | **None** (unlimited) |
| Monthly limit | **None** (unlimited) |

### Impact on Long Runs

| Days | API Calls | Time at Max Rate | Actual Time* |
|------|-----------|------------------|--------------|
| 3 | 15 | ~8 sec | ~15 sec |
| 50 | 250 | ~2 min | ~4 min |
| 250 | 1,250 | ~10 min | ~20 min |
| 375 | 1,875 | ~16 min | ~30 min |

*Actual time includes network latency and processing overhead.

### Will 375 Days Hit Rate Limits?

**Short answer: No, you should be fine.**

- **375 days × 5 calls = 1,875 API requests**
- At 120 req/min limit = ~16 minutes if running at full speed
- The system naturally throttles due to LLM processing time between days
- Each day's LLM analysis takes ~30-60 seconds, giving FRED API plenty of breathing room

---

## Caching Behavior

### In-Memory Cache (Current Session)

- Data is cached per date: `cache_key = date.strftime('%Y-%m-%d')`
- Same date won't hit API twice in one session
- Cache resets when script restarts

### No Persistent Cache

Currently, FRED data is **not persisted to disk**. Each new backtest run will re-fetch all data.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Daily Backtest Loop                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Day 1 (2024-01-02)                                            │
│  ├── MacroInfobot.fetch_data(date=2024-01-02)                  │
│  │   ├── Check cache → Miss                                    │
│  │   ├── _fetch_from_fred(date)                                │
│  │   │   ├── GET FEDFUNDS (90-day window)                      │
│  │   │   ├── GET DGS10 (90-day window)                         │
│  │   │   ├── GET DGS2 (90-day window)                          │
│  │   │   ├── GET UNRATE (90-day window)                        │
│  │   │   └── GET CPIAUCSL (400-day window for YoY)             │
│  │   └── Cache result for 2024-01-02                           │
│  └── Return macro context to agents                            │
│                                                                 │
│  Day 2 (2024-01-03)                                            │
│  ├── MacroInfobot.fetch_data(date=2024-01-03)                  │
│  │   ├── Check cache → Miss (different date)                   │
│  │   └── _fetch_from_fred() → 5 more API calls                 │
│  ...                                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Not Fetch All Data Upfront?

The current approach fetches data day-by-day. Alternative approaches:

| Approach | Pros | Cons |
|----------|------|------|
| **Per-Day (Current)** | Simple, no preprocessing | More API calls |
| **Bulk Prefetch** | Fewer API calls, faster | Requires startup delay |
| **Disk Cache** | Survives restarts | File management overhead |

### Potential Optimization

A bulk prefetch approach could fetch all series once:
```
1 call per series × 5 series = 5 total API calls
```

This would reduce **1,875 calls → 5 calls** for a 375-day backtest.

---

## FRED Data Update Frequency

| Series | Update Frequency | Release Lag |
|--------|------------------|-------------|
| FEDFUNDS | Monthly | ~2 weeks |
| DGS10 | Daily | Same day |
| DGS2 | Daily | Same day |
| UNRATE | Monthly | ~1 week |
| CPIAUCSL | Monthly | ~2 weeks |

This means for historical backtests, the data is stable and already available.

---

## Summary

| Question | Answer |
|----------|--------|
| Is data fetched day-by-day? | **Yes**, 5 API calls per day |
| Will 375 days hit rate limits? | **Unlikely** - LLM processing creates natural throttling |
| Is data cached? | **Yes**, in-memory per session |
| Can this be optimized? | **Yes**, bulk prefetch would reduce calls by 99% |

---

## Related Files

- [`src/infobots.py`](src/infobots.py) - MacroInfobot implementation
- [`src/config.py`](src/config.py) - FRED_API_KEY and series configuration
