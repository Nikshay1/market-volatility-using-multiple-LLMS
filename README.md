# Usage Instructions
1. Install Dependencies

pip install -r requirements.txt

2. Set API Key

$env:GROQ_API_KEY = "your-groq-api-key"

3. Run Backtest

Quick test (5 days)
```python
python -m src.backtest --test --days 5
```

Full backtest
```pyhton
python -m src.backtest
```

4. Run Analysis
```python
python -m src.analysis
``` 