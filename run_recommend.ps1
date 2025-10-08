# C:\Users\carlo\Documents\Portfolio Projects\trading-ai\run_recommend.ps1
# Runs the S&P 500 recommendation scraper and writes results to analysis_recommendations.csv.

$env:PYTHONUNBUFFERED = "1"

& "C:\Users\carlo\Documents\Portfolio Projects\trading-ai\.venv\Scripts\python.exe" `
  "C:\Users\carlo\Documents\Portfolio Projects\trading-ai\src\recommend.py" `
  --years 20 `
  --prob-threshold 0.55 `
  --min-hold-days 252 `
  --max-tickers 500 `
  --top 25 `
  --ticker-cache "C:\Users\carlo\Documents\Portfolio Projects\trading-ai\sp500_tickers.txt" `
  --output "C:\Users\carlo\Documents\Portfolio Projects\trading-ai\analysis_recommendations.csv"
