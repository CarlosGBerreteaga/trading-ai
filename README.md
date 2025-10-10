# Trading AI Pipeline

Automated workflow for downloading market data, engineering predictive features, training a gradient boosting classifier, and backtesting trading signals with optional portfolio hedging and alerting.

## Key Capabilities
- End-to-end orchestration via `src/pipeline.py` for a single ticker (fetch → features → train → backtest).
- Rich feature set built with the `ta` technical-analysis library and multiple return horizons.
- Backtesting with slippage/cost modelling, minimum-hold enforcement, hedging support, and performance stats (CAGR, Sharpe, drawdowns).
- Notification hooks through Twilio SMS or [`ntfy`](https://ntfy.sh) push topics so you can receive trade alerts.
- Batch recommendation runner (`src/recommend.py`) that scores S&P 500 constituents and writes ranked output to `analysis_recommendations.csv`.

## Repository Layout
```text
├── data/                 # CSV downloads, engineered parquet features, backtest outputs
├── models/               # Saved scikit-learn models (`*.pkl`)
├── src/
│   ├── data_download.py  # Fetches historical prices from Yahoo! Finance (yfinance)
│   ├── clean_csv.py      # Sanitises raw CSV files (date parsing, numeric coercion)
│   ├── features.py       # Builds technical indicators & targets and writes parquet features
│   ├── train.py          # Trains GradientBoostingClassifier and reports metrics
│   ├── backtest.py       # Generates trading signals, equity curve, stats, and alert CSV
│   ├── pipeline.py       # Glue script that runs the full single-symbol flow
│   ├── recommend.py      # Evaluates many tickers and ranks by CAGR/Sharpe
│   ├── notify.py         # Twilio SMS and ntfy push helpers
│   └── utils.py          # Shared utilities (datetime indexing helpers)
├── run_pipeline.ps1      # Example PowerShell wrapper for the pipeline
├── run_recommend.ps1     # Example PowerShell wrapper for the recommendation sweep
├── requirements.txt      # Python dependencies
└── sp500_tickers.txt     # Optional cached ticker universe (generated on demand)
```

## Prerequisites
- Python 3.10+
- `pip` / `python3-venv` (install via `sudo apt install python3-venv python3-pip` on WSL/Ubuntu)
- Optional: Twilio credentials and/or ntfy topic if you want live notifications

## Installation

### WSL / Linux / macOS
```bash
python3 -m venv .venv-linux
source .venv-linux/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Tip: if you only work inside WSL, you can delete the Windows `.venv` after migrating scripts to `.venv-linux`.

## Running the Single-Symbol Pipeline
```powershell
python src/pipeline.py --symbol SPY --years 5 --prob-threshold 0.55 --min-hold-days 21
```

What happens:
1. `data_download.py` downloads daily OHLCV data into `data/<symbol>.csv`.
2. `features.py` engineers indicators and writes `data/<symbol>_features.parquet`.
3. `train.py` trains a `GradientBoostingClassifier`, prints metrics for train/valid/test splits, and saves the model to `models/<symbol>_gbdt.pkl`.
4. `backtest.py` scores probabilities, applies a long-only strategy with optional hedging, and writes:
   - Equity curve CSV: `data/<symbol>_backtest.csv`
   - Alert stream CSV: `data/<symbol>_alerts.csv`
   - Performance summary printed to stdout.

Command-line flags let you customise the lookback window (`--years`), probability threshold, holding period, hedge ticker/weight, output directories, and notification settings. See `python src/pipeline.py --help` for the full list.

### Enabling Notifications
- **Twilio SMS**: set `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, and `TWILIO_FROM_NUMBER` env vars (or pass the corresponding CLI flags) plus `--notify-phone` with your destination number.
- **ntfy push**: provide `--ntfy-topic` (and optionally `--ntfy-token`, `--ntfy-title`, `--ntfy-priority`). Alerts are throttled by `--ntfy-limit`.

The bundled `run_pipeline.ps1` demonstrates wiring these environment variables before launching the pipeline. It now shells into WSL and activates `.venv-linux` automatically—adjust the `$distribution` variable at the top of the script if you use a non-default WSL distro name.

## Cleaning Existing CSV Files
If you have historical CSVs from other sources, `clean_csv.py` can tidy formatting before feature generation:
```powershell
python src/clean_csv.py --in path\to\raw.csv --out path\to\clean.csv
```
It removes stray header lines, normalises dates/numerics, and sorts by date.

## Generating Bulk Recommendations
```powershell
python src/recommend.py --years 20 --prob-threshold 0.55 --min-hold-days 252 --max-tickers 100 --min-trading-days 756
```

`recommend.py` will:
1. Load tickers from `sp500_tickers.txt` (or scrape the S&P 500 list from Wikipedia if the file is missing).
2. Run the full pipeline for each symbol (respecting `--max-tickers` to bound runtime).
3. Skip symbols whose backtests cover fewer trading days than `--min-trading-days` (default 504) so newly listed tickers don’t dominate the rankings.
4. Save the aggregated metrics to `analysis_recommendations.csv`.
5. Print the top-ranked ideas ordered by CAGR and Sharpe ratio.

See `run_recommend.ps1` for a turnkey example with default arguments.

## Interactive Web UI

A lightweight FastAPI server exposes the pipeline through a browser dashboard.

```powershell
.\\.venv\\Scripts\\Activate   # or `source .venv-linux/bin/activate` when using WSL/Linux
uvicorn src.ui_server:app --reload
```

Then open http://127.0.0.1:8000/ to:
- launch single-symbol backtests without touching the CLI,
- kick off the S&P 500 recommendation sweep,
- monitor queued/running/completed jobs and inspect stdout/stderr logs,
- review the latest `analysis_recommendations.csv` table directly in the UI.

Jobs run in the background so the interface stays responsive. The API is exposed under `/api/*` if you prefer to automate submissions from scripts.

## Tracking Owned Positions

Sell notifications now respect a persistent `portfolio.json` file stored in the repo root. The pipeline still surfaces every BUY opportunity, but a SELL alert is only emitted when the symbol appears in `portfolio.json`.

```powershell
# show all currently owned tickers
python src/portfolio.py list

# add or remove positions as you trade
python src/portfolio.py add TSLA MSFT
python src/portfolio.py remove TSLA
```

`run_pipeline.ps1` and the FastAPI UI both default to updating `portfolio.json` automatically when a buy or sell signal occurs (toggle this behaviour with `--portfolio-update` on the CLI or the “Auto-update portfolio” checkbox in the UI). Pass `--portfolio ""` if you want to disable filtering for a one-off run.

### One-Click Portfolio Rebalance

`src/auto_trade.py` bundles the recommendation sweep, ranks the ideas, and updates `portfolio.json` in one pass:

```powershell
python src/auto_trade.py --years 10 --max-tickers 50 --top 10
```

### Bankroll Simulation

Project the growth of a $1,000 bankroll by ranking every evaluated strategy and allocating capital to the strongest performers:

```powershell
python src/simulation.py --initial-capital 1000 --years 20 --max-tickers 200 --top 5
```

The command prints a JSON summary with per-strategy allocations, trade statistics, and aggregate returns. The FastAPI dashboard now includes a **Capital Growth Simulation** section so you can launch the same analysis from the browser, watch progress in real time, and review the executed trades in a formatted table.

By default it replaces the current holdings with the top `--top` symbols and writes a trade summary to `data/auto_trades/<timestamp>.json` (check the `CAGR_mean` in that file for a quick 12‑month return proxy). Add `--dry-run` to preview without touching the portfolio, or `--keep-existing` if you only want to append new buys.

## Data & Output Directories
- Raw data: `data/<symbol>.csv`
- Feature store: `data/<symbol>_features.parquet`
- Backtests: `data/<symbol>_backtest.csv`
- Alerts: `data/<symbol>_alerts.csv`
- Models: `models/<symbol>_gbdt.pkl`

These folders are created automatically. Remove files manually if you want to rerun from scratch.

## Troubleshooting
- Ensure Yahoo! Finance (`yfinance`) returns data for your ticker and date range; thinly traded symbols may return empty frames.
- If you see missing feature columns during training/backtests, confirm `generate_features` completed successfully.
- Twilio notifications require the `twilio` package and valid credentials; ntfy requires outbound HTTPS connectivity.

## Next Steps
1. Experiment with alternative model families (e.g., `HistGradientBoostingClassifier`, XGBoost) by editing `train.py`.
2. Extend `features.py` with domain-specific indicators or macro inputs.
3. Build a FastAPI/uvicorn service (dependencies already present) to expose forecasting endpoints if you plan to operationalise the model.
