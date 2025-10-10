from __future__ import annotations

import io
import threading
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from .pipeline import run_pipeline
from .recommend import evaluate_tickers, load_tickers

try:
    from .portfolio import DEFAULT_PORTFOLIO_PATH
except ImportError:  # pragma: no cover
    from portfolio import DEFAULT_PORTFOLIO_PATH  # type: ignore[no-redef]
try:
    from .simulation import run_simulation
except ImportError:  # pragma: no cover
    from simulation import run_simulation  # type: ignore[no-redef]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_TICKER_CACHE = PROJECT_ROOT / "sp500_tickers.txt"
DEFAULT_RECOMMENDATIONS = PROJECT_ROOT / "analysis_recommendations.csv"
DEFAULT_PORTFOLIO = DEFAULT_PORTFOLIO_PATH


def _now() -> datetime:
    return datetime.utcnow()


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.replace(microsecond=0).isoformat() + "Z"


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, dict):
        return {key: _jsonify(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    return value


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobKind(str, Enum):
    PIPELINE = "pipeline"
    RECOMMEND = "recommend"
    SIMULATION = "simulation"


class PipelineRequest(BaseModel):
    symbol: str = Field(default="SPY", min_length=1, max_length=8)
    years: int = Field(default=5, ge=1, le=30)
    prob_threshold: float = Field(default=0.55, gt=0.0, lt=1.0)
    min_hold_days: int = Field(default=21, ge=1, le=252)
    hedge_symbol: Optional[str] = Field(default=None, max_length=8)
    hedge_weight: float = Field(default=1.0, ge=0.0, le=3.0)
    data_dir: Path = Field(default=DEFAULT_DATA_DIR)
    models_dir: Path = Field(default=DEFAULT_MODELS_DIR)
    portfolio_path: Optional[Path] = Field(default=DEFAULT_PORTFOLIO)
    portfolio_update: bool = Field(default=True)

    @field_validator("portfolio_path", mode="before")
    @classmethod
    def _blank_portfolio(cls, value: Any) -> Any:
        if value in (None, "", "null", "None"):
            return None
        return value


class RecommendRequest(BaseModel):
    years: int = Field(default=20, ge=1, le=30)
    prob_threshold: float = Field(default=0.55, gt=0.0, lt=1.0)
    min_hold_days: int = Field(default=252, ge=1, le=504)
    max_tickers: int = Field(default=50, ge=1, le=500)
    top: int = Field(default=25, ge=1, le=100)
    min_trading_days: int = Field(
        default=504,
        ge=0,
        le=2520,
        description="Skip recommendations whose backtests span fewer trading days than this threshold.",
    )
    data_dir: Path = Field(default=DEFAULT_DATA_DIR)
    models_dir: Path = Field(default=DEFAULT_MODELS_DIR)
    ticker_cache: Path = Field(default=DEFAULT_TICKER_CACHE)
    output: Path = Field(default=DEFAULT_RECOMMENDATIONS)


class SimulationRequest(BaseModel):
    initial_capital: float = Field(default=1000.0, gt=0)
    years: int = Field(default=20, ge=1, le=30)
    prob_threshold: float = Field(default=0.55, gt=0.0, lt=1.0)
    min_hold_days: int = Field(default=252, ge=1, le=504)
    max_tickers: int = Field(default=200, ge=1, le=500)
    top: int = Field(default=5, ge=1, le=50)
    min_trading_days: int = Field(default=756, ge=0, le=2520)
    data_dir: Path = Field(default=DEFAULT_DATA_DIR)
    models_dir: Path = Field(default=DEFAULT_MODELS_DIR)
    ticker_cache: Path = Field(default=DEFAULT_TICKER_CACHE)


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, kind: JobKind, params: Dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "type": kind.value,
            "status": JobStatus.QUEUED.value,
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "params": params,
            "result": None,
            "stdout": "",
            "stderr": "",
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        return job_id

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = JobStatus.RUNNING.value
            job["started_at"] = _now()

    def mark_completed(self, job_id: str, result: Dict[str, Any], stdout: str, stderr: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = JobStatus.COMPLETED.value
            job["finished_at"] = _now()
            job["result"] = _jsonify(result)
            job["stdout"] = stdout[-20000:]
            job["stderr"] = stderr[-20000:]

    def mark_failed(self, job_id: str, error: str, stdout: str, stderr: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = JobStatus.FAILED.value
            job["finished_at"] = _now()
            job["error"] = error
            job["stdout"] = stdout[-20000:]
            job["stderr"] = stderr[-20000:]

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda item: item["created_at"], reverse=True)
        return [self._serialize(job, include_logs=False) for job in jobs]

    def get(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
        return self._serialize(job, include_logs=True)

    @staticmethod
    def _serialize(job: Dict[str, Any], include_logs: bool) -> Dict[str, Any]:
        payload = {
            "id": job["id"],
            "type": job["type"],
            "status": job["status"],
            "created_at": _iso(job["created_at"]),
            "started_at": _iso(job["started_at"]),
            "finished_at": _iso(job["finished_at"]),
            "params": _jsonify(job["params"]),
            "result": _jsonify(job["result"]),
            "error": job["error"],
        }
        if include_logs:
            payload["stdout"] = job["stdout"]
            payload["stderr"] = job["stderr"]
        return payload


manager = JobManager()
app = FastAPI(title="Trading AI UI", version="1.0.0")


def run_pipeline_job(job_id: str, request: PipelineRequest) -> None:
    manager.mark_running(job_id)
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            summary = run_pipeline(
                symbol=request.symbol,
                years=request.years,
                data_dir=str(request.data_dir),
                models_dir=str(request.models_dir),
                prob_threshold=request.prob_threshold,
                min_hold_days=request.min_hold_days,
                hedge_symbol=request.hedge_symbol or None,
                hedge_weight=request.hedge_weight,
                portfolio_path=str(request.portfolio_path) if request.portfolio_path else None,
                portfolio_update=request.portfolio_update,
            )
        result = {
            "symbol": request.symbol,
            "total_return": summary.get("total_return"),
            "stats": summary.get("stats"),
            "model_path": summary.get("model_path"),
            "backtest_csv": summary.get("backtest_csv"),
            "alerts_csv": summary.get("alerts_csv"),
            "portfolio_path": summary.get("portfolio_path"),
            "portfolio_owned_before": summary.get("portfolio_owned_before"),
            "portfolio_owned_after": summary.get("portfolio_owned_after"),
            "trade_summary": summary.get("trade_summary"),
            "trades": summary.get("trades"),
        }
        manager.mark_completed(job_id, result, stdout.getvalue(), stderr.getvalue())
    except Exception as exc:
        manager.mark_failed(job_id, str(exc), stdout.getvalue(), stderr.getvalue())


def run_recommend_job(job_id: str, request: RecommendRequest) -> None:
    manager.mark_running(job_id)
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            tickers = load_tickers(request.ticker_cache, request.max_tickers)
            results = evaluate_tickers(
                symbols=tickers,
                years=request.years,
                prob_threshold=request.prob_threshold,
                min_hold_days=request.min_hold_days,
                data_dir=str(request.data_dir),
                models_dir=str(request.models_dir),
                min_trading_days=request.min_trading_days,
            )
            results.to_csv(request.output, index=False)
        ok_results = results[results["status"] == "ok"].copy()
        ok_results = ok_results.sort_values(by=["CAGR", "Sharpe"], ascending=False)
        top = ok_results.head(request.top)
        result = {
            "evaluated": int(len(results)),
            "successful": int(len(ok_results)),
            "output": str(request.output),
            "top": top.to_dict(orient="records"),
        }
        manager.mark_completed(job_id, result, stdout.getvalue(), stderr.getvalue())
    except Exception as exc:
        manager.mark_failed(job_id, str(exc), stdout.getvalue(), stderr.getvalue())


def run_simulation_job(job_id: str, request: SimulationRequest) -> None:
    manager.mark_running(job_id)
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = run_simulation(
                initial_capital=request.initial_capital,
                years=request.years,
                prob_threshold=request.prob_threshold,
                min_hold_days=request.min_hold_days,
                max_tickers=request.max_tickers,
                top=request.top,
                data_dir=request.data_dir,
                models_dir=request.models_dir,
                ticker_cache=request.ticker_cache,
                min_trading_days=request.min_trading_days,
            )
        manager.mark_completed(job_id, result, stdout.getvalue(), stderr.getvalue())
    except Exception as exc:
        manager.mark_failed(job_id, str(exc), stdout.getvalue(), stderr.getvalue())


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Trading AI Control Panel</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      color-scheme: dark;
    }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      line-height: 1.5;
    }
    header {
      padding: 24px;
      background: #1e293b;
      border-bottom: 1px solid #334155;
    }
    h1 {
      margin: 0 0 8px 0;
      font-size: 1.8rem;
      font-weight: 600;
    }
    main {
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
    }
    section {
      margin-bottom: 32px;
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 12px;
      padding: 20px 24px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.48);
    }
    section h2 {
      margin-top: 0;
      font-size: 1.3rem;
      font-weight: 600;
      color: #93c5fd;
    }
    form {
      display: grid;
      gap: 16px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
    }
    label {
      display: flex;
      flex-direction: column;
      font-size: 0.9rem;
      color: #cbd5f5;
      gap: 6px;
    }
    label.checkbox {
      flex-direction: row;
      align-items: center;
    }
    label.checkbox span {
      flex: 1;
    }
    label.checkbox input[type="checkbox"] {
      width: 18px;
      height: 18px;
      accent-color: #38bdf8;
    }
    input, select {
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid #334155;
      background: #0f172a;
      color: inherit;
    }
    input:focus {
      outline: 2px solid #38bdf8;
      border-color: #38bdf8;
    }
    button {
      border: none;
      background: #38bdf8;
      color: #0f172a;
      padding: 12px 18px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s ease;
    }
    button:hover {
      background: #0ea5e9;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
      font-size: 0.9rem;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid #1f2937;
      text-align: left;
    }
    tr:hover {
      background: rgba(56, 189, 248, 0.05);
    }
    .status {
      font-size: 0.9rem;
      color: #bfdbfe;
    }
    .positive { color: #4ade80; }
    .negative { color: #f87171; }
    pre {
      background: #0f172a;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid #1f2937;
      overflow-x: auto;
      font-size: 0.85rem;
    }
    .jobs-table tr {
      cursor: pointer;
    }
    .tag {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-weight: 600;
    }
    .tag.pipeline { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
    .tag.recommend { background: rgba(16, 185, 129, 0.2); color: #5eead4; }
    .tag.simulation { background: rgba(244, 114, 182, 0.2); color: #fbcfe8; }
    .tag.completed { background: rgba(74, 222, 128, 0.18); color: #86efac; }
    .tag.running { background: rgba(165, 180, 252, 0.2); color: #c4b5fd; }
    .tag.failed { background: rgba(248, 113, 113, 0.2); color: #fca5a5; }
    .tag.queued { background: rgba(226, 232, 240, 0.1); color: #cbd5f5; }
    .tag.action-buy { background: rgba(34, 197, 94, 0.22); color: #4ade80; }
    .tag.action-sell { background: rgba(248, 113, 113, 0.22); color: #f87171; }
    .tag.action-hold { background: rgba(148, 163, 184, 0.22); color: #e2e8f0; }
    section details.panel {
      display: block;
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 12px;
      padding: 4px 16px 16px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.48);
    }
    section details.panel:not([open]) {
      padding-bottom: 4px;
    }
    section details.panel summary {
      cursor: pointer;
      list-style: none;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 0;
      font-size: 1.2rem;
      font-weight: 600;
      color: #93c5fd;
    }
    section details.panel summary::-webkit-details-marker {
      display: none;
    }
    section details.panel summary::after {
      content: "\25BC";
      font-size: 0.9rem;
      transition: transform 0.2s ease;
      color: #64748b;
    }
    section details.panel[open] summary::after {
      transform: rotate(180deg);
    }
    .summary-cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 18px 0;
    }
    .summary-card {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .summary-card .label {
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #94a3b8;
    }
    .summary-card .value {
      font-size: 1.1rem;
      font-weight: 600;
      color: #e2e8f0;
    }
    .summary-card .value.positive {
      color: #4ade80;
    }
    .summary-card .value.negative {
      color: #f87171;
    }
    .summary-card small {
      font-size: 0.75rem;
      color: #a5b4fc;
    }
    .flex {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
    }
    @media (max-width: 640px) {
      main {
        padding: 16px;
      }
      section {
        padding: 16px;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Trading AI Control Panel</h1>
    <p>Run the backtest pipeline, launch recommendation sweeps, and review the latest signals.</p>
  </header>
  <main>
    <section>
      <details class="panel" open>
        <summary>Single Symbol Pipeline</summary>
        <form id="pipeline-form">
          <div class="grid">
            <label>Symbol
              <input type="text" name="symbol" value="SPY" maxlength="8" required>
            </label>
          <label>Lookback Years
            <input type="number" name="years" value="5" min="1" max="30" required>
          </label>
          <label>Probability Threshold
            <input type="number" name="prob_threshold" value="0.55" step="0.01" min="0.05" max="0.95" required>
          </label>
          <label>Min Hold (days)
            <input type="number" name="min_hold_days" value="21" min="1" max="252" required>
          </label>
          <label>Hedge Symbol (optional)
            <input type="text" name="hedge_symbol" maxlength="8">
          </label>
          <label>Hedge Weight
            <input type="number" name="hedge_weight" value="1.0" step="0.1" min="0" max="3">
          </label>
          <label>Portfolio JSON
            <input type="text" name="portfolio_path" value="portfolio.json" placeholder="portfolio.json">
          </label>
          <label class="checkbox">
            <span>Auto-update portfolio</span>
            <input type="checkbox" name="portfolio_update" checked>
          </label>
        </div>
        <button type="submit">Run Pipeline</button>
        <div class="status" id="pipeline-status"></div>
        </form>
      </details>
    </section>

    <section>
      <details class="panel" open>
        <summary>Capital Growth Simulation</summary>
        <form id="simulation-form">
          <p style="margin: 0 0 12px 0; color: #94a3b8;">
            Allocate a simulated bankroll across the strongest machine-learning strategies over the past 20 years.
          </p>
          <div class="grid">
            <label>Initial Capital ($)
              <input type="number" name="initial_capital" value="1000" min="100" step="50" required>
            </label>
          <label>Lookback Years
            <input type="number" name="years" value="20" min="5" max="30" required>
          </label>
          <label>Probability Threshold
            <input type="number" name="prob_threshold" value="0.55" step="0.01" min="0.05" max="0.95" required>
          </label>
          <label>Min Hold (days)
            <input type="number" name="min_hold_days" value="252" min="1" max="504" required>
          </label>
          <label>Max Tickers
            <input type="number" name="max_tickers" value="200" min="10" max="500" required>
          </label>
          <label>Top Strategies
            <input type="number" name="top" value="5" min="1" max="50" required>
          </label>
          <label>Min Trading Days
            <input type="number" name="min_trading_days" value="756" min="0" max="5000">
          </label>
        </div>
        <button type="submit">Run Simulation</button>
        <div class="status" id="simulation-status"></div>
        </form>
        <div id="simulation-results"></div>
      </details>
    </section>

    <section>
      <details class="panel" open>
        <summary>Recommendation Sweep</summary>
        <form id="recommend-form">
          <div class="grid">
            <label>Lookback Years
              <input type="number" name="years" value="20" min="1" max="30" required>
            </label>
          <label>Probability Threshold
            <input type="number" name="prob_threshold" value="0.55" step="0.01" min="0.05" max="0.95" required>
          </label>
          <label>Min Hold (days)
            <input type="number" name="min_hold_days" value="252" min="1" max="504" required>
          </label>
          <label>Min Trading Days
            <input type="number" name="min_trading_days" value="504" min="0" max="2520" required>
          </label>
          <label>Max Tickers
            <input type="number" name="max_tickers" value="50" min="1" max="500" required>
          </label>
          <label>Top Results
            <input type="number" name="top" value="25" min="1" max="100" required>
          </label>
        </div>
        <button type="submit">Run Recommendations</button>
        <div class="status" id="recommend-status"></div>
        </form>
      </details>
    </section>

    <section>
      <details class="panel" open>
        <summary>Latest Recommendations</summary>
        <div class="flex" style="flex-wrap: wrap; gap: 12px; align-items: flex-end;">
          <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; flex: 2 1 320px;">
            <label>Symbol Search
              <input type="text" id="recs-search" placeholder="Filter by symbol (e.g. AMZN)">
            </label>
            <label>Action Filter
              <select id="recs-action-filter">
                <option value="">All Actions</option>
                <option value="BUY">Buy</option>
                <option value="SELL">Sell</option>
                <option value="HOLD">Hold</option>
              </select>
            </label>
          </div>
          <button id="refresh-recs" type="button">Refresh</button>
        </div>
        <div class="status" id="recs-meta"></div>
        <table id="recs-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Symbol</th>
              <th>Status</th>
              <th>CAGR</th>
              <th>Sharpe</th>
              <th>Total Return %</th>
              <th>Max DD</th>
              <th>Trading Days</th>
              <th>Latest Action</th>
              <th>Action Date</th>
              <th>Latest Prob.</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </details>
    </section>

    <section>
      <details class="panel" open>
        <summary>Job Monitor</summary>
        <table id="jobs-table" class="jobs-table">
          <thead>
            <tr>
              <th>Job</th>
              <th>Type</th>
              <th>Status</th>
              <th>Created</th>
              <th>Started</th>
              <th>Finished</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </details>
    </section>

    <section>
      <details class="panel" open>
        <summary>Job Details</summary>
        <div id="job-details">
          <p>Select a job from the table to view logs and outputs.</p>
        </div>
      </details>
    </section>
  </main>

  <script>
    function formatDate(value) {
      if (!value) return "—";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return value;
      return date.toLocaleString();
    }

    function makeTag(text, cls) {
      return `<span class="tag ${cls}">${text}</span>`;
    }

    function formatCurrency(value) {
      if (value === null || value === undefined || value === "") return "-";
      const num = Number(value);
      if (Number.isNaN(num)) return "-";
      const fixed = num.toFixed(2);
      const [intPart, decimalPart] = fixed.split(".");
      const withCommas = intPart.replace(/\\B(?=(\\d{3})+(?!\\d))/g, ",");
      return `$${withCommas}.${decimalPart}`;
    }

    function formatPercent(value) {
      if (value === null || value === undefined || value === "") return "-";
      const num = Number(value);
      if (Number.isNaN(num)) return "-";
      return `${num.toFixed(2)}%`;
    }

    function formatNumber(value, decimals = 2) {
      if (value === null || value === undefined || value === "") return "—";
      const num = Number(value);
      if (!Number.isFinite(num)) return "—";
      return num.toFixed(decimals);
    }

    function formatDays(value) {
      if (value === null || value === undefined || value === "") return "—";
      const num = Number(value);
      if (Number.isNaN(num)) return "—";
      const units = num === 1 ? "day" : "days";
      const display = Number.isInteger(num) ? num : num >= 10 ? Math.round(num) : Number(num.toFixed(1));
      return `${display} ${units}`;
    }

    function classifyProbability(prob) {
      if (prob === null || prob === undefined || prob === "") return "";
      const num = Number(prob);
      if (Number.isNaN(num)) return "";
      if (num >= 0.55) return "positive";
      if (num <= 0.45) return "negative";
      return "";
    }

    function formatProbability(value) {
      if (value === null || value === undefined || value === "") return "—";
      const num = Number(value);
      if (Number.isNaN(num)) return "—";
      return formatPercent(num * 100);
    }

    function getActionTag(action, fallbackLabel = "—", fallbackClass = "") {
      if (!action || !action.toString().trim()) {
        if (fallbackClass) {
          return makeTag(fallbackLabel, fallbackClass);
        }
        return fallbackLabel;
      }
      const normalized = action.toString().trim().toUpperCase();
      return makeTag(normalized, `action-${normalized.toLowerCase()}`);
    }

    function formToJSON(form) {
      const data = {};
      const formData = new FormData(form);
      formData.forEach((value, key) => {
        const field = form.querySelector(`[name="${key}"]`);
        if (field && field.type === "checkbox") {
          data[key] = field.checked;
          return;
        }
        if (value === "" && key === "portfolio_path") {
          data[key] = null;
          return;
        }
        if (value === "") return;
        if (["years", "min_hold_days", "max_tickers", "top"].includes(key)) {
          data[key] = Number(value);
        } else if (["prob_threshold", "hedge_weight", "initial_capital"].includes(key)) {
          data[key] = Number(value);
        } else {
          data[key] = value.trim();
        }
      });
      form.querySelectorAll('input[type="checkbox"]').forEach((checkbox) => {
        if (!(checkbox.name in data)) {
          data[checkbox.name] = checkbox.checked;
        }
      });
      return data;
    }

    async function submitForm(event, endpoint, statusEl) {
      event.preventDefault();
      statusEl.textContent = "Submitting…";
      try {
        const payload = formToJSON(event.target);
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || "Request failed");
        }
        const data = await response.json();
        statusEl.textContent = `Job ${data.job_id} queued.`;
        loadJobs();
        setTimeout(loadJobs, 2500);
      } catch (error) {
        statusEl.textContent = `Error: ${error.message}`;
      }
    }

    async function submitSimulationForm(event) {
      event.preventDefault();
      const statusEl = document.getElementById("simulation-status");
      const resultEl = document.getElementById("simulation-results");
      resultEl.innerHTML = "";
      statusEl.textContent = "Submitting…";
      try {
        const payload = formToJSON(event.target);
        const response = await fetch("/api/simulate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || "Request failed");
        }
        const data = await response.json();
        statusEl.textContent = `Simulation job ${data.job_id} queued.`;
        loadJobs();
        pollSimulationJob(data.job_id, statusEl, resultEl);
      } catch (error) {
        statusEl.textContent = `Error: ${error.message}`;
      }
    }

    async function pollSimulationJob(jobId, statusEl, resultEl) {
      let attempts = 0;
      while (attempts < 720) {
        if (attempts > 0) {
          await new Promise((resolve) => setTimeout(resolve, 5000));
        }
        try {
          const response = await fetch(`/api/jobs/${jobId}`);
          if (!response.ok) throw new Error("Job not found");
          const job = await response.json();
          statusEl.textContent = `Simulation status: ${job.status}`;
          if (job.status === "completed") {
            renderSimulationResults(job.result, resultEl);
            statusEl.textContent = "Simulation completed.";
            loadJobs();
            return;
          }
          if (job.status === "failed") {
            const errorMessage = job.error || "Unknown error.";
            statusEl.textContent = `Simulation failed: ${errorMessage}`;
            resultEl.innerHTML = "";
            return;
          }
        } catch (error) {
          statusEl.textContent = `Error retrieving job: ${error.message}`;
          return;
        }
        attempts += 1;
      }
      statusEl.textContent = "Simulation polling timed out.";
    }

    function buildTradesTable(trades, options = {}) {
      const { showShares = false, caption = "Executed Trades", limit = 200, note = "" } = options;
      const items = Array.isArray(trades) ? trades : [];
      const headers = ["Symbol", "Entry", "Exit", "Time Held", "Entry Price", "Exit Price"];
      if (showShares) headers.push("Shares");
      headers.push("P/L");
      headers.push("Return %");
      const headerHtml = headers.map((label) => `<th>${label}</th>`).join("");
      const rowsHtml = items.slice(0, limit).map((trade) => {
        const profit = Number(trade.profit ?? 0);
        const ret = Number(trade.return_pct ?? 0);
        const sharesCell = showShares ? `<td>${Number(trade.shares ?? 0).toFixed(4)}</td>` : "";
        const holdValue = trade.hold_days ?? trade.holdDays ?? null;
        const holdCell = holdValue !== null && holdValue !== undefined ? formatDays(holdValue) : "—";
        return `
          <tr>
            <td>${trade.symbol || "—"}</td>
            <td>${formatDate(trade.entry_date || trade.entryDate)}</td>
            <td>${formatDate(trade.exit_date || trade.exitDate)}</td>
            <td>${holdCell}</td>
            <td>${formatCurrency(trade.entry_price ?? trade.entryPrice)}</td>
            <td>${formatCurrency(trade.exit_price ?? trade.exitPrice)}</td>
            ${sharesCell}
            <td class="${profit >= 0 ? "positive" : "negative"}">${formatCurrency(profit)}</td>
            <td class="${ret >= 0 ? "positive" : "negative"}">${formatPercent(ret)}</td>
          </tr>
        `;
      }).join("");
      const hasMore = items.length > limit;
      const colspan = headers.length;
      const emptyRow = `<tr><td colspan="${colspan}">No trades recorded.</td></tr>`;
      const noteHtml = note ? `<p class="status">${note}</p>` : "";
      const moreNote = hasMore ? `<p class="status">Showing first ${limit} trades. Inspect job details for the full list.</p>` : "";
      return `
        <h3>${caption}</h3>
        <table>
          <thead><tr>${headerHtml}</tr></thead>
          <tbody>${rowsHtml || emptyRow}</tbody>
        </table>
        ${noteHtml}
        ${moreNote}
      `;
    }

    function buildSimulationMarkup(result, options = {}) {
      if (!result) return "<em>No simulation data available.</em>";
      const { tradeLimit } = options;
      const tradeSummary = result.trade_summary || {};
      const trades = Array.isArray(result.trades) ? result.trades : [];
      const cumulativeSeries = [];
      let cumulative = 0;
      trades.forEach((trade) => {
        const profit = Number(trade.profit ?? 0);
        cumulative += profit;
        cumulativeSeries.push({
          date: trade.exit_date || trade.exitDate,
          value: cumulative,
        });
      });
      const summary = `
        <div class="grid" style="margin-top: 18px; gap: 12px;">
          <div><strong>Initial Capital:</strong> ${formatCurrency(result.initial_capital)}</div>
          <div><strong>Final Capital:</strong> ${formatCurrency(result.final_capital)}</div>
          <div><strong>Return:</strong> <span class="${Number(result.return_pct ?? 0) >= 0 ? "positive" : "negative"}">${formatPercent(result.return_pct)}</span></div>
          <div><strong>Total Trades:</strong> ${tradeSummary.count ?? 0}</div>
          <div><strong>Wins:</strong> ${tradeSummary.wins ?? 0}</div>
          <div><strong>Losses:</strong> ${tradeSummary.losses ?? 0}</div>
          <div><strong>Win Rate:</strong> ${formatPercent(tradeSummary.win_rate)}</div>
          <div><strong>Total Profit:</strong> <span class="${Number(tradeSummary.total_profit ?? 0) >= 0 ? "positive" : "negative"}">${formatCurrency(tradeSummary.total_profit)}</span></div>
          <div><strong>Avg Hold:</strong> ${formatDays(tradeSummary.avg_hold_days)}</div>
          <div><strong>Median Hold:</strong> ${formatDays(tradeSummary.median_hold_days)}</div>
          <div><strong>Max Hold:</strong> ${formatDays(tradeSummary.max_hold_days)}</div>
        </div>
      `;
      const perSymbolRows = (result.per_symbol || []).map((row, idx) => {
        const ret = Number(row.return_pct ?? 0);
        const cagr = row.CAGR !== null && row.CAGR !== undefined ? Number(row.CAGR).toFixed(4) : "-";
        const sharpe = row.Sharpe !== null && row.Sharpe !== undefined ? Number(row.Sharpe).toFixed(4) : "-";
        const maxdd = row.MaxDD !== null && row.MaxDD !== undefined ? Number(row.MaxDD).toFixed(4) : "-";
        return `
          <tr>
            <td>${idx + 1}</td>
            <td>${row.symbol}</td>
            <td>${formatCurrency(row.initial)}</td>
            <td>${formatCurrency(row.final)}</td>
            <td class="${ret >= 0 ? "positive" : "negative"}">${formatPercent(ret)}</td>
            <td>${cagr}</td>
            <td>${sharpe}</td>
            <td>${maxdd}</td>
          </tr>
        `;
      }).join("");
      const perSymbolTable = `
        <h3>Strategy Allocation</h3>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Symbol</th>
              <th>Initial</th>
              <th>Final</th>
              <th>Return %</th>
              <th>CAGR</th>
              <th>Sharpe</th>
              <th>Max DD</th>
            </tr>
          </thead>
          <tbody>${perSymbolRows || "<tr><td colspan='8'>No symbols selected.</td></tr>"}</tbody>
        </table>
      `;
      const tradesTable = buildTradesTable(result.trades, {
        showShares: true,
        limit: tradeLimit ?? 200,
      });
      const chartBlock = cumulativeSeries.length
        ? `
          <h3>Cumulative P/L</h3>
          <div class="pl-chart" id="simulation-pl-chart" data-series='${JSON.stringify(cumulativeSeries)}'></div>
        `
        : "";
      return summary + chartBlock + perSymbolTable + tradesTable;
    }

    function renderSimulationChart(container) {
      const chartHost = container.querySelector("#simulation-pl-chart");
      if (!chartHost) return;
      const rawSeries = chartHost.getAttribute("data-series");
      if (!rawSeries) return;
      let series;
      try {
        series = JSON.parse(rawSeries);
      } catch (error) {
        console.error("Failed to parse simulation series", error);
        return;
      }
      if (!Array.isArray(series) || !series.length) return;

      chartHost.innerHTML = "";
      const width = chartHost.clientWidth || 780;
      const height = 220;
      const padding = { top: 24, right: 32, bottom: 24, left: 60 };
      const innerWidth = width - padding.left - padding.right;
      const innerHeight = height - padding.top - padding.bottom;

      const svgNS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(svgNS, "svg");
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
      svg.style.width = "100%";
      svg.style.height = "auto";
      svg.style.background = "#0f172a";
      svg.style.border = "1px solid #1f2937";
      svg.style.borderRadius = "12px";

      const values = series.map((item) => Number(item.value || 0));
      let min = Math.min(...values, 0);
      let max = Math.max(...values, 0);
      if (min === max) {
        max = min + 1;
        min = min - 1;
      }

      const points = series.map((item, index) => {
        const xRatio = series.length === 1 ? 0.5 : index / (series.length - 1);
        const value = Number(item.value || 0);
        const yRatio = (value - min) / (max - min);
        return {
          x: padding.left + xRatio * innerWidth,
          y: padding.top + (1 - yRatio) * innerHeight,
          value,
          date: item.date,
        };
      });

      const gridLines = 4;
      for (let i = 0; i <= gridLines; i += 1) {
        const y = padding.top + (innerHeight / gridLines) * i;
        const line = document.createElementNS(svgNS, "line");
        line.setAttribute("x1", padding.left);
        line.setAttribute("x2", width - padding.right);
        line.setAttribute("y1", y);
        line.setAttribute("y2", y);
        line.setAttribute("stroke", "#1f2937");
        line.setAttribute("stroke-width", "1");
        line.setAttribute("opacity", "0.6");
        svg.appendChild(line);
      }

      const path = document.createElementNS(svgNS, "path");
      const d = points
        .map((pt, idx) => `${idx === 0 ? "M" : "L"}${pt.x.toFixed(2)},${pt.y.toFixed(2)}`)
        .join(" ");
      path.setAttribute("d", d);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "#38bdf8");
      path.setAttribute("stroke-width", "2");
      path.setAttribute("stroke-linejoin", "round");
      path.setAttribute("stroke-linecap", "round");
      svg.appendChild(path);

      if (min < 0 && max > 0) {
        const zeroRatio = (0 - min) / (max - min);
        const zeroY = padding.top + (1 - zeroRatio) * innerHeight;
        const zeroLine = document.createElementNS(svgNS, "line");
        zeroLine.setAttribute("x1", padding.left);
        zeroLine.setAttribute("x2", width - padding.right);
        zeroLine.setAttribute("y1", zeroY);
        zeroLine.setAttribute("y2", zeroY);
        zeroLine.setAttribute("stroke", "#475569");
        zeroLine.setAttribute("stroke-dasharray", "4 4");
        zeroLine.setAttribute("stroke-width", "1");
        svg.appendChild(zeroLine);
      }

      points.forEach((pt) => {
        const dot = document.createElementNS(svgNS, "circle");
        dot.setAttribute("cx", pt.x);
        dot.setAttribute("cy", pt.y);
        dot.setAttribute("r", "2.4");
        dot.setAttribute("fill", "#38bdf8");
        svg.appendChild(dot);
      });

      chartHost.appendChild(svg);
    }

    function buildPipelineMarkup(result, options = {}) {
      if (!result) return "<em>No pipeline result available.</em>";
      const { tradeLimit } = options;
      const stats = result.stats || {};
      const tradeSummary = result.trade_summary || {};
      const totalReturn = Number(result.total_return ?? 0);
      const totalReturnPctValue = totalReturn * 100;
      const latestSignal = result.latest_signal || {};
      const probabilityValue = latestSignal.probability !== undefined && latestSignal.probability !== null ? Number(latestSignal.probability) : null;
      const probabilityClass = classifyProbability(probabilityValue);
      const probabilityDisplay = formatProbability(probabilityValue);
      const actionTag = getActionTag(latestSignal.action, "HOLD", "action-hold");
      const latestDateDisplay = latestSignal.date ? formatDate(latestSignal.date) : "—";
      const totalReturnClass = totalReturn >= 0 ? "positive" : "negative";
      const summaryBlock = `
        <div class="summary-cards">
          <div class="summary-card">
            <span class="label">Latest Action</span>
            <span class="value">${actionTag}</span>
          </div>
          <div class="summary-card">
            <span class="label">Signal Probability</span>
            <span class="value ${probabilityClass}">${probabilityDisplay}</span>
          </div>
          <div class="summary-card">
            <span class="label">Total Return</span>
            <span class="value ${totalReturnClass}">${formatPercent(totalReturnPctValue)}</span>
          </div>
          <div class="summary-card">
            <span class="label">Last Updated</span>
            <span class="value">${latestDateDisplay}</span>
          </div>
        </div>
      `;
      const lastAlertText = latestSignal.last_alert_action
        ? `${latestSignal.last_alert_action.toUpperCase()}${latestSignal.last_alert_date ? ` (${formatDate(latestSignal.last_alert_date)})` : ""}`
        : "—";
      const latestDetailsBlock = latestSignal && Object.keys(latestSignal).length
        ? `
          <h3>Signal Snapshot</h3>
          <div class="grid" style="margin-top: 12px; gap: 12px;">
            <div><strong>Current Action:</strong> ${actionTag}</div>
            <div><strong>Probability:</strong> <span class="${probabilityClass}">${probabilityDisplay}</span></div>
            <div><strong>Signal State:</strong> ${latestSignal.signal !== undefined && latestSignal.signal !== null ? latestSignal.signal : "—"}</div>
            <div><strong>Previous Signal:</strong> ${latestSignal.previous_signal !== undefined && latestSignal.previous_signal !== null ? latestSignal.previous_signal : "—"}</div>
            <div><strong>Last Alert:</strong> ${lastAlertText}</div>
          </div>
        `
        : "";
      const perfBlock = `
        <h3>Performance</h3>
        <div class="grid" style="margin-top: 12px; gap: 12px;">
          <div><strong>Symbol:</strong> ${result.symbol || "—"}</div>
          <div><strong>Total Return:</strong> <span class="${totalReturn >= 0 ? "positive" : "negative"}">${formatPercent(totalReturnPctValue)}</span></div>
          <div><strong>CAGR:</strong> ${stats.CAGR !== undefined && stats.CAGR !== null ? formatPercent(Number(stats.CAGR) * 100) : "-"}</div>
          <div><strong>Sharpe:</strong> ${stats.Sharpe !== undefined && stats.Sharpe !== null && !Number.isNaN(Number(stats.Sharpe)) ? Number(stats.Sharpe).toFixed(2) : "-"}</div>
          <div><strong>Max Drawdown:</strong> <span class="${Number(stats.MaxDD ?? 0) < 0 ? "negative" : "positive"}">${stats.MaxDD !== undefined && stats.MaxDD !== null ? formatPercent(Number(stats.MaxDD) * 100) : "-"}</span></div>
          <div><strong>Volatility:</strong> ${stats.Vol !== undefined && stats.Vol !== null ? formatPercent(Number(stats.Vol) * 100) : "-"}</div>
        </div>
      `;
      const tradeSummaryBlock = `
        <h3>Trade Summary</h3>
        <div class="grid" style="margin-top: 12px; gap: 12px;">
          <div><strong>Total Trades:</strong> ${tradeSummary.count ?? 0}</div>
          <div><strong>Win Rate:</strong> ${formatPercent(tradeSummary.win_rate)}</div>
          <div><strong>Total P/L (per share):</strong> <span class="${Number(tradeSummary.total_profit ?? 0) >= 0 ? "positive" : "negative"}">${formatCurrency(tradeSummary.total_profit)}</span></div>
          <div><strong>Avg P/L:</strong> ${formatCurrency(tradeSummary.avg_profit)}</div>
          <div><strong>Avg Hold:</strong> ${formatDays(tradeSummary.avg_hold_days)}</div>
          <div><strong>Max Hold:</strong> ${formatDays(tradeSummary.max_hold_days)}</div>
        </div>
      `;
      const outputs = `
        <h3>Generated Files</h3>
        <p><strong>Model:</strong> ${result.model_path || "—"}</p>
        <p><strong>Backtest CSV:</strong> ${result.backtest_csv || "—"}</p>
        <p><strong>Alerts CSV:</strong> ${result.alerts_csv || "—"}</p>
      `;
      const portfolioInfo = result.portfolio_path
        ? `
          <h3>Portfolio</h3>
          <p><strong>State File:</strong> ${result.portfolio_path}</p>
          <p><strong>Owned Before:</strong> ${result.portfolio_owned_before === true ? "Yes" : result.portfolio_owned_before === false ? "No" : "—"}</p>
          <p><strong>Owned After:</strong> ${result.portfolio_owned_after === true ? "Yes" : result.portfolio_owned_after === false ? "No" : "—"}</p>
        `
        : "";
      const tradesTable = buildTradesTable(result.trades, {
        showShares: false,
        limit: tradeLimit ?? 250,
        note: "P/L values reflect price change per share between entry and exit closes.",
      });
      return summaryBlock + perfBlock + latestDetailsBlock + tradeSummaryBlock + outputs + portfolioInfo + tradesTable;
    }

    function renderSimulationResults(result, container) {
      if (!container) return;
      if (!result) {
        container.innerHTML = "<em>No simulation data available.</em>";
        return;
      }
      container.innerHTML = buildSimulationMarkup(result);
      renderSimulationChart(container);
    }


    async function loadJobs() {
      try {
        const response = await fetch("/api/jobs");
        if (!response.ok) throw new Error("Failed to fetch jobs");
        const data = await response.json();
        const tbody = document.querySelector("#jobs-table tbody");
        tbody.innerHTML = "";
        data.jobs.forEach((job) => {
          const row = document.createElement("tr");
          row.dataset.jobId = job.id;
          row.innerHTML = `
            <td>${job.id}</td>
            <td>${makeTag(job.type, job.type)}</td>
            <td>${makeTag(job.status, job.status)}</td>
            <td>${formatDate(job.created_at)}</td>
            <td>${formatDate(job.started_at)}</td>
            <td>${formatDate(job.finished_at)}</td>
          `;
          row.addEventListener("click", () => loadJobDetails(job.id));
          tbody.appendChild(row);
        });
      } catch (error) {
        console.error(error);
      }
    }

    function renderLogs(title, content) {
      if (!content) return "";
      const safeContent = content.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
      return `
        <details>
          <summary><strong>${title}</strong></summary>
          <pre>${safeContent}</pre>
        </details>
      `;
    }

    async function loadJobDetails(jobId) {
      try {
        const response = await fetch(`/api/jobs/${jobId}`);
        if (!response.ok) throw new Error("Job not found");
        const job = await response.json();
        const container = document.querySelector("#job-details");
        let resultHtml = "<em>No result payload.</em>";
        if (job.result) {
          if (job.type === "pipeline") {
            resultHtml = buildPipelineMarkup(job.result, { tradeLimit: 500 });
          } else if (job.type === "simulation") {
            resultHtml = buildSimulationMarkup(job.result, { tradeLimit: 500 });
          } else {
            resultHtml = `<pre>${JSON.stringify(job.result, null, 2)}</pre>`;
          }
        }
        const errorHtml = job.error ? `<p class="status negative">Error: ${job.error}</p>` : "";
        container.innerHTML = `
          <div class="flex">
            <div>${makeTag(job.type, job.type)}</div>
            <div>${makeTag(job.status, job.status)}</div>
          </div>
          <div class="grid" style="margin: 12px 0; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));">
            <div><strong>Job ID:</strong> ${job.id}</div>
            <div><strong>Created:</strong> ${formatDate(job.created_at)}</div>
            <div><strong>Started:</strong> ${formatDate(job.started_at)}</div>
            <div><strong>Finished:</strong> ${formatDate(job.finished_at)}</div>
          </div>
          ${errorHtml}
          <details open>
            <summary><strong>Parameters</strong></summary>
            <pre>${JSON.stringify(job.params, null, 2)}</pre>
          </details>
          <details open>
            <summary><strong>Result</strong></summary>
            ${resultHtml}
          </details>
          ${renderLogs("Stdout", job.stdout)}
          ${renderLogs("Stderr", job.stderr)}
        `;
      } catch (error) {
        console.error(error);
      }
    }

    let currentRecommendations = [];

    function renderRecommendationTable(rows) {
      const tbody = document.querySelector("#recs-table tbody");
      tbody.innerHTML = "";
      rows.forEach((row, idx) => {
        const actionTag = getActionTag(row.latest_action, "—");
        const latestDate = row.latest_action_date ? formatDate(row.latest_action_date) : "—";
        const probabilityValue = row.latest_probability;
        const probabilityDisplay = formatProbability(probabilityValue);
        const probabilityClass = classifyProbability(probabilityValue);
        const statusText = row.status || "—";
        let tradingDays = "—";
        if (row.trading_days !== undefined && row.trading_days !== null) {
          const tdNum = Number(row.trading_days);
          if (Number.isFinite(tdNum)) {
            tradingDays = tdNum.toLocaleString();
          }
        }
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${idx + 1}</td>
          <td>${row.symbol || "—"}</td>
          <td>${statusText}</td>
          <td>${formatNumber(row.CAGR, 4)}</td>
          <td>${formatNumber(row.Sharpe, 4)}</td>
          <td>${formatNumber(row.total_return_pct, 2)}</td>
          <td>${formatNumber(row.MaxDD, 4)}</td>
          <td>${tradingDays}</td>
          <td>${actionTag}</td>
          <td>${latestDate}</td>
          <td class="${probabilityClass}">${probabilityDisplay}</td>
        `;
        tbody.appendChild(tr);
      });
      if (!rows.length) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="11" style="text-align:center; padding: 12px 0; color: #94a3b8;">No recommendations matched your filters.</td>`;
        tbody.appendChild(tr);
      }
    }

    function applyRecommendationFilters() {
      const searchTerm = (document.getElementById("recs-search").value || "").trim().toUpperCase();
      const actionFilter = (document.getElementById("recs-action-filter").value || "").trim().toUpperCase();

      const filtered = currentRecommendations.filter((row) => {
        const symbol = (row.symbol || "").toUpperCase();
        const action = (row.latest_action || "").toUpperCase();
        const matchesSymbol = !searchTerm || symbol.includes(searchTerm);
        const matchesAction = !actionFilter || action === actionFilter;
        return matchesSymbol && matchesAction;
      });

      renderRecommendationTable(filtered);
    }

    async function loadRecommendations() {
      try {
        const response = await fetch("/api/recommendations?limit=500");
        if (!response.ok) throw new Error("Failed to fetch recommendations");
        const data = await response.json();
        const meta = document.querySelector("#recs-meta");
        currentRecommendations = data.rows || [];
        meta.textContent = data.updated_at
          ? `Loaded ${currentRecommendations.length} rows from ${data.source} (updated ${formatDate(data.updated_at)})`
          : "No recommendation file found.";
        applyRecommendationFilters();
      } catch (error) {
        console.error(error);
      }
    }

    document.getElementById("recs-search").addEventListener("input", () => {
      applyRecommendationFilters();
    });
    document.getElementById("recs-action-filter").addEventListener("change", () => {
      applyRecommendationFilters();
    });
    document.getElementById("pipeline-form").addEventListener("submit", (event) => {
      submitForm(event, "/api/pipeline", document.getElementById("pipeline-status"));
    });
    document.getElementById("simulation-form").addEventListener("submit", submitSimulationForm);
    document.getElementById("recommend-form").addEventListener("submit", (event) => {
      submitForm(event, "/api/recommend", document.getElementById("recommend-status"));
    });
    document.getElementById("refresh-recs").addEventListener("click", loadRecommendations);

    loadJobs();
    loadRecommendations();
    setInterval(loadJobs, 5000);
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(content=HTML_TEMPLATE)


@app.post("/api/pipeline")
def queue_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    job_id = manager.create(JobKind.PIPELINE, request.dict())
    background_tasks.add_task(run_pipeline_job, job_id, request)
    return {"job_id": job_id}


@app.post("/api/simulate")
def queue_simulation(request: SimulationRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    job_id = manager.create(JobKind.SIMULATION, request.dict())
    background_tasks.add_task(run_simulation_job, job_id, request)
    return {"job_id": job_id}


@app.post("/api/recommend")
def queue_recommend(request: RecommendRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    job_id = manager.create(JobKind.RECOMMEND, request.dict())
    background_tasks.add_task(run_recommend_job, job_id, request)
    return {"job_id": job_id}


@app.get("/api/jobs")
def jobs() -> Dict[str, Any]:
    return {"jobs": manager.list_jobs()}


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str) -> Dict[str, Any]:
    try:
        return manager.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None


@app.get("/api/recommendations")
def get_recommendations(limit: int = 500) -> Dict[str, Any]:
    output_path = DEFAULT_RECOMMENDATIONS
    if not output_path.exists():
        return {"rows": [], "source": str(output_path), "updated_at": None}
    df = pd.read_csv(output_path)
    sort_cols = [col for col in ["status", "CAGR", "Sharpe"] if col in df.columns]
    if sort_cols:
        ascending = [True] + [False] * (len(sort_cols) - 1)
        df = df.sort_values(by=sort_cols, ascending=ascending)
    subset = df.head(limit)
    subset = subset.replace({pd.NA: None})
    subset = subset.where(pd.notna(subset), None)
    return {
        "rows": subset.to_dict(orient="records"),
        "source": str(output_path),
        "updated_at": _iso(datetime.utcfromtimestamp(output_path.stat().st_mtime)),
    }


@app.get("/healthz")
def healthcheck() -> Dict[str, Any]:
    return {"status": "ok", "time": time.time()}
