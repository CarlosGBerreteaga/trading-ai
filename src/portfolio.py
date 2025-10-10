from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Set

DEFAULT_PORTFOLIO_PATH = Path(__file__).resolve().parent.parent / "portfolio.json"


def _normalise(symbol: str) -> str:
    return symbol.strip().upper()


@dataclass
class PortfolioState:
    """Simple persistent registry of owned tickers backed by JSON on disk."""

    path: Path = field(default_factory=lambda: DEFAULT_PORTFOLIO_PATH)
    owned: Set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path | None = None) -> "PortfolioState":
        target = path or DEFAULT_PORTFOLIO_PATH
        if target.exists():
            data = json.loads(target.read_text(encoding="utf-8"))
            raw = data.get("owned", [])
            owned = {_normalise(sym) for sym in raw if isinstance(sym, str) and sym.strip()}
        else:
            owned = set()
        return cls(path=target, owned=owned)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"owned": sorted(self.owned)}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def is_owned(self, symbol: str) -> bool:
        return _normalise(symbol) in self.owned

    def add(self, symbols: Iterable[str]) -> None:
        for sym in symbols:
            if not isinstance(sym, str):
                continue
            normalised = _normalise(sym)
            if normalised:
                self.owned.add(normalised)

    def set_single(self, symbol: str | None) -> None:
        """Replace current holdings with at most one normalised symbol."""
        if symbol is None:
            self.owned = set()
            return
        if not isinstance(symbol, str):
            return
        normalised = _normalise(symbol)
        if normalised:
            self.owned = {normalised}
        else:
            self.owned = set()

    def remove(self, symbols: Iterable[str]) -> None:
        for sym in symbols:
            if not isinstance(sym, str):
                continue
            normalised = _normalise(sym)
            self.owned.discard(normalised)

    def set_owned(self, symbols: Iterable[str]) -> None:
        cleaned = []
        for sym in symbols:
            if isinstance(sym, str) and sym.strip():
                cleaned.append(_normalise(sym))
        self.owned = set(cleaned)

    def list_owned(self) -> List[str]:
        return sorted(self.owned)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the persistent portfolio state.")
    parser.add_argument(
        "--portfolio",
        type=Path,
        default=DEFAULT_PORTFOLIO_PATH,
        help="JSON file that stores owned tickers (default: portfolio.json in project root).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Print all owned tickers.")

    add_parser = sub.add_parser("add", help="Add one or more tickers to the owned list.")
    add_parser.add_argument("symbols", nargs="+", help="Ticker symbols to mark as owned.")

    rm_parser = sub.add_parser("remove", help="Remove one or more tickers from the owned list.")
    rm_parser.add_argument("symbols", nargs="+", help="Ticker symbols to unmark.")

    sub.add_parser("clear", help="Remove all tickers from the owned list.")

    set_parser = sub.add_parser("set", help="Replace the owned list with the provided tickers.")
    set_parser.add_argument("symbols", nargs="+", help="Ticker symbols to own.")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    portfolio = PortfolioState.load(args.portfolio)

    if args.command == "list":
        holdings = portfolio.list_owned()
        if holdings:
            print("\n".join(holdings))
        else:
            print("No owned tickers recorded.")
        return

    if args.command == "add":
        portfolio.add(args.symbols)
    elif args.command == "remove":
        portfolio.remove(args.symbols)
    elif args.command == "clear":
        portfolio.set_owned([])
    elif args.command == "set":
        portfolio.set_owned(args.symbols)

    portfolio.save()
    print("Owned tickers:", ", ".join(portfolio.list_owned()) or "(none)")


if __name__ == "__main__":
    main()
