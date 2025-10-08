from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple, Union

import requests

try:
    from twilio.rest import Client
except ImportError:  # pragma: no cover
    Client = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


class NotificationError(RuntimeError):
    """Raised when an alert notification cannot be sent."""


def _extract_components(entry: Union[dict, "pd.Series", Tuple[object, Union[dict, "pd.Series"]]]) -> Tuple[str, float, float, str]:
    idx: Optional[object]
    payload: Union[dict, "pd.Series"]

    if isinstance(entry, tuple) and len(entry) == 2:
        idx, payload = entry
    else:
        idx, payload = None, entry  # type: ignore[assignment]

    if pd is not None and isinstance(payload, pd.Series):
        action = payload.get("action")
        price = payload.get("close")
        proba = payload.get("proba")
        timestamp = payload.name if hasattr(payload, "name") else None
    elif isinstance(payload, dict):
        action = payload.get("action")
        price = payload.get("close")
        proba = payload.get("proba")
        timestamp = payload.get("Date")
    else:
        action = getattr(payload, "action", None)
        price = getattr(payload, "close", None)
        proba = getattr(payload, "proba", None)
        timestamp = getattr(payload, "Date", None)

    ts = idx or timestamp
    ts_str = str(ts) if ts is not None else ""

    if action is None:
        raise NotificationError("Alert payload missing 'action'.")
    if price is None:
        raise NotificationError("Alert payload missing 'close'.")
    if proba is None:
        raise NotificationError("Alert payload missing 'proba'.")

    return str(action), float(price), float(proba), ts_str


# --- Twilio helpers -----------------------------------------------------------------

def _require_twilio() -> None:
    if Client is None:
        raise NotificationError(
            "twilio package is not installed. Run `pip install twilio` or update requirements."
        )


def _build_twilio_client(
    account_sid: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> Client:
    _require_twilio()

    account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
    if not account_sid or not auth_token:
        raise NotificationError(
            "Twilio credentials missing. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN."
        )
    return Client(account_sid, auth_token)


def _resolve_from_number(from_number: Optional[str]) -> str:
    from_number = from_number or os.getenv("TWILIO_FROM_NUMBER")
    if not from_number:
        raise NotificationError(
            "Origin phone missing. Provide --notify-from or set TWILIO_FROM_NUMBER."
        )
    return from_number


def send_twilio_messages(
    symbol: str,
    alerts: Iterable,
    to_number: str,
    *,
    from_number: Optional[str] = None,
    account_sid: Optional[str] = None,
    auth_token: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[str]:
    client = _build_twilio_client(account_sid, auth_token)
    from_number = _resolve_from_number(from_number)

    alerts_list = list(alerts)
    if limit is not None and limit > 0:
        alerts_list = alerts_list[-limit:]

    if not alerts_list:
        return []

    sids: List[str] = []
    for entry in alerts_list:
        action, price, proba, timestamp = _extract_components(entry)
        date_snippet = timestamp[:10] if timestamp else ""
        body = f"{symbol} alert: {action} @ {price:.2f} (prob={proba:.3f}) {date_snippet}".strip()

        message = client.messages.create(
            to=to_number,
            from_=from_number,
            body=body,
        )
        sids.append(message.sid)

    return sids


# --- ntfy helpers -------------------------------------------------------------------

def send_ntfy_messages(
    symbol: str,
    alerts: Iterable,
    topic: str,
    *,
    limit: Optional[int] = None,
    server: str = "https://ntfy.sh",
    token: Optional[str] = None,
    title: str = "Trading Alert",
    priority: Optional[str] = None,
) -> List[int]:
    alerts_list = list(alerts)
    if limit is not None and limit > 0:
        alerts_list = alerts_list[-limit:]

    if not alerts_list:
        return []

    server = server.rstrip("/")
    url = f"{server}/{topic}"
    headers = {"Title": title}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if priority:
        headers["Priority"] = priority

    statuses: List[int] = []
    session = requests.Session()
    for entry in alerts_list:
        action, price, proba, timestamp = _extract_components(entry)
        date_snippet = timestamp[:10] if timestamp else ""
        body = f"{symbol} alert: {action} @ {price:.2f} (prob={proba:.3f}) {date_snippet}".strip()
        response = session.post(url, data=body.encode("utf-8"), headers=headers, timeout=10)
        if not response.ok:
            raise NotificationError(f"ntfy request failed with HTTP {response.status_code}: {response.text}")
        statuses.append(response.status_code)

    return statuses