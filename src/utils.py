@'
from __future__ import annotations
import pandas as pd
import numpy as np

def ensure_ts_index(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    if date_col in df.columns:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], utc=False)
        df = df.sort_values(date_col).set_index(date_col)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex or a 'Date' column.")
    return df
'@ | Set-Content src\utils.py
