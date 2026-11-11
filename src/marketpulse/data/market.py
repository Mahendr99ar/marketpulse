"""Download daily market levels from FRED and Yahoo Finance into one aligned table.

Output: a DataFrame indexed by business day, one column per instrument id,
holding the closing level (price, index level, or yield in %).
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd

from marketpulse.config import Config


def _fetch_fred(series: dict[str, str], start: str, end: str | None) -> pd.DataFrame:
    from fredapi import Fred

    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set. Copy .env.example to .env and add your key.")
    fred = Fred(api_key=key)
    out = {}
    for inst_id, code in series.items():
        s = fred.get_series(code, observation_start=start, observation_end=end)
        out[inst_id] = pd.to_numeric(s, errors="coerce")
    return pd.DataFrame(out)


def _fetch_yfinance(tickers: dict[str, str], start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        list(tickers.values()), start=start, end=end, auto_adjust=True, progress=False
    )["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=list(tickers.values())[0])
    reverse = {v: k for k, v in tickers.items()}
    raw = raw.rename(columns=reverse)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    return raw[list(tickers.keys())]


def align(frames: list[pd.DataFrame], max_ffill: int = 5) -> pd.DataFrame:
    """Put all series on one business-day calendar.

    Markets have different holidays (Nifty closes on Diwali, FRED skips US holidays).
    A short forward fill means a closed market shows "no change" that day instead of
    dropping the whole row for every other market.
    """
    df = pd.concat(frames, axis=1).sort_index()
    df.index = pd.to_datetime(df.index)
    bdays = pd.bdate_range(df.index.min(), df.index.max())
    df = df.reindex(bdays).ffill(limit=max_ffill)
    df.index.name = "date"
    return df.dropna(how="any")


def fetch_market(cfg: Config) -> pd.DataFrame:
    start, end = cfg["start_date"], cfg["end_date"] or date.today().isoformat()
    fred = {i.id: i.ticker for i in cfg.instruments if i.source == "fred"}
    yfin = {i.id: i.ticker for i in cfg.instruments if i.source == "yfinance"}
    frames = []
    if fred:
        frames.append(_fetch_fred(fred, start, end))
    if yfin:
        frames.append(_fetch_yfinance(yfin, start, end))
    raw = pd.concat(frames, axis=1)
    first = {c: raw[c].first_valid_index() for c in raw.columns}
    print("[data] first date per series: " + ", ".join(
        f"{c} {d.date() if d is not None else 'NO DATA'}" for c, d in first.items()))
    missing = [c for c, d in first.items() if d is None]
    if missing:
        raise RuntimeError(f"No data for {missing}. Check the ticker in configs/config.yaml.")
    latest = max(first, key=first.get)
    if first[latest] - min(first.values()) > pd.Timedelta(days=365):
        print(f"[data] WARNING: {latest} starts {first[latest].date()}, so every series is cut to "
              f"that date. Replace or drop it in configs/config.yaml if you need more history.")
    levels = align(frames)
    return levels[[i.id for i in cfg.instruments]]


def compute_changes(levels: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Daily change per instrument in the unit a desk would quote.

    price -> % log return (x100), rate -> basis points (x100 of the % change).
    """
    import numpy as np

    out = {}
    for inst in cfg.instruments:
        s = levels[inst.id]
        if inst.kind == "rate":
            out[inst.id] = s.diff() * 100.0
        else:
            out[inst.id] = np.log(s).diff() * 100.0
    return pd.DataFrame(out, index=levels.index).iloc[1:]
