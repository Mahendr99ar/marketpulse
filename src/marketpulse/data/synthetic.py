"""Synthetic market and news data for demo mode and tests.

The generator uses a hidden 3-state Markov chain (calm, normal, stressed). Each state
has its own volatility. A few shock days get a large move plus a matching headline.
Because the true regimes and shocks are known, you can check whether the models
recover them before trusting them on real data.

This data is fake. Never report results from it as findings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from marketpulse.config import Config

TRANSITION = np.array([
    [0.985, 0.013, 0.002],
    [0.020, 0.970, 0.010],
    [0.010, 0.040, 0.950],
])

# daily mean and vol per regime (price: %, rate: bp)
PARAMS = {
    "SPX":    {"mu": [0.05, 0.00, -0.15], "sd": [0.7, 1.2, 2.8]},
    "NIFTY":  {"mu": [0.06, 0.00, -0.15], "sd": [0.8, 1.3, 2.6]},
    "USDINR": {"mu": [0.01, 0.01, 0.04], "sd": [0.20, 0.30, 0.60]},
    "EURUSD": {"mu": [0.00, 0.00, -0.02], "sd": [0.40, 0.55, 0.90]},
    "GOLD":   {"mu": [0.02, 0.02, 0.06], "sd": [0.8, 1.0, 1.6]},
    "UST2Y":  {"mu": [0.0, 0.0, -1.5], "sd": [4.0, 6.0, 12.0]},
    "UST10Y": {"mu": [0.0, 0.0, -1.0], "sd": [4.0, 5.5, 10.0]},
    "CREDIT": {"mu": [-0.1, 0.1, 1.0], "sd": [2.0, 4.0, 12.0]},
}
VIX_TARGET = [13.0, 20.0, 35.0]
START_LEVELS = {"SPX": 1200.0, "NIFTY": 2000.0, "USDINR": 45.0, "EURUSD": 1.30,
                "GOLD": 420.0, "UST2Y": 3.0, "UST10Y": 4.2, "CREDIT": 2.2, "VIX": 14.0}

SHOCK_HEADLINES = {
    "SPX": ("Wall Street {dir_word} as S&P 500 posts its biggest move in months",
            "US stocks {dir_word} sharply after surprise economic data"),
    "NIFTY": ("Nifty 50 {dir_word} as foreign investors shift positions",
              "Indian stocks {dir_word} on heavy volumes"),
    "USDINR": ("Rupee swings sharply against the dollar; RBI seen active in market",),
    "EURUSD": ("Euro {dir_word} after ECB comments surprise traders",),
    "GOLD": ("Gold {dir_word} as investors reassess safe havens",),
    "UST2Y": ("2-year Treasury yield {dir_word} after Fed officials signal policy shift",),
    "UST10Y": ("10-year Treasury yield {dir_word} on inflation surprise",),
    "CREDIT": ("Corporate bond spreads {dir_word} as credit worries return",),
}
NOISE_HEADLINES = [
    "Company announces new chief financial officer",
    "Retail sales data due later this week, economists say",
    "Tech firm unveils updated smartphone lineup",
    "Central bank minutes to be published next month",
    "Oil producers meet to discuss output targets",
    "Airline reports record passenger numbers for the holiday season",
    "Housing starts in line with forecasts",
    "Government announces infrastructure spending review",
    "Analysts expect steady earnings from consumer goods sector",
    "Shipping costs ease as port congestion clears",
    "Monsoon arrives on schedule, weather office says",
    "Startup raises funding in latest round",
]


def generate(cfg: Config, n_days: int = 3000, n_shocks: int = 60, seed: int = 7):
    """Return (levels, news, truth) where truth holds true regimes and shock days."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-01", periods=n_days)

    regimes = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        regimes[t] = rng.choice(3, p=TRANSITION[regimes[t - 1]])

    # common market factor gives realistic co-movement between assets
    factor = rng.standard_normal(n_days)
    loadings = {"SPX": 0.8, "NIFTY": 0.6, "USDINR": -0.3, "EURUSD": 0.2, "GOLD": -0.1,
                "UST2Y": 0.4, "UST10Y": 0.5, "CREDIT": -0.6}
    changes = {}
    for inst_id, p in PARAMS.items():
        mu = np.array(p["mu"])[regimes]
        sd = np.array(p["sd"])[regimes]
        b = loadings[inst_id]
        eps = b * factor + np.sqrt(1 - b ** 2) * rng.standard_normal(n_days)
        changes[inst_id] = mu + sd * eps

    shock_days = rng.choice(np.arange(300, n_days), size=n_shocks, replace=False)
    shock_rows = []
    for t in sorted(shock_days):
        inst_id = rng.choice(list(PARAMS))
        sign = rng.choice([-1, 1])
        size = PARAMS[inst_id]["sd"][regimes[t]] * rng.uniform(4.0, 6.0)
        changes[inst_id][t] = sign * size
        shock_rows.append({"date": dates[t], "instrument": inst_id, "direction": int(sign)})

    levels = {}
    for inst_id, ch in changes.items():
        kind = cfg.instrument(inst_id).kind
        if kind == "rate":
            # mean reversion keeps yields in a realistic range instead of drifting to zero
            path = np.empty(n_days)
            path[0] = START_LEVELS[inst_id]
            for t in range(1, n_days):
                pull = 0.004 * (START_LEVELS[inst_id] - path[t - 1])
                path[t] = max(path[t - 1] + ch[t] / 100.0 + pull, 0.05)
            levels[inst_id] = path
        else:
            levels[inst_id] = START_LEVELS[inst_id] * np.exp(np.cumsum(ch) / 100.0)

    vix = np.empty(n_days)
    vix[0] = START_LEVELS["VIX"]
    for t in range(1, n_days):
        target = VIX_TARGET[regimes[t]]
        vix[t] = max(9.0, vix[t - 1] + 0.15 * (target - vix[t - 1]) - 0.4 * changes["SPX"][t]
                     + rng.normal(0, 0.8))
    levels["VIX"] = vix

    levels_df = pd.DataFrame(levels, index=dates)[[i.id for i in cfg.instruments]]
    levels_df.index.name = "date"

    news = _make_news(dates, shock_rows, rng)
    truth = {
        "regimes": pd.Series(regimes, index=dates, name="true_regime"),
        "shocks": pd.DataFrame(shock_rows),
    }
    return levels_df, news, truth


def _make_news(dates, shock_rows, rng) -> pd.DataFrame:
    rows = []
    for d in dates:
        for title in rng.choice(NOISE_HEADLINES, size=4, replace=False):
            rows.append({"date": d, "title": title, "source": "synthetic", "url": "", "text": title})
    for s in shock_rows:
        inst_id, sign = s["instrument"], s["direction"]
        if inst_id in ("UST2Y", "UST10Y", "CREDIT"):
            dir_word = "jumps" if sign > 0 else "drops"
        else:
            dir_word = "rallies" if sign > 0 else "slides"
        template = rng.choice(SHOCK_HEADLINES[inst_id])
        title = template.format(dir_word=dir_word)
        url = f"synthetic://{s['date'].date()}/{inst_id}"
        rows.append({"date": s["date"], "title": title, "source": "synthetic", "url": url, "text": title})
    df = pd.DataFrame(rows)
    df["title"] = df["title"].astype(str)
    df["text"] = df["text"].astype(str)
    return df.drop_duplicates(subset=["date", "title"]).reset_index(drop=True)
