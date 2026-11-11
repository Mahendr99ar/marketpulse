"""Regime features. Every feature at day t uses data up to and including day t only."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_BUILDERS = {
    "spx_ret_5d": lambda lv: np.log(lv["SPX"]).diff(5) * 100,
    "spx_vol_20d": lambda lv: (np.log(lv["SPX"]).diff() * 100).rolling(20).std() * np.sqrt(252),
    "log_vix": lambda lv: np.log(lv["VIX"]),
    "credit_chg_20d": lambda lv: lv["CREDIT"].diff(20) * 100,
    "curve_10s2s": lambda lv: (lv["UST10Y"] - lv["UST2Y"]) * 100,
}


def build_features(levels: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    unknown = set(names) - set(FEATURE_BUILDERS)
    if unknown:
        raise KeyError(f"No builder for features {unknown}. Add one to FEATURE_BUILDERS.")
    feats = pd.DataFrame({n: FEATURE_BUILDERS[n](levels) for n in names}, index=levels.index)
    return feats.dropna()
