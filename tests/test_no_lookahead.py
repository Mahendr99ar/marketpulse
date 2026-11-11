"""Leakage tests. If one of these fails, every backtest number in the project is suspect."""

import numpy as np
import pandas as pd

from marketpulse.detector import regime_conditional_z, rolling_z
from marketpulse.features import build_features
from marketpulse.regimes import RegimeModel


def _shock_future(df: pd.DataFrame, t: int) -> pd.DataFrame:
    out = df.copy()
    out.iloc[t:] = out.iloc[t:] * 3.0 + 50
    return out


def test_features_do_not_use_future(cfg, demo):
    names = cfg["regime"]["features"]
    base = build_features(demo["levels"], names)
    t = 1000
    changed = build_features(_shock_future(demo["levels"], t), names)
    cut = demo["levels"].index[t - 1]
    pd.testing.assert_frame_equal(base.loc[:cut], changed.loc[:cut])


def test_filtered_regimes_do_not_use_future(cfg, demo):
    feats = build_features(demo["levels"], cfg["regime"]["features"])
    model = RegimeModel(3, n_init=2).fit(feats.iloc[:800])
    t = 1100
    a = model.filter_proba(feats.iloc[:t])
    b = model.filter_proba(pd.concat([feats.iloc[:t], _shock_future(feats, t).iloc[t:]]))
    np.testing.assert_allclose(a.to_numpy(), b.iloc[:t].to_numpy())


def test_smoothed_regimes_do_use_future(cfg, demo):
    """Documents why filter_proba exists: predict_proba changes when the future changes."""
    feats = build_features(demo["levels"], cfg["regime"]["features"])
    model = RegimeModel(3, n_init=2).fit(feats.iloc[:800])
    t = 1100
    a = model.smoothed_proba(feats.iloc[:t])
    b = model.smoothed_proba(pd.concat([feats.iloc[:t], _shock_future(feats, t).iloc[t:]]))
    assert not np.allclose(a.to_numpy()[-20:], b.iloc[:t].to_numpy()[-20:])


def test_z_scores_do_not_use_today_or_future(demo):
    ch = demo["changes"]
    regime = pd.Series(np.arange(len(ch)) % 2, index=ch.index)
    t = 900
    base = regime_conditional_z(ch, regime, 500, 30)
    changed = ch.copy()
    changed.iloc[t:] *= 10
    z2 = regime_conditional_z(changed, regime, 500, 30)
    pd.testing.assert_frame_equal(base.iloc[:t], z2.iloc[:t])
    # day t's own move must not shrink its own z-score
    r1, r2 = rolling_z(ch, 252), rolling_z(changed, 252)
    np.testing.assert_allclose(r2.iloc[t].to_numpy(), 10 * r1.iloc[t].to_numpy())
