"""Flag unusual daily moves and compare detectors.

Main idea: a 1% fall in the S&P 500 is big in a calm market and ordinary in a crisis.
So the move is scaled by the volatility seen on past days of the same regime.

Three detectors:
    regime   change / std of past changes in the same regime   (the project's method)
    rolling  change / std of the last N days, ignoring regime  (strong baseline)
    fixed    |change| above a fixed size, e.g. 1.5% or 10bp     (naive baseline)

All volatility estimates use .shift(1), so day t is scored with data up to t-1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from marketpulse.config import Config


def regime_conditional_z(changes: pd.DataFrame, regime: pd.Series, lookback: int,
                         min_obs: int, fallback_window: int = 252) -> pd.DataFrame:
    idx = changes.index.intersection(regime.index)
    changes, regime = changes.loc[idx], regime.loc[idx]
    z = pd.DataFrame(index=idx, columns=changes.columns, dtype=float)
    for col in changes.columns:
        x = changes[col]
        sigma = pd.Series(np.nan, index=idx)
        for r in np.unique(regime):
            same = x.where(regime == r)
            sigma_r = same.shift(1).rolling(lookback, min_periods=min_obs).std()
            mask = regime == r
            sigma[mask] = sigma_r[mask]
        # A regime seen fewer than min_obs times in the lookback has no reliable vol yet.
        fallback = x.shift(1).rolling(fallback_window, min_periods=60).std()
        sigma = sigma.fillna(fallback)
        z[col] = x / sigma
    return z


def rolling_z(changes: pd.DataFrame, window: int) -> pd.DataFrame:
    sigma = changes.shift(1).rolling(window, min_periods=60).std()
    return changes / sigma


def fixed_flags(changes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    thr = cfg["detector"]["fixed_thresholds"]
    out = {}
    for col in changes.columns:
        out[col] = changes[col].abs() >= thr[cfg.instrument(col).kind]
    return pd.DataFrame(out, index=changes.index)


def flags_from_z(z: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return z.abs() >= threshold


def run_all_detectors(changes: pd.DataFrame, regime: pd.Series, cfg: Config) -> dict:
    d = cfg["detector"]
    z_reg = regime_conditional_z(changes, regime, d["lookback_days"], d["min_obs"],
                                 d["baseline_rolling_days"])
    idx = z_reg.index
    z_roll = rolling_z(changes, d["baseline_rolling_days"]).loc[idx]
    return {
        "z": {"regime": z_reg, "rolling": z_roll},
        "flags": {
            "regime": flags_from_z(z_reg, d["z_threshold"]),
            "rolling": flags_from_z(z_roll, d["z_threshold"]),
            "fixed": fixed_flags(changes.loc[idx], cfg),
        },
    }


def flagged_moves(changes: pd.DataFrame, z: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per flagged (date, instrument)."""
    rows = flags.stack()
    rows = rows[rows].index
    out = pd.DataFrame({
        "date": rows.get_level_values(0),
        "instrument": rows.get_level_values(1),
    })
    out["change"] = [changes.at[d, i] for d, i in rows]
    out["z"] = [z.at[d, i] for d, i in rows]
    return out.sort_values(["date", "z"], key=lambda s: s.abs() if s.name == "z" else s,
                           ascending=[True, False]).reset_index(drop=True)


# ---------------- evaluation ----------------

def _lookup(flags: pd.DataFrame, labels: pd.DataFrame) -> np.ndarray:
    return np.array([bool(flags.at[pd.Timestamp(d), i]) for d, i in zip(labels["date"], labels["instrument"])])


def evaluate_flags(flags_by_method: dict[str, pd.DataFrame], labels: pd.DataFrame) -> pd.DataFrame:
    """Precision / recall / F1 on labelled (date, instrument, notable) rows."""
    labels = labels[labels["date"].isin(next(iter(flags_by_method.values())).index)]
    y = labels["notable"].astype(bool).to_numpy()
    rows = []
    for name, flags in flags_by_method.items():
        pred = _lookup(flags, labels)
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"method": name, "flags": int(pred.sum()), "tp": tp, "fp": fp, "fn": fn,
                     "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows).set_index("method")


def mcnemar_test(flags_a: pd.DataFrame, flags_b: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Do detectors A and B make different numbers of mistakes on the same rows?

    H0: both detectors have the same error rate. Only rows where exactly one of them
    is right carry information, which is what McNemar's test uses.
    """
    from statsmodels.stats.contingency_tables import mcnemar

    labels = labels[labels["date"].isin(flags_a.index)]
    y = labels["notable"].astype(bool).to_numpy()
    a_ok = _lookup(flags_a, labels) == y
    b_ok = _lookup(flags_b, labels) == y
    table = [[int((a_ok & b_ok).sum()), int((a_ok & ~b_ok).sum())],
             [int((~a_ok & b_ok).sum()), int((~a_ok & ~b_ok).sum())]]
    exact = (table[0][1] + table[1][0]) < 25
    res = mcnemar(table, exact=exact)
    return {"table": table, "a_right_b_wrong": table[0][1], "a_wrong_b_right": table[1][0],
            "statistic": float(res.statistic), "p_value": float(res.pvalue), "exact": exact}


def labelling_sheet(changes: pd.DataFrame, flags_by_method: dict[str, pd.DataFrame],
                    n_random: int = 150, seed: int = 0) -> pd.DataFrame:
    """Rows for you to label by hand: everything any detector flagged plus a random sample.

    Including unflagged rows is what lets you measure recall. Fill the `notable`
    column with 1 or 0 after reading that day's news.
    """
    any_flag = None
    for f in flags_by_method.values():
        any_flag = f if any_flag is None else (any_flag | f)
    stacked = any_flag.stack()
    flagged = stacked[stacked].index
    unflagged = stacked[~stacked].index
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(unflagged), size=min(n_random, len(unflagged)), replace=False)
    rows = list(flagged) + [unflagged[i] for i in pick]
    sheet = pd.DataFrame(rows, columns=["date", "instrument"])
    sheet["change"] = [round(changes.at[d, i], 3) for d, i in rows]
    sheet["notable"] = ""
    sheet["note"] = ""
    # shuffle so you cannot tell which rows were flagged while labelling
    return sheet.sample(frac=1.0, random_state=seed).reset_index(drop=True)
