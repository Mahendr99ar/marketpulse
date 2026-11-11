"""Test the fact-checker itself by planting errors in correct briefs.

If the checker misses planted errors, its "passed" label means little. This gives a
catch rate per error type, which is a number you can defend in an interview.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from marketpulse.config import Config
from marketpulse.generation.factcheck import NUMBER_RE, _strip_names, fact_check
from marketpulse.generation.facts import DailyFacts
from marketpulse.regimes import REGIME_NAMES

SWAPS = [("rose", "fell"), ("fell", "rose"), ("rallied", "slid"), ("jumped", "dropped"),
         ("climbed", "fell"), ("widened", "narrowed"), ("narrowed", "widened"), ("dropped", "rose")]


def corrupt_number(text: str, facts: DailyFacts, cfg: Config, rng) -> str | None:
    checkable = set(NUMBER_RE.findall(_strip_names(text, cfg, facts)))
    candidates = [m for m in NUMBER_RE.finditer(text) if m.group() in checkable]
    if not candidates:
        return None
    m = candidates[rng.integers(len(candidates))]
    raw = m.group().replace(",", "")
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    value = float(raw) * (1 + rng.choice([-1, 1]) * rng.uniform(0.15, 0.6))
    if round(value, decimals) == round(float(raw), decimals):
        value += 10 ** (-decimals) * 3
    new = f"{value:.{decimals}f}"
    return text[: m.start()] + new + text[m.end():]


def corrupt_direction(text: str, rng) -> str | None:
    options = [(a, b) for a, b in SWAPS if re.search(rf"\b{a}\b", text)]
    if not options:
        return None
    a, b = options[rng.integers(len(options))]
    return re.sub(rf"\b{a}\b", b, text, count=1)


def corrupt_regime(text: str, facts: DailyFacts) -> str | None:
    others = [n for n in REGIME_NAMES[3] if n not in (facts.regime, facts.previous_regime)]
    pat = rf"\b{facts.regime}\s+regime\b"
    if not others or not re.search(pat, text):
        return None
    return re.sub(pat, f"{others[0]} regime", text, count=1)


def evaluate_checker(pairs: list[tuple[str, DailyFacts]], cfg: Config, n_per_brief: int = 3,
                     seed: int = 0) -> pd.DataFrame:
    """pairs: (brief that passed the check, its facts). Returns catch rate per error type."""
    rng = np.random.default_rng(seed)
    results = {"number": [], "direction": [], "regime": []}
    for text, facts in pairs:
        for _ in range(n_per_brief):
            for kind in results:
                if kind == "number":
                    bad = corrupt_number(text, facts, cfg, rng)
                elif kind == "direction":
                    bad = corrupt_direction(text, rng)
                else:
                    bad = corrupt_regime(text, facts)
                if bad is None or bad == text:
                    continue
                results[kind].append(not fact_check(bad, facts, cfg).passed)
    return pd.DataFrame([
        {"error_type": k, "planted": len(v), "caught": int(sum(v)),
         "catch_rate": float(np.mean(v)) if v else float("nan")}
        for k, v in results.items()
    ]).set_index("error_type")
