"""The fact sheet: every number the brief is allowed to contain.

Rule of the project: Python computes, the LLM only writes. Numbers are rounded here,
once, and the fact-checker accepts only these numbers.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from marketpulse.config import Config
from marketpulse.regimes import regime_names


class Move(BaseModel):
    instrument_id: str
    name: str
    kind: str                 # price | rate
    unit: str                 # % or bp, unit of `change`
    level: float
    change: float
    z: float | None = None
    flagged: bool = False


class NewsItem(BaseModel):
    title: str
    source: str
    url: str = ""


class DailyFacts(BaseModel):
    date: str
    regime: str
    regime_probability: int = Field(description="0-100, filtered probability of the regime")
    previous_regime: str | None = None
    moves: list[Move]
    news: dict[str, list[NewsItem]] = Field(default_factory=dict)

    @property
    def flagged(self) -> list[Move]:
        return [m for m in self.moves if m.flagged]

    def prompt_view(self) -> dict:
        """What the LLM sees. Numbers are pre-formatted strings so it copies, not computes."""
        return {
            "date": self.date,
            "regime": self.regime,
            "regime_probability_pct": self.regime_probability,
            "regime_changed_from": self.previous_regime,
            "unusual_moves": [
                {"market": m.name, "level": fmt_level(m), "change": fmt_change(m),
                 "z_score": f"{m.z:.1f}" if m.z is not None else None,
                 "headlines": [n.title for n in self.news.get(m.instrument_id, [])]}
                for m in self.flagged
            ],
            "other_markets": [
                {"market": m.name, "level": fmt_level(m), "change": fmt_change(m)}
                for m in self.moves if not m.flagged
            ],
        }


def fmt_change(m: Move) -> str:
    return f"{m.change:+.1f}bp" if m.kind == "rate" else f"{m.change:+.2f}%"


def fmt_level(m: Move) -> str:
    return f"{m.level:.2f}%" if m.kind == "rate" else f"{m.level:,.2f}"


def build_daily_facts(date, levels: pd.DataFrame, changes: pd.DataFrame, z: pd.DataFrame,
                      flags: pd.DataFrame, regime_proba: pd.DataFrame, cfg: Config,
                      retriever=None, max_flagged: int = 5) -> DailyFacts:
    from marketpulse.retrieval import move_query

    date = pd.Timestamp(date)
    proba = regime_proba.loc[:date]
    names = regime_names(proba.filter(like="p_").shape[1])
    p_today = proba.filter(like="p_").iloc[-1].to_numpy()
    r_today = int(p_today.argmax())
    prev = None
    if len(proba) > 1:
        r_prev = int(proba.filter(like="p_").iloc[-2].to_numpy().argmax())
        prev = names[r_prev] if r_prev != r_today else None

    moves = []
    for inst in cfg.instruments:
        zval = z.at[date, inst.id] if inst.id in z.columns else None
        zval = None if zval is None or pd.isna(zval) else round(float(zval), 1)
        change = float(changes.at[date, inst.id])
        moves.append(Move(
            instrument_id=inst.id, name=inst.name, kind=inst.kind, unit=inst.unit,
            level=round(float(levels.at[date, inst.id]), 2),
            change=round(change, 1 if inst.kind == "rate" else 2),
            z=zval, flagged=bool(flags.at[date, inst.id]),
        ))
    moves.sort(key=lambda m: -abs(m.z or 0))
    flagged_ids = [m.instrument_id for m in moves if m.flagged][:max_flagged]
    for m in moves:  # cap the number of flagged moves the brief has to discuss
        m.flagged = m.instrument_id in flagged_ids

    news = {}
    if retriever is not None:
        for m in moves:
            if m.flagged:
                hits = retriever.query(move_query(m.instrument_id, m.change, cfg), date,
                                       cfg["news"]["window_days"], k=3)
                # an unrelated top hit is worse than none: the brief would invent a cause
                hits = hits[hits["score"] >= cfg["retrieval"].get("min_score", 0.0)]
                news[m.instrument_id] = [NewsItem(title=r.title, source=r.source, url=r.url)
                                         for r in hits.itertuples()]
    return DailyFacts(date=date.strftime("%Y-%m-%d"), regime=names[r_today],
                      regime_probability=int(round(100 * p_today[r_today])),
                      previous_regime=prev, moves=moves, news=news)
