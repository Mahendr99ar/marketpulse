"""Check a generated brief against its fact sheet.

Three checks:
    numbers    every number in the text must match a number in the facts
    direction  a sentence about one market must not say "rose" when it fell
    regime     the text must not name a different regime than the model output

This is deliberately strict and simple. It will miss some errors (for example a wrong
causal claim about the news) and that limit belongs in your README.
"""

from __future__ import annotations

import re

import pandas as pd
from pydantic import BaseModel

from marketpulse.config import Config
from marketpulse.generation.facts import DailyFacts
from marketpulse.regimes import REGIME_NAMES

NUMBER_RE = re.compile(r"(?<![\w.])[+\-−]?\d[\d,]*(?:\.\d+)?")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
UP_WORDS = r"\b(rose|rises|rising|gained|gains|rallied|rallies|jumped|jumps|climbed|climbs|surged|surges|higher|advanced|widened|widens|increased)\b"
DOWN_WORDS = r"\b(fell|falls|falling|dropped|drops|declined|declines|slid|slides|slipped|tumbled|tumbles|lower|eased|eases|narrowed|narrows|decreased|retreated)\b"
# numbers that are part of names or tenors, not data: "2-year", "10Y", "20-day"
TENOR_RE = re.compile(r"\b\d+\s?-?\s?(?:year|yr|day|week|month)s?\b|\b\d+[YyMm]\b", re.I)


class FactCheckReport(BaseModel):
    numbers_checked: int
    unsupported_numbers: list[str]
    direction_errors: list[str]
    regime_errors: list[str]

    @property
    def number_accuracy(self) -> float:
        if self.numbers_checked == 0:
            return 1.0
        return 1 - len(self.unsupported_numbers) / self.numbers_checked

    @property
    def passed(self) -> bool:
        return not (self.unsupported_numbers or self.direction_errors or self.regime_errors)

    def feedback(self) -> str:
        parts = []
        if self.unsupported_numbers:
            parts.append("These numbers are not in FACTS, remove or correct them: "
                         + ", ".join(self.unsupported_numbers))
        parts += [f"Direction error: {e}" for e in self.direction_errors]
        parts += [f"Regime error: {e}" for e in self.regime_errors]
        return "\n".join(parts)


def _num(tok: str) -> float | None:
    try:
        return float(tok.replace(",", "").replace("\u2212", "-"))
    except ValueError:
        return None


def global_numbers(facts: DailyFacts) -> list[float]:
    """Numbers that may appear anywhere: regime probability, count of moves, headline numbers."""
    vals = [float(facts.regime_probability), float(len(facts.flagged))]
    for items in facts.news.values():  # "Fed cuts by 25bp" in a headline is sourced
        for item in items:
            vals += [abs(v) for v in map(_num, NUMBER_RE.findall(item.title)) if v is not None]
    return vals


def instrument_numbers(facts: DailyFacts) -> dict[str, list[float]]:
    out = {}
    for m in facts.moves:
        vals = [m.level, m.change, abs(m.change)]
        if m.z is not None:
            vals += [m.z, abs(m.z)]
        out[m.instrument_id] = vals
    return out


def allowed_numbers(facts: DailyFacts) -> list[float]:
    return global_numbers(facts) + [v for vals in instrument_numbers(facts).values() for v in vals]


def _alias_index(cfg: Config) -> list[tuple[str, str]]:
    pairs = {(a, inst.id) for inst in cfg.instruments for a in (inst.name, *inst.aliases)}
    return sorted(pairs, key=lambda p: -len(p[0]))


def _mask(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for a, b in spans:
        chars[a:b] = " " * (b - a)
    return "".join(chars)


def _date_strings(facts: DailyFacts) -> set[str]:
    d = pd.Timestamp(facts.date)
    # built by hand because strftime("%-d") does not work on Windows
    month, mon, day = d.strftime("%B"), d.strftime("%b"), d.day
    return {facts.date, f"{day:02d} {month} {d.year}", f"{day} {month} {d.year}",
            f"{month} {day:02d}, {d.year}", f"{month} {day}, {d.year}",
            f"{day:02d} {mon} {d.year}", f"{day} {mon} {d.year}"}


def _scan(text: str, cfg: Config, facts: DailyFacts):
    """Return (masked text, [(start, end, instrument_id)] for every market name found).

    Masking keeps character positions, so numbers can be linked to their sentence.
    """
    spans, mentions = [], []
    taken = [False] * len(text)
    for alias, iid in _alias_index(cfg):
        for m in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, flags=re.I):
            if any(taken[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            spans.append((m.start(), m.end()))
            mentions.append((m.start(), m.end(), iid))
    for s in _date_strings(facts):
        spans += [(m.start(), m.end()) for m in re.finditer(re.escape(s), text)]
    spans += [(m.start(), m.end()) for m in TENOR_RE.finditer(text)]
    return _mask(text, spans), mentions


def _strip_names(text: str, cfg: Config, facts: DailyFacts) -> str:
    """Text with market names, the date and tenors blanked out ('500' in 'S&P 500')."""
    return _scan(text, cfg, facts)[0]


def _matches(value: float, decimals: int, allowed: list[float]) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(a - value) <= tol for a in allowed)


def check_numbers(text: str, facts: DailyFacts, cfg: Config) -> tuple[int, list[str]]:
    """Each number must belong to a market named in the same sentence.

    If a sentence names no market, any number from the facts is accepted. This catches
    the common LLM slip of giving one market's move to another.
    """
    masked, mentions = _scan(text, cfg, facts)
    glob, per_inst = global_numbers(facts), instrument_numbers(facts)
    everything = allowed_numbers(facts)
    bounds = [0] + [m.end() for m in SENTENCE_RE.finditer(text)] + [len(text) + 1]
    found, bad = 0, []
    for m in NUMBER_RE.finditer(masked):
        raw = m.group().replace(",", "").replace("\u2212", "-")
        value = _num(raw)
        if value is None:
            continue
        found += 1
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        lo = max(b for b in bounds if b <= m.start())
        hi = min(b for b in bounds if b > m.start())
        named = {iid for a, _, iid in mentions if lo <= a < hi}
        allowed = glob + [v for iid in named for v in per_inst[iid]] if named else everything
        if not _matches(value, decimals, allowed):
            bad.append(m.group())
    return found, bad


def check_directions(text: str, facts: DailyFacts, cfg: Config) -> list[str]:
    errors = []
    by_alias = {}
    for inst in cfg.instruments:
        for a in (inst.name, *inst.aliases):
            by_alias[a.lower()] = inst.id
    change = {m.instrument_id: m.change for m in facts.moves}
    for sent in SENTENCE_RE.split(text):
        low = sent.lower()
        mentioned = {iid for alias, iid in by_alias.items()
                     if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", low)}
        if len(mentioned) != 1:
            continue  # sentences about several markets are too ambiguous for a regex check
        iid = mentioned.pop()
        up, down = bool(re.search(UP_WORDS, low)), bool(re.search(DOWN_WORDS, low))
        if up == down or change.get(iid, 0) == 0:
            continue
        if up and change[iid] < 0:
            errors.append(f"'{sent.strip()}' says up, actual change {change[iid]}")
        if down and change[iid] > 0:
            errors.append(f"'{sent.strip()}' says down, actual change {change[iid]}")
    return errors


def check_regime(text: str, facts: DailyFacts) -> list[str]:
    all_names = {n for names in REGIME_NAMES.values() for n in names}
    errors = []
    for name in all_names - {facts.regime, facts.previous_regime}:
        if re.search(rf"\b{name}\s+(regime|market regime|state)\b", text, re.I):
            errors.append(f"text says '{name} regime' but the model says '{facts.regime}'")
    return errors


def fact_check(text: str, facts: DailyFacts, cfg: Config) -> FactCheckReport:
    n, bad = check_numbers(text, facts, cfg)
    return FactCheckReport(numbers_checked=n, unsupported_numbers=bad,
                           direction_errors=check_directions(text, facts, cfg),
                           regime_errors=check_regime(text, facts))
