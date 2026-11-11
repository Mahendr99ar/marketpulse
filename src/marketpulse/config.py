"""Load configs/config.yaml and resolve paths relative to the project root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class Instrument:
    id: str
    name: str
    source: str
    ticker: str
    kind: str  # "price" or "rate"
    aliases: tuple[str, ...]

    @property
    def unit(self) -> str:
        return "bp" if self.kind == "rate" else "%"


class Config:
    def __init__(self, raw: dict[str, Any], root: Path = PROJECT_ROOT):
        self.raw = raw
        self.root = root
        self.instruments = [
            Instrument(
                id=i["id"],
                name=i["name"],
                source=i["source"],
                ticker=i["ticker"],
                kind=i["kind"],
                aliases=tuple(i.get("aliases", [i["name"]])),
            )
            for i in raw["instruments"]
        ]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, name: str) -> Path:
        p = self.root / self.raw["paths"][name]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def instrument(self, inst_id: str) -> Instrument:
        for inst in self.instruments:
            if inst.id == inst_id:
                return inst
        raise KeyError(inst_id)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
