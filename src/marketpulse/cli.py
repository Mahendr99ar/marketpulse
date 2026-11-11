"""Command line entry point.

    python -m marketpulse.cli all --demo                 # whole pipeline on synthetic data
    python -m marketpulse.cli data --no-news             # download market data (2 minutes)
    python -m marketpulse.cli news                       # download headlines (resumable)
    python -m marketpulse.cli regimes --select           # fit regimes, compare state counts
    python -m marketpulse.cli detect
    python -m marketpulse.cli label-sheet                # rows for you to label by hand
    python -m marketpulse.cli brief --date 2024-08-05 --backend ollama
    python -m marketpulse.cli evaluate --backend ollama --n-briefs 30
"""

from __future__ import annotations

import argparse

from marketpulse import pipeline
from marketpulse.config import load_config


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="marketpulse")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--demo", action="store_true", help="use synthetic data")
        return sp

    add("data", "download or generate data").add_argument(
        "--no-news", action="store_true", help="market data only; add news later with `news`")
    add("news", "download news headlines (resumable)")
    add("regimes", "fit walk-forward regime model").add_argument("--select", action="store_true")
    add("detect", "score and flag unusual moves")
    add("label-sheet", "write a CSV of rows to label by hand").add_argument("--n-random", type=int, default=150)
    b = add("brief", "write a fact-checked brief for one date")
    b.add_argument("--date", default=None)
    b.add_argument("--backend", default=None, choices=["template", "ollama", "groq", "hf"])
    e = add("evaluate", "evaluate detectors, retrieval and briefs")
    e.add_argument("--backend", default=None, choices=["template", "ollama", "groq", "hf"])
    e.add_argument("--n-briefs", type=int, default=30)
    a = add("all", "run every step")
    a.add_argument("--backend", default=None, choices=["template", "ollama", "groq", "hf"])

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    ds = "demo" if args.demo else "real"

    if args.cmd == "data":
        pipeline.step_data(cfg, ds, news=not args.no_news)
    elif args.cmd == "news":
        pipeline.step_news(cfg, ds)
    elif args.cmd == "regimes":
        pipeline.step_regimes(cfg, ds, select=args.select)
    elif args.cmd == "detect":
        pipeline.step_detect(cfg, ds)
    elif args.cmd == "label-sheet":
        pipeline.step_label_sheet(cfg, ds, args.n_random)
    elif args.cmd == "brief":
        pipeline.step_brief(cfg, ds, args.date, args.backend)
    elif args.cmd == "evaluate":
        pipeline.step_evaluate(cfg, ds, args.backend, args.n_briefs)
    elif args.cmd == "all":
        pipeline.step_data(cfg, ds)
        pipeline.step_regimes(cfg, ds, select=True)
        pipeline.step_detect(cfg, ds)
        pipeline.step_brief(cfg, ds, None, args.backend)
        pipeline.step_evaluate(cfg, ds, args.backend)


if __name__ == "__main__":
    main()
