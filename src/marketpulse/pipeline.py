"""End-to-end steps. Each step reads the previous step's files from data/processed/<dataset>/.

dataset = "real" for downloaded data, "demo" for synthetic data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.getLogger("hmmlearn").setLevel(logging.ERROR)  # hides "Model is not converging" noise
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

from marketpulse.config import Config
from marketpulse.regimes import RegimeModel, regime_names, select_n_states, walk_forward


class Store:
    def __init__(self, cfg: Config, dataset: str):
        self.dir = cfg.path("processed") / dataset
        self.reports = cfg.path("reports") / dataset
        self.dir.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, df: pd.DataFrame) -> None:
        df.to_parquet(self.dir / f"{name}.parquet")

    def load(self, name: str) -> pd.DataFrame:
        path = self.dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run the earlier pipeline step first.")
        return pd.read_parquet(path)

    def exists(self, name: str) -> bool:
        return (self.dir / f"{name}.parquet").exists()


# ---------------- step 1: data ----------------

def step_data(cfg: Config, dataset: str, news: bool = True) -> None:
    """Market data first, saved right away. News second, so a slow or failed news
    download never costs you the market data."""
    from marketpulse.data.market import compute_changes

    store = Store(cfg, dataset)
    if dataset == "demo":
        from marketpulse.data.synthetic import generate

        levels, news_df, truth = generate(cfg)
        store.save("true_regimes", truth["regimes"].to_frame())
        store.save("true_shocks", truth["shocks"])
        store.save("news", news_df)
    else:
        from dotenv import load_dotenv

        from marketpulse.data.market import fetch_market

        load_dotenv()
        levels = fetch_market(cfg)
    store.save("levels", levels)
    store.save("changes", compute_changes(levels, cfg))
    print(f"[data] {len(levels)} days x {levels.shape[1]} instruments "
          f"({levels.index.min().date()} to {levels.index.max().date()}) -> {store.dir}")
    if dataset != "demo":
        if news:
            step_news(cfg, dataset)
        elif not store.exists("news"):
            from marketpulse.data.news import NEWS_COLUMNS

            store.save("news", pd.DataFrame(columns=NEWS_COLUMNS))
            print("[news] skipped. Run `cli news` later to add headlines.")


def step_news(cfg: Config, dataset: str = "real") -> None:
    from marketpulse.data.news import fetch_gdelt, fetch_rss, load_statements

    store = Store(cfg, dataset)
    levels = store.load("levels")
    n = cfg["news"]
    end = levels.index.max()
    start = end - pd.DateOffset(months=int(12 * n.get("years", 1)))
    print(f"[news] GDELT headlines {start.date()} to {end.date()}")
    parts = [fetch_gdelt(n["gdelt_keywords"], start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                         chunk_days=n.get("chunk_days", 3), pause_s=n.get("pause_seconds", 6.0),
                         cache_dir=cfg.path("raw") / "gdelt")]
    if n["rss_feeds"]:
        parts.append(fetch_rss(n["rss_feeds"]))
    statements_dir = cfg.path("raw") / "statements"
    statements_dir.mkdir(parents=True, exist_ok=True)
    parts.append(load_statements(statements_dir))
    parts = [p for p in parts if len(p)]
    from marketpulse.data.news import NEWS_COLUMNS

    news_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=NEWS_COLUMNS)
    store.save("news", news_df)
    print(f"[news] {len(news_df)} headlines and statement paragraphs -> {store.dir}")


# ---------------- step 2: regimes ----------------

def step_regimes(cfg: Config, dataset: str, select: bool = False) -> None:
    from marketpulse.features import build_features

    store = Store(cfg, dataset)
    r = cfg["regime"]
    feats = build_features(store.load("levels"), r["features"])
    store.save("features", feats)

    if select:
        train = feats.iloc[: int(len(feats) * 0.6)]
        bic = select_n_states(train, random_state=r["random_state"])
        bic.to_csv(store.reports / "bic_by_n_states.csv")
        print("[regimes] BIC on first 60% of data (lower is better):\n", bic.round(1))

    proba = walk_forward(feats, r["n_states"], r["min_train_days"], r["refit_every_days"], r["random_state"])
    store.save("regime_proba", proba)

    # in-sample model, only for describing regimes (means, durations, transition matrix)
    full = RegimeModel(r["n_states"], random_state=r["random_state"]).fit(feats)
    full.describe().to_csv(store.reports / "regime_profile.csv")
    full.transition_matrix().to_csv(store.reports / "regime_transitions.csv")
    smooth = full.smoothed_proba(feats)
    smooth["regime"] = smooth.to_numpy().argmax(axis=1)
    store.save("regime_insample", smooth)

    names = regime_names(r["n_states"])
    share = proba["regime"].value_counts(normalize=True).sort_index()
    print("[regimes] out-of-sample share of days:",
          ", ".join(f"{names[i]} {v:.0%}" for i, v in share.items()))
    print(full.describe().round(2))


# ---------------- step 3: detection ----------------

def step_detect(cfg: Config, dataset: str) -> None:
    from marketpulse.detector import flagged_moves, run_all_detectors

    store = Store(cfg, dataset)
    changes = store.load("changes")
    regime = store.load("regime_proba")["regime"]
    res = run_all_detectors(changes, regime, cfg)
    for name, z in res["z"].items():
        store.save(f"z_{name}", z)
    for name, f in res["flags"].items():
        store.save(f"flags_{name}", f)
    moves = flagged_moves(changes, res["z"]["regime"], res["flags"]["regime"])
    store.save("flagged_moves", moves)
    days = len(res["flags"]["regime"])
    counts = {k: int(v.to_numpy().sum()) for k, v in res["flags"].items()}
    print(f"[detect] {days} days scored. Flags per method: {counts}")


def step_label_sheet(cfg: Config, dataset: str, n_random: int = 150) -> Path:
    from marketpulse.detector import labelling_sheet

    store = Store(cfg, dataset)
    flags = {k: store.load(f"flags_{k}") for k in ("regime", "rolling", "fixed")}
    # label only the period you downloaded news for: you need the news to judge each move
    cutoff = flags["regime"].index.max() - pd.DateOffset(months=int(12 * cfg["news"].get("years", 1)))
    flags = {k: v.loc[cutoff:] for k, v in flags.items()}
    sheet = labelling_sheet(store.load("changes"), flags, n_random=n_random)

    # Context for the person labelling: market name, unit, and the top headlines of that day.
    # Nothing here says which detector flagged the row, so the labels stay blind.
    sheet.insert(2, "market", [cfg.instrument(i).name for i in sheet["instrument"]])
    sheet.insert(4, "unit", [cfg.instrument(i).unit for i in sheet["instrument"]])
    retriever = _retriever(cfg, store.load("news"))
    if retriever is not None:
        from marketpulse.retrieval import move_query

        heads = []
        for d, inst, ch in zip(sheet["date"], sheet["instrument"], sheet["change"]):
            hits = retriever.query(move_query(inst, ch, cfg), d, cfg["news"]["window_days"], k=3)
            heads.append(" | ".join(hits["title"].tolist()))
        sheet["headlines"] = heads
    # sorted by date so you can read one day's news once for all its rows
    sheet = sheet.sort_values(["date", "market"]).reset_index(drop=True)
    sheet["date"] = pd.to_datetime(sheet["date"]).dt.strftime("%Y-%m-%d")
    # weights go to a separate file: in the sheet they would reveal which rows were flagged
    weights = sheet[["date", "instrument", "weight"]]
    weights.to_csv(cfg.path("labels") / f"sample_weights_{dataset}.csv", index=False)
    sheet = sheet.drop(columns=["weight"])
    out = cfg.path("labels") / f"to_label_{dataset}.csv"
    sheet.to_csv(out, index=False, encoding="utf-8-sig")  # utf-8-sig: Excel shows symbols correctly
    print(f"[labels] {len(sheet)} rows -> {out}. Fill 'notable' with 1/0, save as notable_moves.csv")
    return out


# ---------------- step 4: briefs ----------------

def _retriever(cfg: Config, news: pd.DataFrame):
    """None when there is no news yet: briefs are then written from market data only."""
    from marketpulse.retrieval import NewsRetriever

    if news is None or len(news) == 0:
        return None
    rcfg = cfg["retrieval"]
    return NewsRetriever(rcfg["backend"], rcfg["embedding_model"]).fit(news)


def make_facts(cfg: Config, store: Store, date, retriever=None):
    from marketpulse.generation.facts import build_daily_facts

    return build_daily_facts(
        date, store.load("levels"), store.load("changes"), store.load("z_regime"),
        store.load("flags_regime"), store.load("regime_proba"), cfg, retriever,
        cfg["generation"]["max_movers_in_brief"])


def step_brief(cfg: Config, dataset: str, date: str | None = None, backend: str | None = None) -> dict:
    from marketpulse.generation.brief import generate_brief
    from marketpulse.generation.writers import get_writer

    store = Store(cfg, dataset)
    z = store.load("z_regime")
    if date is None:  # default: the latest day with at least one unusual move
        flagged = store.load("flags_regime").any(axis=1)
        date = flagged[flagged].index.max() if flagged.any() else z.index.max()
    date = pd.Timestamp(date)
    if date not in z.index:
        date = z.index[z.index.get_indexer([date], method="pad")[0]]
    facts = make_facts(cfg, store, date, _retriever(cfg, store.load("news")))
    result = generate_brief(facts, get_writer(cfg, backend), cfg)
    out = {"date": facts.date, "status": result["status"], "attempts": result["attempts"],
           "brief": result["draft"], "number_accuracy": result["report"].number_accuracy,
           "problems": result["report"].feedback(), "facts": facts.model_dump()}
    path = store.reports / f"brief_{facts.date}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[{facts.date}] status={out['status']} attempts={out['attempts']}\n\n{out['brief']}\n")
    return out


# ---------------- step 5: evaluation ----------------

def step_evaluate(cfg: Config, dataset: str, backend: str | None = None, n_briefs: int = 30) -> None:
    from marketpulse.detector import evaluate_flags, mcnemar_test
    from marketpulse.generation.brief import generate_brief
    from marketpulse.generation.writers import get_writer
    from marketpulse.retrieval import NewsRetriever, evaluate_retrieval

    store = Store(cfg, dataset)
    changes = store.load("changes")
    flags = {k: store.load(f"flags_{k}") for k in ("regime", "rolling", "fixed")}
    lines = [f"# Evaluation report ({dataset})", ""]
    if dataset == "demo":
        lines += ["Synthetic data. These numbers check that the code works. They are not findings.", ""]

    # regimes vs truth (demo only: real markets have no ground-truth regimes)
    if store.exists("true_regimes"):
        proba = store.load("regime_proba")
        truth = store.load("true_regimes")["true_regime"].loc[proba.index]
        acc = float((truth.to_numpy() == proba["regime"].to_numpy()).mean())
        conf = pd.crosstab(truth, proba["regime"], rownames=["true"], colnames=["predicted"])
        conf.to_csv(store.reports / "regime_confusion.csv")
        lines += ["## Regimes (out-of-sample, filtered)", f"Accuracy vs true regime: {acc:.1%}", "",
                  conf.to_markdown() if _has_tabulate() else conf.to_string(), ""]

    # detector labels
    labels = _detector_labels(cfg, store, changes, flags["regime"])
    if labels is not None:
        ev = evaluate_flags(flags, labels)
        ev.to_csv(store.reports / "detector_eval.csv")
        mc = mcnemar_test(flags["regime"], flags["rolling"], labels)
        mc_fixed = mcnemar_test(flags["regime"], flags["fixed"], labels)
        (store.reports / "mcnemar.json").write_text(json.dumps({"regime_vs_rolling": mc, "regime_vs_fixed": mc_fixed}, indent=2))
        lines += ["## Unusual-move detectors", _table(ev.round(3)), "",
                  f"McNemar regime vs rolling: regime right & rolling wrong = {mc['a_right_b_wrong']}, "
                  f"opposite = {mc['a_wrong_b_right']}, p = {mc['p_value']:.4f}",
                  f"McNemar regime vs fixed: regime right & fixed wrong = {mc_fixed['a_right_b_wrong']}, "
                  f"opposite = {mc_fixed['a_wrong_b_right']}, p = {mc_fixed['p_value']:.4f}", ""]
    else:
        lines += ["## Unusual-move detectors", "No labels yet. Run `label-sheet`, label, save as data/labels/notable_moves.csv.", ""]

    # retrieval
    rlabels = _retrieval_labels(cfg, store)
    news = store.load("news")
    if rlabels is not None and len(news):
        rows = [evaluate_retrieval(NewsRetriever("tfidf").fit(news), rlabels, changes, cfg,
                                   window_days=cfg["news"]["window_days"])]
        if cfg["retrieval"]["backend"] == "embeddings":
            rows.append(evaluate_retrieval(_retriever(cfg, news), rlabels, changes, cfg,
                                           window_days=cfg["news"]["window_days"]))
        rv = pd.concat(rows)
        rv.to_csv(store.reports / "retrieval_eval.csv")
        lines += ["## News retrieval", _table(rv.round(3)), ""]
    else:
        lines += ["## News retrieval", "No labels yet. Create data/labels/retrieval_labels.csv (date, instrument, relevant_url).", ""]

    # briefs
    writer = get_writer(cfg, backend)
    retriever = _retriever(cfg, news)
    any_flag = flags["regime"].any(axis=1)
    candidates = any_flag[any_flag].index
    rng = np.random.default_rng(0)
    dates = sorted(rng.choice(candidates, size=min(n_briefs, len(candidates)), replace=False))
    recs, passed_pairs = [], []
    for d in dates:
        facts = make_facts(cfg, store, d, retriever)
        res = generate_brief(facts, writer, cfg)
        if res["status"] == "passed":
            passed_pairs.append((res["draft"], facts))
        first = res["history"][0]
        recs.append({"date": facts.date, "status": res["status"], "attempts": res["attempts"],
                     "first_try_passed": first["passed"], "first_try_number_accuracy": first["number_accuracy"],
                     "final_number_accuracy": res["report"].number_accuracy, "brief": res["draft"],
                     "rating_1_to_5": None})
    br = pd.DataFrame(recs)
    br.to_json(store.reports / f"briefs_{writer.name}.jsonl", orient="records", lines=True)
    summary = {
        "writer": writer.name, "n": len(br),
        "first_try_pass_rate": float(br["first_try_passed"].mean()),
        "final_pass_rate": float((br["status"] == "passed").mean()),
        "mean_first_try_number_accuracy": float(br["first_try_number_accuracy"].mean()),
        "mean_attempts": float(br["attempts"].mean()),
    }
    lines += ["## Briefs", _table(pd.DataFrame([summary]).set_index("writer").round(3)), "",
              f"Rate each brief in briefs_{writer.name}.jsonl with docs/rubric.md.", ""]

    from marketpulse.generation.checker_eval import evaluate_checker

    if passed_pairs:
        ce = evaluate_checker(passed_pairs, cfg)
        ce.to_csv(store.reports / "factchecker_eval.csv")
        lines += ["## Fact-checker (planted errors in briefs that passed)", _table(ce.round(3)), ""]
    (store.reports / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def _detector_labels(cfg, store, changes, flags_regime):
    # hand labels are for real market dates; the demo data has its own planted shocks
    path = cfg.path("labels") / "notable_moves.csv"
    if path.exists() and not store.exists("true_shocks"):
        lab = pd.read_csv(path, parse_dates=["date"])
        lab = lab[lab["notable"].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
        lab["notable"] = lab["notable"].astype(float).astype(int)
        wpath = cfg.path("labels") / f"sample_weights_{store.dir.name}.csv"
        if wpath.exists():
            w = pd.read_csv(wpath, parse_dates=["date"])
            lab = lab.drop(columns=["weight"], errors="ignore").merge(w, on=["date", "instrument"], how="left")
            lab["weight"] = lab["weight"].fillna(1.0)
        return lab
    if store.exists("true_shocks"):  # demo: every scored cell is labelled
        shocks = store.load("true_shocks")
        idx = flags_regime.index
        cells = pd.MultiIndex.from_product([idx, flags_regime.columns], names=["date", "instrument"])
        lab = cells.to_frame(index=False)
        key = set(zip(shocks["date"], shocks["instrument"]))
        lab["notable"] = [int((d, i) in key) for d, i in zip(lab["date"], lab["instrument"])]
        return lab
    return None


def _retrieval_labels(cfg, store):
    path = cfg.path("labels") / "retrieval_labels.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    if store.exists("true_shocks"):
        s = store.load("true_shocks")
        return pd.DataFrame({"date": s["date"], "instrument": s["instrument"],
                             "relevant_url": [f"synthetic://{d.date()}/{i}" for d, i in zip(s["date"], s["instrument"])]})
    return None


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def _table(df: pd.DataFrame) -> str:
    return df.to_markdown() if _has_tabulate() else "```\n" + df.to_string() + "\n```"
