import numpy as np
import pandas as pd

from marketpulse.detector import evaluate_flags, labelling_sheet, mcnemar_test, run_all_detectors
from marketpulse.features import build_features
from marketpulse.generation.brief import generate_brief
from marketpulse.generation.facts import build_daily_facts
from marketpulse.generation.writers import TemplateWriter
from marketpulse.regimes import RegimeModel, walk_forward
from marketpulse.retrieval import NewsRetriever, evaluate_retrieval


def test_regime_labels_sorted_by_volatility(cfg, demo):
    feats = build_features(demo["levels"], cfg["regime"]["features"])
    desc = RegimeModel(3, n_init=2).fit(feats).describe()
    assert desc["spx_vol_20d"].is_monotonic_increasing


def test_regimes_beat_chance_on_synthetic(cfg, demo):
    feats = build_features(demo["levels"], cfg["regime"]["features"])
    proba = walk_forward(feats, 3, 600, 250)
    truth = demo["truth"]["regimes"].loc[proba.index]
    acc = (truth.to_numpy() == proba["regime"].to_numpy()).mean()
    majority = truth.value_counts(normalize=True).max()
    assert acc > majority, (acc, majority)


def test_detector_and_evaluation_run(cfg, demo):
    feats = build_features(demo["levels"], cfg["regime"]["features"])
    regime = walk_forward(feats, 3, 600, 250)["regime"]
    res = run_all_detectors(demo["changes"], regime, cfg)
    flags = res["flags"]
    assert flags["fixed"].to_numpy().sum() > flags["regime"].to_numpy().sum()
    shocks = demo["truth"]["shocks"]
    cells = pd.MultiIndex.from_product([flags["regime"].index, flags["regime"].columns]).to_frame(index=False)
    cells.columns = ["date", "instrument"]
    key = set(zip(shocks["date"], shocks["instrument"]))
    cells["notable"] = [int(k in key) for k in zip(cells["date"], cells["instrument"])]
    ev = evaluate_flags(flags, cells)
    assert ev.loc["regime", "recall"] > 0.4
    mc = mcnemar_test(flags["regime"], flags["fixed"], cells)
    assert 0 <= mc["p_value"] <= 1
    sheet = labelling_sheet(demo["changes"].loc[flags["regime"].index], flags, n_random=20)
    assert {"date", "instrument", "change", "notable"} <= set(sheet.columns)


def test_retrieval_finds_planted_headlines(cfg, demo):
    r = NewsRetriever("tfidf").fit(demo["news"])
    s = demo["truth"]["shocks"]
    labels = pd.DataFrame({"date": s["date"], "instrument": s["instrument"],
                           "relevant_url": [f"synthetic://{d.date()}/{i}" for d, i in zip(s["date"], s["instrument"])]})
    out = evaluate_retrieval(r, labels, demo["changes"], cfg)
    assert out["recall@3"].iloc[0] > 0.9


def test_brief_graph_retries_then_flags_for_review(cfg, demo):
    feats = build_features(demo["levels"], cfg["regime"]["features"])
    proba = walk_forward(feats, 3, 600, 250)
    res = run_all_detectors(demo["changes"], proba["regime"], cfg)
    day = res["flags"]["regime"].any(axis=1)
    day = day[day].index[-1]
    facts = build_daily_facts(day, demo["levels"], demo["changes"], res["z"]["regime"],
                              res["flags"]["regime"], proba, cfg)

    ok = generate_brief(facts, TemplateWriter(), cfg, max_retries=2)
    assert ok["status"] == "passed" and ok["attempts"] == 1

    class AlwaysWrong:
        name = "bad"

        def write(self, facts, feedback=None):
            return "The S&P 500 rose 99.99% to 123,456.78."

    class FixesAfterFeedback:
        name = "learner"

        def write(self, facts, feedback=None):
            return TemplateWriter().write(facts) if feedback else "Gold rose 42.42%."

    bad = generate_brief(facts, AlwaysWrong(), cfg, max_retries=2)
    assert bad["status"] == "needs_review" and bad["attempts"] == 3

    fixed = generate_brief(facts, FixesAfterFeedback(), cfg, max_retries=2)
    assert fixed["status"] == "passed" and fixed["attempts"] == 2
    assert np.isclose(fixed["history"][0]["number_accuracy"], 0.0)
