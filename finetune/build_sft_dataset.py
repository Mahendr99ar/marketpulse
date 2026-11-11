"""Build a fine-tuning dataset by distillation with a fact-check filter.

There are no "gold" briefs to learn from. So a bigger teacher model (Groq or a large
Ollama model) writes briefs, the fact-checker keeps only the ones that pass, and the
small student model learns from those. The checker is the quality gate.

Split by time: training dates come before --split-date, test dates after it. The
student must never see the days it is evaluated on.

    python finetune/build_sft_dataset.py --teacher groq --n 600 --split-date 2024-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketpulse.config import load_config  # noqa: E402
from marketpulse.generation.brief import generate_brief  # noqa: E402
from marketpulse.generation.writers import SYSTEM_PROMPT, get_writer, user_prompt  # noqa: E402
from marketpulse.pipeline import Store, _retriever, make_facts  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="groq", choices=["groq", "ollama", "template"])
    p.add_argument("--dataset", default="real")
    p.add_argument("--n", type=int, default=600, help="number of training days to try")
    p.add_argument("--split-date", required=True)
    p.add_argument("--out", default="finetune/data")
    args = p.parse_args()

    cfg = load_config()
    store = Store(cfg, args.dataset)
    flags = store.load("flags_regime")
    # mix days with and without unusual moves so the model learns both cases
    days = flags.index
    split = pd.Timestamp(args.split_date)
    train_days, test_days = days[days < split], days[days >= split]
    rng = np.random.default_rng(0)
    train_pick = sorted(rng.choice(train_days, size=min(args.n, len(train_days)), replace=False))

    writer = get_writer(cfg, args.teacher)
    retriever = _retriever(cfg, store.load("news"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    kept, dropped = 0, 0
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for d in train_pick:
            facts = make_facts(cfg, store, d, retriever)
            res = generate_brief(facts, writer, cfg, max_retries=0)  # first try only
            if res["status"] != "passed":
                dropped += 1
                continue
            f.write(json.dumps({"date": facts.date, "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(facts)},
                {"role": "assistant", "content": res["draft"]},
            ]}) + "\n")
            kept += 1
    (out / "test_dates.json").write_text(json.dumps([d.strftime("%Y-%m-%d") for d in test_days]))
    print(f"kept {kept}, dropped {dropped} (failed fact check). Teacher first-try pass rate: "
          f"{kept / max(kept + dropped, 1):.1%}. Test dates: {len(test_days)} after {args.split_date}")


if __name__ == "__main__":
    main()
