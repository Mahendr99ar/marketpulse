# MarketPulse

Regime-aware detection of unusual market moves, with a daily brief that is fact-checked against the data before anyone reads it.

## The problem

Every morning, analysts on a sales and trading desk write a short note on what moved overnight and why. Two parts of that job are slow or risky:

1. **Finding what matters.** Dozens of markets move every day. A 1% drop in the S&P 500 is news in a quiet market and noise in a crisis, so a fixed threshold either misses real moves or floods the desk with false alarms.
2. **Trusting automated text.** An LLM can write the note in seconds, but it sometimes invents or swaps numbers. One wrong figure in a client note costs more than the time saved.

MarketPulse handles both: it judges each move against the current market regime, and it lets an LLM write only after Python has computed every number, then checks the text before release.

## How it works

```
FRED + Yahoo Finance ──> features ──> HMM regimes (walk-forward, filtered)
                                              │
daily changes ───────────────────────────────>├──> regime-conditional z-scores ──> unusual moves
                                              │
GDELT / RSS / central bank statements ──> retrieval (TF-IDF or embeddings) ──> likely driver
                                                                                     │
                     fact sheet (Python computes and rounds every number) <──────────┘
                                   │
                     LLM writes ──> fact-checker ──> pass: publish
                         ^                │
                         └── feedback ────┤ fail, retries left
                                          └ fail, no retries: needs human review
```

| Module | File | What it does |
|---|---|---|
| Data | `data/market.py`, `data/news.py` | 9 markets (S&P 500, Nifty 50, USD/INR, EUR/USD, gold, US 2Y and 10Y yields, US Baa credit spread, VIX) on one business-day calendar; headlines and statements in one table |
| Regimes | `regimes.py` | Gaussian HMM on 4 features, refit yearly on an expanding window, forward-filtered probabilities |
| Detector | `detector.py` | z-score of each move against past volatility in the same regime; two baselines; precision, recall and McNemar's test |
| Retrieval | `retrieval.py` | Finds headlines that explain a flagged move; recall@k and MRR on labelled days |
| Generation | `generation/` | Pydantic fact sheet, LangGraph write and check loop, four writers (template, Ollama, Groq, fine-tuned model) |
| Fact-checker | `generation/factcheck.py` | Every number must belong to a market named in its sentence; checks up/down direction and regime name |
| Fine-tuning | `finetune/` | Distils a large model's fact-checked briefs into a small model with Unsloth LoRA |

## Design decisions

**No look-ahead anywhere.** hmmlearn's `predict` and `predict_proba` use the full sequence, so the label for March 2020 would use April 2020 data. `RegimeModel.filter_proba` runs the forward algorithm only. Every volatility estimate is shifted by one day. `tests/test_no_lookahead.py` changes future data and asserts that past outputs stay identical; one test also shows that the smoothed probabilities do change, which is why the filter exists.

**Regime labels are stable across refits.** HMM states come out in random order. They are sorted by the mean of 20-day realised volatility, so state 0 is always the calmest.

**Python computes, the LLM writes.** The model receives pre-formatted strings (`-3.04%`, `+12.5bp`) and is told not to calculate. This turns number errors into copying errors, which a checker can catch.

**The fact-checker is tested, not trusted.** `generation/checker_eval.py` plants errors in briefs that passed (changed numbers, flipped directions, wrong regime) and measures how many it catches.

**Unrelated news is dropped.** A headline below `retrieval.min_score` is not shown to the writer. Otherwise the brief would attach a cause to a move from whatever headline ranked first.

**Fine-tuning data is split by time.** The student model trains on days before the split date and is evaluated only on days after it.

## Quick start (synthetic data, no API keys)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

python -m marketpulse.cli all --demo     # about 30 seconds
pytest                                   # 16 tests
streamlit run app/streamlit_app.py
```

Reports go to `reports/demo/`. The demo data is synthetic: it checks that the code works and nothing more.

## Running on real data

1. Get a free FRED API key at fred.stlouisfed.org, copy `.env.example` to `.env`, and add the key.
2. Optional: save central bank statements as `data/raw/statements/YYYY-MM-DD_fed.txt` (or `_rbi`, `_ecb`), copied from the official sites.
3. Run the steps:

```bash
python -m marketpulse.cli data --no-news        # prices from Sep 2007 (Nifty history starts then), about 2 minutes
python -m marketpulse.cli news                  # GDELT headlines, last year, about 25 minutes, resumable
python -m marketpulse.cli regimes --select      # BIC for 2 to 5 states on the first 60% of data
python -m marketpulse.cli detect
python -m marketpulse.cli label-sheet           # writes data/labels/to_label_real.csv
```

4. Label the sheet (see below), then:

```bash
python -m marketpulse.cli brief --date 2024-08-05 --backend ollama
python -m marketpulse.cli evaluate --backend ollama --n-briefs 30
```

For `--backend ollama`, install Ollama and pull the model named in `configs/config.yaml`. For `groq`, add `GROQ_API_KEY` to `.env`. Model names change often; check what is current before you run.

## Labelling

Real markets have no answer key, so you build one. This is the most valuable part of the project and the part a copied repo cannot have.

**Detector labels.** `label-sheet` writes every row any detector flagged in the period you have news for (`news.years`, 1 year by default), plus 150 random unflagged rows, shuffled so you cannot tell which is which. For each row, read that day's news and set `notable` to 1 if a desk analyst would have mentioned the move, else 0. Save as `data/labels/notable_moves.csv`. Write down your rule for "notable" before you start, and keep it fixed.

**Retrieval labels.** For about 100 notable moves, find the headline that explains it and save `data/labels/retrieval_labels.csv` with columns `date, instrument, relevant_url`.

## Results

### Sanity check on synthetic data

These numbers show the code recovers structure that was planted on purpose. They are not findings about markets.

| Check | Result |
|---|---|
| Out-of-sample regime accuracy (3 states, majority class is 50.3%) | 75.0% |
| Detector F1: regime / rolling / fixed | 0.202 / 0.134 / 0.027 |
| McNemar, regime vs rolling detector | regime right where rolling wrong on 351 cells, opposite on 109, p < 0.001 |
| Fact-checker catch rate: numbers / direction / regime | 98.9% / 100% / 100% (90 planted errors each) |

The regime detector wins mainly by raising fewer false alarms (375 flags against 621), while catching 2 fewer planted shocks (43 against 45). Adding the 10s2s curve slope as a feature dropped regime accuracy from 75.0% to 45.0%, below the 50.3% majority baseline: a slow-moving level lets the HMM split history by era instead of by stress.

### Real data

[Fill in after running on real data and labelling. Do not copy numbers from the synthetic run.]

| Check | Result |
|---|---|
| Number of regimes chosen by BIC | |
| Detector precision / recall: regime vs rolling vs fixed | |
| McNemar p-value, regime vs rolling | |
| Retrieval recall@3: TF-IDF vs embeddings | |
| Brief first-try pass rate: prompted model vs fine-tuned model | |
| Mean human rating (docs/rubric.md) | |

## Limitations

- The fact-checker verifies numbers, direction words and the regime name. It cannot tell whether a stated cause is true. A brief can pass and still give a wrong reason for a move.
- In a sentence that names several markets, numbers can be swapped among those markets without being caught. That is where the missed planted number error came from.
- Detector labels come from one person's judgment of what is notable.
- GDELT coverage varies by source and period. Headlines only; no article bodies.
- A Gaussian HMM assumes normal returns within each regime. Real returns have fatter tails.
- The credit input is Moody's Baa spread (BAA10Y), not the ICE BofA high-yield spread: since April 2026 FRED serves only 3 years of ICE index history. Baa is investment grade, so it reacts less than high yield in stress.

## Project structure

```
configs/config.yaml          every setting: markets, features, thresholds, models
src/marketpulse/
  data/market.py             FRED and Yahoo Finance download, calendar alignment
  data/news.py               GDELT, RSS, local statements
  data/synthetic.py          regime-switching demo data with planted shocks
  features.py                regime features
  regimes.py                 HMM, forward filter, walk-forward, BIC
  detector.py                z-scores, baselines, evaluation, labelling sheet
  retrieval.py               TF-IDF and embedding retrieval, recall@k
  generation/facts.py        Pydantic fact sheet
  generation/factcheck.py    number, direction and regime checks
  generation/checker_eval.py planted-error test of the checker
  generation/writers.py      template, Ollama, Groq, Hugging Face writers
  generation/brief.py        LangGraph write and check loop
  pipeline.py, cli.py        steps and command line
app/streamlit_app.py         demo app
finetune/                    dataset builder and Unsloth LoRA training script
notebooks/                   exploration notebook
tests/                       leakage, fact-checker and model tests
docs/                        roadmap, experiment log, rating rubric
```
