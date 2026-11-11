# Roadmap

Ten weeks at roughly 15 hours a week. Each week ends with something committed to Git and one entry in `experiment_log.md`.

## Week 1: run it and read it
- Run the demo, the tests and the app.
- Read every file in `src/`. For each function, write one line in your own words on what it does and why. If you cannot, that is the next thing to study.
- Get a FRED key and run `cli data` on real data.

## Week 2: data and EDA
- Notebook `01_explore`: plot each market, daily change distributions, missing days per market, correlation in calm vs stressed periods.
- Check the calendar alignment: how many days were forward-filled for Nifty?
- Decide whether to keep VIX as a brief market or only as a regime feature. Write down why.

## Week 3: regimes
- `cli regimes --select`. Pick the number of states with BIC and with your own reading of the state profiles.
- Check that 2008, March 2020 and 2022 land in stressed states out of sample. Note how many days the filter lags.
- Experiment: add `curve_10s2s`, then try a 20-day change in the curve instead of its level. Log both.

## Week 4: detector
- `cli detect`, then `cli label-sheet`. Label the sheet. This takes several hours; it is the core of the project.
- `cli evaluate`. Is the regime detector better than the rolling baseline on your labels? Report the McNemar result even if it is not significant.
- Tune `z_threshold` on the first half of your labelled period only, then report on the second half.

## Week 5: news
- Collect GDELT headlines and a set of Fed and RBI statements.
- Label about 100 retrieval rows.
- Compare TF-IDF with embeddings. Tune `min_score` on half the labels.

## Weeks 6 and 7: LLM briefs
- Run the Ollama or Groq writer on 30 days. Record first-try pass rate and the most common failure types.
- Change the system prompt one rule at a time and measure again. Keep the prompt under version control.
- Rate 30 briefs with `rubric.md`. Rate them before looking at the fact-check result.

## Week 8: fine-tuning
- `finetune/build_sft_dataset.py` with a large teacher model on days before your split date.
- Train with `finetune/train_lora.py` on Colab or Kaggle.
- Compare the student with the prompted teacher on test dates: pass rate, number accuracy, rating, speed.

## Week 9: app and write-up
- Finish the Streamlit app. Deploy on Streamlit Community Cloud if you want a public link.
- Fill the real-data results table in the README.

## Week 10: interview preparation
- Practise a two-minute explanation: problem, method, one result, one failure.
- Be ready to explain the forward algorithm, why smoothing leaks future data, BIC, McNemar's test, LoRA and why you split by time.
