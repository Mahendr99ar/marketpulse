# Experiment log

One entry per experiment, including the ones that did not work. Interviewers ask about failures; this file is where the answers come from.

## Template

```
### YYYY-MM-DD: short title
Question:   what I wanted to find out
Change:     the one thing I changed (config value, feature, prompt, model)
Data:       real or demo, date range, which labels
Result:     numbers, with the baseline next to them
Conclusion: keep / revert / unclear, and why
Next:       what this suggests trying
```

## Entries

### Example (synthetic data): curve slope as a regime feature
Question:   Does the 10s2s slope help the HMM find stress regimes?
Change:     added `curve_10s2s` to `regime.features`
Data:       demo, 3,000 days
Result:     out-of-sample regime accuracy 75.0% without it, 45.0% with it (majority baseline 50.3%)
Conclusion: revert. The slope is a slow-moving level, so the HMM split the sample by period instead of by stress.
Next:       try the 20-day change in the slope instead of its level, on real data
