import numpy as np
import pandas as pd

from marketpulse.detector import evaluate_flags


def test_labels_outside_flag_dates_do_not_crash():
    idx = pd.bdate_range("2020-01-01", periods=5)
    flags = pd.DataFrame(False, index=idx, columns=["SPX"])
    labels = pd.DataFrame({"date": [pd.Timestamp("2026-11-05")], "instrument": ["SPX"], "notable": [1]})
    ev = evaluate_flags({"regime": flags}, labels)
    assert ev.loc["regime", "flags"] == 0
    assert np.isclose(ev.loc["regime", "recall"], 0.0)
