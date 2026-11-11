"""Market regime detection with a Gaussian Hidden Markov Model.

Two things here matter more than the model itself:

1. No look-ahead. hmmlearn's predict() and predict_proba() use the whole sequence
   (Viterbi / forward-backward), so the label for 2020-03-01 would use data from
   April 2020. That is fine for a description of history and wrong for anything you
   call a forecast or a signal. `filter_proba` runs the forward pass only, so the
   probability at day t uses data up to day t.

2. Stable labels. HMM states come out in random order on every fit. We sort them by
   the mean of a volatility feature, so state 0 is always the calmest.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

REGIME_NAMES = {2: ["calm", "stressed"], 3: ["calm", "normal", "stressed"],
                4: ["calm", "normal", "stressed", "crisis"]}


def regime_names(k: int) -> list[str]:
    return REGIME_NAMES.get(k, [f"regime_{i}" for i in range(k)])


class RegimeModel:
    def __init__(self, n_states: int = 3, order_by: str | None = "spx_vol_20d",
                 n_init: int = 5, random_state: int = 42):
        self.k = n_states
        self.order_by = order_by
        self.n_init = n_init
        self.random_state = random_state

    # ---------- fitting ----------
    def fit(self, features: pd.DataFrame) -> "RegimeModel":
        from hmmlearn.hmm import GaussianHMM

        self.columns_ = list(features.columns)
        self.mean_ = features.mean()
        self.std_ = features.std().replace(0, 1.0)
        X = self._scale(features)

        best, best_ll = None, -np.inf
        for i in range(self.n_init):
            hmm = GaussianHMM(n_components=self.k, covariance_type="full", n_iter=300,
                              tol=1e-4, random_state=self.random_state + i, min_covar=1e-4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hmm.fit(X)
            ll = hmm.score(X)
            if ll > best_ll:
                best, best_ll = hmm, ll
        self.hmm_ = best
        self.loglik_ = best_ll
        self._set_order()
        return self

    def _scale(self, features: pd.DataFrame) -> np.ndarray:
        return ((features[self.columns_] - self.mean_) / self.std_).to_numpy()

    def _set_order(self) -> None:
        col = self.order_by if self.order_by in self.columns_ else self.columns_[0]
        j = self.columns_.index(col)
        # perm[new_label] = old_state
        self.perm_ = np.argsort(self.hmm_.means_[:, j])

    # ---------- inference ----------
    def filter_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """P(regime at t | data up to t), forward algorithm in log space."""
        X = self._scale(features)
        hmm = self.hmm_
        log_b = np.column_stack([
            multivariate_normal.logpdf(X, mean=hmm.means_[j], cov=hmm.covars_[j], allow_singular=True)
            for j in range(self.k)
        ])
        log_a = np.log(hmm.transmat_ + 1e-300)
        log_alpha = np.empty_like(log_b)
        prev = np.log(hmm.startprob_ + 1e-300) + log_b[0]
        log_alpha[0] = prev - logsumexp(prev)
        for t in range(1, len(X)):
            prev = logsumexp(log_alpha[t - 1][:, None] + log_a, axis=0) + log_b[t]
            log_alpha[t] = prev - logsumexp(prev)
        proba = np.exp(log_alpha)[:, self.perm_]
        return pd.DataFrame(proba, index=features.index, columns=[f"p_{i}" for i in range(self.k)])

    def smoothed_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Forward-backward probabilities. Uses future data: for describing history only."""
        proba = self.hmm_.predict_proba(self._scale(features))[:, self.perm_]
        return pd.DataFrame(proba, index=features.index, columns=[f"p_{i}" for i in range(self.k)])

    # ---------- description ----------
    def n_params(self) -> int:
        d, k = len(self.columns_), self.k
        return (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2

    def bic(self, features: pd.DataFrame) -> float:
        ll = self.hmm_.score(self._scale(features))
        return -2 * ll + self.n_params() * np.log(len(features))

    def describe(self) -> pd.DataFrame:
        """State means in original units, plus expected duration in days."""
        means = self.hmm_.means_[self.perm_] * self.std_.to_numpy() + self.mean_.to_numpy()
        df = pd.DataFrame(means, columns=self.columns_, index=regime_names(self.k))
        stay = np.diag(self.hmm_.transmat_)[self.perm_]
        df["expected_duration_days"] = 1.0 / (1.0 - stay)
        return df

    def transition_matrix(self) -> pd.DataFrame:
        names = regime_names(self.k)
        tm = self.hmm_.transmat_[np.ix_(self.perm_, self.perm_)]
        return pd.DataFrame(tm, index=names, columns=names)


def walk_forward(features: pd.DataFrame, n_states: int, min_train_days: int,
                 refit_every_days: int, random_state: int = 42) -> pd.DataFrame:
    """Out-of-sample regime probabilities.

    Fit on days [0, start), filter forward through [start, start + refit_every),
    move start forward, refit on the longer history, repeat.
    Returns columns p_0..p_{k-1} and regime (argmax) for every out-of-sample day.
    """
    if len(features) <= min_train_days:
        raise ValueError(f"Need more than {min_train_days} rows, got {len(features)}")
    blocks = []
    for start in range(min_train_days, len(features), refit_every_days):
        end = min(start + refit_every_days, len(features))
        model = RegimeModel(n_states, random_state=random_state).fit(features.iloc[:start])
        proba = model.filter_proba(features.iloc[:end]).iloc[start:end]
        blocks.append(proba)
    out = pd.concat(blocks)
    out["regime"] = out.to_numpy().argmax(axis=1)
    return out


def select_n_states(features: pd.DataFrame, candidates=(2, 3, 4, 5), random_state: int = 42) -> pd.DataFrame:
    """Compare state counts by BIC. Pass training data only, not the full sample."""
    rows = []
    for k in candidates:
        m = RegimeModel(k, random_state=random_state).fit(features)
        rows.append({"n_states": k, "loglik": m.hmm_.score(m._scale(features)),
                     "n_params": m.n_params(), "bic": m.bic(features)})
    return pd.DataFrame(rows).set_index("n_states")
