import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

REGIME_LABELS = {0: "range_bound", 1: "trending", 2: "high_vol"}


class RegimeDetector:
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.model: GaussianHMM | None = None

    def fit(self, returns: pd.Series):
        X = returns.values.reshape(-1, 1)
        self.model = GaussianHMM(n_components=self.n_states, covariance_type="diag", n_iter=1000)
        self.model.fit(X)
        return self

    def current_regime(self, returns: pd.Series) -> str:
        if self.model is None:
            self.fit(returns)
        X = returns.values.reshape(-1, 1)
        hidden_states = self.model.predict(X)
        latest_state = hidden_states[-1]

        # map state index -> label by volatility ranking (low vol = range_bound, etc.)
        state_vols = {s: returns[hidden_states == s].std() for s in set(hidden_states)}
        ranked = sorted(state_vols, key=state_vols.get)
        label_map = {ranked[0]: "range_bound"}
        if len(ranked) > 1:
            label_map[ranked[-1]] = "high_vol"
        for s in ranked[1:-1]:
            label_map[s] = "trending"

        return label_map.get(latest_state, "unknown")
