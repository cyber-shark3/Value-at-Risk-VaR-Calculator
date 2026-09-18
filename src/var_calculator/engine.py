"""Core VaR and CVaR calculations."""
from __future__ import annotations
import numpy as np
from scipy import stats

class VaREngine:
    """Calculate portfolio VaR using historical, parametric and Monte Carlo methods."""
    def __init__(self, returns, weights, investment=1.0, confidence=0.95, horizon=1):
        self.returns = np.asarray(returns, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.investment = float(investment)
        self.confidence = float(confidence)
        self.horizon = int(horizon)
        if self.returns.ndim != 2:
            raise ValueError("returns must be a 2D array of observations x assets")
        if self.returns.shape[1] != len(self.weights):
            raise ValueError("Number of weights must match number of assets")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1 day")
        if self.investment < 0:
            raise ValueError("investment cannot be negative")
        self.port_returns = self.returns @ self.weights
        self.mean_ret = float(np.mean(self.port_returns))
        self.std_ret = float(np.std(self.port_returns, ddof=1))
        self.cov_matrix = np.cov(self.returns, rowvar=False, ddof=1)

    def _scale(self, daily_var_return):
        return float(daily_var_return * self.investment * np.sqrt(self.horizon))

    def historical(self):
        alpha = 1 - self.confidence
        return self._scale(-float(np.percentile(self.port_returns, alpha * 100)))

    def parametric(self, dist="normal"):
        alpha = 1 - self.confidence
        if dist == "normal":
            z = stats.norm.ppf(alpha)
        elif dist == "t":
            z = stats.t.ppf(alpha, df=5)
        else:
            raise ValueError("dist must be 'normal' or 't'")
        return self._scale(-(self.mean_ret + z * self.std_ret))

    def monte_carlo(self, n_sims=10000, dist="normal", seed=None):
        if n_sims < 1:
            raise ValueError("n_sims must be positive")
        rng = np.random.default_rng(seed)
        mean = np.mean(self.returns, axis=0)
        if dist == "normal":
            sims = rng.multivariate_normal(mean, self.cov_matrix, n_sims)
        elif dist == "t":
            chol = np.linalg.cholesky(self.cov_matrix + np.eye(len(mean)) * 1e-12)
            z = rng.standard_t(df=5, size=(n_sims, len(mean)))
            z /= np.sqrt(5 / rng.chisquare(5, size=(n_sims, 1)))
            sims = z @ chol.T + mean
        else:
            raise ValueError("dist must be 'normal' or 't'")
        sim_port = sims @ self.weights
        return self._scale(-float(np.percentile(sim_port, (1 - self.confidence) * 100)))

    def cvar_historical(self):
        alpha = 1 - self.confidence
        cutoff = np.percentile(self.port_returns, alpha * 100)
        tail = self.port_returns[self.port_returns <= cutoff]
        return self._scale(-float(np.mean(tail))) if len(tail) else self.historical()
