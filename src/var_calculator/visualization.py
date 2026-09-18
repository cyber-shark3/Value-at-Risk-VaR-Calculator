"""Plotting utilities for VaR analysis."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

def plot_distribution(portfolio_returns, confidence_level, show=True):
    portfolio_returns = np.asarray(portfolio_returns)
    alpha = 1 - confidence_level
    var_limit = np.percentile(portfolio_returns, alpha * 100)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(portfolio_returns, bins=60, edgecolor="black", alpha=0.6, density=True)
    ax.axvline(var_limit, linestyle="--",
               label=str(round(confidence_level * 100, 1)) + "% VaR threshold")
    ax.set_title("Portfolio Return Distribution (Tail: " + str(round(alpha * 100, 1)) + "%)")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
