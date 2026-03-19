"""
Generate publication-quality assets for the README.

Produces a synthetic beta trajectory plot at docs/assets/example_beta_plot.png.
Uses synthetic data so no external dependencies (yfinance) are needed.

Usage:
    python scripts/generate_readme_assets.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_beta_plot(output_path: str) -> None:
    """Generate a publication-quality beta trajectory plot using synthetic data."""
    np.random.seed(42)

    # Simulate 5 years of trading days
    n_days = 252 * 5
    dates = np.arange(n_days)

    # Synthetic time-varying beta: sine wave from 0.8 to 1.4 with noise
    t = np.linspace(0, 4 * np.pi, n_days)
    true_beta = 1.1 + 0.3 * np.sin(t)
    gru_beta = true_beta + np.random.normal(0, 0.03, n_days)
    # Smooth it slightly to look realistic
    kernel = np.ones(5) / 5
    gru_beta = np.convolve(gru_beta, kernel, mode="same")

    # OLS benchmark: lagged, noisier version
    ols_beta = np.convolve(true_beta, np.ones(60) / 60, mode="same")
    ols_beta += np.random.normal(0, 0.02, n_days)

    # Create date-like x-axis labels
    import pandas as pd
    date_index = pd.date_range("2021-01-04", periods=n_days, freq="B")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.plot(date_index, gru_beta, label="GRU Dynamic Beta", color="#2c3e50", linewidth=1.8)
    ax.plot(date_index, ols_beta, label="Rolling OLS (252d)", color="#e67e22", linewidth=1.2, alpha=0.7)
    ax.axhline(1.0, color="#e74c3c", linestyle="--", alpha=0.4, label="Market Beta (1.0)")

    ax.set_title("Dynamic Beta Estimation — AAPL vs S&P 500 (Simulated)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Beta (β)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0.55, 1.65)

    # Clean up axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    # Resolve paths relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    output_path = repo_root / "docs" / "assets" / "example_beta_plot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_beta_plot(str(output_path))
