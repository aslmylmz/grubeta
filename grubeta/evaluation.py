"""
Evaluation and diagnostic tools for dynamic beta models.

This module provides metrics, visualization, and validation tools
for assessing the quality of estimated dynamic betas.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class BetaEvaluator:
    """
    Comprehensive evaluation suite for dynamic beta estimates.

    Provides metrics for assessing:
    - Prediction accuracy (return prediction)
    - Beta stability and temporal consistency
    - CAPM model fit
    - Comparison with benchmark models

    Parameters
    ----------
    output_dir : str or Path, optional
        Directory for saving reports and plots.

    Examples
    --------
    >>> evaluator = BetaEvaluator(output_dir='./results')
    >>>
    >>> # Full evaluation
    >>> metrics = evaluator.evaluate(
    ...     betas=results['beta'],
    ...     stock_returns=results['stock_return'],
    ...     market_returns=results['market_return'],
    ...     dates=results['date']
    ... )
    >>>
    >>> print(metrics['systematic_r2'])
    >>> print(metrics['beta_stability'])
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        betas: np.ndarray,
        stock_returns: np.ndarray,
        market_returns: np.ndarray,
        alphas: Optional[np.ndarray] = None,
        dates: Optional[np.ndarray] = None,
        name: str = "evaluation",
    ) -> Dict:
        """
        Comprehensive evaluation of dynamic beta estimates.

        Parameters
        ----------
        betas : array-like
            Estimated dynamic betas.
        stock_returns : array-like
            Actual stock returns.
        market_returns : array-like
            Market returns.
        alphas : array-like, optional
            Estimated dynamic alphas.
        dates : array-like, optional
            Date index.
        name : str, default="evaluation"
            Name for reports/plots.

        Returns
        -------
        metrics : dict
            Dictionary containing all computed metrics:
            - systematic_r2: R² of systematic component
            - total_r2: R² including alpha
            - beta_mean: Average beta
            - beta_std: Beta volatility
            - beta_stability: Measure of temporal smoothness
            - mae: Mean absolute error of return prediction
            - rmse: Root mean squared error
        """
        # Convert to arrays and handle NaN
        betas = np.asarray(betas)
        stock_returns = np.asarray(stock_returns)
        market_returns = np.asarray(market_returns)

        # Create mask for valid observations
        mask = ~np.isnan(betas) & ~np.isnan(stock_returns) & ~np.isnan(market_returns)

        if mask.sum() < 10:
            warnings.warn("Too few valid observations for evaluation")
            return {}

        b = betas[mask]
        y = stock_returns[mask]
        m = market_returns[mask]

        metrics = {}

        # Systematic component analysis
        y_pred_systematic = b * m
        metrics["systematic_r2"] = r2_score(y, y_pred_systematic)
        metrics["systematic_mse"] = mean_squared_error(y, y_pred_systematic)
        metrics["systematic_mae"] = mean_absolute_error(y, y_pred_systematic)

        # Total prediction (if alphas provided)
        if alphas is not None:
            a = np.asarray(alphas)[mask]
            y_pred_total = a + b * m
            metrics["total_r2"] = r2_score(y, y_pred_total)
            metrics["total_mse"] = mean_squared_error(y, y_pred_total)
            metrics["total_mae"] = mean_absolute_error(y, y_pred_total)
            metrics["alpha_mean"] = np.mean(a)
            metrics["alpha_std"] = np.std(a)

        # Beta statistics
        metrics["beta_mean"] = np.mean(b)
        metrics["beta_std"] = np.std(b)
        metrics["beta_min"] = np.min(b)
        metrics["beta_max"] = np.max(b)
        metrics["beta_median"] = np.median(b)

        # Beta stability (lower = more stable)
        beta_changes = np.diff(b)
        metrics["beta_stability"] = np.std(beta_changes)
        metrics["beta_mean_abs_change"] = np.mean(np.abs(beta_changes))

        # Rolling window comparison
        metrics["rolling_252_corr"] = self._rolling_correlation(b, y, m, 252)

        # Save report if output_dir set
        if self.output_dir:
            self._save_report(metrics, name)
            if dates is not None:
                self._plot_beta(betas, dates, name)

        return metrics

    def compare_models(
        self,
        betas_dict: Dict[str, np.ndarray],
        stock_returns: np.ndarray,
        market_returns: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compare multiple beta estimation methods.

        Parameters
        ----------
        betas_dict : dict
            Dictionary mapping model names to beta arrays.
            Example: {'GRU': gru_betas, 'OLS': ols_betas, 'Kalman': kalman_betas}
        stock_returns : array-like
            Actual stock returns.
        market_returns : array-like
            Market returns.

        Returns
        -------
        comparison : pd.DataFrame
            DataFrame comparing metrics across models.
        """
        results = []

        for name, betas in betas_dict.items():
            metrics = self.evaluate(betas, stock_returns, market_returns, name=name)
            metrics["model"] = name
            results.append(metrics)

        df = pd.DataFrame(results)
        df = df.set_index("model")

        return df

    def _rolling_correlation(
        self,
        betas: np.ndarray,
        returns: np.ndarray,
        market: np.ndarray,
        window: int = 252,
    ) -> float:
        """Compute correlation between predicted and actual systematic returns."""
        if len(betas) < window:
            return np.nan

        y_pred = betas * market

        correlations = []
        for i in range(window, len(betas)):
            y_slice = returns[i - window : i]
            pred_slice = y_pred[i - window : i]
            corr = np.corrcoef(y_slice, pred_slice)[0, 1]
            correlations.append(corr)

        return float(np.nanmean(correlations))

    def _save_report(self, metrics: Dict, name: str) -> None:
        """Save evaluation report to text file."""
        if self.output_dir is None:
            return
        report_path = self.output_dir / f"{name}_report.txt"

        with open(report_path, "w") as f:
            f.write(f"Dynamic Beta Evaluation Report: {name}\n")
            f.write("=" * 50 + "\n\n")

            f.write("Prediction Accuracy\n")
            f.write("-" * 30 + "\n")
            f.write(f"Systematic R²: {metrics.get('systematic_r2', 'N/A'):.4f}\n")
            if "total_r2" in metrics:
                f.write(f"Total R²: {metrics['total_r2']:.4f}\n")
            f.write(f"MAE: {metrics.get('systematic_mae', 'N/A'):.6f}\n")
            f.write(f"RMSE: {np.sqrt(metrics.get('systematic_mse', 0)):.6f}\n\n")

            f.write("Beta Statistics\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mean: {metrics.get('beta_mean', 'N/A'):.4f}\n")
            f.write(f"Std Dev: {metrics.get('beta_std', 'N/A'):.4f}\n")
            f.write(f"Min: {metrics.get('beta_min', 'N/A'):.4f}\n")
            f.write(f"Max: {metrics.get('beta_max', 'N/A'):.4f}\n")
            f.write(f"Stability (Δ std): {metrics.get('beta_stability', 'N/A'):.6f}\n")

            if "alpha_mean" in metrics:
                f.write("\nAlpha Statistics\n")
                f.write("-" * 30 + "\n")
                f.write(f"Mean: {metrics['alpha_mean']:.6f}\n")
                f.write(f"Std Dev: {metrics['alpha_std']:.6f}\n")

    def _plot_beta(self, betas: np.ndarray, dates: np.ndarray, name: str) -> None:
        """Save beta trajectory plot."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        if self.output_dir is None:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        mask = ~np.isnan(betas)
        ax.plot(dates[mask], betas[mask], color="#2c3e50", linewidth=1.5)
        ax.axhline(1.0, color="#e74c3c", linestyle="--", alpha=0.5)

        ax.set_title(f"{name} - Dynamic Beta Evolution")
        ax.set_xlabel("Date")
        ax.set_ylabel("Beta")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{name}_beta_evolution.png", dpi=300)
        plt.close()


def compute_rolling_ols_beta(
    stock_returns: np.ndarray,
    market_returns: np.ndarray,
    window: int = 252,
) -> np.ndarray:
    """
    Compute rolling OLS beta as a benchmark.

    Parameters
    ----------
    stock_returns : array-like
        Stock return series.
    market_returns : array-like
        Market return series.
    window : int, default=252
        Rolling window size.

    Returns
    -------
    betas : np.ndarray
        Rolling beta estimates (NaN-padded for burn-in period).
    """
    n = len(stock_returns)
    betas = np.full(n, np.nan)

    for i in range(window, n):
        y = stock_returns[i - window : i]
        x = market_returns[i - window : i]

        x_dm = x - np.mean(x)
        cov: float = float(np.sum(x_dm * (y - np.mean(y))))
        var: float = float(np.sum(x_dm**2))

        if var > 1e-10:
            betas[i] = cov / var

    return cast(np.ndarray, betas)


def compute_ewma_beta(
    stock_returns: np.ndarray,
    market_returns: np.ndarray,
    halflife: int = 63,
) -> np.ndarray:
    """
    Compute exponentially weighted moving average beta.

    Parameters
    ----------
    stock_returns : array-like
        Stock return series.
    market_returns : array-like
        Market return series.
    halflife : int, default=63
        Halflife for exponential weighting (in periods).

    Returns
    -------
    betas : np.ndarray
        EWMA beta estimates.
    """
    stock_returns_s = pd.Series(stock_returns)
    market_returns_s = pd.Series(market_returns)

    cov = stock_returns_s.ewm(halflife=halflife).cov(market_returns_s)
    var = market_returns_s.ewm(halflife=halflife).var()

    betas = (cov / var).values
    betas = np.where(np.isinf(betas), np.nan, betas)

    return cast(np.ndarray, betas)


def compute_static_beta(
    stock_returns: np.ndarray,
    market_returns: np.ndarray,
) -> float:
    """
    Compute static (full-sample) OLS beta.

    Parameters
    ----------
    stock_returns : array-like
        Stock return series.
    market_returns : array-like
        Market return series.

    Returns
    -------
    beta : float
        Static beta estimate.
    """
    mask = ~np.isnan(stock_returns) & ~np.isnan(market_returns)
    y = stock_returns[mask]
    x = market_returns[mask]

    cov = np.cov(y, x)[0, 1]
    var = np.var(x)

    return cov / var if var > 1e-10 else 1.0


class DiagnosticTests:
    """
    Statistical diagnostic tests for beta estimates.

    Provides tests for lookahead bias detection, stationarity,
    autocorrelation, and heteroskedasticity.
    """

    @staticmethod
    def test_lookahead_bias(
        betas: np.ndarray,
        returns: np.ndarray,
        max_lag: int = 20,
        threshold: float = 0.15,
    ) -> Dict:
        """
        Test for lookahead bias by checking correlation with future returns.

        If beta at time t is correlated with returns at t+k, this suggests
        lookahead bias (information leakage from the future).

        Parameters
        ----------
        betas : array-like
            Beta estimates.
        returns : array-like
            Stock returns.
        max_lag : int, default=20
            Maximum forward lag to test.
        threshold : float, default=0.15
            Correlation threshold for suspicion.

        Returns
        -------
        results : dict
            Dictionary with correlations, suspicious_lags, and passed status.
        """
        mask = ~np.isnan(betas)
        b = betas[mask]
        r = returns[mask]

        correlations = {}
        suspicious = []

        for lag in [1, 5, 10, 20]:
            if lag >= max_lag or len(r) <= lag:
                continue

            future_ret = np.roll(r, -lag)[:-lag]
            beta_aligned = b[:-lag]

            corr = np.corrcoef(beta_aligned, future_ret)[0, 1]
            correlations[lag] = corr

            if abs(corr) > threshold:
                suspicious.append(lag)

        return {
            "correlations": correlations,
            "suspicious_lags": suspicious,
            "passed": len(suspicious) == 0,
        }

    @staticmethod
    def test_beta_stationarity(
        betas: np.ndarray,
        significance: float = 0.05,
    ) -> Dict:
        """
        Test beta series for stationarity using ADF test.

        Parameters
        ----------
        betas : array-like
            Beta estimates.
        significance : float, default=0.05
            Significance level.

        Returns
        -------
        results : dict
            ADF test results including statistic, p-value, and conclusion.
        """
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            return {"error": "statsmodels not installed"}

        mask = ~np.isnan(betas)
        b = betas[mask]

        if len(b) < 20:
            return {"error": "Insufficient data for ADF test"}

        result = adfuller(b, autolag="AIC")

        return {
            "adf_statistic": result[0],
            "p_value": result[1],
            "critical_values": result[4],
            "is_stationary": result[1] < significance,
        }

    @staticmethod
    def test_residual_autocorrelation(
        betas: np.ndarray,
        stock_returns: np.ndarray,
        market_returns: np.ndarray,
        lags: int = 10,
    ) -> Dict:
        """
        Test residuals for autocorrelation using Ljung-Box test.

        Parameters
        ----------
        betas : array-like
            Beta estimates.
        stock_returns : array-like
            Actual stock returns.
        market_returns : array-like
            Market returns.
        lags : int, default=10
            Number of lags to test.

        Returns
        -------
        results : dict
            Ljung-Box test results.
        """
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
        except ImportError:
            return {"error": "statsmodels not installed"}

        mask = ~np.isnan(betas) & ~np.isnan(stock_returns) & ~np.isnan(market_returns)
        b = betas[mask]
        y = stock_returns[mask]
        m = market_returns[mask]

        residuals = y - b * m

        if len(residuals) < lags + 1:
            return {"error": "Insufficient data"}

        lb_result = acorr_ljungbox(residuals, lags=lags, return_df=True)

        return {
            "lb_statistic": lb_result["lb_stat"].values,
            "p_values": lb_result["lb_pvalue"].values,
            "has_autocorrelation": (lb_result["lb_pvalue"] < 0.05).any(),
        }
