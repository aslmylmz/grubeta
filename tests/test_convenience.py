"""Tests for the convenience API (grubeta.convenience)."""

import numpy as np
import pandas as pd
import pytest

from grubeta.convenience import estimate_beta, compare_betas, format_summary


class TestFormatSummary:
    """Test format_summary output."""

    def test_basic_summary(self):
        """format_summary returns a string with key metrics."""
        betas = np.array([np.nan] * 100 + [1.1, 1.15, 1.2, 1.18, 1.22])
        dates = pd.date_range("2020-01-01", periods=105, freq="B")
        summary = format_summary(betas, dates=dates, stock_name="TEST")
        assert "TEST" in summary
        assert "Current Beta" in summary
        assert "Average Beta" in summary
        assert "Beta Range" in summary
        assert "Stability" in summary

    def test_summary_with_r2(self):
        """format_summary includes R² when returns are provided."""
        n = 200
        np.random.seed(42)
        betas = np.concatenate([np.full(100, np.nan), np.ones(100) * 1.1])
        market = np.random.normal(0, 0.01, n)
        stock = 1.1 * market + np.random.normal(0, 0.005, n)
        summary = format_summary(betas, stock_returns=stock, market_returns=market)
        assert "Systematic" in summary

    def test_summary_interpretation_high_beta(self):
        """High beta shows 'amplifies' interpretation."""
        betas = np.array([1.2, 1.25, 1.3])
        summary = format_summary(betas, stock_name="GROWTH")
        assert "amplifies" in summary

    def test_summary_interpretation_low_beta(self):
        """Low beta shows 'dampens' interpretation."""
        betas = np.array([0.7, 0.75, 0.8])
        summary = format_summary(betas, stock_name="DEFENSIVE")
        assert "dampens" in summary

    def test_summary_interpretation_neutral_beta(self):
        """Neutral beta shows 'in line' interpretation."""
        betas = np.array([0.98, 1.0, 1.02])
        summary = format_summary(betas, stock_name="INDEX")
        assert "in line" in summary

    def test_empty_betas(self):
        """All-NaN betas returns a fallback message."""
        betas = np.array([np.nan, np.nan, np.nan])
        summary = format_summary(betas, stock_name="EMPTY")
        assert "No valid beta" in summary


class TestEstimateBeta:
    """Test estimate_beta with pd.Series input (no yfinance needed)."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic stock and market returns."""
        np.random.seed(42)
        n = 700  # Enough for default preset (500 + 60 lookback)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        market = pd.Series(np.random.normal(0.0005, 0.01, n), index=dates, name="MKT")
        stock = pd.Series(1.2 * market + np.random.normal(0, 0.005, n), index=dates, name="TEST")
        return stock, market

    def test_returns_dict_with_expected_keys(self, synthetic_data):
        """estimate_beta returns dict with all expected keys."""
        stock, market = synthetic_data
        result = estimate_beta(stock, market, plot=False, verbose=False)
        assert isinstance(result, dict)
        for key in ["beta", "alpha", "dates", "summary", "model", "results", "fig"]:
            assert key in result, f"Missing key: {key}"

    def test_summary_is_string(self, synthetic_data):
        """Summary is a non-empty string."""
        stock, market = synthetic_data
        result = estimate_beta(stock, market, plot=False, verbose=False)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 50

    def test_beta_series_length(self, synthetic_data):
        """Beta series has same length as input data."""
        stock, market = synthetic_data
        result = estimate_beta(stock, market, plot=False, verbose=False)
        assert len(result["beta"]) == len(stock)

    def test_fig_none_when_plot_false(self, synthetic_data):
        """Fig is None when plot=False."""
        stock, market = synthetic_data
        result = estimate_beta(stock, market, plot=False, verbose=False)
        assert result["fig"] is None

    def test_preset_responsive(self, synthetic_data):
        """Responsive preset works (shorter lookback)."""
        stock, market = synthetic_data
        result = estimate_beta(stock, market, preset="responsive", plot=False, verbose=False)
        assert result["model"].config.lookback == 30

    def test_invalid_preset_raises(self, synthetic_data):
        """Invalid preset raises ValueError."""
        stock, market = synthetic_data
        with pytest.raises(ValueError, match="Unknown preset"):
            estimate_beta(stock, market, preset="nonexistent", plot=False, verbose=False)

    def test_insufficient_data_raises(self):
        """Too-short data raises InsufficientDataError."""
        from grubeta.exceptions import InsufficientDataError

        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        stock = pd.Series(np.random.normal(0, 0.01, 50), index=dates)
        market = pd.Series(np.random.normal(0, 0.01, 50), index=dates)
        with pytest.raises(InsufficientDataError):
            estimate_beta(stock, market, plot=False, verbose=False)

    def test_string_ticker_without_yfinance(self):
        """String ticker without yfinance raises ImportError with helpful message."""
        try:
            import yfinance
            pytest.skip("yfinance is installed")
        except ImportError:
            with pytest.raises(ImportError, match="grubeta\\[data\\]"):
                estimate_beta("AAPL", "SPY", plot=False)


class TestPresets:
    """Test preset configurations."""

    def test_list_presets(self):
        from grubeta.presets import list_presets

        presets = list_presets()
        assert "default" in presets
        assert "responsive" in presets
        assert "smooth" in presets
        assert "research" in presets

    def test_get_preset_returns_copy(self):
        from grubeta.presets import get_preset

        p1 = get_preset("default")
        p2 = get_preset("default")
        assert p1 is not p2  # Different objects
        assert p1.lookback == p2.lookback  # Same values

    def test_preset_values(self):
        from grubeta.presets import get_preset

        responsive = get_preset("responsive")
        assert responsive.lookback == 30
        assert responsive.wf_step_size == 21

        smooth = get_preset("smooth")
        assert smooth.lookback == 120
        assert smooth.lambda_beta == 0.15

    def test_invalid_preset(self):
        from grubeta.presets import get_preset

        with pytest.raises(ValueError):
            get_preset("nonexistent")


class TestExceptions:
    """Test custom exception classes."""

    def test_insufficient_data_error_message(self):
        from grubeta.exceptions import InsufficientDataError

        err = InsufficientDataError(100, 560, ticker="AAPL")
        msg = str(err)
        assert "AAPL" in msg
        assert "100" in msg
        assert "560" in msg
        assert "years" in msg

    def test_data_fetch_error_message(self):
        from grubeta.exceptions import DataFetchError

        err = DataFetchError("INVALID", reason="Ticker not found")
        msg = str(err)
        assert "INVALID" in msg
        assert "Ticker not found" in msg
        assert "Common fixes" in msg

    def test_missing_dependency_error(self):
        from grubeta.exceptions import MissingDependencyError

        err = MissingDependencyError(
            "yfinance", "pip install grubeta[data]", "to fetch ticker data"
        )
        msg = str(err)
        assert "yfinance" in msg
        assert "pip install" in msg
