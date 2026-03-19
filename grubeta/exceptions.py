"""
Custom exceptions for grubeta.

Provides finance-friendly error messages that help non-technical users
diagnose and fix common issues.
"""


class GrubetaError(Exception):
    """Base exception for grubeta."""


class InsufficientDataError(GrubetaError):
    """Not enough historical data for reliable estimation."""

    def __init__(self, n_samples, min_required, ticker=None):
        ticker_str = f" for '{ticker}'" if ticker else ""
        years_needed = min_required / 252
        years_have = n_samples / 252
        super().__init__(
            f"Not enough data{ticker_str}: got {n_samples} trading days "
            f"(~{years_have:.1f} years), need at least {min_required} "
            f"(~{years_needed:.1f} years).\n"
            f"Try an earlier start date or use a stock with longer trading history."
        )


class DataFetchError(GrubetaError):
    """Could not fetch data for the given ticker."""

    def __init__(self, ticker, reason=""):
        msg = f"Could not fetch data for '{ticker}'."
        if reason:
            msg += f" {reason}"
        msg += (
            "\n\nCommon fixes:\n"
            "  - Check the ticker symbol (e.g., 'AAPL' not 'Apple', 'BAS.DE' for Frankfurt)\n"
            "  - Ensure you have internet access\n"
            "  - Try: pip install --upgrade yfinance"
        )
        super().__init__(msg)


class MissingDependencyError(GrubetaError):
    """A required optional dependency is not installed."""

    def __init__(self, package, install_cmd, purpose):
        super().__init__(
            f"'{package}' is required {purpose} but is not installed.\n"
            f"Install it with: {install_cmd}"
        )
