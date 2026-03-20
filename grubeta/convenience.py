"""
High-level convenience API for grubeta.

These functions are the "front door" for non-technical users who want
to estimate dynamic beta without understanding ML concepts.

Usage:
    >>> from grubeta import estimate_beta
    >>> result = estimate_beta("AAPL", "SPY")
    >>> print(result["summary"])
"""

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def _fetch_returns(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch daily returns for a ticker via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "Install yfinance to use ticker symbols: pip install grubeta[data]\n"
            "Or pass return data directly as pd.Series."
        )

    from grubeta.exceptions import DataFetchError

    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as e:
        raise DataFetchError(ticker, reason=str(e))

    if data is None or data.empty:
        raise DataFetchError(ticker, reason="No data returned for the given date range.")

    # Handle MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
    else:
        close = data["Close"]

    returns = close.pct_change().dropna()
    if len(returns) == 0:
        raise DataFetchError(ticker, reason="No valid return data after processing.")

    returns.name = ticker
    return returns


def _resolve_returns(
    data: Union[str, pd.Series],
    start: Optional[str],
    end: Optional[str],
    label: str,
    verbose: bool,
) -> pd.Series:
    """Resolve ticker string or pd.Series into a returns series."""
    if isinstance(data, str):
        if verbose:
            date_range = f" ({start} to {end})" if start else ""
            print(f"Fetching {data} data{date_range}...")
        return _fetch_returns(data, start, end)
    elif isinstance(data, pd.Series):
        return data
    else:
        raise TypeError(
            f"'{label}' must be a ticker string (e.g., 'AAPL') or a pd.Series of returns, "
            f"got {type(data).__name__}."
        )


def _align_returns(
    stock_returns: pd.Series, market_returns: pd.Series
) -> tuple:
    """Align two return series on their common date index."""
    common = stock_returns.index.intersection(market_returns.index)
    if len(common) == 0:
        raise ValueError("Stock and market return series have no overlapping dates.")
    stock_returns = stock_returns.loc[common].sort_index()
    market_returns = market_returns.loc[common].sort_index()
    return stock_returns, market_returns


def format_summary(
    beta_series: Union[pd.Series, np.ndarray],
    dates: Optional[pd.DatetimeIndex] = None,
    stock_returns: Optional[Union[pd.Series, np.ndarray]] = None,
    market_returns: Optional[Union[pd.Series, np.ndarray]] = None,
    stock_name: str = "Stock",
    market_name: str = "Market",
) -> str:
    """
    Format a human-readable summary of dynamic beta results.

    Parameters
    ----------
    beta_series : pd.Series or np.ndarray
        Dynamic beta estimates (may contain NaN for burn-in).
    dates : pd.DatetimeIndex, optional
        Date index corresponding to beta_series.
    stock_returns : pd.Series or np.ndarray, optional
        Stock returns for R² calculation.
    market_returns : pd.Series or np.ndarray, optional
        Market returns for R² calculation.
    stock_name : str
        Ticker or label for the stock.
    market_name : str
        Ticker or label for the market.

    Returns
    -------
    str
        Human-readable summary string.
    """
    betas = np.asarray(beta_series, dtype=float)
    valid = betas[~np.isnan(betas)]

    if len(valid) == 0:
        return f"{stock_name}: No valid beta estimates available."

    current_beta = valid[-1]
    avg_beta = float(np.mean(valid))
    min_beta = float(np.min(valid))
    max_beta = float(np.max(valid))
    stability = float(np.std(np.diff(valid))) if len(valid) > 1 else 0.0

    # Date range
    if dates is not None:
        dates_arr = pd.DatetimeIndex(dates)
        valid_dates = dates_arr[~np.isnan(betas)]
        if len(valid_dates) > 0:
            date_str = f" ({valid_dates[0].strftime('%Y-%m-%d')} to {valid_dates[-1].strftime('%Y-%m-%d')})"
        else:
            date_str = ""
    else:
        date_str = ""

    # R² calculation
    r2_line = ""
    if stock_returns is not None and market_returns is not None:
        sr = np.asarray(stock_returns, dtype=float)
        mr = np.asarray(market_returns, dtype=float)
        mask = ~np.isnan(betas) & ~np.isnan(sr) & ~np.isnan(mr)
        if mask.sum() > 0:
            pred = betas[mask] * mr[mask]
            ss_res = np.sum((sr[mask] - pred) ** 2)
            ss_tot = np.sum((sr[mask] - np.mean(sr[mask])) ** 2)
            if ss_tot > 0:
                r2 = 1 - ss_res / ss_tot
                r2_line = f"\n  Systematic R²:    {r2:.2f}"

    # Interpretation
    if current_beta > 1.05:
        pct = (current_beta - 1) * 100
        interp = (
            f"{stock_name} currently amplifies {market_name} moves by ~{pct:.0f}%.\n"
            f"  A 1% market drop implies a ~{current_beta:.2f}% drop in {stock_name}."
        )
    elif current_beta < 0.95:
        pct = (1 - current_beta) * 100
        interp = (
            f"{stock_name} currently dampens {market_name} moves by ~{pct:.0f}%.\n"
            f"  A 1% market drop implies a ~{current_beta:.2f}% drop in {stock_name}."
        )
    else:
        interp = f"{stock_name} currently moves roughly in line with {market_name}."

    header = f"{stock_name} Dynamic Beta Summary{date_str}"
    separator = "─" * len(header)

    summary = (
        f"{header}\n"
        f"{separator}\n"
        f"  Current Beta:     {current_beta:.2f}\n"
        f"  Average Beta:     {avg_beta:.2f}\n"
        f"  Beta Range:       {min_beta:.2f} → {max_beta:.2f}\n"
        f"  Stability:        {stability:.3f} (daily change std)"
        f"{r2_line}\n"
        f"\n"
        f"  Interpretation: {interp}"
    )

    return summary


def estimate_beta(
    stock: Union[str, pd.Series],
    market: Union[str, pd.Series] = "SPY",
    start: Optional[str] = None,
    end: Optional[str] = None,
    preset: str = "default",
    plot: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Estimate time-varying beta for a stock relative to a market index.

    This is the simplest way to use grubeta. Pass ticker symbols or return
    series, and get back a complete analysis with summary and plot.

    Parameters
    ----------
    stock : str or pd.Series
        Stock ticker (e.g., "AAPL") or daily return series.
    market : str or pd.Series
        Market ticker (e.g., "SPY") or daily return series.
    start : str, optional
        Start date as "YYYY-MM-DD". Default: 10 years ago.
    end : str, optional
        End date as "YYYY-MM-DD". Default: today.
    preset : str
        Configuration preset: "default", "responsive", "smooth", "research".
    plot : bool
        Whether to display a plot of the beta trajectory.
    verbose : bool
        Whether to print progress updates.

    Returns
    -------
    dict
        Keys: 'beta', 'alpha', 'dates', 'summary', 'model', 'results', 'fig'
        - beta: pd.Series of dynamic beta estimates
        - alpha: pd.Series of dynamic alpha estimates
        - dates: DatetimeIndex
        - summary: human-readable summary string
        - model: fitted DynamicBeta object
        - results: full DataFrame from fit_predict
        - fig: matplotlib Figure (if plot=True, else None)

    Examples
    --------
    >>> from grubeta import estimate_beta
    >>> result = estimate_beta("AAPL", "SPY")
    >>> print(result["summary"])
    >>> result["beta"].tail()
    """
    from grubeta.core import DynamicBeta
    from grubeta.exceptions import InsufficientDataError
    from grubeta.presets import get_preset

    # Defaults
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.now() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")

    # Get config
    config = get_preset(preset)

    # Resolve stock/market names for display
    stock_name = stock if isinstance(stock, str) else getattr(stock, "name", "Stock")
    market_name = market if isinstance(market, str) else getattr(market, "name", "Market")

    # Fetch data
    stock_returns = _resolve_returns(stock, start, end, "stock", verbose)
    market_returns = _resolve_returns(market, start, end, "market", verbose)

    # Align
    stock_returns, market_returns = _align_returns(stock_returns, market_returns)

    # Check data sufficiency
    min_required = config.initial_train_size + config.lookback
    if len(stock_returns) < min_required:
        raise InsufficientDataError(
            n_samples=len(stock_returns),
            min_required=min_required,
            ticker=stock_name,
        )

    if verbose:
        print(f"Preparing features ({len(stock_returns)} trading days)...")

    if preset == "research" and verbose:
        print(
            "Note: The 'research' preset uses enhanced model capacity (128 GRU units, "
            "longer training). For full OHLCV feature engineering, use DynamicBeta with "
            "DataPreprocessor directly."
        )

    # All presets use simple mode through convenience API
    from grubeta.preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor()
    features = preprocessor.prepare_simple(
        stock_returns=stock_returns,
        market_returns=market_returns,
        dates=stock_returns.index,
    )

    if verbose:
        print(f"Training model with walk-forward validation (preset='{preset}')...")

    # Fit model
    model = DynamicBeta(config=config)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = model.fit_predict(**features)

    # Extract series
    beta_series = results["beta"]
    alpha_series = results["alpha"]
    dates = results["date"] if "date" in results.columns else results.index

    # Build summary
    summary = format_summary(
        beta_series=beta_series.values,
        dates=dates,
        stock_returns=results["stock_return"].values,
        market_returns=results["market_return"].values,
        stock_name=stock_name,
        market_name=market_name,
    )

    if verbose:
        valid_betas = beta_series.dropna()
        if len(valid_betas) > 0:
            print(
                f"Done! {stock_name} beta ranges from {valid_betas.min():.2f} to "
                f"{valid_betas.max():.2f} (current: {valid_betas.iloc[-1]:.2f})"
            )

    # Plot
    fig = None
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        valid_mask = ~beta_series.isna()
        ax.plot(
            dates[valid_mask],
            beta_series[valid_mask],
            label=f"{stock_name} Dynamic Beta",
            color="#2c3e50",
            linewidth=1.5,
        )
        ax.axhline(1.0, color="#e74c3c", linestyle="--", alpha=0.5, label="Market Beta (1.0)")
        ax.set_title(f"{stock_name} Dynamic Beta Over Time")
        ax.set_ylabel("Beta (β)")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {
        "beta": beta_series,
        "alpha": alpha_series,
        "dates": dates,
        "summary": summary,
        "model": model,
        "results": results,
        "fig": fig,
    }


def compare_betas(
    stocks: list,
    market: Union[str, pd.Series] = "SPY",
    start: Optional[str] = None,
    end: Optional[str] = None,
    preset: str = "default",
) -> dict:
    """
    Compare dynamic betas across multiple stocks.

    Parameters
    ----------
    stocks : list of str
        List of stock tickers (e.g., ["AAPL", "MSFT", "GOOGL"]).
    market : str or pd.Series
        Market ticker or return series.
    start : str, optional
        Start date. Default: 10 years ago.
    end : str, optional
        End date. Default: today.
    preset : str
        Configuration preset.

    Returns
    -------
    dict
        Keys: 'betas' (DataFrame), 'summary', 'fig', 'results' (dict per stock)
    """
    import matplotlib.pyplot as plt

    all_results = {}
    failed = {}

    for ticker in stocks:
        print(f"\n{'='*50}")
        print(f"Estimating beta for {ticker}...")
        print(f"{'='*50}")
        try:
            result = estimate_beta(
                stock=ticker,
                market=market,
                start=start,
                end=end,
                preset=preset,
                plot=False,
                verbose=True,
            )
            all_results[ticker] = result
        except Exception as e:
            print(f"  WARNING: Failed for {ticker}: {e}")
            failed[ticker] = str(e)

    if not all_results:
        raise ValueError(
            f"All stocks failed. Errors:\n"
            + "\n".join(f"  {t}: {e}" for t, e in failed.items())
        )

    # Use only successful tickers going forward
    stocks = [t for t in stocks if t in all_results]

    # Build DataFrame with proper date alignment
    beta_dict = {}
    for ticker in stocks:
        r = all_results[ticker]
        beta_s = r["beta"].copy()
        beta_s.index = r["dates"]
        beta_s.name = ticker
        beta_dict[ticker] = beta_s

    beta_df = pd.DataFrame(beta_dict)  # Auto-aligns on index

    # Build comparison summary
    summary_lines = ["Beta Comparison Summary", "=" * 50]
    for ticker in stocks:
        r = all_results[ticker]
        valid = r["beta"].dropna()
        if len(valid) > 0:
            summary_lines.append(
                f"  {ticker:8s}  Current: {valid.iloc[-1]:6.2f}  "
                f"Avg: {valid.mean():6.2f}  "
                f"Range: {valid.min():.2f} → {valid.max():.2f}"
            )
    summary = "\n".join(summary_lines)

    # Comparison plot
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#2c3e50", "#e74c3c", "#27ae60", "#8e44ad", "#f39c12", "#1abc9c", "#e67e22", "#3498db"]
    for i, ticker in enumerate(stocks):
        beta = all_results[ticker]["beta"]
        dates = all_results[ticker]["dates"]
        valid_mask = ~beta.isna()
        color = colors[i % len(colors)]
        ax.plot(dates[valid_mask], beta[valid_mask], label=ticker, color=color, linewidth=1.5)

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Market Beta (1.0)")
    market_name = market if isinstance(market, str) else "Market"
    ax.set_title(f"Dynamic Beta Comparison (vs {market_name})")
    ax.set_ylabel("Beta (β)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "betas": beta_df,
        "summary": summary,
        "fig": fig,
        "results": all_results,
    }


def quick_report(
    stock: Union[str, pd.Series],
    market: Union[str, pd.Series] = "SPY",
    output: str = "report.html",
    start: Optional[str] = None,
    end: Optional[str] = None,
    preset: str = "default",
) -> str:
    """
    Generate a formatted HTML report with plots, metrics, and interpretation.

    Parameters
    ----------
    stock : str or pd.Series
        Stock ticker or return series.
    market : str or pd.Series
        Market ticker or return series.
    output : str
        Output file path for the HTML report.
    start : str, optional
        Start date.
    end : str, optional
        End date.
    preset : str
        Configuration preset.

    Returns
    -------
    str
        Path to the generated report file.
    """
    import base64
    import io

    import matplotlib.pyplot as plt

    result = estimate_beta(
        stock=stock, market=market, start=start, end=end,
        preset=preset, plot=False, verbose=True,
    )

    stock_name = stock if isinstance(stock, str) else getattr(stock, "name", "Stock")
    market_name = market if isinstance(market, str) else getattr(market, "name", "Market")

    # Generate plot as base64
    fig, ax = plt.subplots(figsize=(12, 6))
    beta = result["beta"]
    dates = result["dates"]
    valid_mask = ~beta.isna()
    ax.plot(dates[valid_mask], beta[valid_mask], color="#2c3e50", linewidth=1.5)
    ax.axhline(1.0, color="#e74c3c", linestyle="--", alpha=0.5)
    ax.set_title(f"{stock_name} Dynamic Beta Over Time")
    ax.set_ylabel("Beta (β)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # Build HTML
    import html as html_mod

    stock_name = html_mod.escape(stock_name)
    market_name = html_mod.escape(market_name)
    summary_html = html_mod.escape(result["summary"]).replace("\n", "<br>")

    valid_betas = beta.dropna()
    stats_rows = ""
    if len(valid_betas) > 0:
        stats = {
            "Current Beta": f"{valid_betas.iloc[-1]:.4f}",
            "Average Beta": f"{valid_betas.mean():.4f}",
            "Min Beta": f"{valid_betas.min():.4f}",
            "Max Beta": f"{valid_betas.max():.4f}",
            "Stability (Δβ std)": f"{np.std(np.diff(valid_betas.values)):.4f}",
        }
        for k, v in stats.items():
            stats_rows += f"<tr><td>{k}</td><td>{v}</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>{stock_name} Dynamic Beta Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
    h1 {{ color: #2c3e50; }}
    .summary {{ background: #f8f9fa; padding: 20px; border-radius: 8px;
               font-family: monospace; white-space: pre-wrap; margin: 20px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
    th, td {{ padding: 10px 16px; text-align: left; border-bottom: 1px solid #ddd; }}
    th {{ background: #2c3e50; color: white; }}
    tr:hover {{ background: #f5f5f5; }}
    img {{ max-width: 100%; border-radius: 8px; margin: 20px 0; }}
    .footer {{ color: #888; font-size: 0.85em; margin-top: 40px; }}
</style>
</head>
<body>
<h1>{stock_name} Dynamic Beta Report</h1>
<p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} &mdash;
   Market: {market_name} &mdash; Preset: {preset}</p>

<h2>Summary</h2>
<div class="summary">{summary_html}</div>

<h2>Beta Trajectory</h2>
<img src="data:image/png;base64,{img_b64}" alt="Beta trajectory plot">

<h2>Statistics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{stats_rows}
</table>

<div class="footer">
    Report generated by <strong>grubeta</strong> v{_get_version()}.
    See <a href="https://grubeta.readthedocs.io">documentation</a> for details.
</div>
</body>
</html>"""

    with open(output, "w") as f:
        f.write(html)

    print(f"Report saved to {output}")
    return output


def _get_version() -> str:
    """Get grubeta version string."""
    try:
        from grubeta import __version__
        return __version__
    except ImportError:
        return "0.1.3"
