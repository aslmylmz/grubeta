"""
Utility functions for GRU Dynamic Beta.

This module provides helper functions for data validation,
lookahead bias testing, and benchmark beta calculations.
"""

from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd


def validate_no_lookahead(
    betas: np.ndarray,
    returns: np.ndarray,
    market_returns: np.ndarray,
    initial_size: int,
    verbose: bool = True,
) -> bool:
    """
    Sanity check: verify beta at t is not correlated with future returns.
    
    This is a critical validation to ensure the model doesn't suffer from
    lookahead bias. If betas are correlated with future returns (t+k),
    it indicates information leakage.
    
    Parameters
    ----------
    betas : array-like
        Estimated beta series.
    returns : array-like
        Stock return series.
    market_returns : array-like
        Market return series (unused but kept for API consistency).
    initial_size : int
        Initial training window size (skipped in analysis).
    verbose : bool, default=True
        Print diagnostic output.
        
    Returns
    -------
    passed : bool
        True if no suspicious correlations detected.
        
    Examples
    --------
    >>> from grubeta.utils import validate_no_lookahead
    >>> 
    >>> results = model.fit_predict(stock_returns, market_returns)
    >>> passed = validate_no_lookahead(
    ...     results['beta'].values,
    ...     results['stock_return'].values,
    ...     results['market_return'].values,
    ...     initial_size=500
    ... )
    >>> if not passed:
    ...     print("Warning: Potential lookahead bias detected!")
    """
    # Filter valid data
    valid_mask = ~np.isnan(betas)
    b = betas[valid_mask]
    r = returns[valid_mask]
    
    suspicious_count = 0
    
    for lag in [1, 5, 10, 20]:
        if len(r) <= lag:
            continue
            
        future_ret = np.roll(r, -lag)[:-lag]
        beta_aligned = b[:-lag]
        
        corr = np.corrcoef(beta_aligned, future_ret)[0, 1]
        
        is_suspicious = abs(corr) > 0.15
        if is_suspicious:
            suspicious_count += 1
        
        if verbose:
            status = "⚠️ SUSPICIOUS" if is_suspicious else "✓"
            print(f"  {status} Beta correlation with return t+{lag}: {corr:.4f}")
    
    return suspicious_count == 0


def rolling_ols_beta(
    stock_returns: Union[np.ndarray, pd.Series],
    market_returns: Union[np.ndarray, pd.Series],
    window: int = 252,
    min_periods: Optional[int] = None,
) -> np.ndarray:
    """
    Compute rolling OLS beta.
    
    This provides a simple benchmark for comparison with the GRU model.
    Uses an expanding window at the start until `window` observations
    are available.
    
    Parameters
    ----------
    stock_returns : array-like
        Stock return series.
    market_returns : array-like
        Market/index return series.
    window : int, default=252
        Rolling window size (approximately 1 year of trading days).
    min_periods : int, optional
        Minimum observations required. Defaults to window // 2.
        
    Returns
    -------
    betas : np.ndarray
        Rolling beta estimates. NaN for periods with insufficient data.
        
    Examples
    --------
    >>> from grubeta.utils import rolling_ols_beta
    >>> 
    >>> # Compute benchmark
    >>> ols_beta = rolling_ols_beta(stock_returns, market_returns, window=252)
    >>> 
    >>> # Compare with GRU beta
    >>> from grubeta import DynamicBeta
    >>> model = DynamicBeta()
    >>> results = model.fit_predict(stock_returns, market_returns)
    >>> 
    >>> # Calculate correlation
    >>> valid = ~np.isnan(ols_beta) & ~np.isnan(results['beta'])
    >>> corr = np.corrcoef(ols_beta[valid], results['beta'].values[valid])[0, 1]
    >>> print(f"Correlation with OLS: {corr:.4f}")
    """
    # Convert to arrays
    if isinstance(stock_returns, pd.Series):
        stock_returns = stock_returns.values
    if isinstance(market_returns, pd.Series):
        market_returns = market_returns.values
    
    stock_returns = np.asarray(stock_returns, dtype=float)
    market_returns = np.asarray(market_returns, dtype=float)
    
    n = len(stock_returns)
    min_periods = min_periods or (window // 2)
    
    betas = np.full(n, np.nan)
    
    for i in range(min_periods, n):
        # Expanding window until we reach full window size
        start = max(0, i - window)
        
        y = stock_returns[start:i]
        x = market_returns[start:i]
        
        # Handle NaN values
        mask = ~np.isnan(y) & ~np.isnan(x)
        if mask.sum() < min_periods:
            continue
        
        y_clean = y[mask]
        x_clean = x[mask]
        
        # Simple OLS: beta = cov(y, x) / var(x)
        x_demean = x_clean - np.mean(x_clean)
        cov = np.sum(x_demean * (y_clean - np.mean(y_clean)))
        var = np.sum(x_demean ** 2)
        
        if var > 1e-10:
            betas[i] = cov / var
    
    return betas


def create_sequences(
    data: np.ndarray,
    lookback: int,
    target_col: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for time series modeling.
    
    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Input time series data.
    lookback : int
        Number of past timesteps to include in each sequence.
    target_col : int, default=0
        Column index for the target variable.
        
    Returns
    -------
    X : np.ndarray of shape (n_sequences, lookback, n_features)
        Input sequences.
    y : np.ndarray of shape (n_sequences,)
        Target values.
        
    Examples
    --------
    >>> data = np.column_stack([returns, features])
    >>> X, y = create_sequences(data, lookback=60)
    >>> print(X.shape)  # (n_samples - 60, 60, n_features)
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    n_samples, n_features = data.shape
    n_sequences = n_samples - lookback
    
    X = np.zeros((n_sequences, lookback, n_features))
    y = np.zeros(n_sequences)
    
    for i in range(n_sequences):
        X[i] = data[i:i + lookback]
        y[i] = data[i + lookback, target_col]
    
    return X, y


def align_series(
    *series: Union[np.ndarray, pd.Series, pd.DataFrame],
    dropna: bool = True
) -> Tuple[np.ndarray, ...]:
    """
    Align multiple time series by removing NaN values across all series.
    
    Parameters
    ----------
    *series : array-like
        Variable number of series to align.
    dropna : bool, default=True
        Whether to drop rows with any NaN values.
        
    Returns
    -------
    aligned : tuple of np.ndarray
        Aligned arrays.
        
    Examples
    --------
    >>> stock_ret, mkt_ret, betas = align_series(
    ...     stock_returns, market_returns, dynamic_betas
    ... )
    >>> # All arrays now have same length with no NaN
    """
    arrays = []
    for s in series:
        if isinstance(s, (pd.Series, pd.DataFrame)):
            arrays.append(s.values)
        else:
            arrays.append(np.asarray(s))
    
    # Ensure same length
    min_len = min(len(a) for a in arrays)
    arrays = [a[:min_len] for a in arrays]
    
    if dropna:
        # Create combined mask
        mask = np.ones(min_len, dtype=bool)
        for a in arrays:
            if a.ndim == 1:
                mask &= ~np.isnan(a)
            else:
                mask &= ~np.isnan(a).any(axis=1)
        
        arrays = [a[mask] for a in arrays]
    
    return tuple(arrays)


def compute_information_coefficient(
    predicted: np.ndarray,
    actual: np.ndarray,
    method: str = 'pearson',
) -> float:
    """
    Compute Information Coefficient (IC) between predictions and actuals.
    
    The IC is commonly used in quantitative finance to measure
    the correlation between predicted and actual values.
    
    Parameters
    ----------
    predicted : array-like
        Predicted values (e.g., beta * market_return).
    actual : array-like
        Actual values (e.g., stock_return).
    method : str, default='pearson'
        Correlation method: 'pearson', 'spearman', or 'kendall'.
        
    Returns
    -------
    ic : float
        Information coefficient.
    """
    predicted, actual = align_series(predicted, actual)
    
    if len(predicted) < 2:
        return np.nan
    
    if method == 'pearson':
        return np.corrcoef(predicted, actual)[0, 1]
    elif method == 'spearman':
        from scipy.stats import spearmanr
        return spearmanr(predicted, actual)[0]
    elif method == 'kendall':
        from scipy.stats import kendalltau
        return kendalltau(predicted, actual)[0]
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_hit_rate(
    predicted_direction: np.ndarray,
    actual_direction: np.ndarray,
) -> float:
    """
    Compute directional hit rate.
    
    Measures how often the predicted direction matches the actual direction.
    
    Parameters
    ----------
    predicted_direction : array-like
        Predicted directions (positive/negative).
    actual_direction : array-like
        Actual directions.
        
    Returns
    -------
    hit_rate : float
        Proportion of correct directional predictions (0 to 1).
    """
    predicted_direction, actual_direction = align_series(
        predicted_direction, actual_direction
    )
    
    pred_sign = np.sign(predicted_direction)
    actual_sign = np.sign(actual_direction)
    
    # Exclude zeros
    mask = (pred_sign != 0) & (actual_sign != 0)
    
    if mask.sum() == 0:
        return np.nan
    
    return (pred_sign[mask] == actual_sign[mask]).mean()


def annualize_beta(
    daily_beta: float,
    trading_days: int = 252,
) -> float:
    """
    Note: Beta does not need annualization as it's a ratio.
    
    This function is provided for completeness but simply returns
    the input beta unchanged. Beta is scale-invariant.
    
    Parameters
    ----------
    daily_beta : float
        Daily beta estimate.
    trading_days : int, default=252
        Trading days per year (unused).
        
    Returns
    -------
    beta : float
        Same as input (beta is scale-invariant).
    """
    return daily_beta  # Beta is already scale-invariant


def beta_to_hedge_ratio(
    beta: float,
    stock_value: float,
    market_value: float,
) -> float:
    """
    Convert beta to a hedge ratio for portfolio management.
    
    Parameters
    ----------
    beta : float
        Stock's beta relative to market.
    stock_value : float
        Current value of stock position.
    market_value : float
        Value of one unit of market hedge instrument.
        
    Returns
    -------
    hedge_ratio : float
        Number of market units to short to hedge the position.
        
    Examples
    --------
    >>> # Stock position worth $100,000 with beta of 1.2
    >>> # Each SPY share is $450
    >>> hedge = beta_to_hedge_ratio(1.2, 100000, 450)
    >>> print(f"Short {hedge:.0f} shares of SPY")
    """
    return (beta * stock_value) / market_value
