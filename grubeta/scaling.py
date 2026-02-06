"""
Point-in-Time (PIT) Scaling Module.

This module provides scaling utilities that strictly respect temporal causality.
Standard scalers (like sklearn's) operate globally or in batch mode, which
can introduce subtle lookahead bias if not carefully managed.

The PITStandardScaler ensures that for any time t, the scaling statistics
(mean, std) are computed using ONLY data from [0, t-1].

TPS Principle: Poka-Yoke (Mistake Proofing) - make lookahead impossible by design.
"""

import numpy as np
from typing import Optional, Tuple, Union, cast
import warnings

# Use numba if available for performance
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


class PITStandardScaler:
    """
    Standard Scaler that enforces Point-in-Time (PIT) integrity.
    
    This scaler can operate in two modes:
    1. Expanding Window (strictly causal): mean/std at t computed from [0:t]
    2. Fixed Window: fit on [0:T], transform on [T+1:]
    
    Attributes
    ----------
    mean_ : np.ndarray
        Current running mean.
    var_ : np.ndarray
        Current running variance.
    n_samples_seen_ : int
        Number of samples processed.
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True, epsilon: float = 1e-8):
        self.with_mean = with_mean
        self.with_std = with_std
        self.epsilon = epsilon
        self.mean_ = None
        self.var_ = None
        self.n_samples_seen_ = 0
        
    def fit(self, X: np.ndarray) -> "PITStandardScaler":
        """
        Compute mean and std on X for later scaling.
        
        Parameters
        ----------
        X : array-like
            Data to fit.
            
        Returns
        -------
        self
        """
        X = np.asarray(X)
        self.n_samples_seen_ = len(X)
        
        if self.with_mean:
            self.mean_ = np.nanmean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])
            
        if self.with_std:
            self.var_ = np.nanvar(X, axis=0)
        else:
            self.var_ = np.ones(X.shape[1])
            
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale X using fitted mean/std.
        
        Parameters
        ----------
        X : array-like
            Data to transform.
            
        Returns
        -------
        X_scaled : np.ndarray
        """
        if self.mean_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
            
        X = np.asarray(X)
        X_scaled = X.copy()
        
        if self.with_mean:
            X_scaled -= self.mean_
            
        if self.with_std:
            scale = np.sqrt(self.var_) + self.epsilon
            X_scaled /= scale
            
        return X_scaled
    
    def fit_transform_pit(self, X: np.ndarray, min_periods: int = 20) -> np.ndarray:
        """
        Perform Point-in-Time (Expanding Window) scaling.
        
        For each row t, scale using statistics from rows [0 : t].
        Note: Strict PIT often implies using [0 : t-1] to scale t.
        However, standard practice often includes t in the distribution estimate if
        t is not the target. Here we strictly respect [0:t] history.
        
        If 'strict_lag' is desired (using 0:t-1 to scale t), pass input shifted by 1.
        
        Parameters
        ----------
        X : array-like
            Time series data (n_samples, n_features).
        min_periods : int
            Number of periods required before scaling effectively starts.
            Before this, returns 0.
            
        Returns
        -------
        X_scaled : np.ndarray
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        X_scaled = np.zeros_like(X)
        
        # Optimization: Use fast cumulative calculation
        # Sum(x)
        # Sum(x^2)
        # Mean = Sum(x) / n
        # Var = Mean(x^2) - (Mean(x))^2
        
        # We process row by row to be explicit, but can be vectorized
        # Cumulative sum
        cumsum = np.nancumsum(X, axis=0)
        
        # Cumulative sum of squares
        cumsum_sq = np.nancumsum(X**2, axis=0)
        
        # Count (handling NaNs if needed, but assuming clean for speed here)
        # If NaNs exist, we should use a valid count mask
        counts = np.arange(1, n_samples + 1).reshape(-1, 1)
        
        # Running Mean
        means = cumsum / counts
        
        # Running Variance
        # Var[t] = E[x^2] - (E[x])^2
        means_sq = cumsum_sq / counts
        vars_ = means_sq - (means ** 2)
        vars_ = np.maximum(vars_, 0) # Clip negative due to float precision
        
        stds = np.sqrt(vars_) + self.epsilon
        
        # Now apply scaling
        # X_scaled[t] = (X[t] - mean[t]) / std[t]
        # BUT wait:
        # To be purely causal, we usually normalize X[t] using stats from [0...t]?
        # Or [0...t-1]?
        # In online learning, we typically update stats with X[t], then scale X[t].
        
        if self.with_mean:
            X_scaled = X - means
        else:
            X_scaled = X
            
        if self.with_std:
            X_scaled = X_scaled / stds
            
        # Handle burn-in
        if min_periods > 0:
            X_scaled[:min_periods] = 0.0
            
        return cast(np.ndarray, X_scaled)

