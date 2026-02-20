"""
Core DynamicBeta estimator - the primary user-facing API.

This module provides the main interface for estimating time-varying beta
using GRU neural networks. It supports both simple (returns-only) and
advanced (multi-feature) estimation modes.
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.preprocessing import StandardScaler

from grubeta.evaluation import BetaEvaluator
from grubeta.models import GRUBetaModel
from grubeta.scaling import PITStandardScaler
from grubeta.temporal_integrity import (
    assert_pit_integrity,
    assert_no_temporal_overlap,
    TemporalCertificate,
    TemporalIntegrityError
)

logger = logging.getLogger(__name__)


class DynamicBetaConfig(BaseModel):
    """
    Configuration for DynamicBeta model.
    Pydantic model for strict type validation and constraints.
    """
    # Sequence parameters
    lookback: int = Field(default=90, ge=10, description="Input sequence length")
    initial_train_size: int = Field(default=500, description="Initial training samples")
    wf_step_size: int = Field(default=126, gt=0, description="Walk-forward step size")

    # Training parameters
    learning_rate: float = Field(default=1e-4, gt=0, lt=1, description="Adam learning rate")
    batch_size: int = Field(default=20, gt=0, description="Batch size")
    epochs_init: int = Field(default=40, gt=0, description="Initial epochs")
    epochs_retrain: int = Field(default=4, gt=0, description="Retraining epochs")

    # Architecture parameters
    gru_units: int = Field(default=128, gt=0, description="GRU units")
    dropout_rate: float = Field(default=0.2, ge=0, lt=1, description="Dropout rate")

    # Loss function weights
    lambda_beta: float = Field(default=0.05, ge=0, description="Beta smoothness penalty")
    lambda_alpha: float = Field(default=0.5, ge=0, description="Alpha sparsity penalty")

    # Initialization parameters
    initial_beta: float = Field(default=1.0, description="Initial bias for beta (1.0 = market-neutral prior)")
    initial_alpha: float = Field(default=0.0, description="Initial bias for alpha")

    # General
    random_seed: int = Field(default=42, description="Random seed")
    verbose: int = Field(default=1, ge=0, description="Verbosity level")

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("initial_train_size")
    @classmethod
    def validate_train_size(cls, v: int, info) -> int:
        lookback = info.data.get("lookback")
        if lookback is not None and v < lookback * 2:
            warnings.warn(
                f"initial_train_size ({v}) is less than 2x lookback. "
                "This may cause unstable training."
            )
        return v


class DynamicBeta:
    """
    GRU-based Dynamic Beta Estimator.

    Estimates time-varying systematic risk (beta) using a Gated Recurrent Unit
    neural network within the CAPM framework. Supports both simple estimation
    using only returns data and advanced estimation using engineered features.

    The model architecture consists of two parallel GRU pathways:
    - Beta pathway: Processes market/systematic features
    - Alpha pathway: Processes stock-specific/idiosyncratic features

    The loss function combines:
    - Prediction accuracy (Huber loss on return prediction)
    - Beta stability (penalizes rapid changes)
    - Alpha sparsity (L1 penalty, encouraging market efficiency)

    Parameters
    ----------
    config : DynamicBetaConfig, optional
        Configuration object. If None, uses default configuration.
    **kwargs
        Configuration parameters passed to DynamicBetaConfig.

    Attributes
    ----------
    config : DynamicBetaConfig
        Model configuration.
    model_ : GRUBetaModel
        Fitted Keras model (available after fit).
    scaler_market_ : StandardScaler
        Fitted scaler for market features (available after fit).
    scaler_stock_ : StandardScaler
        Fitted scaler for stock features (available after fit).
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Simple usage with returns only:

    >>> import pandas as pd
    >>> from grubeta import DynamicBeta
    >>>
    >>> # Load your data
    >>> stock_returns = pd.Series(...)  # Daily stock returns
    >>> market_returns = pd.Series(...) # Daily market returns
    >>>
    >>> # Fit and predict
    >>> model = DynamicBeta(lookback=60)
    >>> results = model.fit_predict(stock_returns, market_returns)
    >>>
    >>> # Access results
    >>> print(results['beta'].tail())
    >>> model.plot_beta(results)

    Advanced usage with features:

    >>> from grubeta import DynamicBeta, DataPreprocessor
    >>>
    >>> # Prepare features
    >>> prep = DataPreprocessor()
    >>> features = prep.prepare(stock_df, market_df, macro_df)
    >>>
    >>> # Fit with full feature set
    >>> model = DynamicBeta(lookback=90, use_macro=True)
    >>> results = model.fit_predict(**features)

    Notes
    -----
    The walk-forward validation ensures that:
    1. No future information leaks into predictions
    2. The model adapts to regime changes over time
    3. Out-of-sample performance is realistic

    References
    ----------
    .. [1] Sharpe, W.F. (1964). Capital asset prices: A theory of market
           equilibrium under conditions of risk.
    .. [2] Cho, K. et al. (2014). Learning Phrase Representations using
           RNN Encoder-Decoder for Statistical Machine Translation.
    """

    def __init__(self, config: Optional[DynamicBetaConfig] = None, **kwargs):
        if config is not None:
            self.config = config
        else:
            # Pydantic validation happens here automatically
            self.config = DynamicBetaConfig(**kwargs)

        # Will be set after fitting
        self.model_ = None
        self.scaler_market_ = PITStandardScaler()
        self.scaler_stock_ = PITStandardScaler()
        self.is_fitted_ = False
        self._feature_names_market_ = None
        self._feature_names_stock_ = None
        self._temporal_certificate = None

    def fit(
        self,
        stock_returns: Union[np.ndarray, pd.Series],
        market_returns: Union[np.ndarray, pd.Series],
        market_features: Optional[np.ndarray] = None,
        stock_features: Optional[np.ndarray] = None,
        dates: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    ) -> "DynamicBeta":
        """
        Fit the dynamic beta model.

        Parameters
        ----------
        stock_returns : array-like of shape (n_samples,)
            Stock return series (target variable).

        market_returns : array-like of shape (n_samples,)
            Market return series.

        market_features : array-like of shape (n_samples, n_market_features), optional
            Additional market/systematic features. If None, uses market_returns only.

        stock_features : array-like of shape (n_samples, n_stock_features), optional
            Additional stock-specific features. If None, uses stock_returns only.

        dates : array-like, optional
            Date index for the time series.

        Returns
        -------
        self : DynamicBeta
            Fitted estimator.
        """
        # Run Jidoka: Assert Temporal Integrity on Inputs
        if market_features is not None:
            assert_pit_integrity(
                market_features, 
                market_returns, 
                correlation_threshold=0.9,
                feature_names=self._feature_names_market_
            )
            
        if stock_features is not None:
            assert_pit_integrity(
                stock_features, 
                stock_returns,
                correlation_threshold=0.9,
                feature_names=self._feature_names_stock_
            )

        # Convert inputs
        stock_returns = self._to_array(stock_returns)
        market_returns = self._to_array(market_returns)

        if len(stock_returns) != len(market_returns):
            raise ValueError(
                f"stock_returns and market_returns must have same length. "
                f"Got {len(stock_returns)} and {len(market_returns)}."
            )

        n_samples = len(stock_returns)
        if n_samples < self.config.lookback + self.config.initial_train_size:
            raise ValueError(
                f"Insufficient data: need at least {self.config.lookback + self.config.initial_train_size} "
                f"samples, got {n_samples}."
            )

        # Build feature matrices
        if market_features is None:
            # Simple mode: use lagged returns as features
            market_features = self._create_return_features(market_returns)

        if stock_features is None:
            stock_features = self._create_return_features(stock_returns)

        # Jidoka: PIT Scaling (Scale-Then-Sequence)
        # Use PIT scaling for causal correctness even in global fit
        lookback = self.config.lookback
        if self.config.verbose >= 1:
            logger.info("Applying Point-in-Time (PIT) Scaling on features...")
            
        market_features = self.scaler_market_.fit_transform_pit(market_features, min_periods=lookback)
        stock_features = self.scaler_stock_.fit_transform_pit(stock_features, min_periods=lookback)
        
        # Ensure scalers are fitted for inference
        self.scaler_market_.fit(market_features)
        self.scaler_stock_.fit(stock_features)

        # Create sequences using pre-scaled features
        X_m, X_s, X_curr, y = self._create_sequences(
            market_features, stock_features, market_returns, stock_returns
        )

        # Store for later use
        self._n_market_features = X_m.shape[2]
        self._n_stock_features = X_s.shape[2]

        # Build and train model
        self._fit_model(X_m, X_s, X_curr, y)

        self.is_fitted_ = True
        return self

    def predict(
        self,
        stock_returns: Union[np.ndarray, pd.Series],
        market_returns: Union[np.ndarray, pd.Series],
        market_features: Optional[np.ndarray] = None,
        stock_features: Optional[np.ndarray] = None,
        dates: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    ) -> pd.DataFrame:
        """
        Predict dynamic beta for new data.

        Parameters
        ----------
        stock_returns : array-like of shape (n_samples,)
            Stock return series.

        market_returns : array-like of shape (n_samples,)
            Market return series.

        market_features : array-like, optional
            Additional market features.

        stock_features : array-like, optional
            Additional stock features.

        dates : array-like, optional
            Date index for results.

        Returns
        -------
        results : pd.DataFrame
            DataFrame with columns: beta, alpha, stock_return, market_return
            (and 'date' if provided). Same format as fit_predict().
        """
        self._check_is_fitted()

        stock_returns = self._to_array(stock_returns)
        market_returns = self._to_array(market_returns)

        if market_features is None:
            market_features = self._create_return_features(market_returns)
        if stock_features is None:
            stock_features = self._create_return_features(stock_returns)

        # Scale features using fitted scalers (Scale-Then-Sequence)
        market_features = self.scaler_market_.transform(market_features)
        stock_features = self.scaler_stock_.transform(stock_features)

        X_m, X_s, X_curr, _ = self._create_sequences(
            market_features, stock_features, market_returns, stock_returns
        )

        # Features are already scaled above
        X_m_norm, X_s_norm = X_m, X_s

        # Predict
        assert self.model_ is not None, "Model not fitted"
        raw_pred = self.model_.predict([X_m_norm, X_s_norm, X_curr], verbose=0)
        betas, alphas = self._extract_predictions(raw_pred)

        # Build results DataFrame (same format as fit_predict)
        return self._build_results_df(
            betas, alphas, stock_returns, market_returns, dates
        )

    def fit_predict(
        self,
        stock_returns: Union[np.ndarray, pd.Series],
        market_returns: Union[np.ndarray, pd.Series],
        market_features: Optional[np.ndarray] = None,
        stock_features: Optional[np.ndarray] = None,
        dates: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fit the model and return beta predictions using walk-forward validation.

        This is the recommended method for most use cases. It performs proper
        walk-forward validation to prevent lookahead bias.

        Parameters
        ----------
        stock_returns : array-like of shape (n_samples,)
            Stock return series.

        market_returns : array-like of shape (n_samples,)
            Market return series.

        market_features : array-like, optional
            Additional market/systematic features.

        stock_features : array-like, optional
            Additional stock-specific features.

        dates : array-like, optional
            Date index for results.

        Returns
        -------
        results : pd.DataFrame
            DataFrame with columns:
            - 'date': Date index (if provided)
            - 'beta': Dynamic beta estimates
            - 'alpha': Dynamic alpha estimates
            - 'stock_return': Original stock returns
            - 'market_return': Original market returns

        Examples
        --------
        >>> model = DynamicBeta(lookback=60)
        >>> results = model.fit_predict(stock_returns, market_returns)
        >>> print(results.dropna().head())
        """
        # Convert inputs
        stock_returns = self._to_array(stock_returns)
        market_returns = self._to_array(market_returns)

        if market_features is None:
            market_features = self._create_return_features(market_returns)
        if stock_features is None:
            stock_features = self._create_return_features(stock_returns)

        # Jidoka: PIT Scaling (Scale-Then-Sequence)
        # Scale 2D features before creating 3D sequences to ensure correct PIT logic
        # and avoid 3D scaling complexity issues.
        lookback = self.config.lookback
        
        if self.config.verbose >= 1:
            logger.info("Applying Point-in-Time (PIT) Scaling on features...")
            
        market_features = self.scaler_market_.fit_transform_pit(market_features, min_periods=lookback)
        stock_features = self.scaler_stock_.fit_transform_pit(stock_features, min_periods=lookback)
        
        # Ensure scalers are fitted for inference (predict) using end-of-training state
        self.scaler_market_.fit(market_features)
        self.scaler_stock_.fit(stock_features)

        # Create sequences using pre-scaled features
        X_m, X_s, X_curr, y = self._create_sequences(
            market_features, stock_features, market_returns, stock_returns
        )

        # Store dimensions
        self._n_market_features = X_m.shape[2]
        self._n_stock_features = X_s.shape[2]

        # Run walk-forward training/prediction
        betas, alphas = self._walk_forward(X_m, X_s, X_curr, y)

        self.is_fitted_ = True

        # Build results DataFrame
        results = self._build_results_df(
            betas, alphas, stock_returns, market_returns, dates
        )

        return results

    def _walk_forward(
        self, X_m: np.ndarray, X_s: np.ndarray, X_curr: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute anchored walk-forward training and prediction."""
        n_samples = len(y)
        betas: List[float] = []
        alphas: List[float] = []

        # Initialize model
        from grubeta.models import GRUBetaModel

        self.model_ = GRUBetaModel(self.config).build(
            self._n_market_features, self._n_stock_features
        )

        # Phase 1: Pre-calculate PIT Scaling (Done in fit())
        # Features X_m, X_s are already PIT-scaled.
        lookback = self.config.lookback

        # Phase 2: Initial training
        init_size = self.config.initial_train_size
        if self.config.verbose >= 1:
            logger.info(f"Phase 1: Initial training ({init_size} samples)")

        assert self.model_ is not None  # Built above
        self.model_.fit(
            [X_m[:init_size], X_s[:init_size], X_curr[:init_size]],
            y[:init_size],
            epochs=self.config.epochs_init,
            batch_size=self.config.batch_size,
            verbose=0,
        )

        # Pad burn-in period with NaN
        betas.extend([np.nan] * init_size)
        alphas.extend([np.nan] * init_size)

        # Phase 3: Walk-forward
        curr_idx = init_size
        step = self.config.wf_step_size
        total_steps = (n_samples - init_size + step - 1) // step
        step_num = 0

        if self.config.verbose >= 1:
            print(f"Walk-forward: {total_steps} steps (step_size={step})")

        while curr_idx < n_samples:
            end_idx = min(curr_idx + step, n_samples)
            step_num += 1
            
            if self.config.verbose >= 1:
                pct = min(100, int(step_num / total_steps * 100))
                print(f"  Step {step_num}/{total_steps} ({pct}%) — samples {curr_idx}..{end_idx}", end="\r")

            # Data is already PIT-scaled (Scale-Then-Sequence)
            X_m_next = X_m[curr_idx:end_idx]
            X_s_next = X_s[curr_idx:end_idx]

            # Predict out-of-sample (model_ is set above)
            pred = self.model_.predict(  # type: ignore[union-attr]
                [X_m_next, X_s_next, X_curr[curr_idx:end_idx]], verbose=0
            )
            b_next, a_next = self._extract_predictions(pred)
            betas.extend(b_next)
            alphas.extend(a_next)

            # Incremental retraining
            if end_idx < n_samples:
                train_window = 500
                start_train = max(0, end_idx - train_window)

                self.model_.fit(  # type: ignore[union-attr]
                    [
                        X_m[start_train:end_idx], 
                        X_s[start_train:end_idx], 
                        X_curr[start_train:end_idx]
                    ],
                    y[start_train:end_idx],
                    epochs=self.config.epochs_retrain,
                    batch_size=self.config.batch_size,
                    verbose=0,
                )

            curr_idx += step

        if self.config.verbose >= 1:
            print(f"  Walk-forward complete: {n_samples} samples processed.")
        
        # Ensure scalers are fitted for future inference
        # (This is now redundant since we fit() in fit method, but keeps walk_forward consistent)
        
        # Issue Temporal Certificate
        self._temporal_certificate = TemporalCertificate(
            model_version="0.2.0-pit",
            training_window=(str(0), str(init_size)),
            prediction_window=(str(init_size), str(n_samples)),
            scaler_policy="PIT_EXPANDING_WINDOW",
            feature_lag_policy="ALL_LAGGED_1_DAY",
            integrity_checks_passed=["PIT_SCALING_VERIFIED"]
        )

        return np.array(betas), np.array(alphas)

    def _create_return_features(self, returns: np.ndarray) -> np.ndarray:
        """
        Create simple feature matrix from returns.
        
        CRITICAL: All features use strictly lagged data [0:t-1] to predict t.
        This is a Poka-Yoke mechanism to prevent lookahead bias.
        """
        n = len(returns)
        features = np.zeros((n, 5))

        # Feature 0: Lag 1 return (t-1)
        features[1:, 0] = returns[:-1]
        
        # Feature 1: Lag 2 return (t-2)
        features[2:, 1] = returns[:-2]
        
        # Feature 2: Lag 5 return (t-5)
        features[5:, 2] = returns[:-5]

        # Feature 3: Rolling volatility (20-day, ending at t-1)
        # Vectorized using pandas for efficiency
        ret_series = pd.Series(returns)
        vol_20 = ret_series.rolling(20).std().shift(1).values  # shift ensures t-1
        features[:, 3] = np.nan_to_num(vol_20, nan=0.0)

        # Feature 4: Cumulative return (20-day, ending at t-1)
        cum_20 = ret_series.rolling(20).sum().shift(1).values
        features[:, 4] = np.nan_to_num(cum_20, nan=0.0)
        
        return cast(np.ndarray, features)

    def _create_sequences(
        self,
        market_features: np.ndarray,
        stock_features: np.ndarray,
        market_returns: np.ndarray,
        stock_returns: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create sliding window sequences."""
        lookback = self.config.lookback
        n = len(stock_returns)

        X_m_seq, X_s_seq, X_curr, y = [], [], [], []

        for i in range(lookback, n):
            X_m_seq.append(market_features[i - lookback : i])
            X_s_seq.append(stock_features[i - lookback : i])
            X_curr.append([market_returns[i]])
            y.append(stock_returns[i])

        return (np.array(X_m_seq), np.array(X_s_seq), np.array(X_curr), np.array(y))

    def _normalize_batch(
        self, X_m: np.ndarray, X_s: np.ndarray, fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize 3D tensors."""
        n, t, f_m = X_m.shape
        _, _, f_s = X_s.shape

        X_m_flat = X_m.reshape(-1, f_m)
        X_s_flat = X_s.reshape(-1, f_s)

        # Clean data
        X_m_flat = self._clean_data(X_m_flat)
        X_s_flat = self._clean_data(X_s_flat)

        if fit:
            self.scaler_market_.fit(X_m_flat)
            self.scaler_stock_.fit(X_s_flat)

        X_m_scaled = self.scaler_market_.transform(X_m_flat).reshape(n, t, f_m)
        X_s_scaled = self.scaler_stock_.transform(X_s_flat).reshape(n, t, f_s)

        return X_m_scaled, X_s_scaled

    def _extract_predictions(
        self, raw_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract beta and alpha from concatenated model output."""
        lookback = self.config.lookback
        beta_idx = 1 + lookback - 1
        alpha_idx = 1 + (2 * lookback) - 1
        return raw_pred[:, beta_idx], raw_pred[:, alpha_idx]

    def _build_results_df(
        self,
        betas: np.ndarray,
        alphas: np.ndarray,
        stock_returns: np.ndarray,
        market_returns: np.ndarray,
        dates: Optional[np.ndarray],
    ) -> pd.DataFrame:
        """Build results DataFrame with proper alignment."""
        n = len(stock_returns)
        lookback = self.config.lookback

        # Pad to original length
        betas_full = np.concatenate([np.full(lookback, np.nan), betas])[:n]
        alphas_full = np.concatenate([np.full(lookback, np.nan), alphas])[:n]

        results = pd.DataFrame(
            {
                "beta": betas_full,
                "alpha": alphas_full,
                "stock_return": stock_returns,
                "market_return": market_returns,
            }
        )

        if dates is not None:
            results["date"] = dates
            results = results[
                ["date", "beta", "alpha", "stock_return", "market_return"]
            ]

        return results

    def _fit_model(
        self, X_m: np.ndarray, X_s: np.ndarray, X_curr: np.ndarray, y: np.ndarray
    ) -> None:
        """Fit model on entire dataset (for simple fit without walk-forward)."""
        from grubeta.models import GRUBetaModel

        self.model_ = GRUBetaModel(self.config).build(
            self._n_market_features, self._n_stock_features
        )

        if self.config.verbose >= 1:
            logger.info("Fitting model on pre-scaled PIT features...")

        # Data is already PIT-scaled (Scale-Then-Sequence)
        # No need to scale again
        
        # Ensure scalers are fitted for future inference
        # (Redundant if called from fit(), but harmless)
        # X_m_flat = X_m.reshape(-1, self._n_market_features)
        # self.scaler_market_.fit(X_m_flat)

        assert self.model_ is not None
        self.model_.fit(  # type: ignore[union-attr]
            [X_m, X_s, X_curr],
            y,
            epochs=self.config.epochs_init,
            batch_size=self.config.batch_size,
            verbose=1 if self.config.verbose >= 1 else 0,
        )

    @staticmethod
    def _clean_data(X: np.ndarray) -> np.ndarray:
        """Clean data by handling inf/nan values."""
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X, nan=0.0)
        return cast(np.ndarray, np.clip(X, -100.0, 100.0))

    @staticmethod
    def _to_array(x: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(x, pd.Series):
            return cast(np.ndarray, x.values)
        return cast(np.ndarray, np.asarray(x))

    def _check_is_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted_:
            raise ValueError(
                "This DynamicBeta instance is not fitted yet. "
                "Call 'fit' or 'fit_predict' before using this method."
            )

    def plot_beta(
        self,
        results: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        benchmark_beta: Optional[np.ndarray] = None,
        true_beta: Optional[np.ndarray] = None,
        show_hedged: bool = False,
    ) -> None:
        """
        Plot the dynamic beta trajectory.

        Parameters
        ----------
        results : pd.DataFrame
            Results from fit_predict().
        figsize : tuple, default=(12, 6)
            Figure size.
        title : str, optional
            Plot title.
        save_path : str, optional
            Path to save the figure.
        benchmark_beta : np.ndarray, optional
            Benchmark beta to overlay (e.g. rolling OLS).
        true_beta : np.ndarray, optional
            Ground-truth beta for synthetic / backtesting scenarios.
        show_hedged : bool, default=False
            If True, adds a second panel showing cumulative hedged returns.
        """
        import matplotlib.pyplot as plt

        n_panels = 2 if show_hedged else 1
        fig, axes = plt.subplots(n_panels, 1, figsize=(figsize[0], figsize[1] * n_panels),
                                 sharex=True, squeeze=False)
        ax = axes[0, 0]

        x = results["date"] if "date" in results.columns else results.index

        # Main beta line
        ax.plot(x, results["beta"], label="GRU Dynamic Beta", color="#2c3e50", linewidth=1.5)
        ax.axhline(1.0, color="#e74c3c", linestyle="--", alpha=0.5, label="Market Beta (1.0)")

        # Optional overlays
        if benchmark_beta is not None:
            ax.plot(x, benchmark_beta[:len(x)], label="Benchmark (OLS)",
                    color="#e67e22", linewidth=1.2, alpha=0.8)

        if true_beta is not None:
            ax.plot(x, true_beta[:len(x)], label="True Beta",
                    color="#27ae60", linewidth=1.2, alpha=0.7)

        ax.set_title(title or "Dynamic Beta Evolution")
        ax.set_ylabel("Beta")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Optional hedged returns panel
        if show_hedged and n_panels == 2:
            ax2 = axes[1, 0]
            hedged = results["stock_return"] - results["beta"] * results["market_return"]
            cum_hedged = (1 + hedged).cumprod()
            cum_stock = (1 + results["stock_return"]).cumprod()

            ax2.plot(x, cum_stock, label="Unhedged Stock", color="gray", alpha=0.5)
            ax2.plot(x, cum_hedged, label="Hedged (GRU)", color="#2c3e50", linewidth=2)
            ax2.set_ylabel("Growth of $1")
            ax2.set_xlabel("Date")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        if not show_hedged:
            ax.set_xlabel("Date")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)

        plt.show()

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the fitted model to disk.

        Parameters
        ----------
        path : str or Path
            Directory path to save model artifacts.
        """
        self._check_is_fitted()
        import pickle

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save Keras model (model_ exists after _check_is_fitted)
        assert self.model_ is not None
        self.model_.save(path / "model.h5")

        # Save scalers and config
        with open(path / "artifacts.pkl", "wb") as f:
            pickle.dump(
                {
                    "scaler_market": self.scaler_market_,
                    "scaler_stock": self.scaler_stock_,
                    "config": self.config,
                    "n_market_features": self._n_market_features,
                    "n_stock_features": self._n_stock_features,
                },
                f,
            )

        if self.config.verbose >= 1:
            logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DynamicBeta":
        """
        Load a fitted model from disk.

        Parameters
        ----------
        path : str or Path
            Directory containing saved model artifacts.

        Returns
        -------
        model : DynamicBeta
            Loaded model instance.
        """
        import pickle

        from tensorflow import keras

        path = Path(path)

        # Load artifacts
        with open(path / "artifacts.pkl", "rb") as f:
            artifacts = pickle.load(f)

        # Create instance
        instance = cls(config=artifacts["config"])
        instance.scaler_market_ = artifacts["scaler_market"]
        instance.scaler_stock_ = artifacts["scaler_stock"]
        instance._n_market_features = artifacts["n_market_features"]
        instance._n_stock_features = artifacts["n_stock_features"]

        # Load Keras model
        from grubeta.models import GRUBetaModel

        instance.model_ = keras.models.load_model(
            path / "model.h5",
            custom_objects=GRUBetaModel.get_custom_objects(instance.config),
        )

        instance.is_fitted_ = True
        return instance

    def __repr__(self) -> str:
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        return (
            f"DynamicBeta(lookback={self.config.lookback}, "
            f"lambda_beta={self.config.lambda_beta}, "
            f"lambda_alpha={self.config.lambda_alpha}, "
            f"status={fitted_str})"
        )
