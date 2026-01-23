"""
Core DynamicBeta estimator - the primary user-facing API.

This module provides the main interface for estimating time-varying beta
using GRU neural networks. It supports both simple (returns-only) and
advanced (multi-feature) estimation modes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List, Any
import warnings
import logging
import gc

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from grubeta.models import GRUBetaModel
from grubeta.evaluation import BetaEvaluator

logger = logging.getLogger(__name__)


@dataclass
class DynamicBetaConfig:
    """
    Configuration for DynamicBeta model.
    
    This dataclass holds all hyperparameters and settings for the GRU-based
    dynamic beta estimation model.
    
    Parameters
    ----------
    lookback : int, default=90
        Number of historical periods used as input sequence length.
        Typical values: 60 (quarterly), 90 (seasonal), 126 (semi-annual).
        
    initial_train_size : int, default=500
        Number of samples for initial model training before walk-forward begins.
        Should be >= lookback * 3 for stable training.
        
    wf_step_size : int, default=126
        Walk-forward step size (number of predictions before retraining).
        126 ≈ 6 months of trading days.
        
    learning_rate : float, default=1e-4
        Adam optimizer learning rate.
        
    batch_size : int, default=20
        Training batch size.
        
    epochs_init : int, default=40
        Epochs for initial training phase.
        
    epochs_retrain : int, default=4
        Epochs for incremental retraining during walk-forward.
        
    gru_units : int, default=128
        Number of GRU hidden units per pathway.
        
    dropout_rate : float, default=0.2
        Dropout rate for regularization.
        
    lambda_beta : float, default=0.05
        Loss weight for beta stability (temporal smoothness).
        Higher values produce smoother beta trajectories.
        
    lambda_alpha : float, default=0.5
        Loss weight for alpha sparsity (L1 penalty).
        Higher values push alpha toward zero (CAPM assumption).
        
    random_seed : int, default=42
        Random seed for reproducibility.
        
    verbose : int, default=1
        Verbosity level (0=silent, 1=progress, 2=debug).
        
    Examples
    --------
    >>> config = DynamicBetaConfig(lookback=60, lambda_beta=0.1)
    >>> model = DynamicBeta(config=config)
    
    >>> # Or use keyword arguments directly
    >>> model = DynamicBeta(lookback=60, lambda_beta=0.1)
    """
    # Sequence parameters
    lookback: int = 90
    initial_train_size: int = 500
    wf_step_size: int = 126
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 20
    epochs_init: int = 40
    epochs_retrain: int = 4
    
    # Architecture parameters
    gru_units: int = 128
    dropout_rate: float = 0.2
    
    # Loss function weights
    lambda_beta: float = 0.05
    lambda_alpha: float = 0.5
    
    # General
    random_seed: int = 42
    verbose: int = 1
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.lookback < 10:
            raise ValueError("lookback must be >= 10")
        if self.initial_train_size < self.lookback * 2:
            warnings.warn(
                f"initial_train_size ({self.initial_train_size}) is less than "
                f"2x lookback ({self.lookback * 2}). This may cause unstable training."
            )
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be between 0 and 1")


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
    
    def __init__(
        self, 
        config: Optional[DynamicBetaConfig] = None,
        **kwargs
    ):
        if config is not None:
            self.config = config
        else:
            self.config = DynamicBetaConfig(**kwargs)
        
        self.config.validate()
        
        # Will be set after fitting
        self.model_ = None
        self.scaler_market_ = StandardScaler()
        self.scaler_stock_ = StandardScaler()
        self.is_fitted_ = False
        self._feature_names_market_ = None
        self._feature_names_stock_ = None
        
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
            
        # Create sequences
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
    ) -> Dict[str, np.ndarray]:
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
            
        Returns
        -------
        results : dict
            Dictionary containing:
            - 'beta': Time-varying beta estimates
            - 'alpha': Time-varying alpha estimates
        """
        self._check_is_fitted()
        
        stock_returns = self._to_array(stock_returns)
        market_returns = self._to_array(market_returns)
        
        if market_features is None:
            market_features = self._create_return_features(market_returns)
        if stock_features is None:
            stock_features = self._create_return_features(stock_returns)
            
        X_m, X_s, X_curr, _ = self._create_sequences(
            market_features, stock_features, market_returns, stock_returns
        )
        
        # Normalize using fitted scalers
        X_m_norm, X_s_norm = self._normalize_batch(X_m, X_s, fit=False)
        
        # Predict
        raw_pred = self.model_.predict([X_m_norm, X_s_norm, X_curr], verbose=0)
        betas, alphas = self._extract_predictions(raw_pred)
        
        # Pad with NaN for alignment
        n_pad = self.config.lookback
        betas = np.concatenate([np.full(n_pad, np.nan), betas])
        alphas = np.concatenate([np.full(n_pad, np.nan), alphas])
        
        return {'beta': betas, 'alpha': alphas}
    
    def fit_predict(
        self,
        stock_returns: Union[np.ndarray, pd.Series],
        market_returns: Union[np.ndarray, pd.Series],
        market_features: Optional[np.ndarray] = None,
        stock_features: Optional[np.ndarray] = None,
        dates: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
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
        
        # Create sequences
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
        self, 
        X_m: np.ndarray, 
        X_s: np.ndarray, 
        X_curr: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute anchored walk-forward training and prediction."""
        n_samples = len(y)
        betas, alphas = [], []
        
        # Initialize model
        from grubeta.models import GRUBetaModel
        self.model_ = GRUBetaModel(self.config).build(
            self._n_market_features, self._n_stock_features
        )
        
        # Phase 1: Initial training
        init_size = self.config.initial_train_size
        if self.config.verbose >= 1:
            logger.info(f"Phase 1: Initial training ({init_size} samples)")
        
        X_m_init, X_s_init = self._normalize_batch(
            X_m[:init_size], X_s[:init_size], fit=True
        )
        
        self.model_.fit(
            [X_m_init, X_s_init, X_curr[:init_size]],
            y[:init_size],
            epochs=self.config.epochs_init,
            batch_size=self.config.batch_size,
            verbose=0
        )
        
        # Pad burn-in period with NaN
        betas.extend([np.nan] * init_size)
        alphas.extend([np.nan] * init_size)
        
        # Phase 2: Walk-forward
        curr_idx = init_size
        step = self.config.wf_step_size
        
        if self.config.verbose >= 1:
            logger.info(f"Phase 2: Walk-forward (step={step})")
        
        while curr_idx < n_samples:
            end_idx = min(curr_idx + step, n_samples)
            
            # Refit scalers on expanding window
            self._normalize_batch(X_m[:curr_idx], X_s[:curr_idx], fit=True)
            
            # Transform out-of-sample chunk
            X_m_next, X_s_next = self._normalize_batch(
                X_m[curr_idx:end_idx], X_s[curr_idx:end_idx], fit=False
            )
            
            # Predict out-of-sample
            pred = self.model_.predict(
                [X_m_next, X_s_next, X_curr[curr_idx:end_idx]], verbose=0
            )
            b_next, a_next = self._extract_predictions(pred)
            betas.extend(b_next)
            alphas.extend(a_next)
            
            # Incremental retraining
            if end_idx < n_samples:
                train_window = 500
                start_train = max(0, end_idx - train_window)
                
                self._normalize_batch(X_m[:end_idx], X_s[:end_idx], fit=True)
                X_m_train, X_s_train = self._normalize_batch(
                    X_m[start_train:end_idx], X_s[start_train:end_idx], fit=False
                )
                
                self.model_.fit(
                    [X_m_train, X_s_train, X_curr[start_train:end_idx]],
                    y[start_train:end_idx],
                    epochs=self.config.epochs_retrain,
                    batch_size=self.config.batch_size,
                    verbose=0
                )
            
            curr_idx += step
            
            if self.config.verbose >= 2:
                logger.debug(f"Progress: {curr_idx}/{n_samples}")
        
        return np.array(betas), np.array(alphas)
    
    def _create_return_features(self, returns: np.ndarray) -> np.ndarray:
        """Create simple feature matrix from returns."""
        # Use lagged returns, volatility, and momentum as basic features
        n = len(returns)
        features = np.zeros((n, 5))
        
        features[:, 0] = returns  # Current return
        features[1:, 1] = returns[:-1]  # Lag 1
        features[5:, 2] = returns[:-5]  # Lag 5
        
        # Rolling volatility (20-day)
        for i in range(20, n):
            features[i, 3] = np.std(returns[i-20:i])
        
        # Cumulative return (20-day)
        for i in range(20, n):
            features[i, 4] = np.sum(returns[i-20:i])
            
        return features
    
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
            X_m_seq.append(market_features[i-lookback:i])
            X_s_seq.append(stock_features[i-lookback:i])
            X_curr.append([market_returns[i]])
            y.append(stock_returns[i])
        
        return (
            np.array(X_m_seq), 
            np.array(X_s_seq), 
            np.array(X_curr), 
            np.array(y)
        )
    
    def _normalize_batch(
        self, 
        X_m: np.ndarray, 
        X_s: np.ndarray, 
        fit: bool = False
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
    
    def _extract_predictions(self, raw_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        
        results = pd.DataFrame({
            'beta': betas_full,
            'alpha': alphas_full,
            'stock_return': stock_returns,
            'market_return': market_returns,
        })
        
        if dates is not None:
            results['date'] = dates
            results = results[['date', 'beta', 'alpha', 'stock_return', 'market_return']]
        
        return results
    
    def _fit_model(
        self, 
        X_m: np.ndarray, 
        X_s: np.ndarray, 
        X_curr: np.ndarray, 
        y: np.ndarray
    ) -> None:
        """Fit model on entire dataset (for simple fit without walk-forward)."""
        from grubeta.models import GRUBetaModel
        
        self.model_ = GRUBetaModel(self.config).build(
            self._n_market_features, self._n_stock_features
        )
        
        X_m_norm, X_s_norm = self._normalize_batch(X_m, X_s, fit=True)
        
        self.model_.fit(
            [X_m_norm, X_s_norm, X_curr],
            y,
            epochs=self.config.epochs_init,
            batch_size=self.config.batch_size,
            verbose=1 if self.config.verbose >= 1 else 0
        )
    
    @staticmethod
    def _clean_data(X: np.ndarray) -> np.ndarray:
        """Clean data by handling inf/nan values."""
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X, nan=0.0)
        return np.clip(X, -100.0, 100.0)
    
    @staticmethod
    def _to_array(x: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(x, pd.Series):
            return x.values
        return np.asarray(x)
    
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
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = results['date'] if 'date' in results.columns else results.index
        ax.plot(x, results['beta'], label='GRU Dynamic Beta', 
                color='#2c3e50', linewidth=1.5)
        ax.axhline(1.0, color='#e74c3c', linestyle='--', alpha=0.5, 
                   label='Market Beta (1.0)')
        
        ax.set_title(title or 'Dynamic Beta Evolution')
        ax.set_xlabel('Date')
        ax.set_ylabel('Beta')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
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
        
        # Save Keras model
        self.model_.save(path / "model.h5")
        
        # Save scalers and config
        with open(path / "artifacts.pkl", 'wb') as f:
            pickle.dump({
                'scaler_market': self.scaler_market_,
                'scaler_stock': self.scaler_stock_,
                'config': self.config,
                'n_market_features': self._n_market_features,
                'n_stock_features': self._n_stock_features,
            }, f)
        
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
        with open(path / "artifacts.pkl", 'rb') as f:
            artifacts = pickle.load(f)
        
        # Create instance
        instance = cls(config=artifacts['config'])
        instance.scaler_market_ = artifacts['scaler_market']
        instance.scaler_stock_ = artifacts['scaler_stock']
        instance._n_market_features = artifacts['n_market_features']
        instance._n_stock_features = artifacts['n_stock_features']
        
        # Load Keras model
        from grubeta.models import GRUBetaModel
        instance.model_ = keras.models.load_model(
            path / "model.h5",
            custom_objects=GRUBetaModel.get_custom_objects(instance.config)
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
