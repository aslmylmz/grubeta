"""
GRU Dynamic Beta - Time-Varying Systematic Risk Estimation
==========================================================

A neural network-based library for estimating time-varying beta within 
the CAPM framework using Gated Recurrent Units (GRU).

Quick Start
-----------
>>> from grubeta import DynamicBeta
>>> model = DynamicBeta()
>>> betas = model.fit_predict(stock_returns, market_returns)

For advanced usage with macroeconomic features:
>>> from grubeta import DynamicBeta, DataPreprocessor
>>> preprocessor = DataPreprocessor()
>>> features = preprocessor.prepare(stock_df, market_df, macro_df)
>>> model = DynamicBeta(lookback=90, use_macro=True)
>>> betas, alphas = model.fit_predict(**features)

Key Features
------------
- Walk-forward validation with anchored expanding window
- Composite loss function (accuracy + stability + sparsity)
- Multiple input pathways (simple returns or full feature engineering)
- Lookahead bias prevention built-in
- GPU acceleration support

Author: Ahmet Selim Yılmaz
License: MIT
"""

__version__ = "0.1.1"
__author__ = "Ahmet Selim Yılmaz"
__license__ = "MIT"

from grubeta.core import DynamicBeta, DynamicBetaConfig
from grubeta.preprocessing import DataPreprocessor, FeatureConfig
from grubeta.models import GRUBetaModel
from grubeta.evaluation import BetaEvaluator
from grubeta.utils import validate_no_lookahead, rolling_ols_beta

__all__ = [
    # Core API
    "DynamicBeta",
    "DynamicBetaConfig",
    # Preprocessing
    "DataPreprocessor", 
    "FeatureConfig",
    # Advanced
    "GRUBetaModel",
    "BetaEvaluator",
    # Utilities
    "validate_no_lookahead",
    "rolling_ols_beta",
]
