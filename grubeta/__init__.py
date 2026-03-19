"""
grubeta — Dynamic Beta Estimation
==================================

Estimate how a stock's market sensitivity (beta) changes over time.

Quick Start
-----------
>>> from grubeta import estimate_beta
>>> result = estimate_beta("AAPL", "SPY")
>>> print(result["summary"])

For advanced usage:
>>> from grubeta import DynamicBeta, DataPreprocessor, FeatureConfig
>>> model = DynamicBeta(lookback=90)
>>> results = model.fit_predict(stock_returns, market_returns)
"""

__version__ = "0.1.3"
__author__ = "Ahmet Selim Yılmaz"
__license__ = "MIT"

# High-level convenience API (recommended for most users)
from grubeta.convenience import compare_betas, estimate_beta, quick_report
from grubeta.presets import get_preset, list_presets

# Core API (for advanced usage)
from grubeta.core import DynamicBeta, DynamicBetaConfig
from grubeta.evaluation import BetaEvaluator
from grubeta.models import GRUBetaModel, TFEnvironment, extract_last_step
from grubeta.preprocessing import DataPreprocessor, FeatureConfig
from grubeta.utils import rolling_ols_beta, validate_no_lookahead

__all__ = [
    # Convenience API
    "estimate_beta",
    "compare_betas",
    "quick_report",
    "get_preset",
    "list_presets",
    # Core API
    "DynamicBeta",
    "DynamicBetaConfig",
    # Preprocessing
    "DataPreprocessor",
    "FeatureConfig",
    # Evaluation
    "BetaEvaluator",
    # Utilities
    "validate_no_lookahead",
    "rolling_ols_beta",
    # Advanced
    "GRUBetaModel",
    "TFEnvironment",
    "extract_last_step",
]
