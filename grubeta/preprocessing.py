"""
Data preprocessing and feature engineering for dynamic beta estimation.

This module provides tools for preparing financial data for the GRU-based
dynamic beta model, including technical indicators, macro features, and
proper handling of lookahead bias.
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

# Optional dependency for technical analysis
try:
    import ta

    HAS_TA = True
except ImportError:
    HAS_TA = False


@dataclass
class FeatureConfig:
    """
    Configuration for feature engineering.

    Controls which features are computed and how lookahead bias
    is prevented through lagging.

    Parameters
    ----------
    lag_features : bool, default=True
        Whether to lag all features by 1 day to prevent lookahead bias.
        Essential for proper model evaluation.

    include_technicals : bool, default=True
        Include technical analysis indicators (RSI, MACD, etc.).

    include_macro : bool, default=False
        Include macroeconomic features (requires macro data input).

    include_volume : bool, default=True
        Include volume-based indicators.

    include_calendar : bool, default=True
        Include calendar features (day of week, month end, etc.).

    ma_windows : List[int]
        Moving average windows to compute.

    volatility_windows : List[int]
        Volatility calculation windows.

    roc_periods : List[int]
        Rate of change periods.

    Examples
    --------
    >>> config = FeatureConfig(
    ...     include_macro=True,
    ...     ma_windows=[10, 20, 50],
    ... )
    >>> preprocessor = DataPreprocessor(config)
    """

    lag_features: bool = True
    include_technicals: bool = True
    include_macro: bool = False
    include_volume: bool = True
    include_calendar: bool = True

    ma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    roc_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20, 60])


class DataPreprocessor:
    """
    Feature engineering pipeline for dynamic beta estimation.

    Handles data cleaning, technical indicator computation, and proper
    lagging to prevent lookahead bias in sequential models.

    Parameters
    ----------
    config : FeatureConfig, optional
        Feature engineering configuration. Uses defaults if not provided.

    Attributes
    ----------
    config : FeatureConfig
        Current configuration.
    macro_data : pd.DataFrame or None
        Loaded macroeconomic data.
    market_features_ : List[str]
        Names of computed market features.
    stock_features_ : List[str]
        Names of computed stock features.

    Examples
    --------
    Simple preprocessing with OHLCV data:

    >>> from grubeta import DataPreprocessor
    >>>
    >>> prep = DataPreprocessor()
    >>> result = prep.prepare_simple(
    ...     stock_returns=stock_df['Close'].pct_change(),
    ...     market_returns=market_df['Close'].pct_change()
    ... )

    Full preprocessing with OHLCV and macro data:

    >>> prep = DataPreprocessor(FeatureConfig(include_macro=True))
    >>> result = prep.prepare(
    ...     stock_df=stock_ohlcv,
    ...     market_df=market_ohlcv,
    ...     macro_df=macro_data
    ... )
    >>>
    >>> # Use with DynamicBeta
    >>> model = DynamicBeta()
    >>> results = model.fit_predict(**result)

    Notes
    -----
    The preprocessor enforces a strict no-lookahead policy by lagging
    all features by 1 day. When predicting return at time T, features
    contain data from T-1 and earlier ONLY.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.macro_data = None
        self.market_features_: List[str] = []
        self.stock_features_: List[str] = []

        if self.config.include_technicals and not HAS_TA:
            warnings.warn(
                "Technical analysis library 'ta' not installed. "
                "Install with: pip install ta. "
                "Falling back to basic features only."
            )

    def prepare_simple(
        self,
        stock_returns: Union[np.ndarray, pd.Series],
        market_returns: Union[np.ndarray, pd.Series],
        dates: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    ) -> Dict:
        """
        Prepare data using only returns (minimal preprocessing).

        This is the simplest entry point - just provide stock and market
        returns and get back everything needed for DynamicBeta.

        Parameters
        ----------
        stock_returns : array-like
            Stock return series.
        market_returns : array-like
            Market return series.
        dates : array-like, optional
            Date index.

        Returns
        -------
        result : dict
            Dictionary with keys:
            - 'stock_returns': cleaned stock returns
            - 'market_returns': cleaned market returns
            - 'dates': date index (if provided)

        Examples
        --------
        >>> prep = DataPreprocessor()
        >>> data = prep.prepare_simple(stock_ret, market_ret)
        >>> model = DynamicBeta()
        >>> results = model.fit_predict(**data)
        """
        # Convert to arrays
        stock_returns = self._to_array(stock_returns)
        market_returns = self._to_array(market_returns)

        # Clean data
        stock_returns = self._clean_returns(stock_returns)
        market_returns = self._clean_returns(market_returns)

        result = {
            "stock_returns": stock_returns,
            "market_returns": market_returns,
        }

        if dates is not None:
            result["dates"] = dates

        return result

    def prepare(
        self,
        stock_df: pd.DataFrame,
        market_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Full preprocessing pipeline with feature engineering.

        Computes technical indicators, handles macro data, and properly
        lags all features to prevent lookahead bias.

        Parameters
        ----------
        stock_df : pd.DataFrame
            Stock OHLCV data with columns: Date, Open, High, Low, Close, Volume.
        market_df : pd.DataFrame
            Market/index OHLCV data with same structure.
        macro_df : pd.DataFrame, optional
            Macroeconomic data with Date column plus indicator columns.

        Returns
        -------
        result : dict
            Dictionary with keys:
            - 'stock_returns': stock return series
            - 'market_returns': market return series
            - 'market_features': 3D array (n_samples, lookback, n_market_features)
            - 'stock_features': 3D array (n_samples, lookback, n_stock_features)
            - 'dates': date index
            - 'feature_names': dict with market and stock feature names

        Raises
        ------
        ValueError
            If required columns are missing from input DataFrames.

        Examples
        --------
        >>> prep = DataPreprocessor(FeatureConfig(include_macro=True))
        >>> result = prep.prepare(stock_df, market_df, macro_df)
        >>>
        >>> # Access individual components
        >>> print(f"Market features: {len(result['feature_names']['market'])}")
        >>> print(f"Stock features: {len(result['feature_names']['stock'])}")
        """
        # Validate inputs
        stock_df = self._validate_ohlcv(stock_df, "stock")
        market_df = self._validate_ohlcv(market_df, "market")

        # Load macro data if provided
        if macro_df is not None:
            self.macro_data = self._prepare_macro(macro_df)

        # Calculate features
        market_features_df = self._calculate_market_features(market_df)
        stock_features_df = self._calculate_stock_features(stock_df, market_features_df)

        # Merge and finalize
        merged_df, market_cols, stock_cols = self._merge_and_finalize(
            stock_features_df, market_features_df
        )

        # Store feature names
        self.market_features_ = market_cols
        self.stock_features_ = stock_cols

        # Extract arrays
        result = {
            "stock_returns": merged_df["stock_return"].values,
            "market_returns": merged_df["market_return"].values,
            "market_features": merged_df[market_cols].values,
            "stock_features": merged_df[stock_cols].values,
            "dates": merged_df["Date"].values,
            "feature_names": {
                "market": market_cols,
                "stock": stock_cols,
            },
        }

        return result

    def _validate_ohlcv(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Validate and standardize OHLCV DataFrame."""
        df = df.copy()

        # Column mapping for common variations
        column_variations = {
            "Date": ["Date", "date", "Tarih", "datetime", "time"],
            "Open": ["Open", "open", "Açılış", "OPEN"],
            "High": ["High", "high", "Yüksek", "HIGH"],
            "Low": ["Low", "low", "Düşük", "LOW"],
            "Close": ["Close", "close", "Price", "Adj Close", "CLOSE"],
            "Volume": ["Volume", "volume", "Vol", "Vol.", "VOLUME"],
        }

        # Map columns
        for standard, variations in column_variations.items():
            for col in df.columns:
                if col.strip() in variations:
                    df = df.rename(columns={col: standard})
                    break

        # Check required columns
        required = ["Date", "Close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in {name} data: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        # Parse dates
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

        # Clean numeric columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(",", "")
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Add synthetic volume if missing
        if "Volume" not in df.columns:
            df["Volume"] = 1_000_000

        # Sort and deduplicate
        df = df.sort_values("Date")
        df = df.drop_duplicates(subset="Date", keep="last")
        df = df.dropna(subset=["Close"])

        return df

    def _prepare_macro(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare macroeconomic data."""
        df = macro_df.copy()

        if "Date" not in df.columns:
            raise ValueError("Macro data must have 'Date' column")

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        # Prefix columns if needed
        for col in df.columns:
            if col != "Date" and not col.startswith("macro_"):
                df = df.rename(columns={col: f"macro_{col}"})

        return df

    def _calculate_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market/index features."""
        df = df.copy()
        lag = 1 if self.config.lag_features else 0

        # Apply feature groups
        df = self._add_return_features(df, "Close", "market", lag)
        df = self._add_ma_features(df, "Close", "market", lag)
        df = self._add_volatility_features(df, "Close", "market", lag)
        df = self._add_momentum_features(df, "Close", "market", lag)
        df = self._add_distance_features(df, "Close", "market", lag)

        # Technical indicators (if ta library available)
        if HAS_TA and self.config.include_technicals:
            df = self._add_technical_indicators(df, "market", lag)

        # Volume features
        if self.config.include_volume and "Volume" in df.columns:
            df = self._add_volume_features(df, "Volume", "market", lag)

        # Calendar features (known at time T, no lag needed)
        if self.config.include_calendar:
            df = self._add_calendar_features(df, "Date", "market")

        # Merge macro data
        if self.macro_data is not None and self.config.include_macro:
            df = self._add_macro_features(df, "Date", lag)

        return df

    def _add_return_features(
        self, df: pd.DataFrame, col_name: str, prefix: str, lag: int
    ) -> pd.DataFrame:
        """Add percentage and log returns."""
        # Returns themselves are often targets, so they are NOT lagged by default here.
        # If used as features, they should be lagged by the caller or distinct feature column.
        # However, purely for feature engineering (like inputs to model), we might want lags.
        # In this architecture, raw returns are usually targets.
        
        # Note: Original code did NOT lag these. Keeping behavior consistent.
        df[f"{prefix}_return"] = df[col_name].pct_change()
        df[f"{prefix}_log_return"] = np.log(df[col_name] / df[col_name].shift(1))
        return df

    def _add_ma_features(
        self, df: pd.DataFrame, col_name: str, prefix: str, lag: int
    ) -> pd.DataFrame:
        """Add Moving Average ratios."""
        for w in self.config.ma_windows:
            col = f"{prefix}_ma_{w}_ratio"
            df[col] = df[col_name] / df[col_name].rolling(w).mean()
            if lag:
                df[col] = df[col].shift(lag)
        return df

    def _add_volatility_features(
        self, df: pd.DataFrame, col_name: str, prefix: str, lag: int
    ) -> pd.DataFrame:
        """Add realized volatility features."""
        log_ret = np.log(df[col_name] / df[col_name].shift(1))
        for w in self.config.volatility_windows:
            col = f"{prefix}_volatility_{w}d"
            df[col] = log_ret.rolling(w).std() * np.sqrt(252)
            if lag:
                df[col] = df[col].shift(lag)
        return df

    def _add_momentum_features(
        self, df: pd.DataFrame, col_name: str, prefix: str, lag: int
    ) -> pd.DataFrame:
        """Add Rate of Change (Momentum) features."""
        for p in self.config.roc_periods:
            col = f"{prefix}_roc_{p}d"
            df[col] = df[col_name].pct_change(p)
            if lag:
                df[col] = df[col].shift(lag)
        return df

    def _add_distance_features(
        self, df: pd.DataFrame, col_name: str, prefix: str, lag: int
    ) -> pd.DataFrame:
        """Add distance from 52-week highs/lows."""
        col_high = f"{prefix}_distance_52w_high"
        col_low = f"{prefix}_distance_52w_low"
        
        df[col_high] = (df[col_name] / df[col_name].rolling(252).max()) - 1
        df[col_low] = (df[col_name] / df[col_name].rolling(252).min()) - 1
        
        if lag:
            df[col_high] = df[col_high].shift(lag)
            df[col_low] = df[col_low].shift(lag)
        return df

    def _add_volume_features(
        self, df: pd.DataFrame, col_name: str, prefix: str, lag: int
    ) -> pd.DataFrame:
        """Add volume processing features."""
        col = f"{prefix}_volume_ratio"
        df[col] = df[col_name] / df[col_name].rolling(20).mean()
        if lag:
            df[col] = df[col].shift(lag)
        return df

    def _add_calendar_features(
        self, df: pd.DataFrame, date_col: str, prefix: str
    ) -> pd.DataFrame:
        """Add calendar/date features."""
        # Known at time T, so strictly no lag needed even if others are lagged.
        dates = df[date_col]
        df[f"{prefix}_day_of_week"] = dates.dt.dayofweek
        df[f"{prefix}_is_month_end"] = dates.dt.is_month_end.astype(int)
        df[f"{prefix}_is_month_start"] = dates.dt.is_month_start.astype(int)
        df[f"{prefix}_day_of_month"] = dates.dt.day
        df[f"{prefix}_week_of_year"] = dates.dt.isocalendar().week.astype(int)
        return df

    def _add_macro_features(
        self, df: pd.DataFrame, date_col: str, lag: int
    ) -> pd.DataFrame:
        """Merge and process macro-economic features."""
        if self.macro_data is None:
            return df
            
        df = pd.merge(df, self.macro_data, on=date_col, how="left")
        for col in [c for c in df.columns if c.startswith("macro_")]:
            df[col] = df[col].ffill()
            if lag:
                df[col] = df[col].shift(lag)
        return df

    def _calculate_stock_features(
        self, stock_df: pd.DataFrame, market_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate stock-specific features."""
        df = stock_df.copy()
        lag = 1 if self.config.lag_features else 0

        # Merge market data
        market_cols = ["Date", "Close", "Volume", "market_return", "market_log_return"]
        market_cols = [c for c in market_cols if c in market_df.columns]

        df = pd.merge(
            df, market_df[market_cols], on="Date", how="left", suffixes=("", "_market")
        )

        # Copy all market features
        for col in market_df.columns:
            if col.startswith(("market_", "macro_")) and col not in df.columns:
                df = pd.merge(df, market_df[["Date", col]], on="Date", how="left")
                df[col] = df[col].ffill()

        # Stock returns (targets - not lagged)
        df["stock_return"] = df["Close"].pct_change()
        df["stock_log_return"] = np.log(df["Close"] / df["Close"].shift(1))

        # MA ratios
        for w in [20, 50, 200]:
            col = f"stock_ma_{w}_ratio"
            df[col] = df["Close"] / df["Close"].rolling(w).mean()
            if lag:
                df[col] = df[col].shift(lag)

        # Volatility
        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        df["stock_volatility_20d"] = log_ret.rolling(20).std() * np.sqrt(252)
        if lag:
            df["stock_volatility_20d"] = df["stock_volatility_20d"].shift(lag)

        # Rate of change
        for p in [20, 60]:
            col = f"stock_roc_{p}d"
            df[col] = df["Close"].pct_change(p)
            if lag:
                df[col] = df[col].shift(lag)

        # Distance from 52-week high
        df["stock_distance_52w_high"] = (
            df["Close"] / df["Close"].rolling(252).max()
        ) - 1
        if lag:
            df["stock_distance_52w_high"] = df["stock_distance_52w_high"].shift(lag)

        # Technical indicators
        if HAS_TA and self.config.include_technicals:
            df = self._add_technical_indicators(df, "stock", lag)

        # Volume features
        if self.config.include_volume and "Volume" in df.columns:
            df["stock_volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
            if lag:
                df["stock_volume_ratio"] = df["stock_volume_ratio"].shift(lag)

        # Relative features (vs market)
        if "Close_market" in df.columns:
            # Rolling correlation
            df["stock_rolling_correlation_50d"] = (
                df["stock_log_return"].rolling(50).corr(df["market_log_return"])
            )
            if lag:
                df["stock_rolling_correlation_50d"] = df[
                    "stock_rolling_correlation_50d"
                ].shift(lag)

            # Traditional beta
            cov = df["stock_log_return"].rolling(252).cov(df["market_log_return"])
            var = df["market_log_return"].rolling(252).var()
            df["stock_beta_traditional_252d"] = cov / var
            if lag:
                df["stock_beta_traditional_252d"] = df[
                    "stock_beta_traditional_252d"
                ].shift(lag + 1)

        return df

    def _add_technical_indicators(
        self, df: pd.DataFrame, prefix: str, lag: int
    ) -> pd.DataFrame:
        """Add technical analysis indicators using ta library."""
        if not HAS_TA:
            return df

        # RSI
        rsi = ta.momentum.RSIIndicator(df["Close"])
        df[f"{prefix}_rsi"] = rsi.rsi()
        if lag:
            df[f"{prefix}_rsi"] = df[f"{prefix}_rsi"].shift(lag)

        # MACD
        macd = ta.trend.MACD(df["Close"])
        df[f"{prefix}_macd_diff"] = macd.macd_diff()
        if lag:
            df[f"{prefix}_macd_diff"] = df[f"{prefix}_macd_diff"].shift(lag)

        # ADX (if OHLC available)
        if all(c in df.columns for c in ["High", "Low"]):
            adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"])
            df[f"{prefix}_adx"] = adx.adx()
            if lag:
                df[f"{prefix}_adx"] = df[f"{prefix}_adx"].shift(lag)

            # CCI
            cci = ta.trend.CCIIndicator(df["High"], df["Low"], df["Close"])
            df[f"{prefix}_cci"] = cci.cci()
            if lag:
                df[f"{prefix}_cci"] = df[f"{prefix}_cci"].shift(lag)

            # ATR
            atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"])
            df[f"{prefix}_atr_ratio"] = atr.average_true_range() / df["Close"]
            if lag:
                df[f"{prefix}_atr_ratio"] = df[f"{prefix}_atr_ratio"].shift(lag)

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df["Close"])
            df[f"{prefix}_bb_width"] = bb.bollinger_wband()
            df[f"{prefix}_bb_position"] = bb.bollinger_pband()
            if lag:
                df[f"{prefix}_bb_width"] = df[f"{prefix}_bb_width"].shift(lag)
                df[f"{prefix}_bb_position"] = df[f"{prefix}_bb_position"].shift(lag)

        return df

    def _merge_and_finalize(
        self,
        stock_df: pd.DataFrame,
        market_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Finalize feature set and identify columns."""
        # Identify feature columns
        market_features = [
            c for c in stock_df.columns if c.startswith(("market_", "macro_"))
        ]
        stock_features = [
            c
            for c in stock_df.columns
            if c.startswith("stock_") and c not in ["stock_return", "stock_log_return"]
        ]

        # Exclude returns from features
        market_features = [c for c in market_features if "return" not in c.lower()]

        # Required columns
        required = ["Date", "stock_return", "market_return"]
        all_cols = required + market_features + stock_features

        # Keep only existing columns
        existing = [c for c in all_cols if c in stock_df.columns]
        df = stock_df[existing].copy()

        # Forward fill (only)
        df = df.ffill()

        # Drop rows with NaN returns
        df = df.dropna(subset=["stock_return", "market_return"])

        # Fill remaining NaNs with 0
        df = df.fillna(0)

        # Update feature lists to only include existing
        market_features = [c for c in market_features if c in df.columns]
        stock_features = [c for c in stock_features if c in df.columns]

        return df, market_features, stock_features

    def save_processed_data(
        self, result: Dict, output_dir: Union[str, Path], name: str = "processed"
    ) -> None:
        """
        Save processed data to disk.

        Parameters
        ----------
        result : dict
            Output from prepare() or prepare_simple().
        output_dir : str or Path
            Output directory.
        name : str, default="processed"
            Base filename.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        df = pd.DataFrame(
            {
                "Date": result.get("dates", range(len(result["stock_returns"]))),
                "stock_return": result["stock_returns"],
                "market_return": result["market_returns"],
            }
        )

        if "market_features" in result and result["market_features"] is not None:
            for i, col in enumerate(result["feature_names"]["market"]):
                df[col] = result["market_features"][:, i]

        if "stock_features" in result and result["stock_features"] is not None:
            for i, col in enumerate(result["feature_names"]["stock"]):
                df[col] = result["stock_features"][:, i]

        df.to_csv(output_dir / f"{name}.csv", index=False)

        # Save feature names
        if "feature_names" in result:
            with open(output_dir / f"{name}_features.json", "w") as f:
                json.dump(result["feature_names"], f, indent=2)

    @staticmethod
    def _to_array(x: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(x, pd.Series):
            return cast(np.ndarray, x.values)
        return cast(np.ndarray, np.asarray(x))

    @staticmethod
    def _clean_returns(returns: np.ndarray) -> np.ndarray:
        """Clean return series."""
        returns = np.where(np.isinf(returns), np.nan, returns)
        returns = np.nan_to_num(returns, nan=0.0)
        return cast(np.ndarray, np.clip(returns, -100.0, 100.0))  # Cap at ±10000% (was ±100%)


def load_processed_data(filepath: Union[str, Path]) -> Dict:
    """
    Load previously processed data.

    Parameters
    ----------
    filepath : str or Path
        Path to processed CSV file.

    Returns
    -------
    result : dict
        Dictionary compatible with DynamicBeta.fit_predict().
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    # Load feature names if available
    features_path = filepath.with_suffix(".json").with_name(
        filepath.stem + "_features.json"
    )

    market_cols = []
    stock_cols = []

    if features_path.exists():
        with open(features_path, "r") as f:
            features = json.load(f)
            market_cols = features.get("market", [])
            stock_cols = features.get("stock", [])
    else:
        # Infer from column names
        market_cols = [
            c
            for c in df.columns
            if c.startswith(("market_", "macro_")) and "return" not in c.lower()
        ]
        stock_cols = [
            c
            for c in df.columns
            if c.startswith("stock_") and c not in ["stock_return"]
        ]

    result = {
        "stock_returns": df["stock_return"].values,
        "market_returns": df["market_return"].values,
    }

    if "Date" in df.columns:
        result["dates"] = df["Date"].values

    if market_cols:
        result["market_features"] = df[market_cols].values

    if stock_cols:
        result["stock_features"] = df[stock_cols].values

    result["feature_names"] = {"market": market_cols, "stock": stock_cols}

    return result
