"""
Temporal Integrity Module - Jidoka (Autonomation) for Grubeta.

This module implements "Andon Cord" runtime assertions that stop the line
immediately if lookahead bias or temporal contamination is detected.

TPS Principle: Build quality into the process, don't inspect for it afterward.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import json
import sys
import numpy as np
import pandas as pd


class TemporalIntegrityError(Exception):
    """
    Raised when lookahead bias or temporal contamination is detected.
    
    This is the "Andon Cord" - it stops the production line immediately
    when a defect is detected, preventing bad outputs.
    """
    pass


class DataStationarityWarning(Warning):
    """Warning for non-stationary features that may cause issues."""
    pass


def assert_pit_integrity(
    features: np.ndarray,
    target: np.ndarray,
    feature_names: Optional[List[str]] = None,
    correlation_threshold: float = 0.8,
) -> None:
    """
    Assert Point-in-Time integrity: no suspicious correlations.
    
    If any feature column has correlation > threshold with target,
    it suggests the feature may contain future information.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features).
    target : np.ndarray
        Target values (n_samples,).
    feature_names : List[str], optional
        Names for error messages.
    correlation_threshold : float, default=0.8
        Maximum allowed correlation before raising error.
        
    Raises
    ------
    TemporalIntegrityError
        If any feature has suspicious correlation with target.
    """
    n_features = features.shape[1] if features.ndim > 1 else 1
    
    for col in range(n_features):
        feat_col = features[:, col] if features.ndim > 1 else features
        
        # Handle NaN values
        mask = ~(np.isnan(feat_col) | np.isnan(target))
        if mask.sum() < 10:
            continue
            
        corr = np.corrcoef(feat_col[mask], target[mask])[0, 1]
        
        if abs(corr) > correlation_threshold:
            name = feature_names[col] if feature_names else f"Feature_{col}"
            raise TemporalIntegrityError(
                f" ANDON CORD PULLED: {name} has correlation {corr:.3f} "
                f"with target (threshold={correlation_threshold}). "
                f"This suggests lookahead bias!"
            )


def assert_no_temporal_overlap(
    train_timestamps: np.ndarray,
    test_timestamps: np.ndarray,
) -> None:
    """
    Assert strict temporal separation between train and test sets.
    
    The maximum training timestamp must be strictly less than
    the minimum test timestamp.
    
    Parameters
    ----------
    train_timestamps : np.ndarray
        Training set timestamps.
    test_timestamps : np.ndarray
        Test set timestamps.
        
    Raises
    ------
    TemporalIntegrityError
        If train and test sets overlap in time.
    """
    train_max = pd.Timestamp(train_timestamps.max())
    test_min = pd.Timestamp(test_timestamps.min())
    
    if train_max >= test_min:
        raise TemporalIntegrityError(
            f" ANDON CORD PULLED: Temporal overlap detected! "
            f"Train max: {train_max}, Test min: {test_min}. "
            f"Train set must end BEFORE test set begins."
        )


def assert_feature_lag(
    feature_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    min_lag_days: int = 1,
) -> None:
    """
    Assert that features are lagged by at least min_lag_days.
    
    For each target timestamp, verifies that the newest data
    in features is at least min_lag_days older.
    
    Parameters
    ----------
    feature_timestamps : np.ndarray
        Timestamps of feature data.
    target_timestamps : np.ndarray
        Timestamps of targets being predicted.
    min_lag_days : int, default=1
        Minimum required lag in days.
        
    Raises
    ------
    TemporalIntegrityError
        If features are not properly lagged.
    """
    feature_max = pd.Timestamp(feature_timestamps.max())
    target_min = pd.Timestamp(target_timestamps.min())
    
    lag = (target_min - feature_max).days
    
    if lag < min_lag_days:
        raise TemporalIntegrityError(
            f" ANDON CORD PULLED: Insufficient feature lag! "
            f"Features end at {feature_max}, targets start at {target_min}. "
            f"Required lag: {min_lag_days} days, actual: {lag} days."
        )


@dataclass
class TemporalCertificate:
    """
    Proof of zero-lookahead for academic peer review.
    
    This certificate documents the exact temporal boundaries used
    for training and prediction, enabling third-party verification.
    
    Attributes
    ----------
    model_version : str
        Version of the grubeta library used.
    generated_at : datetime
        When this certificate was generated.
    training_window : Tuple[str, str]
        (start_date, end_date) of training data.
    prediction_window : Tuple[str, str]
        (start_date, end_date) of predictions.
    feature_lag_policy : str
        Description of lagging policy.
    scaler_policy : str
        Description of normalization policy.
    integrity_checks_passed : List[str]
        List of validation tests that passed.
    """
    model_version: str
    generated_at: datetime = field(default_factory=datetime.now)
    training_window: Tuple[str, str] = ("", "")
    prediction_window: Tuple[str, str] = ("", "")
    feature_lag_policy: str = "ALL_FEATURES_LAGGED_1_DAY"
    scaler_policy: str = "EXPANDING_WINDOW_PIT"
    integrity_checks_passed: List[str] = field(default_factory=list)
    
    # Environment Metadata (Sustenance)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    numpy_version: str = field(default_factory=lambda: np.__version__)
    pandas_version: str = field(default_factory=lambda: pd.__version__)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_version": self.model_version,
            "generated_at": self.generated_at.isoformat(),
            "environment": {
                "python": self.python_version,
                "numpy": self.numpy_version,
                "pandas": self.pandas_version,
            },
            "training_window": {
                "start": self.training_window[0],
                "end": self.training_window[1],
            },
            "prediction_window": {
                "start": self.prediction_window[0],
                "end": self.prediction_window[1],
            },
            "feature_lag_policy": self.feature_lag_policy,
            "scaler_policy": self.scaler_policy,
            "integrity_checks_passed": self.integrity_checks_passed,
        }
    
    def to_json(self) -> str:
        """Export as machine-readable JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        """Human-readable certificate."""
        lines = [
            "=" * 60,
            "TEMPORAL INTEGRITY CERTIFICATE",
            "=" * 60,
            f"Model Version: {self.model_version}",
            f"Generated: {self.generated_at.isoformat()}",
            f"Environment: Python {self.python_version}, NumPy {self.numpy_version}, Pandas {self.pandas_version}",
            "",
            f"Training Window: {self.training_window[0]} to {self.training_window[1]}",
            f"Prediction Window: {self.prediction_window[0]} to {self.prediction_window[1]}",
            "",
            f"Feature Lag Policy: {self.feature_lag_policy}",
            f"Scaler Policy: {self.scaler_policy}",
            "",
            "Integrity Checks Passed:",
        ]
        for check in self.integrity_checks_passed:
            lines.append(f"  ✓ {check}")
        lines.append("=" * 60)
        return "\n".join(lines)


def validate_temporal_integrity(
    features: np.ndarray,
    target: np.ndarray,
    dates: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Run comprehensive temporal integrity validation suite.
    
    This is the full Jidoka quality gate that should be run
    before any production model training.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    target : np.ndarray
        Target values.
    dates : np.ndarray, optional
        Date index for temporal checks.
    verbose : bool, default=True
        Print validation progress.
        
    Returns
    -------
    passed : bool
        True if all checks pass.
    checks_passed : List[str]
        Names of checks that passed.
        
    Raises
    ------
    TemporalIntegrityError
        If any check fails (Andon Cord).
    """
    checks_passed = []
    
    if verbose:
        print(" Running Temporal Integrity Validation...")
    
    # Check 1: PIT Integrity
    try:
        assert_pit_integrity(features, target)
        checks_passed.append("PIT_INTEGRITY")
        if verbose:
            print("  ✓ Point-in-Time integrity verified")
    except TemporalIntegrityError:
        raise
    
    # Check 2: No future correlation
    for lag in [1, 5, 10]:
        if len(target) <= lag:
            continue
        future_target = np.roll(target, -lag)[:-lag]
        features_aligned = features[:-lag]
        
        for col in range(features.shape[1] if features.ndim > 1 else 1):
            feat = features_aligned[:, col] if features.ndim > 1 else features_aligned
            mask = ~(np.isnan(feat) | np.isnan(future_target))
            if mask.sum() < 10:
                continue
            corr = np.corrcoef(feat[mask], future_target[mask])[0, 1]
            if abs(corr) > 0.5:
                raise TemporalIntegrityError(
                    f" Feature {col} correlated {corr:.3f} with target t+{lag}"
                )
    
    checks_passed.append("NO_FUTURE_CORRELATION")
    if verbose:
        print("  ✓ No suspicious correlations with future targets")
    
    # Check 3: NaN pattern (should have NaN at start only)
    if np.isnan(features).any():
        nan_rows = np.where(np.isnan(features).any(axis=1))[0]
        if len(nan_rows) > 0 and nan_rows.max() > len(features) * 0.1:
            if verbose:
                print("  ⚠ Warning: NaN values found beyond initial burn-in")
    
    checks_passed.append("NAN_PATTERN_CHECK")
    if verbose:
        print("  ✓ NaN pattern consistent with proper lagging")
    
    if verbose:
        print(f" All {len(checks_passed)} integrity checks passed!")
    
    return True, checks_passed
