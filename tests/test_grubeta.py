"""
Test suite for GRU Dynamic Beta.

Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from grubeta import DynamicBeta, DynamicBetaConfig
from grubeta.preprocessing import DataPreprocessor, FeatureConfig
from grubeta.evaluation import BetaEvaluator, compute_rolling_ols_beta, compute_static_beta
from grubeta.utils import validate_no_lookahead, rolling_ols_beta, align_series


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_returns():
    """Generate sample return data."""
    np.random.seed(42)
    n = 800
    market = np.random.randn(n) * 0.01
    beta = 1.2
    stock = beta * market + np.random.randn(n) * 0.005
    return stock, market


@pytest.fixture
def sample_returns_with_dates():
    """Generate sample return data with dates."""
    np.random.seed(42)
    n = 800
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    market = np.random.randn(n) * 0.01
    beta = 1.2
    stock = beta * market + np.random.randn(n) * 0.005
    return stock, market, dates


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    
    # Random walk for close prices
    returns = np.random.randn(n) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_ = (close + np.random.randn(n) * 0.5).clip(low, high)
    volume = np.random.randint(100000, 1000000, n)
    
    return pd.DataFrame({
        'Date': dates,
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    })


# =============================================================================
# Test DynamicBetaConfig
# =============================================================================

class TestDynamicBetaConfig:
    """Tests for DynamicBetaConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DynamicBetaConfig()
        assert config.lookback == 90
        assert config.initial_train_size == 500
        assert config.lambda_beta == 0.05
        assert config.lambda_alpha == 0.5
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = DynamicBetaConfig(
            lookback=60,
            lambda_beta=0.1,
            gru_units=64
        )
        assert config.lookback == 60
        assert config.lambda_beta == 0.1
        assert config.gru_units == 64
    
    def test_validation_lookback(self):
        """Test lookback validation."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DynamicBetaConfig(lookback=5)
    
    def test_validation_learning_rate(self):
        """Test learning rate validation."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DynamicBetaConfig(learning_rate=2.0)


# =============================================================================
# Test DynamicBeta
# =============================================================================

class TestDynamicBeta:
    """Tests for DynamicBeta main class."""
    
    def test_init_default(self):
        """Test default initialization."""
        model = DynamicBeta()
        assert model.config.lookback == 90
        assert not model.is_fitted_
    
    def test_init_with_kwargs(self):
        """Test initialization with keyword arguments."""
        model = DynamicBeta(lookback=60, lambda_beta=0.1)
        assert model.config.lookback == 60
        assert model.config.lambda_beta == 0.1
    
    def test_init_with_config(self):
        """Test initialization with config object."""
        config = DynamicBetaConfig(lookback=45)
        model = DynamicBeta(config=config)
        assert model.config.lookback == 45
    
    def test_fit_predict_shape(self, sample_returns):
        """Test that fit_predict returns correct shape."""
        stock, market = sample_returns
        model = DynamicBeta(
            lookback=30, 
            initial_train_size=100,
            wf_step_size=50
        )
        results = model.fit_predict(stock, market)
        
        assert len(results) == len(stock)
        assert 'beta' in results.columns
        assert 'alpha' in results.columns
        assert 'stock_return' in results.columns
        assert 'market_return' in results.columns
    
    def test_fit_predict_with_dates(self, sample_returns_with_dates):
        """Test fit_predict with date index."""
        stock, market, dates = sample_returns_with_dates
        model = DynamicBeta(lookback=30, initial_train_size=100)
        results = model.fit_predict(stock, market, dates=dates)
        
        assert 'date' in results.columns
        assert len(results['date']) == len(dates)
    
    def test_fit_predict_nan_handling(self, sample_returns):
        """Test that initial period has NaN betas."""
        stock, market = sample_returns
        model = DynamicBeta(
            lookback=30, 
            initial_train_size=100,
            wf_step_size=50
        )
        results = model.fit_predict(stock, market)
        
        # First lookback + initial_train_size should be NaN
        expected_nan = 30 + 100
        assert np.isnan(results['beta'].iloc[:expected_nan]).all()
        assert not np.isnan(results['beta'].iloc[-1])
    
    def test_fit_predict_produces_valid_beta(self, sample_returns):
        """Test that estimated beta is finite and not all the same value."""
        stock, market = sample_returns
        
        model = DynamicBeta(
            lookback=30, 
            initial_train_size=100,
            wf_step_size=50
        )
        results = model.fit_predict(stock, market)
        
        valid_betas = results['beta'].dropna()
        
        # Check betas are finite
        assert np.isfinite(valid_betas).all()
        
        # Check there's some variation (not constant)
        assert valid_betas.std() > 0
        
        # Check betas are in a reasonable range (not exploding)
        assert valid_betas.min() > -10
        assert valid_betas.max() < 10
    
    def test_is_fitted_flag(self, sample_returns):
        """Test is_fitted_ flag."""
        stock, market = sample_returns
        model = DynamicBeta(lookback=30, initial_train_size=100)
        
        assert not model.is_fitted_
        model.fit_predict(stock, market)
        assert model.is_fitted_
    
    def test_predict_before_fit_raises(self, sample_returns):
        """Test that predict before fit raises error."""
        stock, market = sample_returns
        model = DynamicBeta()
        
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(stock, market)
    
    def test_insufficient_data_handling(self):
        """Test that very insufficient data is handled appropriately."""
        # With only 20 samples, should either raise or produce mostly NaN
        np.random.seed(42)
        stock = np.random.randn(20)
        market = np.random.randn(20)
        
        model = DynamicBeta(lookback=30, initial_train_size=100)
        
        # Either raises ValueError or produces all NaN betas
        try:
            results = model.fit_predict(stock, market)
            # If it doesn't raise, check that we get mostly NaN
            valid_count = (~np.isnan(results['beta'])).sum()
            assert valid_count == 0, "Expected all NaN betas for insufficient data"
        except (ValueError, IndexError):
            # This is also acceptable behavior
            pass
    
    def test_repr(self):
        """Test string representation."""
        model = DynamicBeta(lookback=60)
        repr_str = repr(model)
        
        assert "DynamicBeta" in repr_str
        assert "lookback=60" in repr_str
        assert "not fitted" in repr_str


# =============================================================================
# Test DataPreprocessor
# =============================================================================

class TestDataPreprocessor:
    """Tests for DataPreprocessor."""
    
    def test_prepare_simple(self, sample_returns):
        """Test simple preprocessing."""
        stock, market = sample_returns
        prep = DataPreprocessor()
        
        result = prep.prepare_simple(stock, market)
        
        assert 'stock_returns' in result
        assert 'market_returns' in result
        assert len(result['stock_returns']) == len(stock)
    
    def test_prepare_simple_with_series(self, sample_returns):
        """Test preprocessing with pandas Series."""
        stock, market = sample_returns
        stock_series = pd.Series(stock)
        market_series = pd.Series(market)
        
        prep = DataPreprocessor()
        result = prep.prepare_simple(stock_series, market_series)
        
        assert len(result['stock_returns']) == len(stock)
    
    def test_validate_ohlcv(self, sample_ohlcv):
        """Test OHLCV validation."""
        prep = DataPreprocessor()
        validated = prep._validate_ohlcv(sample_ohlcv, "test")
        
        assert 'Date' in validated.columns
        assert 'Close' in validated.columns
        # Check it's a datetime type (works with both ns and us precision)
        assert pd.api.types.is_datetime64_any_dtype(validated['Date'])
    
    def test_feature_config_defaults(self):
        """Test FeatureConfig default values."""
        config = FeatureConfig()
        
        assert config.lag_features == True
        assert config.include_technicals == True
        assert 5 in config.ma_windows


# =============================================================================
# Test Evaluation
# =============================================================================

class TestBetaEvaluator:
    """Tests for BetaEvaluator."""
    
    def test_evaluate_basic(self, sample_returns):
        """Test basic evaluation."""
        stock, market = sample_returns
        betas = np.ones_like(stock) * 1.2
        
        evaluator = BetaEvaluator()
        metrics = evaluator.evaluate(betas, stock, market)
        
        assert 'systematic_r2' in metrics
        assert 'beta_mean' in metrics
        assert 'beta_std' in metrics
        assert 'beta_stability' in metrics
    
    def test_evaluate_with_nan(self, sample_returns):
        """Test evaluation with NaN values."""
        stock, market = sample_returns
        betas = np.ones_like(stock) * 1.2
        betas[:100] = np.nan  # Add NaN values
        
        evaluator = BetaEvaluator()
        metrics = evaluator.evaluate(betas, stock, market)
        
        # Should still compute metrics
        assert 'beta_mean' in metrics
        assert metrics['beta_mean'] == pytest.approx(1.2, abs=0.01)
    
    def test_compare_models(self, sample_returns):
        """Test model comparison."""
        stock, market = sample_returns
        
        betas1 = np.ones_like(stock) * 1.0
        betas2 = np.ones_like(stock) * 1.2
        
        evaluator = BetaEvaluator()
        comparison = evaluator.compare_models(
            {'Model1': betas1, 'Model2': betas2},
            stock, market
        )
        
        assert len(comparison) == 2
        assert 'Model1' in comparison.index
        assert 'Model2' in comparison.index


class TestBenchmarks:
    """Tests for benchmark functions."""
    
    def test_rolling_ols_beta(self, sample_returns):
        """Test rolling OLS beta."""
        stock, market = sample_returns
        
        betas = compute_rolling_ols_beta(stock, market, window=100)
        
        assert len(betas) == len(stock)
        assert np.isnan(betas[:99]).all()  # First window-1 should be NaN
        assert not np.isnan(betas[-1])
    
    def test_static_beta(self, sample_returns):
        """Test static beta calculation."""
        stock, market = sample_returns
        
        beta = compute_static_beta(stock, market)
        
        # Data generated with beta=1.2
        assert 1.0 < beta < 1.4


# =============================================================================
# Test Utils
# =============================================================================

class TestUtils:
    """Tests for utility functions."""
    
    def test_validate_no_lookahead(self, sample_returns):
        """Test lookahead bias validation."""
        stock, market = sample_returns
        # Random betas should not correlate with future returns
        np.random.seed(123)
        betas = np.random.randn(len(stock))
        
        passed = validate_no_lookahead(
            betas, stock, market, 
            initial_size=100, 
            verbose=False
        )
        
        # Random betas should pass (no lookahead)
        assert passed
    
    def test_rolling_ols_beta_shape(self, sample_returns):
        """Test rolling_ols_beta output shape."""
        stock, market = sample_returns
        
        betas = rolling_ols_beta(stock, market, window=50)
        
        assert len(betas) == len(stock)
    
    def test_align_series(self):
        """Test series alignment."""
        a = np.array([1, 2, np.nan, 4, 5])
        b = np.array([1, np.nan, 3, 4, 5])
        
        a_aligned, b_aligned = align_series(a, b)
        
        assert len(a_aligned) == len(b_aligned)
        assert not np.isnan(a_aligned).any()
        assert not np.isnan(b_aligned).any()
    
    def test_align_series_pandas(self):
        """Test alignment with pandas Series."""
        a = pd.Series([1, 2, np.nan, 4, 5])
        b = pd.Series([1, np.nan, 3, 4, 5])
        
        a_aligned, b_aligned = align_series(a, b)
        
        assert isinstance(a_aligned, np.ndarray)
        assert not np.isnan(a_aligned).any()


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full workflow."""
    
    def test_full_workflow(self, sample_returns_with_dates):
        """Test complete workflow from data to evaluation."""
        stock, market, dates = sample_returns_with_dates
        
        # 1. Preprocess
        prep = DataPreprocessor()
        data = prep.prepare_simple(stock, market, dates=dates)
        
        # 2. Estimate
        model = DynamicBeta(
            lookback=30,
            initial_train_size=100,
            wf_step_size=50
        )
        results = model.fit_predict(**data)
        
        # 3. Evaluate
        evaluator = BetaEvaluator()
        metrics = evaluator.evaluate(
            results['beta'].values,
            results['stock_return'].values,
            results['market_return'].values
        )
        
        # 4. Validate
        passed = validate_no_lookahead(
            results['beta'].values,
            results['stock_return'].values,
            results['market_return'].values,
            initial_size=100,
            verbose=False
        )
        
        assert 'systematic_r2' in metrics
        assert passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
