
import numpy as np
import pytest
import tensorflow as tf
from grubeta import DynamicBeta, DynamicBetaConfig, TFEnvironment

class TestReproducibility:
    """Test model reproducibility with random seeds."""
    
    def test_seed_consistency(self):
        """Test that same seed produces identical results."""
        # Fix seeds
        seed = 42
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Generate data
        n = 200
        stock = np.random.randn(n)
        market = np.random.randn(n)
        
        # Run 1
        config1 = DynamicBetaConfig(
            random_seed=seed,
            lookback=20,
            initial_train_size=50,
            wf_step_size=50,
            epochs_init=1,
            epochs_retrain=1,
            verbose=0
        )
        model1 = DynamicBeta(config=config1)
        results1 = model1.fit_predict(stock, market)
        
        # Run 2
        config2 = DynamicBetaConfig(
            random_seed=seed,
            lookback=20,
            initial_train_size=50,
            wf_step_size=50,
            epochs_init=1,
            epochs_retrain=1,
            verbose=0
        )
        model2 = DynamicBeta(config=config2)
        results2 = model2.fit_predict(stock, market)
        
        # Assertions
        # 1. Betas should be identical
        np.testing.assert_allclose(
            results1['beta'].fillna(0),
            results2['beta'].fillna(0),
            err_msg="Betas not identical with same seed"
        )
        
        # 2. Alphas should be identical
        np.testing.assert_allclose(
            results1['alpha'].fillna(0),
            results2['alpha'].fillna(0),
            err_msg="Alphas not identical with same seed"
        )

    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        np.random.seed(42)
        n = 200
        stock = np.random.randn(n)
        market = np.random.randn(n)
        
        # Run 1
        config1 = DynamicBetaConfig(random_seed=123, verbose=0, epochs_init=5)
        model1 = DynamicBeta(config=config1)
        results1 = model1.fit_predict(stock, market)
        
        # Run 2
        config2 = DynamicBetaConfig(random_seed=456, verbose=0, epochs_init=5)
        model2 = DynamicBeta(config=config2)
        results2 = model2.fit_predict(stock, market)
        
        # Should be different (checking non-NaN parts)
        beta1 = results1['beta'].dropna()
        beta2 = results2['beta'].dropna()
        
        if len(beta1) > 0 and len(beta2) > 0:
            assert not np.allclose(beta1, beta2), "Different seeds produced identical betas"
