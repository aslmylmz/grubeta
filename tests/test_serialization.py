
import os
import shutil
import tempfile
import numpy as np
import pytest
import tensorflow as tf
from grubeta import DynamicBeta, DynamicBetaConfig
from grubeta.models import GRUBetaModel

class TestSerialization:
    """Test model serialization and deserialization."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create and fit a small model."""
        np.random.seed(42)
        n = 200
        stock = np.random.randn(n)
        market = np.random.randn(n)
        
        config = DynamicBetaConfig(
            lookback=10,
            initial_train_size=50,
            wf_step_size=20,
            epochs_init=1,
            epochs_retrain=1,
            verbose=0
        )
        
        model = DynamicBeta(config=config)
        model.fit(stock, market)
        
        return model, stock, market
        
    def test_save_and_load(self, fitted_model, tmp_path):
        """Test full save and load cycle."""
        model, stock, market = fitted_model
        save_path = tmp_path / "test_model"
        
        # 1. Prediction before save
        pred_before = model.predict(stock, market)
        
        # 2. Save
        model.save(save_path)
        
        # 3. Load
        loaded_model = DynamicBeta.load(save_path)
        
        # 4. Prediction after load
        pred_after = loaded_model.predict(stock, market)
        
        # Check equality
        np.testing.assert_allclose(
            pred_before['beta'], 
            pred_after['beta'], 
            err_msg="Beta predictions mismatch after loading"
        )
        
        np.testing.assert_allclose(
            pred_before['alpha'], 
            pred_after['alpha'], 
            err_msg="Alpha predictions mismatch after loading"
        )
        
        # Check config
        assert loaded_model.config.lookback == model.config.lookback
        assert loaded_model.config.lambda_beta == model.config.lambda_beta
        
    def test_custom_objects_loading(self, fitted_model, tmp_path):
        """Test that custom objects (loss, metrics) are handled correctly manually."""
        model, _, _ = fitted_model
        model_path = tmp_path / "keras_model.h5"
        
        # Save just the Keras model
        model.model_.save(model_path)
        
        # Try to load with custom scope
        custom_objects = GRUBetaModel.get_custom_objects(model.config)
        
        # Add the extraction function if it was used in Lambda layers
        # Though usually it's saved by name if it's in the scope or module
        from grubeta.models import extract_last_step
        custom_objects['extract_last_step'] = extract_last_step
        
        loaded_keras_model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects
        )
        
        assert loaded_keras_model is not None

    def test_json_config_survival(self, fitted_model, tmp_path):
        """Ensure config survives roundtrip."""
        model, _, _ = fitted_model
        save_path = tmp_path / "config_test"
        model.save(save_path)
        
        loaded = DynamicBeta.load(save_path)
        assert hasattr(loaded, 'config')
        assert loaded.config.lookback == 10
