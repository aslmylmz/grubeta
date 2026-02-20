"""
Neural network model architectures for dynamic beta estimation.

This module contains the GRU-based model architecture and custom loss functions
used for time-varying beta estimation within the CAPM framework.
"""

import gc
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers


def extract_last_step(x):
    """Extract the last timestep from a sequence."""
    return x[:, -1, :]


class TFEnvironment:
    """Manages TensorFlow environment state."""

    @staticmethod
    def reset(seed: int = 42) -> None:
        """
        Reset Keras session, clear graph, and manage GPU memory.

        Parameters
        ----------
        seed : int, default=42
            Random seed for reproducibility.
        """
        K.clear_session()

        # Legacy compatibility
        if hasattr(tf.compat.v1, "reset_default_graph"):
            tf.compat.v1.reset_default_graph()

        gc.collect()

        if str(seed) != os.environ.get("PYTHONHASHSEED"):
            os.environ["PYTHONHASHSEED"] = str(seed)

        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # GPU configuration
        try:
            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.set_memory_growth(gpu, True)
        except Exception:
            pass  # No GPU available


class CAPMLoss:
    """
    Custom loss functions for CAPM-based beta estimation.

    The composite loss combines:
    1. Accuracy: Huber loss on return predictions
    2. Stability: L2 penalty on beta changes over time
    3. Sparsity: L1 penalty on alpha (encouraging CAPM compliance)
    4. Alpha Stability: L2 penalty on alpha changes over time
    """

    @staticmethod
    def create_composite_loss(lookback: int, lambda_beta: float, lambda_alpha: float, lambda_alpha_smooth: float = 0.0):
        """
        Create the CAPM composite loss function.

        Parameters
        ----------
        lookback : int
            Sequence length for temporal analysis.
        lambda_beta : float
            Weight for beta stability term.
        lambda_alpha : float
            Weight for alpha sparsity term.
        lambda_alpha_smooth : float, default=0.0
            Weight for alpha temporal smoothness term.

        Returns
        -------
        loss_fn : callable
            Keras-compatible loss function.
        """
        huber = tf.keras.losses.Huber()

        def capm_loss(y_true, y_pred_concatenated):
            """
            Composite loss: Accuracy + Stability + Sparsity + Alpha Stability.

            Output structure: [Return(1) | Beta_Seq(L) | Alpha_Seq(L)]
            """
            # Deconstruct output tensor
            pred_return = y_pred_concatenated[:, 0:1]
            beta_flat = y_pred_concatenated[:, 1 : 1 + lookback]
            alpha_flat = y_pred_concatenated[:, 1 + lookback :]

            # Reshape for temporal analysis
            beta_seq = tf.reshape(beta_flat, [-1, lookback, 1])
            alpha_seq = tf.reshape(alpha_flat, [-1, lookback, 1])

            # Component 1: Prediction Accuracy (Huber loss)
            loss_accuracy = huber(y_true, pred_return)

            # Component 2: Beta Stability (temporal smoothness)
            # Penalize rapid changes in beta over time
            beta_diff = beta_seq[:, 1:, :] - beta_seq[:, :-1, :]
            loss_stability = tf.reduce_mean(tf.square(beta_diff))

            # Component 3: Alpha Sparsity (L1 penalty)
            # Encourage alpha to be small (CAPM assumption)
            loss_sparsity = tf.reduce_mean(tf.abs(alpha_seq))

            # Component 4: Alpha Stability (temporal smoothness)
            # Penalize rapid changes in alpha over time
            alpha_diff = alpha_seq[:, 1:, :] - alpha_seq[:, :-1, :]
            loss_alpha_stability = tf.reduce_mean(tf.square(alpha_diff))

            return (
                loss_accuracy
                + (lambda_beta * loss_stability)
                + (lambda_alpha * loss_sparsity)
                + (lambda_alpha_smooth * loss_alpha_stability)
            )

        return capm_loss

    @staticmethod
    def mae_metric(y_true, y_pred_concatenated):
        """
        MAE metric that extracts prediction from composite tensor.

        Parameters
        ----------
        y_true : tensor
            True return values.
        y_pred_concatenated : tensor
            Concatenated model output.

        Returns
        -------
        mae : tensor
            Mean absolute error.
        """
        pred_return = y_pred_concatenated[:, 0:1]
        y_true_reshaped = tf.reshape(y_true, tf.shape(pred_return))
        return tf.reduce_mean(tf.abs(y_true_reshaped - pred_return))


class GRUBetaModel:
    """
    GRU-based neural network for dynamic beta estimation.

    Architecture overview:
    - Beta pathway: GRU processing market/systematic features
    - Alpha pathway: GRU processing stock-specific features
    - CAPM reconstruction: R_stock = alpha + beta * R_market

    The model outputs a concatenated tensor containing:
    - Single return prediction
    - Full beta sequence (for stability loss)
    - Full alpha sequence (for sparsity loss)

    Parameters
    ----------
    config : DynamicBetaConfig
        Model configuration object.

    Attributes
    ----------
    model : keras.Model
        Built and compiled Keras model.
    """

    def __init__(self, config):
        """
        Initialize GRUBetaModel.

        Parameters
        ----------
        config : DynamicBetaConfig
            Configuration containing model hyperparameters.
        """
        self.config = config
        self.model: Optional[Model] = None

    def build(self, n_market_features: int, n_stock_features: int) -> Model:
        """
        Build and compile the Keras model.

        Parameters
        ----------
        n_market_features : int
            Number of features in market input.
        n_stock_features : int
            Number of features in stock input.

        Returns
        -------
        model : keras.Model
            Compiled model ready for training.
        """
        TFEnvironment.reset(self.config.random_seed)

        # Input layers
        inp_market = layers.Input(
            shape=(self.config.lookback, n_market_features), name="market_input"
        )
        inp_stock = layers.Input(
            shape=(self.config.lookback, n_stock_features), name="stock_input"
        )
        inp_current_market = layers.Input(shape=(1,), name="current_market_return")

        # Beta Pathway (Systematic Risk)
        # Processes market features to estimate time-varying beta
        x_beta = layers.GRU(
            self.config.gru_units, return_sequences=True, name="beta_gru"
        )(inp_market)
        x_beta = layers.Dropout(self.config.dropout_rate)(x_beta)

        # Output beta for each timestep (for stability loss)
        beta_seq = layers.TimeDistributed(
            layers.Dense(
                1, 
                activation="linear", 
                bias_initializer=tf.keras.initializers.Constant(self.config.initial_beta)
            ), 
            name="beta_sequence"
        )(x_beta)

        # Extract last timestep beta for CAPM calculation
        beta_t = layers.Lambda(extract_last_step, name="beta_current")(beta_seq)

        # Alpha Pathway (Idiosyncratic Risk)
        # Processes stock features to estimate time-varying alpha
        x_alpha = layers.GRU(
            self.config.gru_units, return_sequences=True, name="alpha_gru"
        )(inp_stock)
        x_alpha = layers.Dropout(self.config.dropout_rate)(x_alpha)

        # Output alpha for each timestep (for sparsity loss)
        alpha_seq = layers.TimeDistributed(
            layers.Dense(
                1, 
                activation="linear", 
                bias_initializer=tf.keras.initializers.Constant(self.config.initial_alpha)
            ), 
            name="alpha_sequence"
        )(x_alpha)

        # Extract last timestep alpha
        alpha_t = layers.Lambda(extract_last_step, name="alpha_current")(alpha_seq)

        # CAPM Reconstruction: R_stock = alpha + (beta * R_market)
        systematic = layers.Multiply(name="systematic_risk")(
            [beta_t, inp_current_market]
        )
        predicted_return = layers.Add(name="predicted_return")([alpha_t, systematic])

        # Flatten sequences for loss calculation
        beta_flat = layers.Reshape((-1,), name="beta_flat")(beta_seq)
        alpha_flat = layers.Reshape((-1,), name="alpha_flat")(alpha_seq)

        # Concatenate: [Prediction(1) | Beta_Seq(L) | Alpha_Seq(L)]
        composite_output = layers.Concatenate(name="composite_output")(
            [predicted_return, beta_flat, alpha_flat]
        )

        # Create model
        self.model = Model(
            inputs=[inp_market, inp_stock, inp_current_market],
            outputs=composite_output,
            name="GRU_DynamicBeta",
        )

        # Compile with composite loss
        loss_fn = CAPMLoss.create_composite_loss(
            self.config.lookback, self.config.lambda_beta, self.config.lambda_alpha,
            self.config.lambda_alpha_smooth
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)

        self.model.compile(
            optimizer=optimizer, loss=loss_fn, metrics=[CAPMLoss.mae_metric]
        )

        return self.model

    @staticmethod
    def get_custom_objects(config) -> Dict[str, Any]:
        """
        Get custom objects for model loading.

        Parameters
        ----------
        config : DynamicBetaConfig
            Configuration used to create loss functions.

        Returns
        -------
        custom_objects : dict
            Dictionary of custom objects for keras.models.load_model.
        """
        return {
            "capm_loss": CAPMLoss.create_composite_loss(
                config.lookback, config.lambda_beta, config.lambda_alpha,
                config.lambda_alpha_smooth
            ),
            "mae_metric": CAPMLoss.mae_metric,
            "extract_last_step": extract_last_step,
        }

    def summary(self) -> None:
        """Print model summary."""
        if self.model is not None:
            self.model.summary()
        else:
            print("Model not built yet. Call build() first.")


class DualPathwayGRU(GRUBetaModel):
    """
    Extended GRU model with separate macro feature pathway.

    This architecture adds a third pathway specifically for
    macroeconomic features, allowing the model to capture
    regime-dependent dynamics.

    Architecture:
    - Beta pathway: Market technical features -> GRU -> beta
    - Alpha pathway: Stock-specific features -> GRU -> alpha
    - Regime pathway: Macro features -> Dense -> regime embedding
    - Fusion: Combine pathways with attention or concatenation
    """

    def build(
        self, n_market_features: int, n_stock_features: int, n_macro_features: int = 0
    ) -> Model:
        """
        Build extended model with macro pathway.

        Parameters
        ----------
        n_market_features : int
            Number of market features.
        n_stock_features : int
            Number of stock features.
        n_macro_features : int, default=0
            Number of macroeconomic features.

        Returns
        -------
        model : keras.Model
            Compiled model.
        """
        if n_macro_features == 0:
            # Fall back to standard model
            return super().build(n_market_features, n_stock_features)

        TFEnvironment.reset(self.config.random_seed)

        # Standard inputs
        inp_market = layers.Input(
            shape=(self.config.lookback, n_market_features), name="market_input"
        )
        inp_stock = layers.Input(
            shape=(self.config.lookback, n_stock_features), name="stock_input"
        )
        inp_current_market = layers.Input(shape=(1,), name="current_market_return")

        # Macro input
        inp_macro = layers.Input(
            shape=(self.config.lookback, n_macro_features), name="macro_input"
        )

        # Macro pathway (regime detection)
        x_macro = layers.GRU(32, return_sequences=False, name="macro_gru")(inp_macro)
        regime_embed = layers.Dense(16, activation="relu", name="regime_embed")(x_macro)

        # Beta pathway with regime conditioning
        market_with_regime = layers.Concatenate()(
            [inp_market, layers.RepeatVector(self.config.lookback)(regime_embed)]
        )

        x_beta = layers.GRU(
            self.config.gru_units, return_sequences=True, name="beta_gru"
        )(market_with_regime)
        x_beta = layers.Dropout(self.config.dropout_rate)(x_beta)
        beta_seq = layers.TimeDistributed(
            layers.Dense(
                1, 
                activation="linear",
                bias_initializer=tf.keras.initializers.Constant(self.config.initial_beta)
            ), 
            name="beta_sequence"
        )(x_beta)
        beta_t = layers.Lambda(extract_last_step, name="beta_current")(beta_seq)

        # Alpha pathway
        x_alpha = layers.GRU(
            self.config.gru_units, return_sequences=True, name="alpha_gru"
        )(inp_stock)
        x_alpha = layers.Dropout(self.config.dropout_rate)(x_alpha)
        alpha_seq = layers.TimeDistributed(
            layers.Dense(
                1, 
                activation="linear",
                bias_initializer=tf.keras.initializers.Constant(self.config.initial_alpha)
            ), 
            name="alpha_sequence"
        )(x_alpha)
        alpha_t = layers.Lambda(extract_last_step, name="alpha_current")(alpha_seq)

        # CAPM reconstruction
        systematic = layers.Multiply(name="systematic_risk")(
            [beta_t, inp_current_market]
        )
        predicted_return = layers.Add(name="predicted_return")([alpha_t, systematic])

        # Output
        beta_flat = layers.Reshape((-1,))(beta_seq)
        alpha_flat = layers.Reshape((-1,))(alpha_seq)
        composite_output = layers.Concatenate(name="composite_output")(
            [predicted_return, beta_flat, alpha_flat]
        )

        self.model = Model(
            inputs=[inp_market, inp_stock, inp_current_market, inp_macro],
            outputs=composite_output,
            name="DualPathway_DynamicBeta",
        )

        loss_fn = CAPMLoss.create_composite_loss(
            self.config.lookback, self.config.lambda_beta, self.config.lambda_alpha,
            self.config.lambda_alpha_smooth
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=loss_fn,
            metrics=[CAPMLoss.mae_metric],
        )

        return self.model
