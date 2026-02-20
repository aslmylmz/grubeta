Advanced Features
=================

This guide covers advanced usage patterns including feature engineering,
macroeconomic data integration, and custom configurations.

Using OHLCV Data
----------------

For better results, use the full preprocessing pipeline with OHLCV data:

.. code-block:: python

   import pandas as pd
   from grubeta import DynamicBeta, DataPreprocessor, FeatureConfig
   
   # Load OHLCV data
   stock_df = pd.read_csv('stock_ohlcv.csv', parse_dates=['Date'])
   market_df = pd.read_csv('market_ohlcv.csv', parse_dates=['Date'])
   
   # Configure feature engineering
   config = FeatureConfig(
       lag_features=True,        # Prevent lookahead bias
       include_technicals=True,  # RSI, MACD, ADX, etc.
       include_volume=True,      # Volume-based features
       include_calendar=True,    # Day of week, month end
       ma_windows=[5, 10, 20, 50, 100, 200],
       volatility_windows=[5, 10, 20, 60],
   )
   
   # Preprocess
   preprocessor = DataPreprocessor(config)
   features = preprocessor.prepare(stock_df, market_df)
   
   # Estimate with features
   model = DynamicBeta(lookback=90)
   results = model.fit_predict(**features)

Feature Configuration
~~~~~~~~~~~~~~~~~~~~~

The :class:`~grubeta.FeatureConfig` class controls feature engineering:

.. code-block:: python

   config = FeatureConfig(
       # Lookahead prevention (CRITICAL)
       lag_features=True,
       
       # Feature groups
       include_technicals=True,   # Requires 'ta' library
       include_macro=False,       # Set True if using macro data
       include_volume=True,
       include_calendar=True,
       
       # Moving average windows
       ma_windows=[5, 10, 20, 50, 100, 200],
       
       # Volatility calculation windows
       volatility_windows=[5, 10, 20, 60],
       
       # Rate of change periods
       roc_periods=[1, 2, 3, 5, 10, 20, 60],
   )

Macroeconomic Features
----------------------

Integrate macroeconomic data for regime-aware beta estimation:

Preparing Macro Data
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load macro data
   macro_df = pd.read_csv('macro_data.csv', parse_dates=['Date'])
   
   # Expected format:
   # | Date       | vix    | fed_rate | unemployment | credit_spread | ...
   # |------------|--------|----------|--------------|---------------|
   # | 2020-01-02 | 12.5   | 1.75     | 3.6          | 1.2           |
   
   # Configure with macro enabled
   config = FeatureConfig(
       include_macro=True,
       lag_features=True
   )
   
   preprocessor = DataPreprocessor(config)
   features = preprocessor.prepare(
       stock_df=stock_df,
       market_df=market_df,
       macro_df=macro_df
   )

Recommended Macro Features
~~~~~~~~~~~~~~~~~~~~~~~~~~

For financial beta estimation, consider:

* **VIX** - Market fear/volatility
* **Credit spreads** - Risk appetite
* **Yield curve slope** - Economic expectations
* **Fed funds rate** - Monetary policy
* **Unemployment** - Economic health
* **PMI** - Business conditions
* **Dollar index** - Currency effects

Custom Feature Engineering
--------------------------

Add your own features to the pipeline:

.. code-block:: python

   # Standard preprocessing
   preprocessor = DataPreprocessor()
   features = preprocessor.prepare(stock_df, market_df)
   
   # Add custom features
   import numpy as np
   
   # Example: Sentiment score
   sentiment = load_sentiment_data()  # Your custom data
   
   # Ensure proper alignment and lagging
   custom_feature = sentiment['score'].shift(1).values  # Lag by 1 day
   
   # Append to market features
   n_samples = features['market_features'].shape[0]
   custom_expanded = custom_feature[-n_samples:].reshape(-1, 1)
   
   features['market_features'] = np.concatenate([
       features['market_features'],
       custom_expanded
   ], axis=1)
   
   # Update feature names
   features['feature_names']['market'].append('sentiment_score')
   
   # Fit model
   model = DynamicBeta(lookback=90)
   results = model.fit_predict(**features)

Walk-Forward Configuration
--------------------------

Control the walk-forward validation process:

.. code-block:: python

   from grubeta import DynamicBeta, DynamicBetaConfig
   
   config = DynamicBetaConfig(
       # Initial training
       initial_train_size=500,  # Samples for first training
       epochs_init=40,          # Epochs for initial training
       
       # Walk-forward steps
       wf_step_size=126,        # Predict 126 days, then retrain
       epochs_retrain=4,        # Epochs per retrain
       
       # Lookback
       lookback=90,             # 90-day input sequences
   )
   
   model = DynamicBeta(config=config)

**Choosing ``wf_step_size``:**

* 21 (monthly) - Most responsive, computationally expensive
* 63 (quarterly) - Good balance
* 126 (semi-annual) - Default, computationally efficient
* 252 (annual) - Minimal retraining

Loss Function Tuning
--------------------

The composite loss balances four objectives:

.. math::

   L = L_{accuracy} + \lambda_\beta \cdot L_{\beta\_stability} + \lambda_\alpha \cdot L_{sparsity} + \lambda_{\alpha\_smooth} \cdot L_{\alpha\_stability}

Lambda Beta (Stability)
~~~~~~~~~~~~~~~~~~~~~~~

Controls beta smoothness:

.. code-block:: python

   # High stability (smooth beta, may miss rapid changes)
   model = DynamicBeta(lambda_beta=0.15)
   
   # Low stability (responsive beta, may be noisy)
   model = DynamicBeta(lambda_beta=0.01)
   
   # Default balance
   model = DynamicBeta(lambda_beta=0.05)

Lambda Alpha (Sparsity)
~~~~~~~~~~~~~~~~~~~~~~~

Controls alpha magnitude:

.. code-block:: python

   # High sparsity (alpha pushed toward zero)
   model = DynamicBeta(lambda_alpha=1.0)
   
   # Low sparsity (allows larger alpha values)
   model = DynamicBeta(lambda_alpha=0.1)
   
   # Default
   model = DynamicBeta(lambda_alpha=0.5)

Lambda Alpha Smooth (Temporal Smoothness)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Controls alpha temporal stability:

.. code-block:: python

   # High smoothness (alpha changes slowly over time)
   model = DynamicBeta(lambda_alpha_smooth=0.2)

   # Low smoothness (allows rapid alpha changes)
   model = DynamicBeta(lambda_alpha_smooth=0.02)

   # Default (2x beta smoothness, since alpha should be even smoother)
   model = DynamicBeta(lambda_alpha_smooth=0.1)

Network Architecture
--------------------

Customize the GRU architecture:

.. code-block:: python

   config = DynamicBetaConfig(
       gru_units=128,       # Hidden units per GRU
       dropout_rate=0.2,    # Dropout for regularization
       learning_rate=1e-4,  # Adam learning rate
   )

**Guidelines:**

* Larger ``gru_units`` (128-256): More complex patterns, risk of overfitting
* Smaller ``gru_units`` (32-64): Simpler model, faster training
* Higher ``dropout_rate`` (0.3-0.5): More regularization
* Lower ``learning_rate`` (1e-5): More stable training, slower convergence

Multi-Asset Analysis
--------------------

Estimate beta for multiple assets:

.. code-block:: python

   from grubeta import DynamicBeta
   import pandas as pd
   
   stocks = {
       'AAPL': 'data/aapl.csv',
       'GOOGL': 'data/googl.csv',
       'MSFT': 'data/msft.csv',
   }
   
   # Shared market data
   market_df = pd.read_csv('data/spy.csv', parse_dates=['Date'])
   market_returns = market_df['Close'].pct_change().dropna()
   
   # Estimate for each stock
   all_results = {}
   model = DynamicBeta(lookback=60, initial_train_size=200)
   
   for ticker, filepath in stocks.items():
       stock_df = pd.read_csv(filepath, parse_dates=['Date'])
       stock_returns = stock_df['Close'].pct_change().dropna()
       
       results = model.fit_predict(
           stock_returns.values,
           market_returns.values,
           dates=stock_df['Date'][1:].values
       )
       all_results[ticker] = results
       print(f"{ticker}: Mean beta = {results['beta'].dropna().mean():.3f}")
   
   # Compare
   import matplotlib.pyplot as plt
   
   fig, ax = plt.subplots(figsize=(12, 6))
   for ticker, results in all_results.items():
       mask = ~results['beta'].isna()
       ax.plot(results['date'][mask], results['beta'][mask], label=ticker)
   
   ax.legend()
   ax.set_title('Dynamic Beta Comparison')
   plt.show()

GPU Acceleration
----------------

For large datasets, enable GPU:

.. code-block:: python

   import tensorflow as tf
   
   # Check GPU availability
   print(tf.config.list_physical_devices('GPU'))
   
   # Enable memory growth (prevents OOM errors)
   gpus = tf.config.list_physical_devices('GPU')
   for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
   
   # Model automatically uses GPU if available
   model = DynamicBeta(lookback=90)
   results = model.fit_predict(stock_returns, market_returns)

Batch Processing
----------------

For processing many stocks efficiently:

.. code-block:: python

   from grubeta import DynamicBeta
   import gc
   
   def process_stock(ticker, stock_data, market_data):
       """Process single stock with memory cleanup."""
       model = DynamicBeta(lookback=60, initial_train_size=200)
       results = model.fit_predict(stock_data, market_data)
       
       # Extract what we need
       output = results[['date', 'beta', 'alpha']].copy()
       
       # Clear memory
       del model
       gc.collect()
       
       return output
   
   # Process all stocks
   for ticker in stock_list:
       result = process_stock(ticker, stock_data[ticker], market_data)
       result.to_csv(f'results/{ticker}_beta.csv', index=False)
