Basic Usage
===========

This guide covers the fundamental usage patterns for GRU Dynamic Beta.

The DynamicBeta Class
---------------------

The :class:`~grubeta.DynamicBeta` class is the main interface for beta estimation.

Creating a Model
~~~~~~~~~~~~~~~~

.. code-block:: python

   from grubeta import DynamicBeta
   
   # Default configuration
   model = DynamicBeta()
   
   # Custom parameters
   model = DynamicBeta(
       lookback=60,
       lambda_beta=0.05,
       lambda_alpha=0.3
   )

Using a Configuration Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more control, use :class:`~grubeta.DynamicBetaConfig`:

.. code-block:: python

   from grubeta import DynamicBeta, DynamicBetaConfig
   
   config = DynamicBetaConfig(
       lookback=90,
       initial_train_size=500,
       wf_step_size=126,
       learning_rate=1e-4,
       gru_units=128,
       dropout_rate=0.2,
       lambda_beta=0.05,
       lambda_alpha=0.5,
       lambda_alpha_smooth=0.1,
       epochs_init=40,
       epochs_retrain=4,
       verbose=1
   )
   
   model = DynamicBeta(config=config)

Input Data Formats
------------------

GRU Dynamic Beta accepts various input formats:

NumPy Arrays
~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   stock_returns = np.array([0.01, -0.02, 0.015, ...])
   market_returns = np.array([0.008, -0.015, 0.012, ...])
   
   results = model.fit_predict(stock_returns, market_returns)

Pandas Series
~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   
   stock_returns = df['stock_close'].pct_change()
   market_returns = df['market_close'].pct_change()
   
   results = model.fit_predict(stock_returns, market_returns)

With Date Index
~~~~~~~~~~~~~~~

.. code-block:: python

   results = model.fit_predict(
       stock_returns,
       market_returns,
       dates=df['date'].values
   )
   
   # Results will include 'date' column
   print(results[['date', 'beta']].tail())

The fit_predict Method
----------------------

The recommended method for most use cases:

.. code-block:: python

   results = model.fit_predict(
       stock_returns,      # Required: stock return series
       market_returns,     # Required: market return series
       market_features=None,   # Optional: additional market features
       stock_features=None,    # Optional: additional stock features
       dates=None              # Optional: date index
   )

This method performs walk-forward validation to prevent lookahead bias.

Separate Fit and Predict
------------------------

For more control, use separate fit and predict steps:

.. code-block:: python

   # Fit on training data
   model.fit(
       stock_returns[:800],
       market_returns[:800]
   )
   
   # Predict on new data
   predictions = model.predict(
       stock_returns[800:],
       market_returns[800:]
   )

Understanding Results
---------------------

The results DataFrame contains:

.. code-block:: python

   results.columns
   # ['date', 'beta', 'alpha', 'stock_return', 'market_return']

NaN Values
~~~~~~~~~~

The first ``lookback + initial_train_size`` rows have ``NaN`` beta values:

.. code-block:: python

   # These are NaN (burn-in period)
   results.head(200)
   
   # Valid estimates start here
   results.dropna().head()

Working with Results
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get valid beta series
   valid_betas = results['beta'].dropna()
   
   # Calculate summary statistics
   print(f"Mean beta: {valid_betas.mean():.4f}")
   print(f"Beta volatility: {valid_betas.std():.4f}")
   print(f"Beta range: [{valid_betas.min():.4f}, {valid_betas.max():.4f}]")
   
   # Check for regime changes
   rolling_mean = valid_betas.rolling(60).mean()
   rolling_std = valid_betas.rolling(60).std()

Visualization
-------------

Built-in Plotting
~~~~~~~~~~~~~~~~~

.. code-block:: python

   model.plot_beta(
       results,
       figsize=(12, 6),
       title='AAPL Dynamic Beta',
       save_path='beta_plot.png'
   )

Custom Visualization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   
   fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
   
   # Beta trajectory
   axes[0].plot(results['date'], results['beta'], 'b-', label='GRU Beta')
   axes[0].axhline(1.0, color='red', linestyle='--', alpha=0.5)
   axes[0].fill_between(
       results['date'],
       results['beta'] - 0.1,
       results['beta'] + 0.1,
       alpha=0.2
   )
   axes[0].set_ylabel('Beta')
   axes[0].legend()
   axes[0].grid(True, alpha=0.3)
   
   # Alpha trajectory
   axes[1].plot(results['date'], results['alpha'], 'purple', label='Alpha')
   axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
   axes[1].set_ylabel('Alpha')
   axes[1].set_xlabel('Date')
   axes[1].legend()
   axes[1].grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

Model Persistence
-----------------

Saving Models
~~~~~~~~~~~~~

.. code-block:: python

   # After fitting
   model.fit_predict(stock_returns, market_returns)
   
   # Save to directory
   model.save('./saved_models/aapl_beta')

This saves:

* ``model.h5`` - Keras model weights
* ``artifacts.pkl`` - Scalers and configuration

Loading Models
~~~~~~~~~~~~~~

.. code-block:: python

   # Load saved model
   model = DynamicBeta.load('./saved_models/aapl_beta')
   
   # Use for predictions
   new_results = model.predict(new_stock_returns, new_market_returns)

Error Handling
--------------

Common errors and solutions:

.. code-block:: python

   # Insufficient data
   try:
       model = DynamicBeta(lookback=90, initial_train_size=500)
       model.fit_predict(short_data, short_market)  # < 590 samples
   except ValueError as e:
       print(f"Error: {e}")
       # Solution: Use shorter lookback or more data
   
   # Model not fitted
   try:
       model = DynamicBeta()
       model.predict(data, market)  # Before fitting
   except ValueError as e:
       print(f"Error: {e}")
       # Solution: Call fit_predict() or fit() first
