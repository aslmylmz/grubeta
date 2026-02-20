Quickstart
==========

This guide will get you up and running with GRU Dynamic Beta in 5 minutes.

Your First Beta Estimation
--------------------------

The simplest way to estimate dynamic beta:

.. code-block:: python

   import numpy as np
   from grubeta import DynamicBeta
   
   # Your data (daily returns)
   stock_returns = np.array([...])   # Stock return series
   market_returns = np.array([...])  # Market index returns
   
   # Create model and estimate
   model = DynamicBeta(lookback=60)
   results = model.fit_predict(stock_returns, market_returns)
   
   # View results
   print(results.head())

The ``results`` DataFrame contains:

* ``beta`` - Time-varying beta estimates
* ``alpha`` - Time-varying alpha estimates  
* ``stock_return`` - Original stock returns
* ``market_return`` - Original market returns

Working with Real Data
----------------------

Here's a complete example using pandas:

.. code-block:: python

   import pandas as pd
   from grubeta import DynamicBeta
   
   # Load stock data
   stock_df = pd.read_csv('AAPL.csv', parse_dates=['Date'])
   market_df = pd.read_csv('SPY.csv', parse_dates=['Date'])
   
   # Calculate returns
   stock_returns = stock_df['Close'].pct_change().dropna()
   market_returns = market_df['Close'].pct_change().dropna()
   
   # Align dates
   common_dates = stock_df['Date'][1:].values
   
   # Estimate dynamic beta
   model = DynamicBeta(
       lookback=60,            # 60-day lookback window
       lambda_beta=0.05,       # Beta stability weight
       lambda_alpha=0.5,       # Alpha sparsity weight
       lambda_alpha_smooth=0.1 # Alpha temporal smoothness weight
   )
   
   results = model.fit_predict(
       stock_returns.values,
       market_returns.values,
       dates=common_dates
   )
   
   # Plot the results
   model.plot_beta(results, title='AAPL Dynamic Beta')

Understanding the Output
------------------------

The first ``lookback + initial_train_size`` observations will have ``NaN`` beta values.
This is the "burn-in" period where the model is training.

.. code-block:: python

   # Check valid estimates
   valid_results = results.dropna()
   print(f"Total observations: {len(results)}")
   print(f"Valid beta estimates: {len(valid_results)}")
   
   # Summary statistics
   print(valid_results['beta'].describe())

Typical output:

.. code-block:: text

   Total observations: 1000
   Valid beta estimates: 410
   
   count    410.000000
   mean       1.152340
   std        0.089234
   min        0.923456
   25%        1.089123
   50%        1.145678
   75%        1.212345
   max        1.398765

Key Parameters
--------------

The most important parameters to tune:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``lookback``
     - 90
     - Number of days in input sequence. Higher = more context, slower training.
   * - ``lambda_beta``
     - 0.05
     - Beta stability weight. Higher = smoother beta trajectory.
   * - ``lambda_alpha``
     - 0.5
     - Alpha sparsity weight. Higher = alpha closer to zero.
   * - ``lambda_alpha_smooth``
     - 0.1
     - Alpha temporal smoothness weight. Higher = smoother alpha trajectory.
   * - ``initial_train_size``
     - 500
     - Samples for initial training before walk-forward begins.
   * - ``wf_step_size``
     - 126
     - Days between model retraining (~6 months).

Example configurations:

.. code-block:: python

   # For volatile stocks (more responsive beta)
   model = DynamicBeta(lookback=30, lambda_beta=0.01)
   
   # For stable, large-cap stocks (smoother beta)
   model = DynamicBeta(lookback=90, lambda_beta=0.1)
   
   # For high-frequency analysis
   model = DynamicBeta(lookback=20, wf_step_size=21)

Comparing with Benchmarks
-------------------------

Compare GRU beta against traditional methods:

.. code-block:: python

   from grubeta import DynamicBeta, BetaEvaluator
   from grubeta.utils import rolling_ols_beta
   
   # GRU beta
   model = DynamicBeta(lookback=60)
   results = model.fit_predict(stock_returns, market_returns)
   
   # Rolling OLS benchmark
   ols_beta = rolling_ols_beta(stock_returns, market_returns, window=252)
   
   # Compare
   evaluator = BetaEvaluator()
   comparison = evaluator.compare_models(
       {
           'GRU': results['beta'].values,
           'Rolling OLS': ols_beta,
       },
       stock_returns,
       market_returns
   )
   print(comparison[['systematic_r2', 'beta_stability']])

Saving and Loading Models
-------------------------

Save a trained model for later use:

.. code-block:: python

   # Train and save
   model = DynamicBeta(lookback=60)
   model.fit_predict(stock_returns, market_returns)
   model.save('./my_model')
   
   # Load later
   loaded_model = DynamicBeta.load('./my_model')
   
   # Use for new predictions
   new_results = loaded_model.predict(new_stock_returns, new_market_returns)

Next Steps
----------

* :doc:`user_guide/basic_usage` - More detailed usage examples
* :doc:`user_guide/advanced_features` - Feature engineering and macro data
* :doc:`user_guide/evaluation` - Model evaluation and diagnostics
* :doc:`api/core` - Full API reference
