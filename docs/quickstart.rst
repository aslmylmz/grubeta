Quickstart
==========

This guide will get you up and running with grubeta in 5 minutes.

Estimate Beta in 3 Lines
-------------------------

The simplest way to estimate dynamic beta:

.. code-block:: python

   from grubeta import estimate_beta

   result = estimate_beta("AAPL", "SPY")
   print(result["summary"])

This will:

1. Fetch 10 years of AAPL and SPY data via yfinance
2. Train a GRU model with walk-forward validation
3. Print a human-readable summary and display a plot

The ``result`` dictionary contains:

* ``beta`` - Time-varying beta estimates (pd.Series)
* ``alpha`` - Time-varying alpha estimates (pd.Series)
* ``dates`` - Date index
* ``summary`` - Human-readable summary string
* ``model`` - Fitted DynamicBeta object (for advanced use)
* ``results`` - Full DataFrame with all columns
* ``fig`` - Matplotlib figure (if ``plot=True``)

Using Presets
-------------

Presets let you pick a configuration without understanding ML parameters:

.. code-block:: python

   from grubeta import estimate_beta, list_presets

   # See available presets
   print(list_presets())

   # For event studies (captures rapid beta changes)
   result = estimate_beta("AAPL", "SPY", preset="responsive")

   # For long-term portfolio construction (stable estimates)
   result = estimate_beta("AAPL", "SPY", preset="smooth")

   # For academic research (enhanced model capacity)
   result = estimate_beta("AAPL", "SPY", preset="research")

.. list-table::
   :header-rows: 1
   :widths: 15 30 20 20

   * - Preset
     - Best For
     - Lookback
     - Retraining
   * - ``default``
     - General analysis
     - 3 months (60 days)
     - Semi-annually (126 days)
   * - ``responsive``
     - Event studies, tactical allocation
     - 6 weeks (30 days)
     - Monthly (21 days)
   * - ``smooth``
     - Strategic portfolio construction
     - 6 months (120 days)
     - Annually (252 days)
   * - ``research``
     - Academic papers
     - ~4 months (90 days)
     - Quarterly (63 days)

Comparing Multiple Stocks
--------------------------

.. code-block:: python

   from grubeta import compare_betas

   result = compare_betas(["AAPL", "MSFT", "GOOGL"], market="SPY")
   print(result["summary"])

Using Your Own Data
--------------------

You can pass pandas Series instead of ticker strings:

.. code-block:: python

   import pandas as pd
   from grubeta import estimate_beta

   # Load your own data
   stock_returns = pd.read_csv('my_stock.csv', index_col='Date', parse_dates=True)['Return']
   market_returns = pd.read_csv('my_market.csv', index_col='Date', parse_dates=True)['Return']

   result = estimate_beta(stock_returns, market_returns)

Command Line
------------

.. code-block:: bash

   python -m grubeta AAPL SPY
   python -m grubeta AAPL SPY --preset responsive
   python -m grubeta AAPL MSFT GOOGL --market SPY --compare
   python -m grubeta --list-presets

Advanced Usage
--------------

For fine-grained control, use the core API directly:

.. code-block:: python

   from grubeta import DynamicBeta, DynamicBetaConfig

   model = DynamicBeta(config=DynamicBetaConfig(
       lookback=60,
       lambda_beta=0.05,
       lambda_alpha=0.5,
       lambda_alpha_smooth=0.1,
   ))
   results = model.fit_predict(stock_returns, market_returns, dates=dates)

   # View results
   print(results['beta'].dropna().describe())

Understanding the Output
------------------------

The first ``lookback + initial_train_size`` observations will have ``NaN`` beta values.
This is the "burn-in" period where the model is training.

.. code-block:: python

   valid_results = results.dropna()
   print(f"Total observations: {len(results)}")
   print(f"Valid beta estimates: {len(valid_results)}")

Key Parameters
--------------

The most important parameters (when using the core API):

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
* :doc:`api/convenience` - Convenience API reference
* :doc:`api/core` - Full core API reference
