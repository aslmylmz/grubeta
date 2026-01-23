Utilities API
=============

.. module:: grubeta.utils

The utils module provides helper functions for validation and analysis.

Validation Functions
--------------------

.. autofunction:: grubeta.utils.validate_no_lookahead

.. autofunction:: grubeta.utils.rolling_ols_beta

Data Manipulation
-----------------

.. autofunction:: grubeta.utils.create_sequences

.. autofunction:: grubeta.utils.align_series

Performance Metrics
-------------------

.. autofunction:: grubeta.utils.compute_information_coefficient

.. autofunction:: grubeta.utils.compute_hit_rate

Portfolio Functions
-------------------

.. autofunction:: grubeta.utils.beta_to_hedge_ratio

.. autofunction:: grubeta.utils.annualize_beta

Example Usage
-------------

Lookahead Validation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grubeta.utils import validate_no_lookahead
   
   passed = validate_no_lookahead(
       betas=results['beta'].values,
       returns=results['stock_return'].values,
       market_returns=results['market_return'].values,
       initial_size=500,
       verbose=True
   )
   
   if not passed:
       print("Warning: Potential lookahead bias detected!")

Rolling Beta Benchmark
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grubeta.utils import rolling_ols_beta
   
   # Compute benchmark
   ols_betas = rolling_ols_beta(
       stock_returns,
       market_returns,
       window=252
   )
   
   # Compare with GRU
   correlation = np.corrcoef(
       gru_betas[~np.isnan(gru_betas)],
       ols_betas[~np.isnan(ols_betas)]
   )[0, 1]

Hedging Calculation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grubeta.utils import beta_to_hedge_ratio
   
   # Calculate hedge for $100k position
   current_beta = results['beta'].iloc[-1]
   hedge = beta_to_hedge_ratio(
       beta=current_beta,
       stock_value=100_000,
       market_value=450  # SPY price
   )
   print(f"Short {hedge:.0f} shares of SPY")
