Evaluation API
==============

.. module:: grubeta.evaluation

The evaluation module provides metrics, diagnostics, and benchmarks.

BetaEvaluator
-------------

.. autoclass:: grubeta.BetaEvaluator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

DiagnosticTests
---------------

.. autoclass:: grubeta.evaluation.DiagnosticTests
   :members:
   :undoc-members:

Benchmark Functions
-------------------

.. autofunction:: grubeta.evaluation.compute_rolling_ols_beta

.. autofunction:: grubeta.evaluation.compute_ewma_beta

.. autofunction:: grubeta.evaluation.compute_static_beta

Example Usage
-------------

Basic Evaluation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from grubeta import BetaEvaluator
   
   evaluator = BetaEvaluator(output_dir='./results')
   
   metrics = evaluator.evaluate(
       betas=results['beta'].values,
       stock_returns=results['stock_return'].values,
       market_returns=results['market_return'].values,
       name='AAPL'
   )
   
   print(f"Systematic R²: {metrics['systematic_r2']:.4f}")
   print(f"Beta mean: {metrics['beta_mean']:.4f}")

Model Comparison
~~~~~~~~~~~~~~~~

.. code-block:: python

   from grubeta import BetaEvaluator
   from grubeta.evaluation import compute_rolling_ols_beta
   
   evaluator = BetaEvaluator()
   
   comparison = evaluator.compare_models(
       {
           'GRU': gru_betas,
           'Rolling OLS': compute_rolling_ols_beta(stock, market, 252)
       },
       stock_returns,
       market_returns
   )
   
   print(comparison)

Diagnostic Tests
~~~~~~~~~~~~~~~~

.. code-block:: python

   from grubeta.evaluation import DiagnosticTests
   
   # Check for lookahead bias
   result = DiagnosticTests.test_lookahead_bias(betas, returns)
   print(f"Passed: {result['passed']}")
   
   # Check stationarity
   result = DiagnosticTests.test_beta_stationarity(betas)
   print(f"Is stationary: {result['is_stationary']}")
