Model Evaluation
================

This guide covers evaluation metrics, diagnostic tests, and model comparison.

BetaEvaluator
-------------

The :class:`~grubeta.BetaEvaluator` class provides comprehensive model evaluation:

.. code-block:: python

   from grubeta import BetaEvaluator
   
   evaluator = BetaEvaluator(output_dir='./results')
   
   metrics = evaluator.evaluate(
       betas=results['beta'].values,
       stock_returns=results['stock_return'].values,
       market_returns=results['market_return'].values,
       alphas=results['alpha'].values,
       dates=results['date'].values,
       name='AAPL'
   )
   
   print(metrics)

Key Metrics
-----------

Prediction Accuracy
~~~~~~~~~~~~~~~~~~~

**Systematic R²**

How well does β × R_market explain stock returns?

.. code-block:: python

   # R² of: y_pred = beta * market_return
   metrics['systematic_r2']
   
   # Interpretation:
   # > 0.3: Good systematic explanation
   # 0.1-0.3: Moderate
   # < 0.1: Weak (high idiosyncratic risk)

**Total R²** (if alpha provided)

How well does α + β × R_market explain returns?

.. code-block:: python

   # R² of: y_pred = alpha + beta * market_return
   metrics['total_r2']

**MAE and RMSE**

.. code-block:: python

   metrics['systematic_mae']  # Mean Absolute Error
   metrics['systematic_mse']  # Mean Squared Error (take sqrt for RMSE)

Beta Statistics
~~~~~~~~~~~~~~~

.. code-block:: python

   metrics['beta_mean']     # Average beta
   metrics['beta_std']      # Beta volatility
   metrics['beta_min']      # Minimum beta
   metrics['beta_max']      # Maximum beta
   metrics['beta_median']   # Median beta

**Beta Stability**

Lower values indicate smoother beta trajectory:

.. code-block:: python

   metrics['beta_stability']        # Std of daily beta changes
   metrics['beta_mean_abs_change']  # Mean absolute daily change

Alpha Statistics
~~~~~~~~~~~~~~~~

.. code-block:: python

   metrics['alpha_mean']  # Should be close to 0 for efficient markets
   metrics['alpha_std']   # Alpha volatility

Benchmark Comparison
--------------------

Compare GRU beta against traditional methods:

.. code-block:: python

   from grubeta import BetaEvaluator
   from grubeta.evaluation import (
       compute_rolling_ols_beta,
       compute_ewma_beta,
       compute_static_beta
   )
   
   # Compute benchmarks
   ols_beta = compute_rolling_ols_beta(stock_ret, market_ret, window=252)
   ewma_beta = compute_ewma_beta(stock_ret, market_ret, halflife=63)
   static = compute_static_beta(stock_ret, market_ret)
   
   # Compare all models
   evaluator = BetaEvaluator()
   comparison = evaluator.compare_models(
       {
           'GRU': results['beta'].values,
           'Rolling OLS (252d)': ols_beta,
           'EWMA (63d)': ewma_beta,
           'Static': np.full_like(ols_beta, static),
       },
       stock_ret,
       market_ret
   )
   
   print(comparison[['systematic_r2', 'beta_mean', 'beta_stability']])

Example output:

.. code-block:: text

                        systematic_r2  beta_mean  beta_stability
   model                                                        
   GRU                        0.3521     1.1523          0.0089
   Rolling OLS (252d)         0.3412     1.1456          0.0234
   EWMA (63d)                 0.3389     1.1489          0.0156
   Static                     0.3245     1.1500          0.0000

Diagnostic Tests
----------------

Lookahead Bias Test
~~~~~~~~~~~~~~~~~~~

Check for information leakage:

.. code-block:: python

   from grubeta.evaluation import DiagnosticTests
   
   results = DiagnosticTests.test_lookahead_bias(
       betas=results['beta'].values,
       returns=results['stock_return'].values,
       max_lag=20,
       threshold=0.15
   )
   
   print(f"Passed: {results['passed']}")
   print(f"Correlations: {results['correlations']}")
   
   if results['suspicious_lags']:
       print(f"WARNING: Suspicious lags: {results['suspicious_lags']}")

Stationarity Test
~~~~~~~~~~~~~~~~~

Test if beta series is stationary (ADF test):

.. code-block:: python

   results = DiagnosticTests.test_beta_stationarity(
       betas=results['beta'].values,
       significance=0.05
   )
   
   print(f"ADF Statistic: {results['adf_statistic']:.4f}")
   print(f"P-value: {results['p_value']:.4f}")
   print(f"Is Stationary: {results['is_stationary']}")

Residual Autocorrelation
~~~~~~~~~~~~~~~~~~~~~~~~

Check for autocorrelation in residuals (Ljung-Box test):

.. code-block:: python

   results = DiagnosticTests.test_residual_autocorrelation(
       betas=results['beta'].values,
       stock_returns=results['stock_return'].values,
       market_returns=results['market_return'].values,
       lags=10
   )
   
   print(f"Has Autocorrelation: {results['has_autocorrelation']}")

Visualization
-------------

Beta Trajectory Plot
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   
   fig, ax = plt.subplots(figsize=(12, 6))
   
   # Plot beta with confidence band
   dates = results['date']
   beta = results['beta']
   
   ax.plot(dates, beta, 'b-', linewidth=1.5, label='GRU Beta')
   ax.fill_between(
       dates,
       beta - 2 * metrics['beta_stability'],
       beta + 2 * metrics['beta_stability'],
       alpha=0.2,
       label='±2σ band'
   )
   ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Market Beta')
   ax.axhline(metrics['beta_mean'], color='green', linestyle=':', 
              alpha=0.5, label=f"Mean ({metrics['beta_mean']:.2f})")
   
   ax.set_xlabel('Date')
   ax.set_ylabel('Beta')
   ax.set_title('Dynamic Beta with Uncertainty')
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.tight_layout()

Rolling Performance
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Rolling R² calculation
   window = 252
   rolling_r2 = []
   
   for i in range(window, len(results)):
       y = results['stock_return'].iloc[i-window:i].values
       beta = results['beta'].iloc[i-window:i].values
       mkt = results['market_return'].iloc[i-window:i].values
       
       mask = ~np.isnan(beta)
       if mask.sum() > 50:
           y_pred = beta[mask] * mkt[mask]
           r2 = 1 - np.sum((y[mask] - y_pred)**2) / np.sum((y[mask] - y[mask].mean())**2)
           rolling_r2.append(r2)
       else:
           rolling_r2.append(np.nan)
   
   # Plot
   plt.figure(figsize=(12, 4))
   plt.plot(results['date'].iloc[window:], rolling_r2)
   plt.axhline(metrics['systematic_r2'], color='red', linestyle='--', 
               label=f"Overall R² ({metrics['systematic_r2']:.3f})")
   plt.ylabel('Rolling R² (252d)')
   plt.title('Time-Varying Model Performance')
   plt.legend()
   plt.grid(True, alpha=0.3)

Model Comparison Plot
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fig, axes = plt.subplots(2, 2, figsize=(14, 10))
   
   # 1. Beta trajectories
   ax = axes[0, 0]
   ax.plot(dates, gru_beta, label='GRU', alpha=0.8)
   ax.plot(dates, ols_beta, label='OLS', alpha=0.8)
   ax.set_title('Beta Comparison')
   ax.legend()
   
   # 2. Beta distribution
   ax = axes[0, 1]
   ax.hist(gru_beta[~np.isnan(gru_beta)], bins=50, alpha=0.5, label='GRU')
   ax.hist(ols_beta[~np.isnan(ols_beta)], bins=50, alpha=0.5, label='OLS')
   ax.set_title('Beta Distribution')
   ax.legend()
   
   # 3. Stability comparison
   ax = axes[1, 0]
   gru_changes = np.abs(np.diff(gru_beta[~np.isnan(gru_beta)]))
   ols_changes = np.abs(np.diff(ols_beta[~np.isnan(ols_beta)]))
   ax.hist(gru_changes, bins=50, alpha=0.5, label='GRU')
   ax.hist(ols_changes, bins=50, alpha=0.5, label='OLS')
   ax.set_title('Daily Beta Change Distribution')
   ax.legend()
   
   # 4. Metrics comparison
   ax = axes[1, 1]
   models = ['GRU', 'OLS', 'EWMA', 'Static']
   r2_values = [
       comparison.loc['GRU', 'systematic_r2'],
       comparison.loc['Rolling OLS (252d)', 'systematic_r2'],
       comparison.loc['EWMA (63d)', 'systematic_r2'],
       comparison.loc['Static', 'systematic_r2'],
   ]
   ax.bar(models, r2_values)
   ax.set_ylabel('Systematic R²')
   ax.set_title('Model R² Comparison')
   
   plt.tight_layout()

Generating Reports
------------------

Automated reporting:

.. code-block:: python

   evaluator = BetaEvaluator(output_dir='./reports')
   
   # This generates:
   # - ./reports/AAPL_report.txt (metrics summary)
   # - ./reports/AAPL_beta_evolution.png (trajectory plot)
   
   metrics = evaluator.evaluate(
       betas=results['beta'].values,
       stock_returns=results['stock_return'].values,
       market_returns=results['market_return'].values,
       dates=results['date'].values,
       name='AAPL'
   )

Report contents:

.. code-block:: text

   Dynamic Beta Evaluation Report: AAPL
   ==================================================
   
   Prediction Accuracy
   ------------------------------
   Systematic R²: 0.3521
   Total R²: 0.3789
   MAE: 0.008234
   RMSE: 0.011456
   
   Beta Statistics
   ------------------------------
   Mean: 1.1523
   Std Dev: 0.0892
   Min: 0.9234
   Max: 1.3987
   Stability (Δ std): 0.008900
   
   Alpha Statistics
   ------------------------------
   Mean: 0.000023
   Std Dev: 0.002341
