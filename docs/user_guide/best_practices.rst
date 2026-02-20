Best Practices
==============

This guide covers recommended practices for getting the best results
from GRU Dynamic Beta.

Data Requirements
-----------------

Minimum Data Size
~~~~~~~~~~~~~~~~~

For reliable results:

.. code-block:: python

   # Minimum recommended
   min_samples = lookback + initial_train_size + 200
   
   # Example: lookback=90, initial_train_size=500
   # Minimum: 90 + 500 + 200 = 790 samples (~3 years daily data)
   
   # Ideal for robust estimation
   ideal_samples = 2000+  # ~8 years daily data

Data Quality
~~~~~~~~~~~~

1. **No gaps**: Fill or interpolate missing dates
2. **Clean returns**: Cap extreme values

   .. code-block:: python
   
      returns = np.clip(returns, -0.5, 0.5)  # Cap at ±50%

3. **Consistent dates**: Stock and market must align

   .. code-block:: python
   
      # Align on common dates
      common = stock_df.merge(market_df, on='Date', how='inner')

Parameter Selection
-------------------

Lookback Window
~~~~~~~~~~~~~~~

+------------------+----------+------------------+
| Lookback         | Use Case | Characteristics  |
+==================+==========+==================+
| 20-30 days       | Fast-moving stocks, HFT | Responsive, noisy |
+------------------+----------+------------------+
| 60 days          | Default, most stocks | Balanced |
+------------------+----------+------------------+
| 90 days          | Stable large-caps | Smooth, slower |
+------------------+----------+------------------+
| 120+ days        | Very stable assets | Very smooth |
+------------------+----------+------------------+

Lambda Beta (Stability)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Start with default
   lambda_beta = 0.05
   
   # If beta is too noisy, increase
   lambda_beta = 0.1 to 0.2
   
   # If beta is too smooth (missing regime changes), decrease
   lambda_beta = 0.01 to 0.03

Lambda Alpha (Sparsity)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For efficient markets (alpha should be ~0)
   lambda_alpha = 0.5 to 1.0
   
   # If stock has persistent alpha
   lambda_alpha = 0.1 to 0.3
   
   # For alpha estimation focus
   lambda_alpha = 0.05

Lambda Alpha Smooth (Temporal Smoothness)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For efficient markets (alpha should be stable near zero)
   lambda_alpha_smooth = 0.1 to 0.2

   # If stock has genuine time-varying alpha
   lambda_alpha_smooth = 0.02 to 0.05

   # For maximum smoothness
   lambda_alpha_smooth = 0.2

Walk-Forward Step Size
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # More frequent retraining = more adaptive, slower
   wf_step_size = 21   # Monthly
   wf_step_size = 63   # Quarterly
   wf_step_size = 126  # Semi-annual (default)
   wf_step_size = 252  # Annual

**Recommendation**: Start with 126, reduce if you need more adaptability.

Avoiding Common Pitfalls
------------------------

1. Lookahead Bias
~~~~~~~~~~~~~~~~~

**Always verify**:

.. code-block:: python

   from grubeta.utils import validate_no_lookahead
   
   passed = validate_no_lookahead(
       results['beta'].values,
       results['stock_return'].values,
       results['market_return'].values,
       initial_size=config.initial_train_size
   )
   
   assert passed, "Lookahead bias detected!"

**Common causes**:

* Using current-day features (not lagging)
* Fitting scalers on full dataset
* Using in-sample predictions

2. Overfitting
~~~~~~~~~~~~~~

**Signs of overfitting**:

* In-sample R² much higher than out-of-sample
* Beta volatility too low (memorizing)
* Poor performance on new data

**Solutions**:

.. code-block:: python

   # Increase regularization
   config = DynamicBetaConfig(
       dropout_rate=0.3,  # Increase from 0.2
       lambda_beta=0.1,   # Increase stability penalty
   )
   
   # Use smaller network
   config = DynamicBetaConfig(
       gru_units=64,  # Reduce from 128
   )
   
   # Reduce training epochs
   config = DynamicBetaConfig(
       epochs_init=30,
       epochs_retrain=2,
   )

3. Insufficient Training
~~~~~~~~~~~~~~~~~~~~~~~~

**Signs**:

* Beta stuck near initial values
* High prediction variance
* Poor convergence

**Solutions**:

.. code-block:: python

   # Increase initial training
   config = DynamicBetaConfig(
       initial_train_size=750,  # From 500
       epochs_init=60,          # From 40
   )
   
   # Adjust learning rate
   config = DynamicBetaConfig(
       learning_rate=5e-5,  # Slower but more stable
   )

4. Regime Changes
~~~~~~~~~~~~~~~~~

Beta can change dramatically during market regime changes.

**Detection**:

.. code-block:: python

   # Monitor rolling beta volatility
   rolling_std = results['beta'].rolling(60).std()
   
   # High volatility = potential regime change
   regime_changes = rolling_std > 2 * rolling_std.mean()

**Handling**:

.. code-block:: python

   # Faster adaptation
   config = DynamicBetaConfig(
       wf_step_size=21,  # Monthly retraining
       lambda_beta=0.02, # More responsive
   )

Production Deployment
---------------------

Model Versioning
~~~~~~~~~~~~~~~~

.. code-block:: python

   from datetime import datetime
   
   # Version your models
   version = datetime.now().strftime("%Y%m%d")
   model.save(f'./models/AAPL_beta_v{version}')
   
   # Keep metadata
   metadata = {
       'version': version,
       'lookback': config.lookback,
       'lambda_beta': config.lambda_beta,
       'train_end_date': str(data['dates'][-1]),
       'performance': metrics
   }

Monitoring
~~~~~~~~~~

.. code-block:: python

   def monitor_model_performance(model, new_data, threshold=0.2):
       """Monitor for model degradation."""
       predictions = model.predict(new_data['stock'], new_data['market'])
       
       # Calculate recent R²
       recent_r2 = compute_r2(
           new_data['stock'][-252:],
           predictions['beta'][-252:] * new_data['market'][-252:]
       )
       
       if recent_r2 < threshold:
           alert("Model performance degraded - consider retraining")
           return False
       return True

Retraining Schedule
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Automated retraining
   def should_retrain(model, metrics_history, threshold=0.1):
       """Determine if model needs retraining."""
       if len(metrics_history) < 2:
           return False
       
       recent = metrics_history[-1]['systematic_r2']
       baseline = np.mean([m['systematic_r2'] for m in metrics_history[:-1]])
       
       degradation = (baseline - recent) / baseline
       return degradation > threshold

Performance Optimization
------------------------

Memory Management
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import gc
   
   def process_batch(stocks, market_data):
       """Process stocks with memory cleanup."""
       results = {}
       
       for stock in stocks:
           model = DynamicBeta(lookback=60)
           result = model.fit_predict(stock_data[stock], market_data)
           
           # Keep only what we need
           results[stock] = result[['date', 'beta', 'alpha']].copy()
           
           # Cleanup
           del model
           gc.collect()
       
       return results

GPU Utilization
~~~~~~~~~~~~~~~

.. code-block:: python

   import tensorflow as tf
   
   # Enable memory growth
   gpus = tf.config.list_physical_devices('GPU')
   for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
   
   # For multiple GPUs
   strategy = tf.distribute.MirroredStrategy()
   with strategy.scope():
       model = DynamicBeta(lookback=90)

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from concurrent.futures import ProcessPoolExecutor
   
   def process_stock(args):
       stock_name, stock_data, market_data = args
       model = DynamicBeta(lookback=60)
       return stock_name, model.fit_predict(stock_data, market_data)
   
   # Parallel processing
   with ProcessPoolExecutor(max_workers=4) as executor:
       tasks = [(name, data, market) for name, data in stocks.items()]
       results = dict(executor.map(process_stock, tasks))

Validation Checklist
--------------------

Before deploying a model, verify:

.. code-block:: python

   def validate_model(model, results, data):
       """Complete validation checklist."""
       checks = []
       
       # 1. Lookahead bias
       from grubeta.utils import validate_no_lookahead
       checks.append(('No lookahead bias', validate_no_lookahead(
           results['beta'].values, 
           results['stock_return'].values,
           results['market_return'].values,
           500, verbose=False
       )))
       
       # 2. Reasonable beta range
       beta_valid = results['beta'].dropna()
       checks.append(('Beta in range [0, 3]', 
                      (beta_valid >= 0).all() and (beta_valid <= 3).all()))
       
       # 3. Sufficient R²
       from sklearn.metrics import r2_score
       mask = ~results['beta'].isna()
       r2 = r2_score(
           results['stock_return'][mask],
           results['beta'][mask] * results['market_return'][mask]
       )
       checks.append(('R² > 0.1', r2 > 0.1))
       
       # 4. Model stability
       beta_stability = results['beta'].diff().std()
       checks.append(('Beta stability < 0.05', beta_stability < 0.05))
       
       # 5. Alpha near zero
       alpha_mean = abs(results['alpha'].dropna().mean())
       checks.append(('|Alpha mean| < 0.001', alpha_mean < 0.001))
       
       # Report
       print("Validation Checklist")
       print("=" * 40)
       all_passed = True
       for name, passed in checks:
           status = "✓" if passed else "✗"
           print(f"{status} {name}")
           all_passed = all_passed and passed
       
       return all_passed
