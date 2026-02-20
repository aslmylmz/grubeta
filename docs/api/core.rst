Core API
========

.. module:: grubeta.core

The core module contains the main classes for dynamic beta estimation.

DynamicBeta
-----------

.. autoclass:: grubeta.DynamicBeta
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __repr__

DynamicBetaConfig
-----------------

.. autoclass:: grubeta.DynamicBetaConfig
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from grubeta import DynamicBeta
   
   # Create model with default settings
   model = DynamicBeta()
   
   # Or with custom parameters
   model = DynamicBeta(
       lookback=60,
       lambda_beta=0.05,
       lambda_alpha=0.5,
       lambda_alpha_smooth=0.1
   )
   
   # Fit and predict
   results = model.fit_predict(stock_returns, market_returns)

Using Configuration Object
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grubeta import DynamicBeta, DynamicBetaConfig
   
   config = DynamicBetaConfig(
       lookback=90,
       initial_train_size=500,
       wf_step_size=126,
       learning_rate=1e-4,
       lambda_beta=0.05,
       lambda_alpha=0.5,
       lambda_alpha_smooth=0.1
   )
   
   model = DynamicBeta(config=config)
   results = model.fit_predict(stock_returns, market_returns)

Save and Load
~~~~~~~~~~~~~

.. code-block:: python

   # Save fitted model
   model.save('./my_model')
   
   # Load later
   model = DynamicBeta.load('./my_model')
