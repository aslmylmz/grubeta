Preprocessing API
=================

.. module:: grubeta.preprocessing

The preprocessing module handles data preparation and feature engineering.

DataPreprocessor
----------------

.. autoclass:: grubeta.DataPreprocessor
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

FeatureConfig
-------------

.. autoclass:: grubeta.FeatureConfig
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: grubeta.preprocessing.load_processed_data

Example Usage
-------------

Simple Preprocessing
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grubeta import DataPreprocessor
   
   prep = DataPreprocessor()
   data = prep.prepare_simple(stock_returns, market_returns)
   
   # Use with DynamicBeta
   model = DynamicBeta()
   results = model.fit_predict(**data)

Full Feature Engineering
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grubeta import DataPreprocessor, FeatureConfig
   
   config = FeatureConfig(
       lag_features=True,
       include_technicals=True,
       include_macro=True
   )
   
   prep = DataPreprocessor(config)
   features = prep.prepare(stock_df, market_df, macro_df)
   
   model = DynamicBeta(lookback=90)
   results = model.fit_predict(**features)

Saving Processed Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   prep.save_processed_data(
       features,
       output_dir='./data',
       name='AAPL'
   )
   
   # Load later
   from grubeta.preprocessing import load_processed_data
   features = load_processed_data('./data/AAPL.csv')
