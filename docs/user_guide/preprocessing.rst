Data Preprocessing
==================

This guide covers the data preprocessing pipeline in detail.

Overview
--------

The preprocessing pipeline transforms raw OHLCV and macro data into 
features suitable for the GRU model while preventing lookahead bias.

.. code-block:: python

   from grubeta import DataPreprocessor, FeatureConfig
   
   # Create preprocessor
   preprocessor = DataPreprocessor(FeatureConfig())
   
   # Two modes:
   # 1. Simple (returns only)
   data = preprocessor.prepare_simple(stock_returns, market_returns)
   
   # 2. Full (OHLCV + macro)
   data = preprocessor.prepare(stock_df, market_df, macro_df)

Lookahead Bias Prevention
-------------------------

Lookahead bias occurs when future information leaks into features used
for prediction. This is **critical** to prevent in time series models.

How GRU Dynamic Beta Prevents Lookahead
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Feature Lagging**: All features are shifted by 1 day

   .. code-block:: python
   
      # When predicting return at time T:
      # - Features use data from T-1 and earlier ONLY
      # - No information from time T is used
      
      config = FeatureConfig(lag_features=True)  # Default

2. **Walk-Forward Validation**: Model only sees historical data

   .. code-block:: text
   
      Training Window -> Predict -> Retrain -> Predict -> ...
      [---Train---]      [OOS]
                    [---Train---]    [OOS]
                               [---Train---]    [OOS]

3. **Expanding Window Scaling**: Scalers use historical data only

   .. code-block:: python
   
      # At each step, scalers are refit on all available history
      # Never includes future data

Verifying No Lookahead
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grubeta.utils import validate_no_lookahead
   
   passed = validate_no_lookahead(
       results['beta'].values,
       results['stock_return'].values,
       results['market_return'].values,
       initial_size=500,
       verbose=True
   )
   
   # Output:
   #   ✓ Beta correlation with return t+1: 0.023
   #   ✓ Beta correlation with return t+5: 0.018
   #   ✓ Beta correlation with return t+10: 0.012
   #   ✓ Beta correlation with return t+20: 0.008

Input Data Requirements
-----------------------

Stock/Market OHLCV Data
~~~~~~~~~~~~~~~~~~~~~~~

Required columns:

.. code-block:: text

   | Column  | Required | Description                    |
   |---------|----------|--------------------------------|
   | Date    | Yes      | Trading date                   |
   | Close   | Yes      | Closing price                  |
   | Open    | No       | Opening price                  |
   | High    | No       | Daily high                     |
   | Low     | No       | Daily low                      |
   | Volume  | No       | Trading volume                 |

Column name variations are automatically detected:

.. code-block:: python

   # These all work:
   df['Date'], df['date'], df['Tarih']
   df['Close'], df['close'], df['Adj Close']
   df['Volume'], df['Vol.'], df['Hac.']

Macro Data
~~~~~~~~~~

.. code-block:: text

   | Column  | Required | Description                    |
   |---------|----------|--------------------------------|
   | Date    | Yes      | Date (will be merged)          |
   | *       | No       | Any macro indicators           |

Example macro data:

.. code-block:: python

   macro_df = pd.DataFrame({
       'Date': dates,
       'vix': vix_values,
       'fed_rate': rates,
       'credit_spread': spreads,
       'unemployment': unemployment,
   })

Feature Groups
--------------

Market Technical Features
~~~~~~~~~~~~~~~~~~~~~~~~~

Computed from market OHLCV:

* **Moving Average Ratios**: ``market_ma_5_ratio``, ``market_ma_20_ratio``, etc.
* **Momentum**: ``market_roc_1d``, ``market_roc_5d``, ``market_rsi``, ``market_macd_diff``
* **Trend**: ``market_adx``, ``market_ma_50_200_cross``, ``market_cci``
* **Volatility**: ``market_volatility_5d``, ``market_atr_ratio``, ``market_bb_width``
* **Volume**: ``market_volume_ratio``, ``market_mfi``, ``market_obv_trend``
* **Position**: ``market_distance_52w_high``, ``market_bb_position``

Stock-Specific Features
~~~~~~~~~~~~~~~~~~~~~~~

Computed from stock OHLCV:

* **Relative Performance**: ``stock_relative_strength_50d``
* **Correlation**: ``stock_rolling_correlation_50d``
* **Traditional Beta**: ``stock_beta_traditional_252d``
* **Relative Volatility**: ``stock_relative_volatility_50d``
* **Volume**: ``stock_volume_ratio``, ``stock_mfi``, ``stock_force_index``

Calendar Features
~~~~~~~~~~~~~~~~~

Known at prediction time (not lagged):

* ``market_day_of_week``
* ``market_is_month_end``
* ``market_is_month_start``
* ``market_day_of_month``
* ``market_week_of_year``

Custom Preprocessing
--------------------

Adding Custom Features
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Standard preprocessing
   preprocessor = DataPreprocessor()
   features = preprocessor.prepare(stock_df, market_df)
   
   # Add custom feature
   custom_values = compute_my_feature(data).shift(1)  # LAG IT!
   
   features['market_features'] = np.column_stack([
       features['market_features'],
       custom_values.values
   ])
   
   features['feature_names']['market'].append('my_custom_feature')

Subclassing DataPreprocessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grubeta.preprocessing import DataPreprocessor
   
   class MyPreprocessor(DataPreprocessor):
       def _calculate_market_features(self, df):
           # Call parent method
           df = super()._calculate_market_features(df)
           
           # Add custom features
           df['market_my_indicator'] = self._compute_my_indicator(df)
           
           # Remember to lag!
           if self.config.lag_features:
               df['market_my_indicator'] = df['market_my_indicator'].shift(1)
           
           return df
       
       def _compute_my_indicator(self, df):
           # Your custom logic
           return df['Close'].rolling(10).skew()

Saving Processed Data
---------------------

Save for later use:

.. code-block:: python

   preprocessor = DataPreprocessor()
   features = preprocessor.prepare(stock_df, market_df)
   
   # Save to disk
   preprocessor.save_processed_data(
       features,
       output_dir='./processed_data',
       name='AAPL'
   )
   
   # Creates:
   # ./processed_data/AAPL.csv
   # ./processed_data/AAPL_features.json

Loading Processed Data
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from grubeta.preprocessing import load_processed_data
   
   features = load_processed_data('./processed_data/AAPL.csv')
   
   model = DynamicBeta()
   results = model.fit_predict(**features)

Data Quality Checks
-------------------

Before training, verify data quality:

.. code-block:: python

   def check_data_quality(features):
       """Verify data quality before training."""
       import numpy as np
       
       stock_ret = features['stock_returns']
       market_ret = features['market_returns']
       
       print("Data Quality Report")
       print("=" * 40)
       
       # Sample size
       n = len(stock_ret)
       print(f"Sample size: {n}")
       
       # Missing values
       stock_nan = np.isnan(stock_ret).sum()
       market_nan = np.isnan(market_ret).sum()
       print(f"Missing - Stock: {stock_nan}, Market: {market_nan}")
       
       # Extreme values
       stock_extreme = (np.abs(stock_ret) > 0.2).sum()
       market_extreme = (np.abs(market_ret) > 0.1).sum()
       print(f"Extreme values - Stock: {stock_extreme}, Market: {market_extreme}")
       
       # Correlation (sanity check)
       mask = ~np.isnan(stock_ret) & ~np.isnan(market_ret)
       corr = np.corrcoef(stock_ret[mask], market_ret[mask])[0, 1]
       print(f"Stock-Market correlation: {corr:.4f}")
       
       # Date range
       if 'dates' in features:
           print(f"Date range: {features['dates'][0]} to {features['dates'][-1]}")
       
       return n >= 500 and stock_nan == 0 and market_nan == 0
   
   # Use before training
   if check_data_quality(features):
       model.fit_predict(**features)
   else:
       print("Data quality issues detected!")
