GRU Dynamic Beta
================

.. image:: https://badge.fury.io/py/grubeta.svg
   :target: https://badge.fury.io/py/grubeta
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/grubeta.svg
   :target: https://pypi.org/project/grubeta/
   :alt: Python versions

.. image:: https://readthedocs.org/projects/grubeta/badge/?version=latest
   :target: https://grubeta.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/github/license/aslmylmz/grubeta.svg
   :target: https://github.com/aslmylmz/grubeta/blob/main/LICENSE
   :alt: License

**GRU Dynamic Beta** is a Python library for estimating time-varying systematic 
risk (beta) using Gated Recurrent Unit neural networks within the CAPM framework.

.. code-block:: python

   from grubeta import DynamicBeta
   
   model = DynamicBeta(lookback=60)
   results = model.fit_predict(stock_returns, market_returns)
   
   print(results['beta'].tail())
   model.plot_beta(results)

Key Features
------------

* **GRU-based estimation** - Captures complex temporal patterns in beta dynamics
* **Walk-forward validation** - Prevents lookahead bias with proper out-of-sample testing  
* **Composite loss function** - Balances prediction accuracy, beta stability, and alpha sparsity
* **Flexible input modes** - Use simple returns or full feature engineering
* **Built-in evaluation** - Comprehensive metrics and benchmark comparisons
* **Production-ready** - GPU support, model persistence, and extensive documentation

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install grubeta

Basic usage:

.. code-block:: python

   import pandas as pd
   from grubeta import DynamicBeta
   
   # Load your data
   stock_returns = pd.read_csv('stock.csv')['return']
   market_returns = pd.read_csv('market.csv')['return']
   
   # Estimate dynamic beta
   model = DynamicBeta(lookback=60)
   results = model.fit_predict(stock_returns, market_returns)
   
   # View results
   print(results[['beta', 'alpha']].dropna().describe())

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/basic_usage
   user_guide/advanced_features
   user_guide/preprocessing
   user_guide/evaluation
   user_guide/best_practices

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/preprocessing
   api/models
   api/evaluation
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog
   contributing

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
--------

If you use GRU Dynamic Beta in your research, please cite:

.. code-block:: bibtex

   @software{grubeta2026,
     author = {Yılmaz, Ahmet Selim},
     title = {GRU Dynamic Beta: Neural Network-Based Time-Varying Systematic Risk Estimation},
     year = {2026},
     url = {https://github.com/aslmylmz/grubeta}
   }

License
-------

GRU Dynamic Beta is released under the MIT License. See the 
`LICENSE <https://github.com/aslmylmz/grubeta/blob/main/LICENSE>`_ file for details.
