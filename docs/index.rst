GRU Dynamic Beta
================

.. image:: https://badge.fury.io/py/grubeta.svg
   :target: https://badge.fury.io/py/grubeta
   :alt: PyPI version

.. image:: https://readthedocs.org/projects/grubeta/badge/?version=latest
   :target: https://grubeta.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/github/license/aslmylmz/grubeta.svg
   :target: https://github.com/aslmylmz/grubeta/blob/main/LICENSE
   :alt: License

Estimate how a stock's market sensitivity (beta) changes over time,
powered by neural networks with built-in safeguards against lookahead bias.

.. code-block:: python

   from grubeta import estimate_beta

   result = estimate_beta("AAPL", "SPY")
   print(result["summary"])

Key Features
------------

* **Simple convenience API** - Estimate beta in 3 lines of code with ``estimate_beta()``
* **Named presets** - ``"default"``, ``"responsive"``, ``"smooth"``, ``"research"`` — no ML tuning needed
* **GRU-based estimation** - Captures complex temporal patterns in beta dynamics
* **Walk-forward validation** - Prevents lookahead bias with proper out-of-sample testing
* **Composite loss function** - Balances prediction accuracy, beta stability, alpha sparsity, and alpha temporal smoothness
* **Built-in evaluation** - Comprehensive metrics and benchmark comparisons
* **CLI support** - ``python -m grubeta AAPL SPY`` from the command line
* **Production-ready** - GPU support, model persistence, and extensive documentation

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install grubeta[data]   # includes yfinance for ticker support

Estimate dynamic beta:

.. code-block:: python

   from grubeta import estimate_beta

   # Estimate AAPL's time-varying beta relative to S&P 500
   result = estimate_beta("AAPL", "SPY")
   print(result["summary"])

   # Access the raw beta series
   result["beta"].tail()

Or use presets for different use cases:

.. code-block:: python

   # For event studies (captures rapid changes)
   result = estimate_beta("AAPL", "SPY", preset="responsive")

   # For long-term portfolio construction
   result = estimate_beta("AAPL", "SPY", preset="smooth")

For advanced usage with full control:

.. code-block:: python

   from grubeta import DynamicBeta, DynamicBetaConfig

   model = DynamicBeta(config=DynamicBetaConfig(lookback=60, lambda_beta=0.05))
   results = model.fit_predict(stock_returns, market_returns, dates=dates)

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

   api/convenience
   api/presets
   api/exceptions
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
