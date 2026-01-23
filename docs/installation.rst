Installation
============

Requirements
------------

GRU Dynamic Beta requires Python 3.8 or later. The core dependencies are:

* numpy >= 1.20.0
* pandas >= 1.3.0
* scikit-learn >= 1.0.0
* tensorflow >= 2.10.0
* matplotlib >= 3.5.0

Installing from PyPI
--------------------

The simplest way to install GRU Dynamic Beta is via pip:

.. code-block:: bash

   pip install grubeta

Full Installation
-----------------

To install with all optional dependencies (technical analysis, statistical tests):

.. code-block:: bash

   pip install grubeta[full]

This includes:

* ``ta`` - Technical analysis indicators
* ``statsmodels`` - Statistical tests (ADF, Ljung-Box)
* ``seaborn`` - Enhanced visualizations
* ``scipy`` - Additional statistical functions

Development Installation
------------------------

For development or contributing:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/grubeta.git
   cd grubeta
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode with all extras
   pip install -e ".[dev]"

GPU Support
-----------

GRU Dynamic Beta automatically uses GPU acceleration if available. To enable GPU support:

**For NVIDIA GPUs:**

.. code-block:: bash

   # Install TensorFlow with GPU support
   pip install tensorflow[and-cuda]

**Verify GPU is detected:**

.. code-block:: python

   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))

For Apple Silicon (M1/M2/M3), TensorFlow will automatically use the Metal backend.

Verifying Installation
----------------------

After installation, verify everything works:

.. code-block:: python

   import grubeta
   print(grubeta.__version__)
   
   # Quick test
   import numpy as np
   from grubeta import DynamicBeta
   
   # Generate test data
   np.random.seed(42)
   market = np.random.randn(1000) * 0.01
   stock = 1.2 * market + np.random.randn(1000) * 0.005
   
   # Fit model
   model = DynamicBeta(lookback=30, initial_train_size=100)
   results = model.fit_predict(stock, market)
   
   print(f"Mean beta: {results['beta'].dropna().mean():.3f}")

Expected output:

.. code-block:: text

   0.1.0
   Mean beta: 1.198

Troubleshooting
---------------

**TensorFlow not found:**

.. code-block:: bash

   pip install --upgrade tensorflow

**Memory issues with large datasets:**

Set TensorFlow to use memory growth:

.. code-block:: python

   import tensorflow as tf
   gpus = tf.config.list_physical_devices('GPU')
   for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)

**Technical analysis features not working:**

.. code-block:: bash

   pip install ta

**Import errors:**

Ensure you're using the correct Python environment:

.. code-block:: bash

   which python  # or `where python` on Windows
   pip list | grep grubeta
