Models API
==========

.. module:: grubeta.models

The models module contains neural network architectures and loss functions.

GRUBetaModel
------------

.. autoclass:: grubeta.models.GRUBetaModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

DualPathwayGRU
--------------

.. autoclass:: grubeta.models.DualPathwayGRU
   :members:
   :undoc-members:
   :show-inheritance:

CAPMLoss
--------

.. autoclass:: grubeta.models.CAPMLoss
   :members:
   :undoc-members:

TFEnvironment
-------------

.. autoclass:: grubeta.models.TFEnvironment
   :members:
   :undoc-members:

Architecture Overview
---------------------

The GRU Dynamic Beta model uses a dual-pathway architecture:

.. code-block:: text

   Market Features ──→ [GRU (128)] ──→ [Dense (1)] ──→ Beta(t)
                                                        ↓
                              R_stock = α + β × R_market
                                                        ↑
   Stock Features ──→ [GRU (128)] ──→ [Dense (1)] ──→ Alpha(t)

Loss Function
~~~~~~~~~~~~~

The composite loss combines four objectives:

.. math::

   L = L_{accuracy} + \lambda_\beta \cdot L_{\beta\_stability} + \lambda_\alpha \cdot L_{sparsity} + \lambda_{\alpha\_smooth} \cdot L_{\alpha\_stability}

Where:

* :math:`L_{accuracy}` = Huber loss on return predictions
* :math:`L_{\beta\_stability}` = L2 penalty on :math:`\beta_{t} - \beta_{t-1}`
* :math:`L_{sparsity}` = L1 penalty on alpha values
* :math:`L_{\alpha\_stability}` = L2 penalty on :math:`\alpha_{t} - \alpha_{t-1}`

Example: Custom Model
---------------------

.. code-block:: python

   from grubeta.models import GRUBetaModel
   from grubeta import DynamicBetaConfig
   
   config = DynamicBetaConfig(
       gru_units=64,
       dropout_rate=0.3
   )
   
   model_builder = GRUBetaModel(config)
   keras_model = model_builder.build(
       n_market_features=30,
       n_stock_features=20
   )
   
   keras_model.summary()
