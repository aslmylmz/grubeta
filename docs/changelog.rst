Changelog
=========

All notable changes to GRU Dynamic Beta will be documented here.

[0.1.0] - 2026-01-23
--------------------

Initial release.

Added
~~~~~

* Core ``DynamicBeta`` class for GRU-based beta estimation
* ``DataPreprocessor`` for feature engineering with lookahead bias prevention
* ``BetaEvaluator`` for comprehensive model evaluation
* Walk-forward validation with anchored expanding window
* Composite loss function (accuracy + stability + sparsity)
* Multiple input modes (simple returns vs. full features)
* Diagnostic tests for lookahead bias detection
* Benchmark comparison utilities (rolling OLS, EWMA, static)
* Model persistence (save/load)
* GPU acceleration support
* Comprehensive documentation and examples

Technical Details
~~~~~~~~~~~~~~~~~

* Dual-pathway GRU architecture (beta + alpha pathways)
* Configurable hyperparameters via ``DynamicBetaConfig``
* Feature engineering via ``FeatureConfig``
* Support for technical indicators (requires ``ta`` library)
* Macroeconomic feature integration

Future Plans
------------

[0.2.0] - Planned
~~~~~~~~~~~~~~~~~

* Attention mechanism for beta pathway
* Transformer-based alternative model
* Regime detection integration
* Real-time prediction mode
* Extended benchmark comparisons

[0.3.0] - Planned
~~~~~~~~~~~~~~~~~

* Multi-asset portfolio beta estimation
* Factor model extension (Fama-French)
* Confidence intervals for beta estimates
* Online learning mode
