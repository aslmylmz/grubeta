# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-02-20

### Added
- `lambda_alpha_smooth` hyperparameter for L2 temporal smoothness penalty on alpha
- 4-component composite loss function (accuracy, beta stability, alpha sparsity, alpha stability)

### Changed
- `CAPMLoss.create_composite_loss` now accepts `lambda_alpha_smooth` parameter
- Default `lambda_alpha_smooth=0.1` (2x beta smoothness weight, reflecting that alpha should be smoother than beta under CAPM)

## [0.1.2] - 2026-02-11

### Fixed
- Critical: `pydantic` dependency now correctly included in published package
- Publish workflow no longer depends on deleted `testpypi` environment
- Migrated to Pydantic V2 validators (`@field_validator`, `ConfigDict`)
- Removed duplicate line in `_create_return_features`
- Aligned minimum Python version to 3.9 across all metadata
- Reverted `initial_beta` default to `0.0` for stability
- Removed stray `verify_revert.py` from repository

## [0.1.1] - 2026-02-06

### Added
- Fixed missing `pydantic` dependency
- Validated "Zero Defects" status
- Improved build configuration

## [0.1.0] - 2026-01-23

### Added
- Core `DynamicBeta` class for GRU-based beta estimation
- `DataPreprocessor` for feature engineering with lookahead bias prevention
- `BetaEvaluator` for comprehensive model evaluation
- Walk-forward validation with anchored expanding window
- Composite loss function (accuracy + stability + sparsity)
- Multiple input modes (simple returns vs. full features)
- Diagnostic tests for lookahead bias detection
- Benchmark comparison utilities (rolling OLS, EWMA, static)
- Model persistence (save/load)
- GPU acceleration support
- Comprehensive documentation and examples

### Technical Details
- Dual-pathway GRU architecture (beta + alpha pathways)
- Configurable hyperparameters via `DynamicBetaConfig`
- Feature engineering via `FeatureConfig`
- Support for technical indicators (requires `ta` library)
- Macroeconomic feature integration

## Future Plans

### [0.2.0] - Planned
- Attention mechanism for beta pathway
- Transformer-based alternative model
- Regime detection integration
- Real-time prediction mode
- Extended benchmark comparisons (DCC-GARCH, Kalman)

### [0.3.0] - Planned
- Multi-asset portfolio beta estimation
- Factor model extension (Fama-French)
- Confidence intervals for beta estimates
- Online learning mode
