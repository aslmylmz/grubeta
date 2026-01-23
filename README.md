# GRU Dynamic Beta

[![PyPI version](https://badge.fury.io/py/grubeta.svg)](https://badge.fury.io/py/grubeta)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/grubeta/badge/?version=latest)](https://grubeta.readthedocs.io)

**GRU Dynamic Beta** is a Python library for estimating time-varying systematic risk (beta) using Gated Recurrent Unit neural networks within the CAPM framework.

## Key Features

- 🧠 **GRU-based estimation**: Captures complex temporal patterns in beta dynamics
- 📊 **Walk-forward validation**: Prevents lookahead bias with proper out-of-sample testing
- ⚖️ **Composite loss function**: Balances prediction accuracy, beta stability, and alpha sparsity
- 🔧 **Flexible input modes**: Use simple returns or full feature engineering
- 📈 **Built-in evaluation**: Comprehensive metrics and benchmark comparisons
- 🚀 **Production-ready**: GPU support, model persistence, and extensive documentation

## Installation

### Basic Installation

```bash
pip install grubeta
```

### Full Installation (with technical analysis features)

```bash
pip install grubeta[full]
```

### Development Installation

```bash
git clone https://github.com/aslmylmz/grubeta.git
cd grubeta
pip install -e ".[dev]"
```

## Quick Start

### Simple Usage (Returns Only)

The simplest way to estimate dynamic beta using only stock and market returns:

```python
import pandas as pd
from grubeta import DynamicBeta

# Load your data
stock_returns = pd.read_csv('stock_data.csv')['return']
market_returns = pd.read_csv('market_data.csv')['return']

# Estimate dynamic beta
model = DynamicBeta(lookback=60)
results = model.fit_predict(stock_returns, market_returns)

# View results
print(results[['beta', 'alpha']].dropna().describe())

# Plot beta evolution
model.plot_beta(results, title='Dynamic Beta - AAPL')
```

### Advanced Usage (With Features)

For better results, use the full preprocessing pipeline:

```python
from grubeta import DynamicBeta, DataPreprocessor, FeatureConfig

# Configure feature engineering
config = FeatureConfig(
    include_technicals=True,
    include_macro=True,
    lag_features=True  # Critical for preventing lookahead bias
)

# Prepare features
preprocessor = DataPreprocessor(config)
features = preprocessor.prepare(
    stock_df=stock_ohlcv,    # OHLCV data
    market_df=market_ohlcv,  # Market index OHLCV
    macro_df=macro_data      # Macroeconomic indicators
)

# Estimate with features
model = DynamicBeta(
    lookback=90,
    lambda_beta=0.05,   # Beta stability weight
    lambda_alpha=0.5,   # Alpha sparsity weight
)
results = model.fit_predict(**features)
```

### Model Evaluation

```python
from grubeta import BetaEvaluator
from grubeta.utils import rolling_ols_beta

# Create evaluator
evaluator = BetaEvaluator(output_dir='./results')

# Compute benchmark
ols_beta = rolling_ols_beta(
    results['stock_return'], 
    results['market_return'],
    window=252
)

# Compare models
comparison = evaluator.compare_models(
    {
        'GRU': results['beta'].values,
        'Rolling OLS': ols_beta,
    },
    results['stock_return'].values,
    results['market_return'].values,
)
print(comparison)
```

## Architecture

The GRU Dynamic Beta model uses a dual-pathway architecture:

```
Market Features ──→ [GRU] ──→ Beta(t)
                              ↓
                    R_stock = α + β × R_market
                              ↑
Stock Features ──→ [GRU] ──→ Alpha(t)
```

### Loss Function

The composite loss balances three objectives:

1. **Accuracy**: Huber loss on return predictions
2. **Stability**: L2 penalty on beta changes (temporal smoothness)
3. **Sparsity**: L1 penalty on alpha (CAPM compliance)

```
L = L_accuracy + λ_β × L_stability + λ_α × L_sparsity
```

## Configuration

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 90 | Sequence length for GRU input |
| `initial_train_size` | 500 | Samples for initial training |
| `wf_step_size` | 126 | Walk-forward step size |
| `learning_rate` | 1e-4 | Adam optimizer learning rate |
| `gru_units` | 128 | GRU hidden units |
| `dropout_rate` | 0.2 | Dropout regularization |
| `lambda_beta` | 0.05 | Beta stability weight |
| `lambda_alpha` | 0.5 | Alpha sparsity weight |

### Feature Configuration

```python
from grubeta import FeatureConfig

config = FeatureConfig(
    lag_features=True,        # Prevent lookahead bias
    include_technicals=True,  # RSI, MACD, ADX, etc.
    include_macro=True,       # Macro indicators
    include_volume=True,      # Volume-based features
    include_calendar=True,    # Day of week, month end, etc.
    ma_windows=[5, 10, 20, 50, 100, 200],
    volatility_windows=[5, 10, 20, 60],
)
```

## API Reference

### Core Classes

#### `DynamicBeta`

Main estimator class for GRU-based dynamic beta estimation.

```python
DynamicBeta(
    config: DynamicBetaConfig = None,
    lookback: int = 90,
    lambda_beta: float = 0.05,
    lambda_alpha: float = 0.5,
    **kwargs
)
```

**Methods:**

- `fit(stock_returns, market_returns, ...)` - Fit the model
- `predict(stock_returns, market_returns, ...)` - Predict beta for new data
- `fit_predict(stock_returns, market_returns, ...)` - Fit and predict with walk-forward validation
- `save(path)` - Save model to disk
- `load(path)` - Load model from disk
- `plot_beta(results, ...)` - Visualize beta trajectory

#### `DataPreprocessor`

Feature engineering pipeline with lookahead bias prevention.

```python
DataPreprocessor(config: FeatureConfig = None)
```

**Methods:**

- `prepare_simple(stock_returns, market_returns, ...)` - Minimal preprocessing
- `prepare(stock_df, market_df, macro_df)` - Full feature engineering

#### `BetaEvaluator`

Evaluation suite for dynamic beta estimates.

```python
BetaEvaluator(output_dir: str = None)
```

**Methods:**

- `evaluate(betas, stock_returns, market_returns, ...)` - Comprehensive evaluation
- `compare_models(betas_dict, stock_returns, market_returns)` - Model comparison

### Utility Functions

```python
from grubeta.utils import (
    validate_no_lookahead,  # Check for lookahead bias
    rolling_ols_beta,       # Benchmark beta calculation
    align_series,           # Align multiple time series
    compute_information_coefficient,  # Calculate IC
)
```

## Input Data Formats

### Simple Mode (Returns Only)

```python
# Pandas Series or NumPy arrays
stock_returns = df['close'].pct_change()
market_returns = market_df['close'].pct_change()
```

### Full Mode (OHLCV + Macro)

**Stock/Market OHLCV DataFrame:**
```
| Date       | Open   | High   | Low    | Close  | Volume    |
|------------|--------|--------|--------|--------|-----------|
| 2020-01-02 | 100.00 | 101.50 | 99.50  | 101.00 | 1000000   |
| 2020-01-03 | 101.00 | 102.00 | 100.50 | 101.50 | 1200000   |
```

**Macro DataFrame (optional):**
```
| Date       | vix    | fed_rate | unemployment | ...  |
|------------|--------|----------|--------------|------|
| 2020-01-02 | 12.5   | 1.75     | 3.6          | ...  |
| 2020-01-03 | 13.2   | 1.75     | 3.6          | ...  |
```

## Lookahead Bias Prevention

GRU Dynamic Beta implements strict lookahead bias prevention:

1. **Feature Lagging**: All features are lagged by 1 day (configurable)
2. **Walk-Forward Validation**: Model only sees historical data at each prediction point
3. **Expanding Window Scaling**: Scalers are refit on expanding historical data only
4. **Built-in Validation**: `validate_no_lookahead()` function to verify

```python
from grubeta.utils import validate_no_lookahead

# Check for lookahead bias
passed = validate_no_lookahead(
    results['beta'].values,
    results['stock_return'].values,
    results['market_return'].values,
    initial_size=500
)
# Output:
#   ✓ Beta correlation with return t+1: 0.023
#   ✓ Beta correlation with return t+5: 0.018
#   ✓ Beta correlation with return t+10: 0.012
#   ✓ Beta correlation with return t+20: 0.008
```

## Examples

### Portfolio Hedging

```python
from grubeta import DynamicBeta
from grubeta.utils import beta_to_hedge_ratio

# Estimate beta
model = DynamicBeta()
results = model.fit_predict(stock_returns, market_returns)

# Current beta
current_beta = results['beta'].iloc[-1]

# Calculate hedge ratio
stock_position = 100_000  # $100k position
spy_price = 450  # SPY price

hedge = beta_to_hedge_ratio(current_beta, stock_position, spy_price)
print(f"To hedge: short {hedge:.0f} shares of SPY")
```

### Multi-Asset Analysis

```python
from grubeta import DynamicBeta, BetaEvaluator

stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
results = {}

model = DynamicBeta(lookback=60)

for stock in stocks:
    stock_ret = load_returns(stock)
    results[stock] = model.fit_predict(stock_ret, market_returns)
    
# Compare beta dynamics
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
for stock, res in results.items():
    ax.plot(res['date'], res['beta'], label=stock)
ax.legend()
ax.set_title('Dynamic Beta Comparison')
plt.show()
```

## Comparison with Other Methods

| Method | Time-Varying | Captures Non-linearity | Lookahead Free | Walk-Forward |
|--------|--------------|------------------------|----------------|--------------|
| Static OLS | ❌ | ❌ | ✅ | N/A |
| Rolling OLS | ✅ | ❌ | ✅ | ✅ |
| EWMA | ✅ | ❌ | ✅ | ✅ |
| Kalman Filter | ✅ | ❌ | ✅ | ✅ |
| DCC-GARCH | ✅ | Partial | ✅ | ✅ |
| **GRU Dynamic Beta** | ✅ | ✅ | ✅ | ✅ |

## Citation

If you use GRU Dynamic Beta in your research, please cite:

```bibtex
@software{grubeta2025,
  author = {Yılmaz, Ahmet Selim},
  title = {GRU Dynamic Beta: Neural Network-Based Time-Varying Systematic Risk Estimation},
  year = {2025},
  url = {https://github.com/aslmylmz/grubeta}
}
```

## Related Work

- [arch](https://github.com/bashtage/arch) - GARCH models
- [pykalman](https://github.com/pykalman/pykalman) - Kalman Filter
- [statsmodels](https://www.statsmodels.org/) - Statistical models
- [ta](https://github.com/bukosabino/ta) - Technical Analysis library

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
