"""
GRU Dynamic Beta - Complete Example
====================================

This example demonstrates the full workflow for estimating
time-varying beta using the GRU Dynamic Beta library.

Requirements:
    pip install grubeta[full]
    
Data:
    This example uses synthetic data. Replace with your own
    stock and market data for real applications.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import GRU Dynamic Beta
from grubeta import DynamicBeta, DynamicBetaConfig, DataPreprocessor, BetaEvaluator
from grubeta.utils import rolling_ols_beta, validate_no_lookahead


def generate_synthetic_data(n_samples: int = 2000, seed: int = 42):
    """
    Generate synthetic data with time-varying beta.
    
    The true beta follows a sine wave pattern, making it
    ideal for testing time-varying beta estimation.
    """
    np.random.seed(seed)
    
    # Generate dates
    dates = pd.date_range('2015-01-01', periods=n_samples, freq='B')
    
    # Generate market returns
    market_returns = np.random.randn(n_samples) * 0.01  # ~1% daily vol
    
    # Time-varying true beta (sine wave between 0.8 and 1.5)
    t = np.linspace(0, 4 * np.pi, n_samples)
    true_beta = 1.15 + 0.35 * np.sin(t)
    
    # Generate stock returns: r_stock = alpha + beta * r_market + epsilon
    alpha = 0.0001  # Small positive alpha
    epsilon = np.random.randn(n_samples) * 0.008  # Idiosyncratic noise
    stock_returns = alpha + true_beta * market_returns + epsilon
    
    return pd.DataFrame({
        'date': dates,
        'stock_return': stock_returns,
        'market_return': market_returns,
        'true_beta': true_beta,
    })


def example_simple_usage():
    """
    Example 1: Simple Usage with Returns Only
    
    The simplest way to estimate dynamic beta - just provide
    stock and market returns.
    """
    print("=" * 60)
    print("Example 1: Simple Usage")
    print("=" * 60)
    
    # Generate data
    data = generate_synthetic_data(n_samples=1500)
    
    # Create and fit model
    model = DynamicBeta(
        lookback=60,            # 60-day lookback window
        initial_train_size=300, # 300 samples for initial training
        wf_step_size=63,        # Retrain every quarter
        lambda_beta=0.05,       # Beta stability weight
        lambda_alpha=0.3,       # Alpha sparsity weight
        verbose=1
    )
    
    # Fit and predict
    results = model.fit_predict(
        stock_returns=data['stock_return'].values,
        market_returns=data['market_return'].values,
        dates=data['date'].values
    )
    
    # Print summary
    print("\nResults Summary:")
    print(results[['beta', 'alpha']].dropna().describe())
    
    # Validate no lookahead bias
    print("\nLookahead Bias Check:")
    validate_no_lookahead(
        results['beta'].values,
        results['stock_return'].values,
        results['market_return'].values,
        initial_size=300
    )
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Beta comparison
    mask = ~np.isnan(results['beta'])
    axes[0].plot(results['date'][mask], data['true_beta'].values[mask], 
                 'g-', label='True Beta', alpha=0.7)
    axes[0].plot(results['date'][mask], results['beta'][mask], 
                 'b-', label='GRU Beta', alpha=0.7)
    axes[0].axhline(1.0, color='red', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('Beta')
    axes[0].legend()
    axes[0].set_title('Dynamic Beta Estimation')
    axes[0].grid(True, alpha=0.3)
    
    # Alpha
    axes[1].plot(results['date'][mask], results['alpha'][mask], 
                 'purple', alpha=0.7)
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Alpha')
    axes[1].set_xlabel('Date')
    axes[1].set_title('Dynamic Alpha Estimation')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_simple_results.png', dpi=150)
    plt.show()
    
    return results, data


def example_model_comparison():
    """
    Example 2: Compare with Benchmark Models
    
    Compare GRU beta against rolling OLS and static beta.
    """
    print("\n" + "=" * 60)
    print("Example 2: Model Comparison")
    print("=" * 60)
    
    # Generate data
    data = generate_synthetic_data(n_samples=1500)
    
    # GRU beta
    model = DynamicBeta(lookback=60, initial_train_size=300)
    gru_results = model.fit_predict(
        data['stock_return'].values,
        data['market_return'].values,
        dates=data['date'].values
    )
    
    # Rolling OLS beta (252-day window)
    ols_beta = rolling_ols_beta(
        data['stock_return'].values,
        data['market_return'].values,
        window=252
    )
    
    # Static beta
    mask = ~np.isnan(data['stock_return']) & ~np.isnan(data['market_return'])
    static_beta = np.cov(
        data['stock_return'][mask], 
        data['market_return'][mask]
    )[0, 1] / np.var(data['market_return'][mask])
    
    # Evaluate
    evaluator = BetaEvaluator()
    
    comparison = evaluator.compare_models(
        {
            'GRU': gru_results['beta'].values,
            'Rolling OLS': ols_beta,
            'Static': np.full_like(ols_beta, static_beta),
            'True': data['true_beta'].values,
        },
        data['stock_return'].values,
        data['market_return'].values,
    )
    
    print("\nModel Comparison:")
    print(comparison[['systematic_r2', 'beta_mean', 'beta_std', 'beta_stability']])
    
    # Compute tracking error vs true beta
    valid = ~np.isnan(gru_results['beta']) & ~np.isnan(ols_beta)
    gru_error = np.mean(np.abs(gru_results['beta'][valid] - data['true_beta'][valid]))
    ols_error = np.mean(np.abs(ols_beta[valid] - data['true_beta'][valid]))
    static_error = np.mean(np.abs(static_beta - data['true_beta'][valid]))
    
    print(f"\nMean Absolute Error vs True Beta:")
    print(f"  GRU:    {gru_error:.4f}")
    print(f"  OLS:    {ols_error:.4f}")
    print(f"  Static: {static_error:.4f}")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dates = data['date'].values
    ax.plot(dates, data['true_beta'], 'g-', label='True Beta', linewidth=2)
    ax.plot(dates[valid], gru_results['beta'][valid], 'b-', 
            label='GRU Beta', alpha=0.8)
    ax.plot(dates[valid], ols_beta[valid], 'orange', 
            label='Rolling OLS (252d)', alpha=0.8)
    ax.axhline(static_beta, color='red', linestyle='--', 
               label=f'Static ({static_beta:.2f})', alpha=0.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Beta')
    ax.set_title('Beta Estimation: GRU vs Traditional Methods')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_comparison.png', dpi=150)
    plt.show()


def example_custom_config():
    """
    Example 3: Custom Configuration
    
    Demonstrate how to customize model hyperparameters.
    """
    print("\n" + "=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)
    
    # Create custom configuration
    config = DynamicBetaConfig(
        lookback=90,            # Longer lookback
        initial_train_size=500,
        wf_step_size=126,       # Semi-annual retraining
        learning_rate=5e-5,     # Lower learning rate
        epochs_init=60,         # More initial epochs
        epochs_retrain=6,
        gru_units=64,           # Smaller network
        dropout_rate=0.3,       # More dropout
        lambda_beta=0.1,        # Higher stability weight
        lambda_alpha=0.3,
        verbose=1
    )
    
    # Pydantic validates configuration on construction automatically
    
    # Create model with config
    model = DynamicBeta(config=config)
    
    print(f"Model configuration:")
    print(f"  Lookback: {model.config.lookback}")
    print(f"  GRU Units: {model.config.gru_units}")
    print(f"  Lambda Beta: {model.config.lambda_beta}")
    print(f"  Lambda Alpha: {model.config.lambda_alpha}")
    
    # Generate data and fit
    data = generate_synthetic_data(n_samples=2000)
    results = model.fit_predict(
        data['stock_return'].values,
        data['market_return'].values
    )
    
    print(f"\nResults shape: {results.shape}")
    print(f"Valid beta estimates: {(~np.isnan(results['beta'])).sum()}")


def example_save_load():
    """
    Example 4: Save and Load Model
    
    Demonstrate model persistence.
    """
    print("\n" + "=" * 60)
    print("Example 4: Save and Load Model")
    print("=" * 60)
    
    # Generate data
    data = generate_synthetic_data(n_samples=1000)
    
    # Fit model
    model = DynamicBeta(lookback=30, initial_train_size=200)
    model.fit(
        data['stock_return'].values[:800],
        data['market_return'].values[:800]
    )
    
    # Save model
    model.save('./saved_model')
    print("Model saved to ./saved_model")
    
    # Load model
    loaded_model = DynamicBeta.load('./saved_model')
    print("Model loaded successfully")
    
    # Predict with loaded model
    predictions = loaded_model.predict(
        data['stock_return'].values[800:],
        data['market_return'].values[800:]
    )
    
    print(f"Predictions shape: beta={len(predictions['beta'])}")
    print(f"Mean beta: {np.nanmean(predictions['beta']):.4f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("GRU Dynamic Beta - Examples")
    print("=" * 60)
    
    # Run examples
    example_simple_usage()
    example_model_comparison()
    example_custom_config()
    example_save_load()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
