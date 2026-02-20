# Contributing to GRU Dynamic Beta

Thank you for your interest in contributing to GRU Dynamic Beta! This document provides guidelines and instructions for contributing.

## Ways to Contribute

- **Bug Reports**: Submit issues for bugs you encounter
- **Feature Requests**: Suggest new features or improvements
- **Documentation**: Help improve or translate documentation
- **Code**: Submit pull requests for bug fixes or new features
- **Examples**: Share notebooks or use cases

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git

### Installation

1. Fork and clone the repository:
```bash
git clone https://github.com/aslmylmz/grubeta.git
cd grubeta
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

We follow these coding standards:

- **Black** for code formatting (line length 88)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **Google-style docstrings**

### Running Code Quality Checks

```bash
# Format code
black grubeta tests
isort grubeta tests

# Lint
flake8 grubeta tests

# Type check
mypy grubeta
```

## Testing

We use pytest for testing. All new features should include tests.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=grubeta --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run specific test
pytest tests/test_core.py::test_fit_predict
```

### Writing Tests

```python
import pytest
import numpy as np
from grubeta import DynamicBeta

class TestDynamicBeta:
    """Tests for DynamicBeta class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n = 1000
        market = np.random.randn(n) * 0.01
        beta_true = 1.2
        stock = beta_true * market + np.random.randn(n) * 0.005
        return stock, market
    
    def test_fit_predict_shape(self, sample_data):
        """Test that fit_predict returns correct shape."""
        stock, market = sample_data
        model = DynamicBeta(lookback=30, initial_train_size=100)
        results = model.fit_predict(stock, market)
        
        assert len(results) == len(stock)
        assert 'beta' in results.columns
        assert 'alpha' in results.columns
```

## Pull Request Process

1. **Create a branch** for your feature or fix:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** with clear, atomic commits

3. **Add tests** for any new functionality

4. **Update documentation** if needed

5. **Run all checks**:
```bash
black grubeta tests
isort grubeta tests
flake8 grubeta tests
pytest
```

6. **Submit a pull request** with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots/examples if applicable

### PR Title Convention

Use conventional commit format:
- `feat: Add new feature`
- `fix: Fix bug in X`
- `docs: Update documentation`
- `test: Add tests for Y`
- `refactor: Refactor Z`

## Issue Guidelines

### Bug Reports

Please include:
- GRU Dynamic Beta version
- Python version
- Operating system
- Minimal code to reproduce
- Full error traceback
- Expected vs actual behavior

### Feature Requests

Please include:
- Use case description
- Proposed API (if applicable)
- Example usage

## Documentation

Documentation is built with Sphinx. To build locally:

```bash
cd docs
make html
# Open docs/_build/html/index.html
```

### Docstring Format

We use Google-style docstrings:

```python
def function(arg1: int, arg2: str = "default") -> bool:
    """Short description.
    
    Longer description if needed.
    
    Parameters
    ----------
    arg1 : int
        Description of arg1.
    arg2 : str, default="default"
        Description of arg2.
        
    Returns
    -------
    bool
        Description of return value.
        
    Raises
    ------
    ValueError
        When arg1 is negative.
        
    Examples
    --------
    >>> function(1, "test")
    True
    """
```

## Code Review

All submissions require review. We look for:

- Code quality and readability
- Test coverage
- Documentation
- Performance considerations
- Backward compatibility

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.

Thank you for contributing!
