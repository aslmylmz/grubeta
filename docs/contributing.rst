Contributing
============

Thank you for your interest in contributing to GRU Dynamic Beta!

Ways to Contribute
------------------

* **Bug Reports**: Submit issues for bugs you encounter
* **Feature Requests**: Suggest new features or improvements
* **Documentation**: Help improve or translate documentation
* **Code**: Submit pull requests for bug fixes or new features
* **Examples**: Share notebooks or use cases

Development Setup
-----------------

Prerequisites
~~~~~~~~~~~~~

* Python 3.8 or higher
* Git

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/aslmylmz/grubeta.git
   cd grubeta
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev]"

Code Style
----------

We follow these standards:

* **Black** for code formatting (line length 88)
* **isort** for import sorting
* **flake8** for linting
* **mypy** for type checking
* **Google-style docstrings**

Running Checks
~~~~~~~~~~~~~~

.. code-block:: bash

   # Format code
   black grubeta tests
   isort grubeta tests
   
   # Lint
   flake8 grubeta tests
   
   # Type check
   mypy grubeta

Testing
-------

We use pytest for testing.

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=grubeta --cov-report=html
   
   # Run specific test
   pytest tests/test_core.py::test_fit_predict

Pull Request Process
--------------------

1. Create a branch for your feature or fix
2. Make your changes with clear, atomic commits
3. Add tests for any new functionality
4. Update documentation if needed
5. Run all checks
6. Submit a pull request

License
-------

By contributing, you agree that your contributions will be licensed 
under the MIT License.
