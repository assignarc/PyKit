# Testing VKPyKit

This document provides instructions for running the VKPyKit test suite.

## Quick Start

```bash
# Install package with test dependencies
pip install -e ".[test]"

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src/VKPyKit --cov-report=html
```

## Test Structure

- `tests/test_EDA.py` - Tests for Exploratory Data Analysis module
- `tests/test_DT.py` - Tests for Decision Tree module  
- `tests/test_LR.py` - Tests for Linear Regression module
- `tests/test_MLM.py` - Tests for Machine Learning Models module
- `tests/conftest.py` - Shared pytest fixtures
- `tests/test_data/` - Synthetic test datasets

## Test Data

Three synthetic datasets are available in `tests/test_data/`:
- `classification_data.csv` - 1,000 rows for classification testing
- `regression_data.csv` - 500 rows for regression testing
- `eda_data.csv` - 300 rows for EDA testing

Regenerate test data:
```bash
python tests/test_data/generate_test_data.py
```

## Running Specific Tests

```bash
# Run single test file
python -m pytest tests/test_LR.py -v

# Run specific test class
python -m pytest tests/test_LR.py::TestLRMapeScore -v

# Run specific test
python -m pytest tests/test_LR.py::TestLRMapeScore::test_mape_score_basic -v
```

## Test Coverage

Generate coverage report:
```bash
python -m pytest tests/ --cov=src/VKPyKit --cov-report=html
open htmlcov/index.html  # View in browser
```

## Dependencies

Test dependencies (installed with `pip install -e ".[test]"`):
- pytest >= 7.0
- pytest-cov >= 4.0
- pytest-mock >= 3.10
