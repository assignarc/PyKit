"""
Shared pytest fixtures for VKPyKit test suite.

This module provides reusable test fixtures including:
- Test datasets (classification, regression, EDA)
- Pre-trained models for testing
- Common test data
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# ============================================================================
# Path Configuration
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "test_data"


# ============================================================================
# Dataset Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def classification_df(test_data_dir):
    """Load the classification test dataset."""
    csv_path = test_data_dir / "classification_data.csv"
    return pd.read_csv(csv_path)


@pytest.fixture(scope="session")
def regression_df(test_data_dir):
    """Load the regression test dataset."""
    csv_path = test_data_dir / "regression_data.csv"
    return pd.read_csv(csv_path)


@pytest.fixture(scope="session")
def eda_df(test_data_dir):
    """Load the EDA test dataset."""
    csv_path = test_data_dir / "eda_data.csv"
    return pd.read_csv(csv_path)


# ============================================================================
# Classification Data Splits
# ============================================================================

@pytest.fixture(scope="session")
def classification_train_test_split(classification_df):
    """Return train/test split for classification data."""
    X = classification_df.drop('target', axis=1).select_dtypes(include=[np.number])
    y = classification_df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X': X,
        'y': y
    }


# ============================================================================
# Regression Data Splits
# ============================================================================

@pytest.fixture(scope="session")
def regression_train_test_split(regression_df):
    """Return train/test split for regression data."""
    X = regression_df.drop('price', axis=1).select_dtypes(include=[np.number])
    y = regression_df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X': X,
        'y': y
    }


# ============================================================================
# Pre-trained Model Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def trained_dt_model(classification_train_test_split):
    """Return a trained Decision Tree classifier."""
    data = classification_train_test_split
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(data['X_train'], data['y_train'])
    return model


@pytest.fixture(scope="session")
def trained_rf_model(classification_train_test_split):
    """Return a trained Random Forest classifier."""
    data = classification_train_test_split
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(data['X_train'], data['y_train'])
    return model


@pytest.fixture(scope="session")
def trained_lr_model(regression_train_test_split):
    """Return a trained Linear Regression model."""
    data = regression_train_test_split
    model = LinearRegression()
    model.fit(data['X_train'], data['y_train'])
    return model


# ============================================================================
# Perfect Classifier Fixture (for edge case testing)
# ============================================================================

@pytest.fixture
def perfect_classifier_data():
    """Generate a simple dataset where a classifier can achieve perfect accuracy."""
    np.random.seed(42)
    
    # Create perfectly separable data
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train a decision tree on this
    model = DecisionTreeClassifier(max_depth=2, random_state=42)
    model.fit(X_train, y_train)
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


# ============================================================================
# Simple Test Data Fixtures
# ============================================================================

@pytest.fixture
def simple_classification_data():
    """Return a minimal classification dataset for quick tests."""
    df = pd.DataFrame({
        'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'feature_2': [2.0, 4.0, 1.0, 3.0, 5.0, 7.0, 6.0, 8.0, 9.0, 10.0],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    return df


@pytest.fixture
def simple_regression_data():
    """Return a minimal regression dataset for quick tests."""
    df = pd.DataFrame({
        'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'x2': [2.0, 4.0, 6.0, 8.0, 10.0],
        'y': [3.0, 7.0, 11.0, 15.0, 19.0]  # y = 2*x1 + x2 + noise
    })
    return df


# ============================================================================
# Mock Display Fixture (to prevent IPython.display errors in tests)
# ============================================================================

@pytest.fixture
def mock_display(monkeypatch):
    """Mock IPython.display to prevent display errors during testing."""
    def mock_display_func(*args, **kwargs):
        pass
    
    try:
        from IPython.display import display
        monkeypatch.setattr("IPython.display.display", mock_display_func)
    except ImportError:
        pass
