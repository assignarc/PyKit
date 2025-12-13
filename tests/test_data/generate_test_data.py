"""
Script to generate synthetic test datasets for VKPyKit testing.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

# Set random seed for reproducibility
np.random.seed(42)

print("Generating test datasets...")

# ============================================================================
# 1. Classification Dataset (for DT and MLM modules)
# ============================================================================
print("Creating classification_data.csv...")

X_class, y_class = make_classification(
    n_samples=1000,
    n_features=8,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    flip_y=0.1,  # Add some noise
    random_state=42
)

# Create feature names
feature_names = [f'feature_{i}' for i in range(1, 9)]

# Create DataFrame
classification_df = pd.DataFrame(X_class, columns=feature_names)

# Add categorical features
classification_df['category_A'] = np.random.choice(['Low', 'Medium', 'High'], size=1000)
classification_df['category_B'] = np.random.choice(['Type1', 'Type2', 'Type3', 'Type4'], size=1000)
classification_df['category_C'] = np.random.choice(['GroupA', 'GroupB'], size=1000)

# Add target variable
classification_df['target'] = y_class

# Add some outliers in feature_1
outlier_indices = np.random.choice(1000, size=20, replace=False)
classification_df.loc[outlier_indices, 'feature_1'] = classification_df['feature_1'].mean() + 5 * classification_df['feature_1'].std()

# Save to CSV
classification_df.to_csv('tests/test_data/classification_data.csv', index=False)
print(f"  ✓ Created classification_data.csv with {len(classification_df)} rows")

# ============================================================================
# 2. Regression Dataset (for LR module)
# ============================================================================
print("Creating regression_data.csv...")

X_reg, y_reg = make_regression(
    n_samples=500,
    n_features=6,
    n_informative=5,
    noise=10.0,
    random_state=42
)

# Create feature names
reg_feature_names = [f'predictor_{i}' for i in range(1, 7)]

# Create DataFrame
regression_df = pd.DataFrame(X_reg, columns=reg_feature_names)

# Add categorical features
regression_df['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=500)
regression_df['product_type'] = np.random.choice(['A', 'B', 'C'], size=500)

# Add target variable (scale it to reasonable range)
regression_df['price'] = y_reg * 10 + 1000

# Save to CSV
regression_df.to_csv('tests/test_data/regression_data.csv', index=False)
print(f"  ✓ Created regression_data.csv with {len(regression_df)} rows")

# ============================================================================
# 3. EDA Dataset (comprehensive dataset for EDA testing)
# ============================================================================
print("Creating eda_data.csv...")

n_samples = 300

eda_df = pd.DataFrame({
    # Numerical features with different distributions
    'normal_dist': np.random.normal(50, 10, n_samples),
    'uniform_dist': np.random.uniform(0, 100, n_samples),
    'skewed_dist': np.random.exponential(20, n_samples),
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.lognormal(10, 1, n_samples),
    
    # Categorical features
    'category': np.random.choice(['A', 'B', 'C', 'D'], size=n_samples, p=[0.4, 0.3, 0.2, 0.1]),
    'color': np.random.choice(['Red', 'Blue', 'Green'], size=n_samples),
    'size': np.random.choice(['Small', 'Medium', 'Large'], size=n_samples),
    
    # Binary target
    'binary_target': np.random.choice([0, 1], size=n_samples),
    
    # Feature with outliers
    'feature_with_outliers': np.concatenate([
        np.random.normal(100, 15, n_samples - 15),
        np.random.uniform(200, 300, 15)  # Add outliers
    ])
})

# Add some missing values (5% missing in specific columns)
missing_indices_age = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
missing_indices_income = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
eda_df.loc[missing_indices_age, 'age'] = np.nan
eda_df.loc[missing_indices_income, 'income'] = np.nan

# Shuffle the DataFrame
eda_df = eda_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
eda_df.to_csv('tests/test_data/eda_data.csv', index=False)
print(f"  ✓ Created eda_data.csv with {len(eda_df)} rows")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("Test Data Generation Complete!")
print("="*60)
print(f"\nDatasets created in: tests/test_data/")
print(f"  1. classification_data.csv - {len(classification_df)} rows, {len(classification_df.columns)} columns")
print(f"  2. regression_data.csv     - {len(regression_df)} rows, {len(regression_df.columns)} columns")
print(f"  3. eda_data.csv            - {len(eda_df)} rows, {len(eda_df.columns)} columns")
print("\nSample statistics:")
print("\nClassification data:")
print(classification_df.describe().iloc[:, :5])
print(f"\nTarget distribution: \n{classification_df['target'].value_counts()}")

print("\n" + "="*60)
