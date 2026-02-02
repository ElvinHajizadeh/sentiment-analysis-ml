import pandas as pd
import numpy as np

# Load datasets
train = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')
val = pd.read_csv('validation.csv')

print("=" * 50)
print("TRAINING DATA")
print("=" * 50)
print(f"Shape: {train.shape}")
print(f"Columns: {train.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(train.head())
print(f"\nData types:\n{train.dtypes}")
print(f"\nMissing values:\n{train.isnull().sum()}")
print(f"\nUnique values in each column:")
for col in train.columns:
    print(f"  {col}: {train[col].nunique()}")

print("\n" + "=" * 50)
print("TEST DATA")
print("=" * 50)
print(f"Shape: {test.shape}")
print(f"Columns: {test.columns.tolist()}")
print(f"First 3 rows:\n{test.head(3)}")

print("\n" + "=" * 50)
print("VALIDATION DATA")
print("=" * 50)
print(f"Shape: {val.shape}")
print(f"Columns: {val.columns.tolist()}")
print(f"First 3 rows:\n{val.head(3)}")
