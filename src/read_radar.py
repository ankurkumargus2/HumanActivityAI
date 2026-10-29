
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"

# Find the first CSV file
csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

if not csv_files:
    print("No CSV files found in data/raw.")
    raise SystemExit

file_path = csv_files[0]

print("Radar Human Activity Recognition")
print("=" * 45)
print("File:", file_path.name)

# Load the radar recording
df = pd.read_csv(file_path)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nNumber of rows and columns:")
print(df.shape)
