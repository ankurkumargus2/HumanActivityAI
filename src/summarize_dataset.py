
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise SystemExit("No CSV files found in data/raw.")

records = []

print(f"Inspecting {len(csv_files)} radar recordings...\n")

for file_path in csv_files:
    try:
        df = pd.read_csv(file_path)

        # Activity name is the part before the first hyphen
        activity = file_path.stem.split("-")[0]

        records.append({
            "filename": file_path.name,
            "activity": activity,
            "points": len(df),
            "frames": (
                df["frame_idx"].nunique()
                if "frame_idx" in df.columns else 0
            ),
            "missing_values": int(df.isna().sum().sum()),
            "columns": ", ".join(df.columns)
        })

    except Exception as error:
        print(f"Could not read {file_path.name}: {error}")

summary = pd.DataFrame(records)

if summary.empty:
    raise SystemExit("No recordings could be read.")

output_file = PROCESSED_DIR / "recording_summary.csv"
summary.to_csv(output_file, index=False)

print("=== RECORDINGS PER ACTIVITY ===")
print(summary["activity"].value_counts().to_string())

print("\n=== DATASET TOTALS ===")
print("Readable recordings:", len(summary))
print("Total radar points:", summary["points"].sum())
print("Total distinct activity labels:", summary["activity"].nunique())
print("Total missing values:", summary["missing_values"].sum())

print("\n=== FIRST 10 RECORDINGS ===")
print(summary.head(10).to_string(index=False))

print(f"\nSummary saved to: {output_file}")
