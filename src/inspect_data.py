
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
raw_data_dir = project_dir / "data" / "raw"

print("=== Human Activity Recognition ===")
print("Project:", project_dir)
print("Raw data folder:", raw_data_dir)

files = [p for p in raw_data_dir.rglob("*") if p.is_file()]

if files:
    print(f"\nFound {len(files)} files:")
    for file in files[:20]:
        print(file.relative_to(project_dir))
else:
    print("\nNo radar data files found yet.")
    print("Download the dataset before continuing.")
