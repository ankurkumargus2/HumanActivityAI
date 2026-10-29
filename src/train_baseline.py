
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_COLUMNS = ["x", "y", "z", "doppler", "snr"]


def extract_features(file_path):
    """Extract statistical features from one radar recording."""
    df = pd.read_csv(file_path)

    required_columns = SIGNAL_COLUMNS + ["frame_idx"]
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing columns: {sorted(missing_columns)}"
        )

    features = {}

    for column in SIGNAL_COLUMNS:
        values = df[column].to_numpy(dtype=float)

        features[f"{column}_mean"] = np.mean(values)
        features[f"{column}_std"] = np.std(values)
        features[f"{column}_min"] = np.min(values)
        features[f"{column}_max"] = np.max(values)
        features[f"{column}_q25"] = np.percentile(values, 25)
        features[f"{column}_q75"] = np.percentile(values, 75)

    points_per_frame = df.groupby("frame_idx").size()

    features["number_of_frames"] = df["frame_idx"].nunique()
    features["total_points"] = len(df)
    features["points_per_frame_mean"] = points_per_frame.mean()
    features["points_per_frame_std"] = points_per_frame.std()

    return features


def create_model():
    """Create a consistent preprocessing and classification pipeline."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )


def main():
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise SystemExit("No CSV files found in data/raw.")

    rows = []

    print(f"Extracting features from {len(csv_files)} recordings...")

    for file_path in csv_files:
        # Expected filename format:
        # activity-facing-person-sequence.csv
        parts = file_path.stem.split("-")

        if len(parts) < 4:
            print("Skipping unexpected filename:", file_path.name)
            continue

        activity = parts[0]
        person = parts[-2]

        features = extract_features(file_path)
        features["activity"] = activity
        features["person"] = person
        features["filename"] = file_path.name

        rows.append(features)

    dataset = pd.DataFrame(rows)

    if dataset.empty:
        raise SystemExit("No usable recordings found.")

    # Select features without labels or metadata.
    feature_columns = [
        column
        for column in dataset.columns
        if column not in ["activity", "person", "filename"]
    ]

    X = dataset[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = dataset["activity"]
    groups = dataset["person"]

    print("\nActivity counts:")
    print(y.value_counts().to_string())

    print("\nParticipants:", sorted(groups.unique()))
    print("Total recordings:", len(dataset))

    # -----------------------------------------
    # 1. Evaluate on participants not in training
    # -----------------------------------------
    logo = LeaveOneGroupOut()

    all_actual = []
    all_predictions = []

    print("\n=== LEAVE-ONE-PARTICIPANT-OUT VALIDATION ===")

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        test_person = groups.iloc[test_idx].iloc[0]
        train_people = sorted(groups.iloc[train_idx].unique())

        model = create_model()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)

        correct = int((predictions == y_test.to_numpy()).sum())

        print(
            f"\nTest participant: {test_person} | "
            f"Training participants: {train_people} | "
            f"Accuracy: {score:.3f} ({correct}/{len(y_test)})"
        )

        all_actual.extend(y_test.tolist())
        all_predictions.extend(predictions.tolist())

    print("\n=== OVERALL VALIDATION RESULTS ===")
    print(
        f"Overall accuracy: "
        f"{accuracy_score(all_actual, all_predictions):.3f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            all_actual,
            all_predictions,
            zero_division=0
        )
    )

    # -----------------------------------------
    # 2. Save extracted features
    # -----------------------------------------
    feature_path = PROCESSED_DIR / "recording_features.csv"
    dataset.to_csv(feature_path, index=False)

    print(f"\nFeatures saved to: {feature_path}")

    # -----------------------------------------
    # 3. Train final model using all recordings
    # -----------------------------------------
    print("\nTraining final model on all recordings...")

    final_model = create_model()
    final_model.fit(X, y)

    model_path = MODEL_DIR / "har_random_forest.joblib"

    joblib.dump(
        {
            "model": final_model,
            "feature_columns": feature_columns,
            "activity_labels": sorted(y.unique().tolist())
        },
        model_path
    )

    print(f"\nFinal model saved to: {model_path}")
    print("Training and evaluation complete.")


if __name__ == "__main__":
    main()
