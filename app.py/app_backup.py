from pathlib import Path
from datetime import datetime
import sqlite3
import json
import hashlib
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# 1. PROJECT PATHS AND PAGE SETTINGS
# =====================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "har_random_forest.joblib"
DB_DIR = PROJECT_DIR / "data" / "processed"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "prediction_history.db"

st.set_page_config(
    page_title="Radar AI | Activity Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Futuristic dashboard styling
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp {
    background:
      radial-gradient(ellipse at 10% 0%, rgba(0, 180, 255, .10), transparent 34%),
      radial-gradient(ellipse at 95% 10%, rgba(0, 255, 210, .06), transparent 28%),
      #07111f;
    color: #e6f2ff;
}
[data-testid="stHeader"] { background: rgba(7,17,31,.82); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1b2d 0%, #07111f 100%);
    border-right: 1px solid rgba(83,190,255,.18);
}
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1600px; }
h1, h2, h3 { color: #eef8ff !important; letter-spacing: -.02em; }
p, label, [data-testid="stCaptionContainer"] { color: #a8bdd2; }
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(18,43,67,.96), rgba(10,27,44,.96));
    border: 1px solid rgba(76,190,255,.20);
    border-radius: 16px;
    padding: 17px 18px;
    box-shadow: 0 8px 28px rgba(0,0,0,.14);
}
div[data-testid="stMetricLabel"] { color: #9ab8d2; }
div[data-testid="stMetricValue"] { color: #59d9ff; }
div[data-testid="stTabs"] button { color: #a8bdd2; }
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #59d9ff;
    border-bottom-color: #59d9ff;
}
div[data-testid="stForm"], div[data-testid="stExpander"] {
    border: 1px solid rgba(76,190,255,.18);
    border-radius: 14px;
}
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
    border-radius: 10px;
    border: 1px solid rgba(89,217,255,.5);
    background: linear-gradient(100deg, #087ea4, #126f89);
    color: white;
    font-weight: 600;
}
.stButton button:hover, .stDownloadButton button:hover {
    border-color: #75e4ff;
    color: white;
}
hr { border-color: rgba(120,175,210,.18); }
.small-muted { color: #8faac2; font-size: .9rem; }
.hero {
    padding: 1.35rem 1.5rem;
    border: 1px solid rgba(89,217,255,.23);
    border-radius: 20px;
    background: linear-gradient(110deg, rgba(15,43,68,.96), rgba(8,25,41,.92));
    margin-bottom: 1rem;
}
.hero-kicker { color: #59d9ff; font-size: .78rem; letter-spacing: .16em; font-weight: 700; }
.hero-title { color: #f2f9ff; font-size: 2.1rem; font-weight: 800; margin: .2rem 0; }
.hero-sub { color: #9eb8cf; }
.status-dot { color: #50e3b1; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. DATABASE
# =====================================================
def get_connection():
    return sqlite3.connect(DB_PATH)

def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                activity TEXT NOT NULL,
                score REAL NOT NULL,
                total_points INTEGER NOT NULL,
                total_frames INTEGER NOT NULL,
                all_scores TEXT NOT NULL
            )
        """)
        conn.commit()

def save_prediction(filename, file_hash, activity, score, total_points, total_frames, all_scores):
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO predictions (
                created_at, filename, file_hash, activity,
                score, total_points, total_frames, all_scores
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().astimezone().isoformat(timespec="seconds"),
            filename, file_hash, activity, float(score), int(total_points),
            int(total_frames), json.dumps(all_scores)
        ))
        conn.commit()
        return cursor.lastrowid

def load_history():
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)

initialize_database()

# =====================================================
# 3. LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["feature_columns"]

# =====================================================
# 4. INPUT VALIDATION + FEATURE EXTRACTION
# Keep feature definitions aligned with training code.
# =====================================================
SIGNAL_COLUMNS = ["x", "y", "z", "doppler", "snr"]
REQUIRED_COLUMNS = ["frame_idx"] + SIGNAL_COLUMNS

def recording_quality_report(raw_df):
    """Summarize input quality before strict validation or prediction."""
    report = {
        "rows": int(len(raw_df)),
        "columns_missing": [],
        "missing_values": {},
        "non_numeric_values": {},
        "unique_frames": 0,
        "frame_gaps": 0,
        "duplicate_rows": int(raw_df.duplicated().sum()) if not raw_df.empty else 0,
        "points_per_frame_min": 0,
        "points_per_frame_max": 0,
        "warnings": [],
    }

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(raw_df.columns))
    report["columns_missing"] = missing_columns

    for col in REQUIRED_COLUMNS:
        if col not in raw_df.columns:
            continue
        original = raw_df[col]
        report["missing_values"][col] = int(original.isna().sum())
        converted = pd.to_numeric(original, errors="coerce")
        report["non_numeric_values"][col] = int((original.notna() & converted.isna()).sum())

    if "frame_idx" in raw_df.columns:
        frames = pd.to_numeric(raw_df["frame_idx"], errors="coerce").dropna()
        if not frames.empty:
            unique_frames = np.sort(frames.unique())
            report["unique_frames"] = int(len(unique_frames))
            if len(unique_frames) > 1:
                # Count absent integer frame indices between the observed minimum and maximum.
                if np.all(np.isclose(unique_frames, np.round(unique_frames))):
                    expected = int(unique_frames[-1] - unique_frames[0] + 1)
                    report["frame_gaps"] = max(0, expected - len(unique_frames))
            points_per_frame = frames.groupby(frames).size()
            report["points_per_frame_min"] = int(points_per_frame.min())
            report["points_per_frame_max"] = int(points_per_frame.max())

    if missing_columns:
        report["warnings"].append("Missing required columns: " + ", ".join(missing_columns))
    missing_total = sum(report["missing_values"].values())
    non_numeric_total = sum(report["non_numeric_values"].values())
    if missing_total:
        report["warnings"].append(f"{missing_total:,} missing required-field value(s) detected.")
    if non_numeric_total:
        report["warnings"].append(f"{non_numeric_total:,} non-numeric required-field value(s) detected.")
    if report["frame_gaps"]:
        report["warnings"].append(f'{report["frame_gaps"]:,} frame index value(s) appear to be missing between observed frames.')
    if report["duplicate_rows"]:
        report["warnings"].append(f'{report["duplicate_rows"]:,} exact duplicate row(s) detected.')
    if report["rows"] == 0:
        report["warnings"].append("The CSV has no data rows.")
    if not report["warnings"]:
        report["warnings"].append("No basic missing-data or frame-gap issues detected.")
    return report


def validate_recording(df):
    if df.empty:
        raise ValueError("The uploaded CSV is empty.")
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Unsupported CSV format. Missing columns: {missing}")
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[REQUIRED_COLUMNS].isna().any().any():
        bad = df[REQUIRED_COLUMNS].isna().sum()
        bad = bad[bad > 0].to_dict()
        raise ValueError(f"Some required values are missing or non-numeric: {bad}")
    if not np.isfinite(df[REQUIRED_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("The CSV contains infinite or invalid numeric values.")
    if (df["frame_idx"] < 0).any():
        raise ValueError("frame_idx cannot contain negative values.")
    if len(df) > 1_000_000:
        raise ValueError("This file has over 1,000,000 rows. Please use a smaller recording.")
    return df

def extract_features(df):
    required = SIGNAL_COLUMNS + ["frame_idx"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    features = {}
    for column in SIGNAL_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            raise ValueError(f"Column '{column}' contains no usable numeric values.")
        features[f"{column}_mean"] = np.mean(values)
        features[f"{column}_std"] = np.std(values)
        features[f"{column}_min"] = np.min(values)
        features[f"{column}_max"] = np.max(values)
        features[f"{column}_q25"] = np.percentile(values, 25)
        features[f"{column}_q75"] = np.percentile(values, 75)

    frame_values = pd.to_numeric(df["frame_idx"], errors="coerce")
    valid_frames = frame_values.notna()
    if not valid_frames.any():
        raise ValueError("No valid frame_idx values found.")
    points_per_frame = frame_values[valid_frames].groupby(frame_values[valid_frames]).size()
    features["number_of_frames"] = frame_values.nunique()
    features["total_points"] = len(df)
    features["points_per_frame_mean"] = points_per_frame.mean()
    # Keep pandas sample-standard-deviation behavior matching the existing app.
    features["points_per_frame_std"] = points_per_frame.std()
    return features

def make_plot_layout(fig, height=390):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,22,36,.55)",
        font=dict(color="#c7d9e8"),
        margin=dict(l=12, r=12, t=45, b=12),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(135,175,205,.12)", zerolinecolor="rgba(135,175,205,.16)")
    fig.update_yaxes(gridcolor="rgba(135,175,205,.12)", zerolinecolor="rgba(135,175,205,.16)")
    return fig

# =====================================================
# 5. HEADER + MODEL STATUS
# =====================================================
st.markdown("""
<div class="hero">
  <div class="hero-kicker">HUMAN ACTIVITY RECOGNITION / RADAR INTELLIGENCE</div>
  <div class="hero-title">📡 RADAR AI MONITOR</div>
  <div class="hero-sub">Point-cloud analysis · Activity classification · Experiment history</div>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.info("Train the baseline model first so models/har_random_forest.joblib exists.")
    st.stop()

try:
    model, feature_columns = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

history_count = len(load_history())
with st.sidebar:
    st.markdown("## 📡 System panel")
    st.markdown('<span class="status-dot">● MODEL LOADED</span>', unsafe_allow_html=True)
    st.caption("Input mode: recorded point-cloud CSV")
    st.metric("Saved predictions", history_count)
    st.divider()
    st.markdown("**Expected input columns**")
    st.code("frame_idx, x, y, z, doppler, snr", language="text")
    st.caption("This prototype does not yet connect to a physical radar sensor.")

prediction_tab, history_tab, analytics_tab = st.tabs([
    "🔎  Analyze recording", "🕘  Prediction history", "📊  Analytics"
])

# =====================================================
# 6. ANALYZE RECORDING
# =====================================================
with prediction_tab:
    st.subheader("Recording analysis")
    st.caption("Upload a CSV in the same processed point-cloud format used to train the model.")

    with st.form("prediction_form", clear_on_submit=False):
        uploaded_file = st.file_uploader("Choose radar CSV", type=["csv"], key="radar_csv")
        analyze_clicked = st.form_submit_button("⚡ Analyze recording", type="primary", width="stretch")

    if analyze_clicked:
        if uploaded_file is None:
            st.warning("Please upload a CSV file first.")
        else:
            try:
                file_bytes = uploaded_file.getvalue()
                file_hash = hashlib.sha256(file_bytes).hexdigest()
                df = pd.read_csv(io.BytesIO(file_bytes))
                quality_report = recording_quality_report(df)
                st.session_state["last_quality_report"] = {
                    "filename": uploaded_file.name,
                    "report": quality_report,
                }
                df = validate_recording(df)
                features = extract_features(df)
                X = pd.DataFrame([features]).reindex(columns=feature_columns)

                missing_features = X.columns[X.isna().any()].tolist()
                if missing_features:
                    raise ValueError(
                        "The model expects features that could not be created from this file: "
                        + ", ".join(missing_features)
                    )

                prediction = str(model.predict(X)[0])
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X)[0]
                    scores = {str(label): float(prob) for label, prob in zip(model.classes_, probabilities)}
                    top_score = max(scores.values()) if scores else 0.0
                else:
                    scores, top_score = {}, 0.0

                total_frames = int(df["frame_idx"].nunique())
                record_id = save_prediction(
                    uploaded_file.name, file_hash, prediction, top_score,
                    len(df), total_frames, scores
                )
                st.session_state["last_result"] = {
                    "id": record_id, "filename": uploaded_file.name,
                    "activity": prediction, "score": top_score,
                    "points": len(df), "frames": total_frames,
                    "scores": scores, "data": df.copy(),
                    "quality": quality_report,
                }
                st.success("Analysis completed. Prediction saved to history.")
            except Exception as error:
                st.error(f"Could not analyze this recording: {error}")

    latest_quality = st.session_state.get("last_quality_report")
    if latest_quality:
        st.markdown("#### Latest uploaded file: pre-analysis quality check")
        st.caption(latest_quality["filename"])
        qr = latest_quality["report"]
        if qr["columns_missing"] or sum(qr["missing_values"].values()) or sum(qr["non_numeric_values"].values()) or qr["frame_gaps"] or qr["duplicate_rows"]:
            st.warning("The uploaded file has quality warnings. Prediction will only run if required fields pass validation.")
        else:
            st.success("No basic missing-data or frame-gap issues detected in the uploaded file.")
        qcols = st.columns(4)
        qcols[0].metric("Rows", f'{qr["rows"]:,}')
        qcols[1].metric("Unique frames", f'{qr["unique_frames"]:,}')
        qcols[2].metric("Missing frame indices", f'{qr["frame_gaps"]:,}')
        qcols[3].metric("Duplicate rows", f'{qr["duplicate_rows"]:,}')
        with st.expander("View pre-analysis warnings"):
            st.write("Missing columns:", ", ".join(qr["columns_missing"]) if qr["columns_missing"] else "None")
            st.write("Missing values:", qr["missing_values"] or "No required columns available")
            st.write("Non-numeric values:", qr["non_numeric_values"] or "No required columns available")
            for warning in qr["warnings"]:
                st.caption("• " + warning)

    result = st.session_state.get("last_result")
    if result:
        st.divider()
        st.subheader("AI classification result")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted activity", result["activity"].replace("_", " ").title())
        c2.metric("Top model score", f'{result["score"]:.1%}')
        c3.metric("Radar points", f'{result["points"]:,}')
        c4.metric("Frames", f'{result["frames"]:,}')
        st.caption(f'Recording: {result["filename"]}  ·  History record ID: {result["id"]}')
        st.info("The top model score is a model probability estimate, not verified prediction accuracy. This model has not yet been validated on your physical 24 GHz sensor.")

        data = result["data"]

        left, right = st.columns([1.25, 1])
        with left:
            st.markdown("#### 3D radar point cloud")
            max_points = min(len(data), 12000)
            if len(data) > max_points:
                cloud = data.sample(n=max_points, random_state=42)
                st.caption(f"Showing a sample of {max_points:,} of {len(data):,} points for performance.")
            else:
                cloud = data
            fig3d = px.scatter_3d(
                cloud, x="x", y="y", z="z", color="doppler",
                color_continuous_scale="Turbo", opacity=0.78,
                hover_data=["frame_idx", "snr"],
                title="Spatial distribution · color represents Doppler",
            )
            fig3d.update_traces(marker=dict(size=2))
            fig3d.update_layout(
                height=520, paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c7d9e8"),
                margin=dict(l=0, r=0, t=48, b=0),
                scene=dict(
                    bgcolor="rgba(8,22,36,.55)",
                    xaxis=dict(title="X", backgroundcolor="rgba(8,22,36,.55)", gridcolor="rgba(135,175,205,.15)"),
                    yaxis=dict(title="Y", backgroundcolor="rgba(8,22,36,.55)", gridcolor="rgba(135,175,205,.15)"),
                    zaxis=dict(title="Z", backgroundcolor="rgba(8,22,36,.55)", gridcolor="rgba(135,175,205,.15)"),
                ),
            )
            st.plotly_chart(fig3d, width="stretch")
        with right:
            st.markdown("#### Activity score distribution")
            if result["scores"]:
                score_df = pd.DataFrame({
                    "Activity": [k.replace("_", " ").title() for k in result["scores"]],
                    "Score": list(result["scores"].values()),
                }).sort_values("Score", ascending=True)
                fig_score = px.bar(
                    score_df, x="Score", y="Activity", orientation="h",
                    text=score_df["Score"].map(lambda v: f"{v:.1%}"),
                    color="Score", color_continuous_scale=["#126f89", "#59d9ff"],
                    title="Model output by class",
                )
                fig_score.update_xaxes(tickformat=".0%", range=[0, 1])
                fig_score.update_layout(coloraxis_showscale=False)
                st.plotly_chart(make_plot_layout(fig_score, 430), width="stretch")
            else:
                st.warning("This model does not provide class probabilities.")

            st.markdown("#### Recording quality checks")
            quality = result.get("quality", recording_quality_report(data))
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("CSV rows / points", f'{quality["rows"]:,}')
            q2.metric("Unique frames", f'{quality["unique_frames"]:,}')
            q3.metric("Missing frame indices", f'{quality["frame_gaps"]:,}')
            q4.metric("Duplicate rows", f'{quality["duplicate_rows"]:,}')

            missing_total = sum(quality["missing_values"].values())
            non_numeric_total = sum(quality["non_numeric_values"].values())
            if quality["columns_missing"] or missing_total or non_numeric_total or quality["frame_gaps"] or quality["duplicate_rows"]:
                st.warning("Quality check found one or more issues. Review the details below before relying on this prediction.")
            else:
                st.success("Basic required-field and frame-index checks passed.")

            with st.expander("View quality-check details", expanded=bool(
                quality["columns_missing"] or missing_total or non_numeric_total or quality["frame_gaps"] or quality["duplicate_rows"]
            )):
                st.write("**Missing required columns:**", ", ".join(quality["columns_missing"]) if quality["columns_missing"] else "None")
                st.write("**Missing values by required column:**", quality["missing_values"] or "No required columns available")
                st.write("**Non-numeric values by required column:**", quality["non_numeric_values"] or "No required columns available")
                st.write("**Points per frame:**",
                         f'min {quality["points_per_frame_min"]:,} · max {quality["points_per_frame_max"]:,}')
                for warning in quality["warnings"]:
                    st.caption("• " + warning)
            st.caption("These are basic CSV/point-cloud checks, not a guarantee that the recording is compatible with a physical radar sensor.")

        st.markdown("#### Signal characteristics")
        sig1, sig2 = st.columns(2)
        with sig1:
            fig_doppler = px.histogram(data, x="doppler", nbins=55, title="Doppler distribution")
            st.plotly_chart(make_plot_layout(fig_doppler, 340), width="stretch")
        with sig2:
            fig_snr = px.histogram(data, x="snr", nbins=55, title="Signal-to-noise ratio (SNR) distribution")
            st.plotly_chart(make_plot_layout(fig_snr, 340), width="stretch")

        with st.expander("Inspect raw rows and frame density"):
            st.dataframe(data.head(30), width="stretch", hide_index=True)
            points_per_frame = data.groupby("frame_idx").size().rename("Points")
            fig_density = px.line(
                points_per_frame.reset_index(), x="frame_idx", y="Points",
                title="Points per frame",
            )
            st.plotly_chart(make_plot_layout(fig_density, 300), width="stretch")
            st.download_button(
                "⬇️ Download uploaded recording",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name=result["filename"],
                mime="text/csv",
            )

# =====================================================
# 7. PREDICTION HISTORY
# =====================================================
with history_tab:
    st.subheader("Prediction history")
    history = load_history()
    if history.empty:
        st.info("No saved predictions yet. Analyze a radar CSV to start building history.")
    else:
        f1, f2, f3 = st.columns([2, 1, 1])
        with f1:
            search_text = st.text_input("Search filename or activity", placeholder="e.g. walking or arm_raise")
        with f2:
            activities = ["All activities"] + sorted(history["activity"].unique().tolist())
            selected_activity = st.selectbox("Activity filter", activities)
        with f3:
            min_score = st.slider("Minimum model score", 0.0, 1.0, 0.0, 0.05)

        filtered = history.copy()
        if search_text:
            mask = (
                filtered["filename"].str.contains(search_text, case=False, na=False)
                | filtered["activity"].str.contains(search_text, case=False, na=False)
            )
            filtered = filtered[mask]
        if selected_activity != "All activities":
            filtered = filtered[filtered["activity"] == selected_activity]
        filtered = filtered[filtered["score"] >= min_score]

        st.caption(f"Showing {len(filtered):,} of {len(history):,} records")
        if filtered.empty:
            st.warning("No history records match these filters.")
        else:
            display = filtered[[
                "id", "created_at", "filename", "activity", "score", "total_points", "total_frames"
            ]].copy()
            display = display.rename(columns={
                "id": "ID", "created_at": "Date/time", "filename": "Filename",
                "activity": "Predicted activity", "score": "Top model score",
                "total_points": "Radar points", "total_frames": "Frames",
            })
            display["Predicted activity"] = display["Predicted activity"].str.replace("_", " ").str.title()
            display["Top model score"] = display["Top model score"].map(lambda x: f"{x:.1%}")
            st.dataframe(display, width="stretch", hide_index=True)

            csv_data = filtered.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Export filtered history as CSV",
                data=csv_data,
                file_name="radar_prediction_history.csv",
                mime="text/csv",
            )

            st.divider()
            st.markdown("#### Inspect a saved prediction")
            selected_id = st.selectbox("Select record ID", filtered["id"].tolist())
            selected_row = filtered[filtered["id"] == selected_id].iloc[0]
            detail1, detail2, detail3 = st.columns(3)
            detail1.metric("Activity", selected_row["activity"].replace("_", " ").title())
            detail2.metric("Top model score", f'{selected_row["score"]:.1%}')
            detail3.metric("Radar points", f'{selected_row["total_points"]:,}')
            st.caption(f'File: {selected_row["filename"]}  ·  Saved at: {selected_row["created_at"]}')
            saved_scores = json.loads(selected_row["all_scores"])
            if saved_scores:
                saved_score_df = pd.DataFrame({
                    "Activity": [k.replace("_", " ").title() for k in saved_scores],
                    "Score": list(saved_scores.values()),
                }).sort_values("Score", ascending=True)
                fig_hist = px.bar(saved_score_df, x="Score", y="Activity", orientation="h", title="Saved class scores")
                fig_hist.update_xaxes(tickformat=".0%", range=[0, 1])
                st.plotly_chart(make_plot_layout(fig_hist, 350), width="stretch")

        st.divider()
        st.markdown("#### Clear history")
        confirm_clear = st.checkbox("I understand that this permanently deletes saved prediction records.")
        if st.button("Delete all prediction history", disabled=not confirm_clear):
            with get_connection() as conn:
                conn.execute("DELETE FROM predictions")
                conn.commit()
            st.session_state.pop("last_result", None)
            st.success("Prediction history cleared.")
            st.rerun()

# =====================================================
# 8. ANALYTICS
# =====================================================
with analytics_tab:
    st.subheader("Experiment analytics")
    history = load_history()
    if history.empty:
        st.info("Analyze some recordings to generate analytics.")
    else:
        total_records = len(history)
        unique_activities = history["activity"].nunique()
        mean_score = history["score"].mean()
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Saved predictions", f"{total_records:,}")
        a2.metric("Predicted classes", unique_activities)
        a3.metric("Average top score", f"{mean_score:.1%}")
        a4.metric("Total radar points", f'{history["total_points"].sum():,}')

        left, right = st.columns(2)
        with left:
            counts = history["activity"].value_counts().rename_axis("Activity").reset_index(name="Count")
            counts["Activity"] = counts["Activity"].str.replace("_", " ").str.title()
            fig_counts = px.bar(counts, x="Activity", y="Count", title="Prediction count by activity")
            st.plotly_chart(make_plot_layout(fig_counts, 360), width="stretch")
        with right:
            avg = history.groupby("activity", as_index=False)["score"].mean()
            avg["activity"] = avg["activity"].str.replace("_", " ").str.title()
            fig_avg = px.bar(avg, x="activity", y="score", title="Average top model score by predicted class")
            fig_avg.update_yaxes(tickformat=".0%", range=[0, 1])
            st.plotly_chart(make_plot_layout(fig_avg, 360), width="stretch")

        history["date"] = pd.to_datetime(history["created_at"], errors="coerce").dt.date
        daily = history.groupby("date").size().reset_index(name="Predictions")
        fig_time = px.line(daily, x="date", y="Predictions", markers=True, title="Predictions over time")
        st.plotly_chart(make_plot_layout(fig_time, 330), width="stretch")
        st.caption("These charts summarize model outputs, not ground-truth accuracy. Real-world accuracy requires labeled test recordings.")

# =====================================================
# 9. FOOTER
# =====================================================
st.divider()
st.caption("RADAR AI MONITOR · Local research prototype · Predictions and history are stored on this computer.")
