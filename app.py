from __future__ import annotations

import hashlib
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pdm.data import generate_demo_fleet, generate_sensor_history
from pdm.diagnostics import maintenance_guidance, risk_band
from pdm.model import (
    FEATURE_LABELS,
    SENSOR_FEATURES,
    fit_failure_model,
    generate_labeled_data,
    normalize_sensor_columns,
    predict_failure_probability,
)

st.set_page_config(page_title="Aegis | Predictive Maintenance", page_icon="⚙️", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
      :root { --ink:#e8eef7; --muted:#92a0b3; --line:rgba(168,188,214,.13); --base:#0a111b; --teal:#46dfbd; }
      .stApp { background: radial-gradient(ellipse at 68% -16%, rgba(24,78,92,.27), transparent 42%), var(--base); color:var(--ink); font-family:'DM Sans',sans-serif; }
      [data-testid="stSidebar"] { background:#0d1622; border-right:1px solid var(--line); }
      [data-testid="stHeader"] { background:rgba(10,17,27,.8); }
      h1,h2,h3 { letter-spacing:-.035em; }
      h1 { font-size:2.15rem !important; font-weight:650 !important; }
      h2 { font-size:1.3rem !important; }
      .hero { padding:14px 0 22px; border-bottom:1px solid var(--line); margin-bottom:22px; }
      .eyebrow { text-transform:uppercase; letter-spacing:.16em; font-size:.68rem; color:var(--teal); font-weight:700; margin-bottom:8px; }
      .subhead { color:var(--muted); font-size:.94rem; max-width:850px; margin-top:-8px; }
      .metric-card { background:linear-gradient(145deg,rgba(21,34,49,.96),rgba(14,24,36,.96)); border:1px solid var(--line); border-radius:14px; padding:17px 18px; min-height:111px; }
      .metric-label { color:var(--muted); font-size:.76rem; letter-spacing:.045em; }
      .metric-value { color:var(--ink); font-size:1.78rem; font-weight:650; padding-top:7px; letter-spacing:-.04em; }
      .metric-foot { color:#76869a; font-size:.72rem; padding-top:2px; }
      .section-label { color:var(--muted); text-transform:uppercase; letter-spacing:.13em; font-size:.68rem; font-weight:700; }
      .risk-critical { color:#ff787d; font-weight:700; } .risk-high { color:#ffad66; font-weight:700; }
      .risk-elevated { color:#f1cf68; font-weight:700; } .risk-low { color:var(--teal); font-weight:700; }
      .note { background:rgba(70,223,189,.055); border:1px solid rgba(70,223,189,.19); border-radius:10px; padding:12px 14px; color:#b8c9d5; font-size:.84rem; }
      .stButton>button { border-radius:8px; border:1px solid rgba(70,223,189,.35); background:rgba(70,223,189,.1); color:#a7f6e1; }
      .stButton>button:hover { border-color:var(--teal); color:white; }
      div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
      .block-container { padding-top:2rem; padding-bottom:3rem; max-width:1500px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training the demonstration model…")
def get_demo_model():
    return fit_failure_model(generate_labeled_data(), "failure_within_24h", source="Synthetic demonstration data")


def metric_card(label: str, value: str, foot: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>',
        unsafe_allow_html=True,
    )


def risk_plot(frame: pd.DataFrame) -> go.Figure:
    order = ["Low", "Elevated", "High", "Critical"]
    counts = frame["risk_band"].value_counts().reindex(order, fill_value=0)
    colors = ["#46dfbd", "#e8ce70", "#ffad66", "#ff7279"]
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=colors, marker_line_width=0, text=counts.values, textposition="outside"))
    fig.update_layout(
        height=250, margin=dict(l=4, r=8, t=14, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aebbd0", family="DM Sans"), xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title="Assets", gridcolor="rgba(168,188,214,.10)", zeroline=False), showlegend=False,
    )
    return fig


def trend_plot(history: pd.DataFrame) -> go.Figure:
    colors = {"temperature": "#ffad66", "vibration": "#be9cff", "pressure": "#50b9ff", "current": "#46dfbd"}
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055, subplot_titles=[FEATURE_LABELS[name] for name in SENSOR_FEATURES])
    for row, feature in enumerate(SENSOR_FEATURES, start=1):
        fig.add_trace(
            go.Scatter(x=history["time"], y=history[feature], mode="lines", line=dict(color=colors[feature], width=1.8), fill="tozeroy", fillcolor=colors[feature] + "14", showlegend=False),
            row=row,
            col=1,
        )
        fig.update_yaxes(gridcolor="rgba(168,188,214,.08)", zeroline=False, row=row, col=1)
    fig.update_layout(height=420, margin=dict(l=0, r=5, t=25, b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#aebbd0", family="DM Sans"))
    fig.update_annotations(font=dict(size=10, color="#8798ad"))
    fig.update_xaxes(showgrid=False)
    return fig


def score_fleet(raw: pd.DataFrame, model) -> tuple[pd.DataFrame, int]:
    data = normalize_sensor_columns(raw)
    missing = [feature for feature in SENSOR_FEATURES if feature not in data.columns]
    if missing:
        raise ValueError("Missing required sensor columns: " + ", ".join(missing))
    for feature in SENSOR_FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    valid = data[SENSOR_FEATURES].notna().all(axis=1) & np.isfinite(data[SENSOR_FEATURES].to_numpy(dtype=float)).all(axis=1)
    skipped = int((~valid).sum())
    scored = data.loc[valid].copy()
    if scored.empty:
        raise ValueError("No rows have four valid numeric sensor readings.")
    if "machine_id" not in scored.columns:
        scored.insert(0, "machine_id", [f"ROW-{index:04d}" for index in range(1, len(scored) + 1)])
    else:
        scored["machine_id"] = scored["machine_id"].fillna("Unnamed asset").astype(str)
    scored["failure_probability"] = predict_failure_probability(model, scored)
    scored["risk_band"] = scored["failure_probability"].map(risk_band)
    return scored, skipped


def display_name(feature: str) -> str:
    return FEATURE_LABELS[feature]


with st.sidebar:
    st.markdown("### ⚙️ AEGIS / PdM")
    st.caption("Industrial asset health console")
    st.markdown("---")
    model_choice = st.radio(
        "Prediction model",
        ["Demonstration model", "Train from labeled CSV"],
        help="The demo model is trained on generated data. For your site, train using historical sensor readings paired with failure outcomes.",
    )

    custom_model = None
    if model_choice == "Train from labeled CSV":
        training_file = st.file_uploader("Historical labeled telemetry", type=["csv"], key="training_csv", help="Required columns: temperature, vibration, pressure, current, plus a binary failure label.")
        if training_file is not None:
            try:
                training_frame = normalize_sensor_columns(pd.read_csv(training_file))
                candidates = [column for column in training_frame.columns if column.lower() in {"failure_within_24h", "failure", "fault", "label", "target"}]
                target_default = candidates[0] if candidates else training_frame.columns[-1]
                target = st.selectbox("Failure outcome column", training_frame.columns, index=list(training_frame.columns).index(target_default))
                file_digest = hashlib.sha256(training_file.getvalue() + target.encode("utf-8")).hexdigest()
                stored = st.session_state.get("custom_model_cache")
                if stored and stored[0] == file_digest:
                    custom_model = stored[1]
                    st.success(f"Model ready · {custom_model.rows_used:,} labeled rows")
                elif st.button("Train model", use_container_width=True):
                    with st.spinner("Training and checking a holdout set…"):
                        candidate = fit_failure_model(training_frame, target, source=training_file.name)
                    st.session_state["custom_model_cache"] = (file_digest, candidate)
                    custom_model = candidate
                    st.success(f"Model trained · {candidate.rows_used:,} labeled rows")
            except Exception as exc:
                st.error(str(exc))
        else:
            st.info("Upload a historical CSV with sensor readings and a failure outcome label.")

    st.markdown("---")
    st.markdown("**Scoring input**")
    score_file = st.file_uploader("Current telemetry CSV (optional)", type=["csv"], key="scoring_csv", help="If omitted, the app displays its reproducible simulated fleet.")
    if score_file is not None:
        st.caption("A separate current-data file keeps training records distinct from scored telemetry.")

    if model_choice == "Demonstration model":
        st.markdown("---")
        if "simulation_seed" not in st.session_state:
            st.session_state["simulation_seed"] = 7
        if st.button("↻  Regenerate demo fleet", use_container_width=True):
            st.session_state["simulation_seed"] += 1
        sample = generate_labeled_data(rows=250, seed=18)
        st.download_button("↓  Sample labeled CSV", sample.to_csv(index=False).encode("utf-8"), "sample_labeled_telemetry.csv", "text/csv", use_container_width=True)


st.markdown(
    '<div class="hero"><div class="eyebrow">Condition monitoring · Industry 4.0</div>'
    '<h1>Predict maintenance before failure.</h1>'
    '<div class="subhead">One view of fleet health, sensor drift, and the next maintenance action. Score live telemetry or explore the simulated factory fleet.</div></div>',
    unsafe_allow_html=True,
)

model = None
if model_choice == "Demonstration model":
    model = get_demo_model()
elif custom_model is not None:
    model = custom_model
else:
    st.warning("Train a model in the sidebar, or switch to the demonstration model to explore the dashboard.")
    st.stop()

try:
    if score_file is not None:
        uploaded = pd.read_csv(score_file)
        fleet, skipped_rows = score_fleet(uploaded, model)
        data_source = f"{score_file.name} · {len(fleet)} scored rows"
    else:
        if model_choice != "Demonstration model":
            st.info("Upload current telemetry in the sidebar to score assets with your trained model.")
            st.stop()
        raw_fleet = generate_demo_fleet(seed=st.session_state.get("simulation_seed", 7))
        fleet, skipped_rows = score_fleet(raw_fleet, model)
        for column in ["runtime_hours", "days_since_service", "asset_type", "site", "_wear"]:
            fleet[column] = raw_fleet.loc[fleet.index, column]
        data_source = "Simulated factory fleet · 36 assets"
except Exception as exc:
    st.error(f"Could not score this dataset: {exc}")
    st.stop()

fleet["risk_percent"] = fleet["failure_probability"] * 100
fleet["priority"] = fleet["risk_band"].map({"Critical": 0, "High": 1, "Elevated": 2, "Low": 3})
using_demo_fleet = score_file is None
attention_count = int((fleet["failure_probability"] >= 0.30).sum())
urgent_count = int((fleet["failure_probability"] >= 0.55).sum())
average_risk = fleet["risk_percent"].mean()

if skipped_rows:
    st.warning(f"Skipped {skipped_rows} row(s) with missing or invalid sensor readings.")
if model_choice == "Demonstration model":
    st.markdown('<div class="note">Demo mode uses generated sensor data and a synthetic failure label for demonstration. The risk score is not a validated plant forecast. Replace the training data, target horizon, and reference limits with site-specific history before operational use.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="note">Model trained from <b>{escape(model.source)}</b> ({model.rows_used:,} complete labeled rows; target: <code>{escape(model.target_name)}</code>). Validate performance on representative, time-separated plant data before acting on alerts.</div>', unsafe_allow_html=True)

st.markdown(f"<div class='section-label' style='margin:18px 0 12px'>Fleet overview &nbsp;·&nbsp; {escape(data_source)}</div>", unsafe_allow_html=True)
card_cols = st.columns(4, gap="medium")
with card_cols[0]:
    metric_card("ASSETS MONITORED", f"{len(fleet):,}", "Valid telemetry records")
with card_cols[1]:
    metric_card("NEEDS ATTENTION", f"{attention_count:,}", "Elevated risk or above · ≥30%")
with card_cols[2]:
    metric_card("HIGH / CRITICAL", f"{urgent_count:,}", "Priority inspection · ≥55%")
with card_cols[3]:
    metric_card("MEAN FAILURE SCORE", f"{average_risk:.1f}%", "Average uncalibrated model score")

st.markdown("")
left, right = st.columns([0.94, 1.45], gap="large")
with left:
    st.markdown("#### Risk distribution")
    st.plotly_chart(risk_plot(fleet), use_container_width=True, config={"displayModeBar": False})
with right:
    st.markdown("#### Fleet triage")
    table = fleet.sort_values(["priority", "failure_probability"], ascending=[True, False]).copy()
    table_view = table[["machine_id", "risk_band", "risk_percent", *SENSOR_FEATURES]].rename(
        columns={"machine_id": "Asset", "risk_band": "Risk", "risk_percent": "Failure score %", **{key: display_name(key) for key in SENSOR_FEATURES}}
    )
    st.dataframe(
        table_view,
        use_container_width=True,
        hide_index=True,
        height=272,
        column_config={
            "Failure score %": st.column_config.ProgressColumn("Failure score %", min_value=0, max_value=100, format="%.1f%%"),
            "Risk": st.column_config.TextColumn("Risk"),
        },
    )

identifiers = fleet["machine_id"].astype(str).tolist()
default_asset = str(table.iloc[0]["machine_id"])
selected_asset = st.selectbox("Inspect an asset", identifiers, index=identifiers.index(default_asset), label_visibility="collapsed")
selected = fleet.loc[fleet["machine_id"].astype(str) == selected_asset].iloc[0]
probability = float(selected["failure_probability"])
band, timing, actions = maintenance_guidance(probability, selected)

st.markdown("---")
st.markdown("#### Asset condition & recommended action")
detail_left, detail_right = st.columns([1.2, 1], gap="large")
with detail_left:
    st.markdown(f"**{escape(str(selected_asset))}** &nbsp; · &nbsp; {escape(str(selected.get('asset_type', 'Asset telemetry')))} &nbsp; · &nbsp; {escape(str(selected.get('site', 'Uploaded dataset')))}")
    if "runtime_hours" in selected.index:
        st.caption(f"{int(selected['runtime_hours']):,} runtime hours &nbsp;·&nbsp; {int(selected['days_since_service'])} days since last service")
    history = None
    if using_demo_fleet:
        history_seed = sum(ord(char) for char in selected_asset)
        history = generate_sensor_history(selected, seed=history_seed)
    else:
        time_column = next((column for column in fleet.columns if str(column).strip().lower() in {"time", "timestamp", "reading_at", "datetime", "date_time"}), None)
        if time_column is not None and "machine_id" in fleet.columns:
            records = fleet.loc[fleet["machine_id"].astype(str) == str(selected_asset)].copy()
            records["time"] = pd.to_datetime(records[time_column], errors="coerce")
            records = records.dropna(subset=["time"]).sort_values("time")
            if len(records) > 1:
                history = records[["time", *SENSOR_FEATURES]]
    if history is not None:
        st.plotly_chart(trend_plot(history), use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Add multiple timestamped readings per machine to the CSV to display its sensor trend.")
with detail_right:
    risk_class = band.lower()
    target_label = escape(model.target_name.replace("_", " ").upper())
    st.markdown(f"<div class='metric-card'><div class='metric-label'>FAILURE SCORE · {target_label}</div><div class='metric-value'>{probability:.1%}</div><div class='metric-foot'><span class='risk-{risk_class}'>{band} priority</span> &nbsp;·&nbsp; {timing}</div></div>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("**Recommended next step**")
    st.info(actions[0])
    st.markdown("**Signals to review**")
    for reason in actions[1:]:
        st.markdown(f"- {reason}")
    st.caption("Reference limits are configurable demo values. Confirm sensor units, alarm thresholds, and safe actions with your reliability team.")

with st.expander("Model validation & feature importance"):
    validation = model.metrics
    st.caption(f"Random forest · {model.rows_used:,} training rows · {int(validation['validation_rows'])} holdout rows · holdout split is stratified and reproducible.")
    metric_cols = st.columns(5)
    labels = [("balanced_accuracy", "Balanced accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("accuracy", "Accuracy")]
    for column, (key, label) in zip(metric_cols, labels):
        with column:
            st.metric(label, f"{validation[key]:.0%}")
    importance = pd.DataFrame({"Sensor": [FEATURE_LABELS[name] for name in model.feature_importance], "Importance": list(model.feature_importance.values())}).sort_values("Importance", ascending=True)
    chart = px.bar(importance, x="Importance", y="Sensor", orientation="h", color="Importance", color_continuous_scale=["#1c5361", "#46dfbd"])
    chart.update_layout(height=220, margin=dict(l=0, r=0, t=5, b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#aebbd0"), coloraxis_showscale=False)
    st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
    st.caption("Feature importance summarizes how the forest used each sensor across the dataset; it is not a causal explanation for an individual alert.")

export = fleet.drop(columns=["priority", "_wear"], errors="ignore").copy()
export["failure_probability"] = export["failure_probability"].round(4)
st.download_button("↓  Download scored fleet CSV", export.to_csv(index=False).encode("utf-8"), "predictive_maintenance_scores.csv", "text/csv")
st.caption("This prototype supports decision-making; it does not directly control machinery or replace site safety procedures.")
