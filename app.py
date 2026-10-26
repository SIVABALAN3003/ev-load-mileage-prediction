import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Optional machine-learning imports. The rest of the app still works without sklearn.
try:
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

st.set_page_config(
    page_title="EV Intelligence Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling: CSS is decorative; the app remains usable if styles fail ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#07111f; --panel:#101d30; --line:#263c57; --text:#eaf3ff; --muted:#9eb0c8; --mint:#62f5bd; }
.stApp { background:radial-gradient(ellipse at 12% 0%,#173657 0%,#0a1424 38%,#060b14 100%); color:var(--text); font-family:'DM Sans',sans-serif; }
.block-container { max-width:1580px; padding-top:1.25rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.025em; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0b1728,#08111e); border-right:1px solid #263c57; }
[data-testid="stMetric"] { background:linear-gradient(135deg,rgba(24,44,69,.96),rgba(13,25,42,.98)); border:1px solid #2a415e; border-radius:16px; padding:15px 17px; }
[data-testid="stMetricLabel"] { color:#a8bbd4; }
[data-testid="stMetricValue"] { color:#f4f8ff; }
.stButton>button,.stDownloadButton>button { border-radius:10px; border:1px solid #385573; font-weight:700; min-height:2.5rem; }
.stButton>button[kind="primary"] { background:linear-gradient(90deg,#62f5bd,#9af7d3); color:#06140e; border:0; }
.hero { padding:28px 30px; border:1px solid #2d4867; border-radius:22px; background:linear-gradient(115deg,rgba(23,54,83,.98),rgba(13,27,46,.92)); margin-bottom:20px; }
.eyebrow { color:#62f5bd; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:800; }
.hero h1 { font-size:clamp(2rem,4vw,3rem); margin:.35rem 0 .5rem; }
.hero p { color:#b7c8dc; margin:0; max-width:780px; }
.pill { display:inline-block; border:1px solid #2e775f; background:rgba(98,245,189,.09); color:#75f4c4; padding:5px 10px; border-radius:30px; font-size:.76rem; font-weight:700; }
.panel { background:linear-gradient(145deg,rgba(18,34,55,.88),rgba(10,20,34,.9)); border:1px solid #293f5a; border-radius:16px; padding:16px 18px; }
.small-note { color:#9eb0c8; font-size:.82rem; }
hr { border-color:#263c57; }
</style>
""", unsafe_allow_html=True)

PLOT_TEMPLATE = "plotly_dark"

def style_chart(fig, height=360):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce8f8"),
        margin=dict(l=12, r=12, t=55, b=12),
        height=height,
        legend_title_text="",
    )
    return fig

def demo_dataset():
    rng = np.random.default_rng(42)
    n = 180
    capacity = rng.choice([35, 40, 50, 60, 75], n)
    charge = rng.integers(15, 101, n)
    load = rng.integers(0, 501, n)
    speed = rng.integers(20, 121, n)
    ac = rng.integers(0, 2, n)
    road = rng.choice(["City", "Mixed", "Highway"], n, p=[0.4, 0.35, 0.25])
    road_penalty = np.select([road == "City", road == "Mixed", road == "Highway"], [14, 0, 20])
    consumption = 145 + speed * 0.35 + load * 0.045 + ac * 17 + road_penalty + rng.normal(0, 12, n)
    range_km = (capacity * charge / 100 * 0.9 * 1000) / np.maximum(consumption, 50)
    return pd.DataFrame({
        "Battery_Capacity_kWh": capacity,
        "Battery_Charge_pct": charge,
        "Extra_Load_kg": load,
        "Speed_kmh": speed,
        "AC_On": ac,
        "Road_Type": road,
        "Consumption_Wh_km": np.round(consumption, 1),
        "Range_km": np.round(range_km, 1),
    })

def numeric_series(frame, column):
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")

def estimate_range(capacity_kwh, charge_pct, usable_pct, baseline_wh_km, payload_kg, speed_kmh, ac_on, road):
    # Simple illustrative calculation only. Not validated vehicle physics.
    load_factor = 1 + max(0, payload_kg) * 0.0002
    speed_factor = 1 + max(0, speed_kmh - 60) * 0.001
    ac_factor = 1.08 if ac_on else 1.0
    road_factor = 1.04 if road == "Highway" else (1.03 if road == "City" and speed_kmh < 35 else 1.0)
    usable_energy = capacity_kwh * charge_pct / 100 * usable_pct / 100
    adjusted_consumption = baseline_wh_km * load_factor * speed_factor * ac_factor * road_factor
    estimated_range = usable_energy * 1000 / adjusted_consumption if adjusted_consumption > 0 else 0
    return usable_energy, adjusted_consumption, estimated_range

# Initialize session state before navigation.
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "model_features" not in st.session_state:
    st.session_state.model_features = []
if "model_target" not in st.session_state:
    st.session_state.model_target = None
if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = None
if "model_test_results" not in st.session_state:
    st.session_state.model_test_results = None

# Sidebar and data loading.
with st.sidebar:
    st.markdown("# ⚡ EV INTELLIGENCE")
    st.caption("PRO · RANGE & ENERGY ANALYTICS")
    st.markdown('<span class="pill">● ANALYTICS WORKSPACE</span>', unsafe_allow_html=True)
    page = st.radio(
        "NAVIGATION",
        ["Executive Overview", "EV Cockpit", "Range Simulator", "Advanced Analytics", "Dataset Lab", "AI Model Studio", "Prediction History"],
    )
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload EV CSV dataset", type=["csv"], help="CSV data is processed for this session.")
    st.caption("Without an upload, the app uses synthetic demo data—not live vehicle telemetry.")
    st.markdown("---")
    st.caption("EV Intelligence Pro · Stable Edition")

data_source = "Synthetic demo dataset"
df = None
if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        if df.empty or len(df.columns) == 0:
            st.warning("The uploaded CSV is empty. Using demo data instead.")
            df = None
        else:
            data_source = f"Uploaded dataset · {uploaded_file.name}"
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.info("Check that the file is a valid comma-separated CSV, then upload it again.")
if df is None:
    df = demo_dataset()

# Avoid failures caused by duplicate or blank column names.
df.columns = [str(c).strip() if str(c).strip() else f"Column_{i+1}" for i, c in enumerate(df.columns)]
if df.columns.duplicated().any():
    df.columns = [f"{c}_{i+1}" if list(df.columns).count(c) > 1 else c for i, c in enumerate(df.columns)]

charge = numeric_series(df, "Battery_Charge_pct")
load = numeric_series(df, "Extra_Load_kg")
range_values = numeric_series(df, "Range_km")
consumption = numeric_series(df, "Consumption_Wh_km")
