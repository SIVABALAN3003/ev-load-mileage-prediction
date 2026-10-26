import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

try:
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

st.set_page_config(
    page_title="EV Intelligence | Range & Energy Analytics",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#eaf1ff; --muted:#9aa9c2; --panel:#111b2e; --line:#263651; --accent:#55e6a5; }
.stApp { background: radial-gradient(circle at 12% 0%, #183354 0%, #0a1220 38%, #080d17 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
.block-container { padding-top:1.6rem; padding-bottom:3rem; max-width:1500px; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.035em; }
[data-testid="stSidebar"] { background:#0c1525; border-right:1px solid #263651; }
[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(25,41,65,.95),rgba(14,24,40,.95)); border:1px solid #263651; border-radius:16px; padding:18px 20px; }
[data-testid="stMetricLabel"] { color:#9aa9c2; }
[data-testid="stMetricValue"] { color:#f2f7ff; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color:#263651 !important; border-radius:16px; }
.stButton>button, .stDownloadButton>button { border-radius:10px; border:1px solid #3a536f; font-weight:700; min-height:2.7rem; }
.stButton>button[kind="primary"] { background:#55e6a5; color:#07150f; border:0; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background:#111b2e; }
hr { border-color:#263651; }
.eyebrow { color:#55e6a5; text-transform:uppercase; letter-spacing:.16em; font-size:.75rem; font-weight:700; }
.hero { padding:26px 28px; border:1px solid #2b405d; border-radius:22px; background:linear-gradient(120deg,rgba(26,49,77,.94),rgba(14,26,44,.88)); margin-bottom:18px; }
.hero h1 { font-size:2.45rem; margin:.25rem 0 .45rem 0; }
.hero p { color:#b2c1d7; margin:0; font-size:1rem; }
.small-note { color:#9aa9c2; font-size:.82rem; }
.status-dot { color:#55e6a5; }
</style>
""", unsafe_allow_html=True)

# ---------- Example data (synthetic workflow demo only) ----------
@st.cache_data
def demo_dataset():
    return pd.DataFrame([
        [40,90,0,40,0,"City",145,248.28],
        [40,90,100,50,0,"Mixed",150,216.00],
        [40,80,200,60,1,"Highway",180,160.00],
        [50,90,150,50,1,"Mixed",165,245.45],
        [35,80,250,40,1,"City",175,144.00],
        [45,90,50,60,0,"Highway",160,227.81],
        [30,75,300,70,1,"City",190,106.58],
        [60,95,100,55,0,"Mixed",155,330.97],
        [50,85,400,80,1,"Highway",205,186.59],
        [40,70,200,45,0,"City",160,157.50],
        [35,90,0,35,0,"Mixed",140,202.50],
        [60,80,350,65,1,"Highway",195,221.54],
        [45,75,100,50,0,"City",150,202.50],
        [50,95,250,70,1,"Mixed",180,237.50],
        [40,85,300,55,1,"Highway",185,165.41],
        [30,90,50,40,0,"City",145,167.59],
        [60,90,200,60,0,"Mixed",160,303.75],
        [45,80,350,75,1,"Highway",200,162.00],
        [50,70,150,45,0,"Mixed",155,203.23],
        [35,85,200,55,1,"City",170,157.94],
    ], columns=["Battery_Capacity_kWh","Battery_Charge_pct","Extra_Load_kg","Speed_kmh","AC_On","Road_Type","Consumption_Wh_km","Range_km"])

def numeric_or_default(df, col, default):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([default] * len(df), index=df.index)

def estimate_range(capacity, charge, usable, consumption, load, speed, ac_on, road):
    # Transparent illustrative assumptions; not validated vehicle physics.
    consumption_factor = 1 + max(load, 0) * 0.0002
    consumption_factor *= 1.08 if ac_on else 1.0
    consumption_factor *= 1.05 if road == "Highway" and speed > 90 else 1.0
    adjusted_wh_km = consumption * consumption_factor
    energy_kwh = capacity * (charge / 100) * (usable / 100)
    return energy_kwh, adjusted_wh_km, (energy_kwh * 1000 / adjusted_wh_km if adjusted_wh_km > 0 else 0)

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "model_features" not in st.session_state:
