import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

st.set_page_config(page_title="EV Intelligence Pro", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# ---------- Premium visual system ----------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#07111f;--panel:#101d30;--line:#263c57;--text:#eaf3ff;--muted:#9eb0c8;--mint:#62f5bd;--blue:#60a5fa;--purple:#b59cff}
.stApp{background:radial-gradient(ellipse at 12% 0%,#173657 0%,#0a1424 38%,#060b14 100%);color:var(--text);font-family:'DM Sans',sans-serif}
.block-container{max-width:1580px;padding-top:1.35rem;padding-bottom:3rem}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;letter-spacing:-.035em}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1728,#08111e);border-right:1px solid #263c57}
[data-testid="stMetric"]{background:linear-gradient(135deg,rgba(24,44,69,.96),rgba(13,25,42,.98));border:1px solid #2a415e;border-radius:18px;padding:18px 20px;box-shadow:0 8px 28px rgba(0,0,0,.12)}
[data-testid="stMetricLabel"]{color:#a8bbd4}[data-testid="stMetricValue"]{color:#f4f8ff;font-family:'Space Grotesk',sans-serif}
div[data-testid="stVerticalBlockBorderWrapper"]{border-color:#263c57!important;border-radius:18px}
.stButton>button,.stDownloadButton>button{border-radius:11px;border:1px solid #385573;font-weight:700;min-height:2.65rem;transition:all .2s}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,#62f5bd,#9af7d3);color:#06140e;border:0}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#62f5bd;transform:translateY(-1px)}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div{background:#111e31;border-color:#2b415d}
hr{border-color:#263c57}.eyebrow{color:#62f5bd;text-transform:uppercase;letter-spacing:.18em;font-size:.72rem;font-weight:800}
.hero{padding:30px 32px;border:1px solid #2d4867;border-radius:24px;background:linear-gradient(115deg,rgba(23,54,83,.98),rgba(13,27,46,.9));margin-bottom:20px;position:relative;overflow:hidden}
.hero:after{content:'⚡';position:absolute;right:5%;top:-35px;font-size:190px;color:rgba(98,245,189,.06);transform:rotate(10deg)}
.hero h1{font-size:clamp(2rem,4vw,3.1rem);margin:.35rem 0 .5rem 0}.hero p{color:#b7c8dc;margin:0;font-size:1rem;max-width:760px}
.section-head{display:flex;align-items:center;gap:10px;color:#f3f7ff;font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:700;margin:.35rem 0 .8rem}
.section-head span{color:#62f5bd}.glass{background:linear-gradient(145deg,rgba(18,34,55,.88),rgba(10,20,34,.9));border:1px solid #293f5a;border-radius:18px;padding:18px 20px}
.pill{display:inline-block;border:1px solid #2e775f;background:rgba(98,245,189,.09);color:#75f4c4;padding:5px 10px;border-radius:30px;font-size:.76rem;font-weight:700}
.small-note{color:#9eb0c8;font-size:.82rem}.stTabs [data-baseweb="tab-list"]{gap:8px}.stTabs [data-baseweb="tab"]{background:#101d30;border-radius:10px 10px 0 0;padding:10px 16px}
</style>
""", unsafe_allow_html=True)

PLOT = "plotly_dark"
def polish(fig, height=None):
    fig.update_layout(template=PLOT, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="DM Sans", color="#dce8f8"), margin=dict(l=12,r=12,t=45,b=12),
                      legend_title_text="", hoverlabel=dict(bgcolor="#13243a"))
    if height: fig.update_layout(height=height)
    return fig

def demo_dataset():
    rng=np.random.default_rng(12)
    n=160
    capacity=rng.choice([40,50,60,75],n)
    charge=rng.integers(20,101,n)
    load=rng.integers(0,501,n)
    speed=rng.integers(20,121,n)
    ac=rng.integers(0,2,n)
    road=rng.choice(["City","Mixed","Highway"],n,p=[.4,.35,.25])
    road_factor=np.select([road=="City",road=="Mixed",road=="Highway"],[1.08,1.0,1.12])
    consumption=145 + speed*.35 + load*.045 + ac*17 + (road_factor-1)*70 + rng.normal(0,12,n)
    range_km=(capacity*charge/100*0.9*1000)/consumption
    return pd.DataFrame({"Battery_Capacity_kWh":capacity,"Battery_Charge_pct":charge,"Extra_Load_kg":load,
                         "Speed_kmh":speed,"AC_On":ac,"Road_Type":road,"Consumption_Wh_km":consumption.round(1),"Range_km":range_km.round(1)})

def as_num(df,col,default=np.nan):
    if col not in df.columns: return pd.Series([default]*len(df),index=df.index,dtype=float)
    return pd.to_numeric(df[col],errors="coerce")

def estimate_range(capacity, charge, usable, consumption, load, speed, ac_on, road):
    # Educational estimate, not validated vehicle physics.
    factor=1+max(load,0)*0.0002
    factor*=1.08 if ac_on else 1.0
    factor*=1.04 if road=="Highway" and speed>90 else 1.0
    factor*=1.03 if road=="City" and speed<35 else 1.0
    energy=capacity*(charge/100)*(usable/100)
    adjusted=consumption*factor
    return energy,adjusted,(energy*1000/adjusted if adjusted>0 else 0)

for key,default in {"prediction_history":[],"trained_model":None,"model_features":None,"model_target":None,"model_metrics":None,"model_test_results":None}.items():
    if key not in st.session_state: st.session_state[key]=default

with st.sidebar:
    st.markdown("# ⚡ EV INTELLIGENCE")
    st.caption("PRO · RANGE & ENERGY ANALYTICS")
    st.markdown('<span class="pill">● ANALYTICS WORKSPACE</span>',unsafe_allow_html=True)
    st.markdown("---")
    page=st.radio("NAVIGATION",["Executive Overview","EV Cockpit","Range Simulator","Advanced Analytics","Dataset Lab","AI Model Studio","Prediction History"],label_visibility="visible")
    st.markdown("---")
    st.markdown("**DATA CONNECTION**")
