import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="EV Intelligence Pro — Enterprise Platform", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #f8fafc; }
    .company-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0284c7 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .company-header h1 { color: #ffffff !important; font-weight: 800; font-size: 2.4rem; margin: 0; }
    .company-header p { color: #93c5fd; margin-top: 6px; font-size: 1.05rem; }
    .stat-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .stat-label { color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; }
    .stat-value { color: #f8fafc; font-size: 1.8rem; font-weight: 700; margin-top: 6px; }
    .stat-badge {
        display: inline-block; padding: 3px 8px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-top: 8px;
    }
    .badge-success { background: rgba(16, 185, 129, 0.15); color: #34d399; }
    .badge-info { background: rgba(14, 165, 233, 0.15); color: #38bdf8; }
    .section-title { color: #38bdf8; font-size: 1.3rem; font-weight: 700; margin-bottom: 1rem; }
    div[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "fleet" not in st.session_state:
    st.session_state.fleet = pd.DataFrame([
        {"vehicle_id": "EV-001", "model": "Model S Long Range", "battery_kwh": 100.0, "status": "Active", "lat": 13.0827, "lon": 80.2707},
        {"vehicle_id": "EV-002", "model": "Model 3 Standard", "battery_kwh": 60.0, "status": "Active", "lat": 13.0500, "lon": 80.2100},
        {"vehicle_id": "EV-003", "model": "SUV Performance", "battery_kwh": 90.0, "status": "Charging", "lat": 13.1000, "lon": 80.3000},
        {"vehicle_id": "EV-004", "model": "Fleet Delivery Van", "battery_kwh": 120.0, "status": "Active", "lat": 13.0200, "lon": 80.2400}
    ])

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "timestamp", "vehicle_id", "battery_kwh", "charge_percent", "extra_load_kg", "speed_kmph", "ac_on", "predicted_range_km"
    ])

# --- ML Model ---
@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 500
    cap = np.random.uniform(50, 110, n_samples)
    charge = np.random.uniform(15, 100, n_samples)
    load = np.random.uniform(0, 300, n_samples)
    speed = np.random.uniform(40, 120, n_samples)
    ac = np.random.choice([0, 1], n_samples)
    
    range_km = (cap * (charge / 100) * 4.5) - (load * 0.08) - ((speed - 60).clip(0) * 0.5) - (ac * 15) + np.random.normal(0, 5, n_samples)
    range_km = np.clip(range_km, 10, 700)
    
    X = pd.DataFrame({"battery_capacity_kwh": cap, "charge_percent": charge, "extra_load_kg": load, "speed_kmph": speed, "ac_on": ac})
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, range_km)
    return model

model = train_model()

# --- Header Banner ---
st.markdown("""
    <div class="company-header">
        <h1>⚡ EV Intelligence Pro</h1>
        <p>Enterprise Hybrid Operations, AI Range Analytics & GIS Command Center</p>
    </div>
""", unsafe_allow_html=True)

nav_selection = st.radio(
    "Navigation", 
    ["🏠 Home & Overview", "🚗 Fleet Dashboard", "🔬 Research & Analytics", "🏢 About Company"], 
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 1. HOME & OVERVIEW
# ==========================================
if nav_selection == "🏠 Home & Overview":
    st.markdown('<div class="section-title">📊 Executive Summary</div>', unsafe_allow_html=True)
    
    total_vehicles = len(st.session_state.fleet)
    active_count = len(st.session_state.fleet[st.session_state.fleet["status"] == "Active"])
    avg_range = st.session_state.history["predicted_range_km"].astype(float).mean() if not st.session_state.history.empty else 342.8
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Total Fleet Units</div><div class="stat-value">{total_vehicles}</div><span class="stat-badge badge-info">Active Operations</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Active Units</div><div class="stat-value">{active_count} / {total_vehicles}</div><span class="stat-badge badge-success">Online</span></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Avg Range</div><div class="stat-value">{avg_range:.1f} km</div><span class="stat-badge badge-info">AI Estimated</span></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="stat-card"><div class="stat-label">Fleet SOH</div><div class="stat-value">96.4%</div><span class="stat-badge badge-success">Optimal Battery</span></div>', unsafe_allow_html=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    # Speed vs Range & GIS Map
    col_map, col_chart = st.columns([1.2, 1])
    with col_map:
        st.markdown('<div class="section-title">🗺️ Active Telemetry GPS Map</div>', unsafe_allow_html=True)
        fig_map = px.scatter_mapbox(
            st.session_state.fleet, lat="lat", lon="lon", color="status", hover_name="vehicle_id",
            zoom=10, height=350, template="plotly_dark"
        )
        fig_map.update_layout(mapbox_style="carto-darkmatter", paper_bgcolor="#1e293b", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)
        
    with col_chart:
        st.markdown('<div class="section-title">📊 Range Decay Curve</div>', unsafe_allow_html=True)
        dummy_df = pd.DataFrame({"Speed": [40, 60, 80, 100, 120], "Range": [450, 410, 360, 290, 210]})
        fig_line = px.line(dummy_df, x="Speed", y="Range", markers=True, template="plotly_dark")
        fig_line.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#0f172a")
        st.plotly_chart(fig_line, use_container_width=True)

# ==========================================
# 2. FLEET DASHBOARD
# ==========================================
elif nav_selection == "🚗 Fleet Dashboard":
    selected_vehicle = st.sidebar.selectbox("Select Active Unit", st.session_state.fleet["vehicle_id"].values)
    selected_row = st.session_state.fleet[st.session_state.fleet["vehicle_id"] == selected_vehicle].iloc[0]

    left_col, right_col = st.columns([1, 1.1])
    with left_col:
        st.markdown(f'<div class="section-title">⚡ Range Prediction ({selected_vehicle})</div>', unsafe_allow_html=True)
        with st.form("pred_form"):
            battery_cap = st.number_input("Battery Capacity (kWh)", value=float(selected_row['battery_kwh']))
            charge_pct = st.slider("Charge (%)", 5.0, 100.0, 80.0)
            extra_load = st.slider("Payload (kg)", 0.0, 500.0, 75.0)
            speed = st.slider("Speed (km/h)", 10.0, 140.0, 80.0)
            ac_status = st.selectbox("AC Control", ["Off", "On"])
            submit_btn = st.form_submit_button("⚡ Predict Range", type="primary")
            
        if submit_btn:
            ac_val = 1 if ac_status == "On" else 0
            prediction = model.predict(pd.DataFrame({
                "battery_capacity_kwh": [battery_cap], "charge_percent": [charge_pct],
                "extra_load_kg": [extra_load], "speed_kmph": [speed], "ac_on": [ac_val]
            }))[0]
            st.success(f"**Predicted Range: {prediction:.1f} km**")

    with right_col:
        st.markdown('<div class="section-title">📋 Registered Inventory</div>', unsafe_allow_html=True)
        st.dataframe(st.session_state.fleet, use_container_width=True)

# ==========================================
# 3. RESEARCH & ANALYTICS
# ==========================================
elif nav_selection == "🔬 Research & Analytics":
    st.markdown('<div class="section-title">🔬 Batch Research Workspace</div>', unsafe_allow_html=True)
    st.info("Upload CSV dataset to execute batch prediction and accuracy evaluations.")

# ==========================================
# 4. ABOUT COMPANY
# ==========================================
elif nav_selection == "🏢 About Company":
    st.markdown('<div class="section-title">🏢 EV Intelligence Pro Enterprise</div>', unsafe_allow_html=True)
    st.markdown("Enterprise fleet analytics platform combining telemetry, machine learning, and battery intelligence.")
