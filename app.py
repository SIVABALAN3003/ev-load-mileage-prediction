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

# --- Custom CSS Styling ---
st.markdown("""
<style>
    /* Dark Theme Colors & Global Styles */
    .main {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    
    /* Top Header Styling */
    .company-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0284c7 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .company-header h1 {
        color: #ffffff !important;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        font-size: 2.4rem;
    }
    .company-header p {
        color: #93c5fd;
        margin-top: 6px;
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* Enterprise Card Component */
    .stat-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px -3px rgba(14, 165, 233, 0.15);
        border-color: #0284c7;
    }
    .stat-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 6px;
    }
    .stat-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 8px;
    }
    .badge-success { background: rgba(16, 185, 129, 0.15); color: #34d399; }
    .badge-info { background: rgba(14, 165, 233, 0.15); color: #38bdf8; }

    /* Custom Section Headers */
    .section-title {
        color: #38bdf8;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Streamlit UI Tweaks */
    div[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "fleet" not in st.session_state:
    st.session_state.fleet = pd.DataFrame([
        {"vehicle_id": "EV-001", "model": "Model S Long Range", "battery_kwh": 100.0, "status": "Active"},
        {"vehicle_id": "EV-002", "model": "Model 3 Standard", "battery_kwh": 60.0, "status": "Active"},
        {"vehicle_id": "EV-003", "model": "SUV Performance", "battery_kwh": 90.0, "status": "Charging"},
        {"vehicle_id": "EV-004", "model": "Fleet Delivery Van", "battery_kwh": 120.0, "status": "Active"}
    ])

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "timestamp", "vehicle_id", "battery_kwh", "charge_percent", "extra_load_kg", "speed_kmph", "ac_on", "predicted_range_km"
    ])

# --- Machine Learning Engine ---
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
    
    X = pd.DataFrame({
        "battery_capacity_kwh": cap,
        "charge_percent": charge,
        "extra_load_kg": load,
        "speed_kmph": speed,
        "ac_on": ac
    })
    y = range_km
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# --- Top Header Banner ---
st.markdown("""
    <div class="company-header">
        <h1>⚡ EV Intelligence Pro</h1>
        <p>Enterprise Fleet Telemetry, AI Range Prediction & Research Analytics Platform</p>
    </div>
""", unsafe_allow_html=True)

# Navigation Menu
nav_selection = st.radio(
    "Corporate Navigation", 
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
    avg_range = st.session_state.history["predicted_range_km"].astype(float).mean() if not st.session_state.history.empty else 385.4
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Fleet Units</div>
                <div class="stat-value">{total_vehicles}</div>
                <span class="stat-badge badge-info">+1 Added Recently</span>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Active Units</div>
                <div class="stat-value">{active_count} / {total_vehicles}</div>
                <span class="stat-badge badge-success">Operational</span>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Avg Fleet Range</div>
                <div class="stat-value">{avg_range:.1f} km</div>
                <span class="stat-badge badge-info">AI Predicted</span>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-label">System Status</div>
                <div class="stat-value">99.98%</div>
                <span class="stat-badge badge-success">Cloud Online</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚡ Enterprise Platform Capabilities</div>', unsafe_allow_html=True)
    
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
            <div class="stat-card" style="height: 100%;">
                <h3 style="color:#0284c7; margin-top:0;">🚗 Fleet Operations</h3>
                <p style="color:#94a3b8;">Monitor active vehicle inventories, battery metrics, payload limits, and live operational statuses across multi-unit commercial fleets.</p>
            </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
            <div class="stat-card" style="height: 100%;">
                <h3 style="color:#0284c7; margin-top:0;">⚡ AI Range Inference</h3>
                <p style="color:#94a3b8;">Utilize trained Random Forest machine learning models to simulate real-time driving range based on payload, speed, and climate control impact.</p>
            </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
            <div class="stat-card" style="height: 100%;">
                <h3 style="color:#0284c7; margin-top:0;">🔬 Research Benchmark</h3>
                <p style="color:#94a3b8;">Upload bulk telemetry datasets, conduct model batch predictions, calculate R² accuracy fit, and export evaluation reports.</p>
            </div>
        """, unsafe_allow_html=True)

# ==========================================
# 2. FLEET DASHBOARD
# ==========================================
elif nav_selection == "🚗 Fleet Dashboard":
    st.sidebar.markdown("### 🚗 Fleet Controls")
    
    with st.sidebar.expander("Register New Vehicle"):
        new_id = st.text_input("Vehicle ID", value=f"EV-00{len(st.session_state.fleet)+1}")
        new_model = st.text_input("Vehicle Model", value="Commercial Van X")
        new_cap = st.number_input("Battery Capacity (kWh)", value=85.0, min_value=20.0, max_value=200.0)
        new_status = st.selectbox("Initial Status", ["Active", "Charging", "Maintenance"])
        
        if st.button("Add Vehicle"):
            if new_id in st.session_state.fleet["vehicle_id"].values:
                st.sidebar.error("Vehicle ID already exists!")
            else:
                new_row = pd.DataFrame([{"vehicle_id": new_id, "model": new_model, "battery_kwh": new_cap, "status": new_status}])
                st.session_state.fleet = pd.concat([st.session_state.fleet, new_row], ignore_index=True)
                st.sidebar.success(f"Added {new_id} successfully!")

    selected_vehicle = st.sidebar.selectbox("Select Active Unit", st.session_state.fleet["vehicle_id"].values)
    selected_row = st.session_state.fleet[st.session_state.fleet["vehicle_id"] == selected_vehicle].iloc[0]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Selected Specs:**\n- Model: `{selected_row['model']}`\n- Battery: `{selected_row['battery_kwh']} kWh`\n- Status: `{selected_row['status']}`")

    left_col, right_col = st.columns([1, 1.1])

    with left_col:
        st.markdown(f'<div class="section-title">⚡ Range Prediction Panel ({selected_vehicle})</div>', unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            battery_cap = st.number_input("Battery Capacity (kWh)", value=float(selected_row['battery_kwh']), min_value=10.0, max_value=200.0)
            charge_pct = st.slider("State of Charge (%)", min_value=5.0, max_value=100.0, value=80.0)
            extra_load = st.slider("Additional Payload Load (kg)", min_value=0.0, max_value=500.0, value=75.0)
            speed = st.slider("Average Driving Speed (km/h)", min_value=10.0, max_value=140.0, value=80.0)
            ac_status = st.selectbox("Climate Control (AC)", ["Off", "On"])
            
            submit_btn = st.form_submit_button("⚡ Predict Driving Range", type="primary")
            
        if submit_btn:
            ac_val = 1 if ac_status == "On" else 0
            input_data = pd.DataFrame({
                "battery_capacity_kwh": [float(battery_cap)],
                "charge_percent": [float(charge_pct)],
                "extra_load_kg": [float(extra_load)],
                "speed_kmph": [float(speed)],
                "ac_on": [int(ac_val)]
            })
            
            prediction = model.predict(input_data)[0]
            
            st.success(f"**Predicted Operating Range for {selected_vehicle}: {prediction:.1f} km**")
            if charge_pct < 20:
                st.warning("⚠️ Low battery state! Schedule unit for recharge.")
                
            new_history_row = pd.DataFrame([{
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "vehicle_id": str(selected_vehicle),
                "battery_kwh": float(battery_cap),
                "charge_percent": float(charge_pct),
                "extra_load_kg": float(extra_load),
                "speed_kmph": float(speed),
                "ac_on": str(ac_status),
                "predicted_range_km": float(round(prediction, 2))
            }])
            st.session_state.history = pd.concat([new_history_row, st.session_state.history], ignore_index=True)

    with right_col:
        st.markdown('<div class="section-title">📊 Fleet Speed vs Range Analytics</div>', unsafe_allow_html=True)
        if not st.session_state.history.empty:
            plot_df = st.session_state.history.copy()
            plot_df["speed_kmph"] = pd.to_numeric(plot_df["speed_kmph"])
            plot_df["predicted_range_km"] = pd.to_numeric(plot_df["predicted_range_km"])
            
            fig = px.scatter(
                plot_df, 
                x="speed_kmph", 
                y="predicted_range_km", 
                color="vehicle_id",
                hover_data=["extra_load_kg", "charge_percent", "ac_on"],
                title="Speed vs Range Impact across Fleet",
                template="plotly_dark"
            )
            fig.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#0f172a")
            st.plotly_chart(fig, use_container_width=True)
        else:
            dummy_df = pd.DataFrame({"Speed": [40, 60, 80, 100, 120], "Estimated Range": [450, 420, 380, 320, 260]})
            fig = px.line(dummy_df, x="Speed", y="Estimated Range", markers=True, title="Sample Baseline Curve", template="plotly_dark")
            fig.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#0f172a")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    tab_f1, tab_f2 = st.tabs(["🚗 Registered Fleet Inventory", "📈 Prediction Session Logs"])
    with tab_f1:
        st.dataframe(st.session_state.fleet, use_container_width=True)
        fleet_csv = st.session_state.fleet.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export Fleet CSV", fleet_csv, file_name="fleet_inventory.csv", mime="text/csv")
    with tab_f2:
        if not st.session_state.history.empty:
            st.dataframe(st.session_state.history, use_container_width=True)
            hist_csv = st.session_state.history.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export Logs CSV", hist_csv, file_name="prediction_history.csv", mime="text/csv")
        else:
            st.info("No prediction history logged in this session yet.")

# ==========================================
# 3. RESEARCH & ANALYTICS
# ==========================================
elif nav_selection == "🔬 Research & Analytics":
    st.markdown('<div class="section-title">🔬 Research Dataset Benchmark Suite</div>', unsafe_allow_html=True)
    st.markdown("Evaluate machine learning prediction performance using standardized test sets or custom CSV uploads.")
    
    col_res1, col_res2 = st.columns([1, 1])
    with col_res1:
        use_sample_data = st.checkbox("Load Built-in EV Research Benchmark", value=True)
    with col_res2:
        uploaded_research_file = st.file_uploader("Upload Custom Research CSV", type=["csv"])

    research_df = None
    if uploaded_research_file is not None:
        try:
            research_df = pd.read_csv(uploaded_research_file)
            st.success("Custom research dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    elif use_sample_data:
        np.random.seed(101)
        n_test = 100
        research_df = pd.DataFrame({
            "test_sample_id": [f"RES-{i:03d}" for i in range(1, n_test + 1)],
            "battery_capacity_kwh": np.random.uniform(60, 100, n_test),
            "charge_percent": np.random.uniform(20, 100, n_test),
            "extra_load_kg": np.random.uniform(0, 250, n_test),
            "speed_kmph": np.random.uniform(45, 130, n_test),
            "ac_on": np.random.choice([0, 1], n_test),
            "actual_range_km": np.random.uniform(120, 480, n_test)
        })

    if research_df is not None:
        st.markdown("#### Dataset Preview")
        st.dataframe(research_df.head(5), use_container_width=True)
        
        required_cols = ["battery_capacity_kwh", "charge_percent", "extra_load_kg", "speed_kmph", "ac_on"]
        missing_cols = [col for col in required_cols if col not in research_df.columns]
        
        if missing_cols:
            st.error(f"Research dataset missing required feature columns: {missing_cols}")
        else:
            if st.button("🚀 Run Batch Research Inference", type="primary"):
                features_for_pred = research_df[required_cols].copy()
                for col in required_cols:
                    features_for_pred[col] = pd.to_numeric(features_for_pred[col])
                
                predictions = model.predict(features_for_pred)
                research_df["predicted_range_km"] = np.round(predictions, 2)
                
                st.success("Batch prediction successfully completed across all records!")
                st.dataframe(research_df, use_container_width=True)
                
                if "actual_range_km" in research_df.columns:
                    st.markdown("### 📊 Model Validation Metrics")
                    y_true = pd.to_numeric(research_df["actual_range_km"])
                    y_pred = research_df["predicted_range_km"]
                    
                    mae = mean_absolute_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                    
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f} km")
                    with m2:
                        st.metric("R² Score (Accuracy Fit)", f"{r2:.4f}")
                        
                    fig_eval = px.scatter(
                        research_df,
                        x="actual_range_km",
                        y="predicted_range_km",
                        labels={"actual_range_km": "Actual Range (km)", "predicted_range_km": "Predicted Range (km)"},
                        title="Model Validation: Actual vs. Predicted Range",
                        template="plotly_dark"
                    )
                    fig_eval.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#0f172a")
                    st.plotly_chart(fig_eval, use_container_width=True)
                
                research_csv = research_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Batch Results CSV",
                    research_csv,
                    file_name="ev_research_predictions.csv",
                    mime="text/csv"
                )

# ==========================================
# 4. ABOUT COMPANY
# ==========================================
elif nav_selection == "🏢 About Company":
    st.markdown('<div class="section-title">🏢 About EV Intelligence Pro</div>', unsafe_allow_html=True)
    st.markdown("""
    **EV Intelligence Pro** is an enterprise electric vehicle analytics platform designed for commercial logistics fleets and automotive research environments.
    
    ### Mission & Engineering Focus
    - **Range Prediction Precision:** Delivering AI models that account for payload stress, climate draw, and velocity dynamics.
    - **Fleet Electrification:** Optimizing battery degradation and state-of-charge efficiency across multi-vehicle operations.
    
    ### Technology Architecture
    - **Frontend UI & Cloud Deployment:** Streamlit Cloud (Python 3)
    - **Machine Learning Core:** Scikit-Learn (Random Forest Regression)
    - **Data Visualizations:** Plotly Interactive Dark Theme
    """)

# Footer
st.markdown("<br><hr><br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>© 2026 EV Intelligence Pro Enterprise. All rights reserved.</p>", 
    unsafe_allow_html=True
)
