import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

st.set_page_config(
    page_title="EV Intelligence Pro — Enterprise Fleet Analytics", 
    page_icon="⚡", 
    layout="wide"
)

# --- Initialize Session State for Fleet and History ---
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

# --- Train a Demo ML Model ---
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

# --- Corporate Website Navigation Bar ---
st.markdown("""
    <div style='background: linear-gradient(90deg, #1e293b, #0f172a); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;'>
        <h1 style='margin: 0; font-size: 2.5rem;'>⚡ EV Intelligence Pro</h1>
        <p style='margin: 5px 0 0 0; color: #94a3b8; font-size: 1.1rem;'>Enterprise Electric Vehicle Fleet Monitoring & AI Range Analytics Platform</p>
    </div>
""", unsafe_allow_html=True)

nav_selection = st.radio(
    "Navigation", 
    ["🏠 Home & Overview", "🚗 Fleet Dashboard", "🔬 Research & Analytics", "🏢 About Company"], 
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# ==========================================
# 1. HOME & OVERVIEW PAGE
# ==========================================
if nav_selection == "🏠 Home & Overview":
    st.subheader("Welcome to EV Intelligence Pro Enterprise")
    st.markdown("""
    We deliver cutting-edge machine learning solutions for commercial electric vehicle fleets. 
    Our platform optimizes battery health, predicts precise driving range under varying loads and speeds, 
    and provides robust dataset analytics to maximize operational efficiency.
    """)
    
    # Executive KPI Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    total_vehicles = len(st.session_state.fleet)
    active_count = len(st.session_state.fleet[st.session_state.fleet["status"] == "Active"])
    avg_range = st.session_state.history["predicted_range_km"].astype(float).mean() if not st.session_state.history.empty else 385.4
    
    with col1:
        st.metric("Total Fleet Units", total_vehicles, delta="+1 this month")
    with col2:
        st.metric("Active Units", f"{active_count} / {total_vehicles}", delta="Operational")
    with col3:
        st.metric("Avg Fleet Range", f"{avg_range:.1f} km", delta="AI Optimized")
    with col4:
        st.metric("System Health", "99.98%", delta="Online")

    st.markdown("---")
    
    # Core Features Grid
    st.subheader("Enterprise Capabilities")
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        st.markdown("### 🚗 Fleet Management")
        st.markdown("Register, track, and monitor individual electric vehicles across your entire corporate network with live status indicators.")
        
    with f_col2:
        st.markdown("### ⚡ AI Range Prediction")
        st.markdown("Advanced Random Forest regression models account for battery capacity, payload weight, velocity, and climate control.")
        
    with f_col3:
        st.markdown("### 🔬 Research Analytics")
        st.markdown("Perform batch evaluations, validate test benchmark datasets, calculate $R^2$ accuracy scores, and export CSV reports.")

    st.markdown("---")
    st.info("💡 **Getting Started:** Use the navigation menu above to access the **Fleet Dashboard** for live range simulations or **Research & Analytics** to evaluate batch data sets.")

# ==========================================
# 2. FLEET DASHBOARD PAGE
# ==========================================
elif nav_selection == "🚗 Fleet Dashboard":
    st.subheader("🚗 Enterprise Fleet & Range Prediction Control Center")
    
    # Sidebar for Fleet Control
    st.sidebar.header("Fleet Controls")
    with st.sidebar.expander("Register New Vehicle"):
        new_id = st.text_input("Vehicle ID", value=f"EV-00{len(st.session_state.fleet)+1}")
        new_model = st.text_input("Vehicle Model", value="Commercial Van X")
        new_cap = st.number_input("Battery Capacity (kWh)", value=85.0, min_value=20.0, max_value=200.0)
        new_status = st.selectbox("Initial Status", ["Active", "Charging", "Maintenance"])
        
        if st.button("Add Vehicle to Fleet"):
            if new_id in st.session_state.fleet["vehicle_id"].values:
                st.sidebar.error("Vehicle ID already exists!")
            else:
                new_row = pd.DataFrame([{"vehicle_id": new_id, "model": new_model, "battery_kwh": new_cap, "status": new_status}])
                st.session_state.fleet = pd.concat([st.session_state.fleet, new_row], ignore_index=True)
                st.sidebar.success(f"Added {new_id} successfully!")

    selected_vehicle = st.sidebar.selectbox("Select Active Vehicle", st.session_state.fleet["vehicle_id"].values)
    selected_row = st.session_state.fleet[st.session_state.fleet["vehicle_id"] == selected_vehicle].iloc[0]

    st.sidebar.markdown("---")
    st.sidebar.info(f"**Vehicle Specs:**\n- Model: {selected_row['model']}\n- Capacity: {selected_row['battery_kwh']} kWh\n- Status: {selected_row['status']}")

    # Main Grid Layout
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown(f"#### Active Simulation Panel ({selected_vehicle})")
        
        with st.form("prediction_form"):
            battery_cap = st.number_input("Battery Capacity (kWh)", value=float(selected_row['battery_kwh']), min_value=10.0, max_value=200.0)
            charge_pct = st.slider("Battery Charge (%)", min_value=5.0, max_value=100.0, value=80.0)
            extra_load = st.slider("Additional Payload Load (kg)", min_value=0.0, max_value=500.0, value=75.0)
            speed = st.slider("Average Speed (km/h)", min_value=10.0, max_value=140.0, value=80.0)
            ac_status = st.selectbox("AC / Climate Control", ["Off", "On"])
            
            submit_btn = st.form_submit_button("⚡ Run AI Range Prediction", type="primary")
            
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
            
            st.success(f"**Estimated Driving Range for {selected_vehicle}: {prediction:.1f} km**")
            
            if charge_pct < 20:
                st.warning("⚠️ Low battery warning! Schedule unit for recharge.")
                
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
        st.markdown("#### Fleet Speed vs Range Analytics")
        if not st.session_state.history.empty:
            plot_df = st.session_state.history.copy()
            plot_df["speed_kmph"] = pd.to_numeric(plot_df["speed_kmph"])
            plot_df["predicted_range_km"] = pd.to_numeric(plot_df["predicted_range_km"])
            plot_df["extra_load_kg"] = pd.to_numeric(plot_df["extra_load_kg"])
            
            fig = px.scatter(
                plot_df, 
                x="speed_kmph", 
                y="predicted_range_km", 
                color="vehicle_id",
                hover_data=["extra_load_kg", "charge_percent", "ac_on"],
                title="Speed vs Range Impact across Fleet"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run predictions on the left panel to generate live fleet analytics charts.")
            dummy_df = pd.DataFrame({"Speed": [40, 60, 80, 100, 120], "Estimated Range": [450, 420, 380, 320, 260]})
            fig = px.line(dummy_df, x="Speed", y="Estimated Range", markers=True, title="Sample Baseline Curve")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # Inventory & History Tables
    tab_f1, tab_f2 = st.tabs(["🚗 Fleet Inventory", "📈 Prediction Logs"])
    with tab_f1:
        st.dataframe(st.session_state.fleet, use_container_width=True)
        fleet_csv = st.session_state.fleet.to_csv(index=False).encode("utf-8")
        st.download_button("Download Fleet CSV", fleet_csv, file_name="fleet_inventory.csv", mime="text/csv")
    with tab_f2:
        if not st.session_state.history.empty:
            st.dataframe(st.session_state.history, use_container_width=True)
            hist_csv = st.session_state.history.to_csv(index=False).encode("utf-8")
            st.download_button("Download History CSV", hist_csv, file_name="prediction_history.csv", mime="text/csv")
        else:
            st.info("No prediction history recorded yet.")

# ==========================================
# 3. RESEARCH & ANALYTICS PAGE
# ==========================================
elif nav_selection == "🔬 Research & Analytics":
    st.subheader("🔬 Research Dataset & Model Evaluation Suite")
    st.markdown("Evaluate machine learning model performance using standardized benchmark test sets or custom telemetry batch files.")
    
    col_res1, col_res2 = st.columns([1, 1])
    with col_res1:
        use_sample_data = st.checkbox("Load Built-in EV Research Test Benchmark", value=True)
    with col_res2:
        uploaded_research_file = st.file_uploader("Upload Custom Research CSV", type=["csv"])

    research_df = None
    if uploaded_research_file is not None:
        try:
            research_df = pd.read_csv(uploaded_research_file)
            st.success("Custom research dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
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
            st.error(f"Research dataset is missing required feature columns: {missing_cols}")
        else:
            if st.button("🚀 Run Batch Research Inference", type="primary"):
                features_for_pred = research_df[required_cols].copy()
                for col in required_cols:
                    features_for_pred[col] = pd.to_numeric(features_for_pred[col])
                
                predictions = model.predict(features_for_pred)
                research_df["predicted_range_km"] = np.round(predictions, 2)
                
                st.success("Batch prediction successfully completed!")
                st.dataframe(research_df, use_container_width=True)
                
                if "actual_range_km" in research_df.columns:
                    st.markdown("### 📊 Model Evaluation Metrics")
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
                        title="Model Validation: Actual vs. Predicted Range"
                    )
                    st.plotly_chart(fig_eval, use_container_width=True)
                
                research_csv = research_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Batch Results CSV",
                    research_csv,
                    file_name="ev_research_predictions.csv",
                    mime="text/csv"
                )

# ==========================================
# 4. ABOUT COMPANY PAGE
# ==========================================
elif nav_selection == "🏢 About Company":
    st.subheader("🏢 About EV Intelligence Pro")
    st.markdown("""
    **EV Intelligence Pro** is a pioneer in enterprise electric vehicle analytics, empowering logistics companies, 
    commercial fleets, and automotive researchers with high-precision machine learning tools.
    
    ### Our Mission
    To accelerate global fleet electrification by providing transparent, data-driven insights into battery life, 
    energy consumption, and operational range limits.
    
    ### Technology Stack
    - **Frontend & Deployment:** Streamlit Cloud & Python
    - **Machine Learning Engine:** Scikit-Learn (Random Forest Regressor)
    - **Data Visualization:** Plotly Interactive Dashboards
    - **Data Processing:** NumPy & Pandas
    
    ### Corporate Contact & Support
    - **Email:** support@evintelligence-pro.com
    - **Global HQ:** CleanTech Innovation Park, Suite 400
    - **Documentation:** Available via internal corporate portal or GitHub repository.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>© 2026 EV Intelligence Pro Enterprise. All rights reserved.</p>", 
    unsafe_allow_html=True
)
