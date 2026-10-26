import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

st.set_page_config(page_title="EV Intelligence Pro — Fleet & Research Dashboard", page_icon="⚡", layout="wide")

# --- Initialize Session State for Fleet and History ---
if "fleet" not in st.session_state:
    st.session_state.fleet = pd.DataFrame([
        {"vehicle_id": "EV-001", "model": "Model S Long Range", "battery_kwh": 100.0, "status": "Active"},
        {"vehicle_id": "EV-002", "model": "Model 3 Standard", "battery_kwh": 60.0, "status": "Active"},
        {"vehicle_id": "EV-003", "model": "SUV Performance", "battery_kwh": 90.0, "status": "Charging"}
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

# --- Top Bar Header ---
st.title("⚡ EV Intelligence Pro — Fleet & Research Dashboard")
st.markdown("Company-level multi-vehicle EV monitoring, AI range prediction, session history, and research dataset evaluation.")
st.markdown("---")

# --- Sidebar: Fleet Management & Controls ---
st.sidebar.header("🚗 Fleet Management")

with st.sidebar.expander("Register New Vehicle"):
    new_id = st.text_input("Vehicle ID", value=f"EV-00{len(st.session_state.fleet)+1}")
    new_model = st.text_input("Vehicle Model", value="Model Y Performance")
    new_cap = st.number_input("Battery Capacity (kWh)", value=75.0, min_value=20.0, max_value=200.0)
    new_status = st.selectbox("Initial Status", ["Active", "Charging", "Maintenance"])
    
    if st.button("Add Vehicle"):
        if new_id in st.session_state.fleet["vehicle_id"].values:
            st.sidebar.error("Vehicle ID already exists!")
        else:
            new_row = pd.DataFrame([{"vehicle_id": new_id, "model": new_model, "battery_kwh": new_cap, "status": new_status}])
            st.session_state.fleet = pd.concat([st.session_state.fleet, new_row], ignore_index=True)
            st.sidebar.success(f"Added {new_id} successfully!")

selected_vehicle = st.sidebar.selectbox("Select Active Vehicle for Prediction", st.session_state.fleet["vehicle_id"].values)
selected_row = st.session_state.fleet[st.session_state.fleet["vehicle_id"] == selected_vehicle].iloc[0]

st.sidebar.markdown("---")
st.sidebar.info(f"**Selected Info:**\n- Model: {selected_row['model']}\n- Capacity: {selected_row['battery_kwh']} kWh\n- Status: {selected_row['status']}")

# --- KPI Cards ---
col1, col2, col3, col4 = st.columns(4)
total_vehicles = len(st.session_state.fleet)

avg_range = st.session_state.history["predicted_range_km"].astype(float).mean() if not st.session_state.history.empty else 350.0
active_count = len(st.session_state.fleet[st.session_state.fleet["status"] == "Active"])

with col1:
    st.metric("Total Fleet Vehicles", total_vehicles)
with col2:
    st.metric("Active Status", f"{active_count} / {total_vehicles}")
with col3:
    st.metric("Avg Predicted Range", f"{avg_range:.1f} km" if not np.isnan(avg_range) else "N/A")
with col4:
    st.metric("System Status", "Operational", "Online")

st.markdown("---")

# --- Main Dashboard Grid ---
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader(f"⚡ Range Prediction Panel ({selected_vehicle})")
    
    with st.form("prediction_form"):
        battery_cap = st.number_input("Battery Capacity (kWh)", value=float(selected_row['battery_kwh']), min_value=10.0, max_value=200.0)
        charge_pct = st.slider("Battery Charge (%)", min_value=5.0, max_value=100.0, value=80.0)
        extra_load = st.slider("Additional Load (kg)", min_value=0.0, max_value=500.0, value=50.0)
        speed = st.slider("Average Speed (km/h)", min_value=10.0, max_value=140.0, value=75.0)
        ac_status = st.selectbox("AC / Climate Control", ["Off", "On"])
        
        submit_btn = st.form_submit_button("⚡ Predict EV Range & Save", type="primary")
        
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
        
        st.success(f"**Estimated Range for {selected_vehicle}: {prediction:.1f} km**")
        
        if charge_pct < 20:
            st.warning("⚠️ Low battery warning! Consider scheduling a recharge soon.")
            
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
    st.subheader("📊 Fleet Overview & Analytics")
    
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
        st.info("Make a few predictions using the panel on the left to see live analytics charts here!")
        
        dummy_df = pd.DataFrame({
            "Speed": [40, 60, 80, 100, 120],
            "Estimated Range": [450, 420, 380, 320, 260]
        })
        fig = px.line(dummy_df, x="Speed", y="Estimated Range", markers=True, title="Sample Baseline Range Curve")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- Tabs for Inventory, History, and Research Evaluation ---
tab1, tab2, tab3 = st.tabs(["🚗 Registered Fleet", "📈 Prediction History", "🔬 Research Dataset & Evaluation"])

with tab1:
    st.subheader("Current Fleet Inventory")
    st.dataframe(st.session_state.fleet, use_container_width=True)
    
    fleet_csv = st.session_state.fleet.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Fleet Inventory CSV",
        fleet_csv,
        file_name="ev_fleet_inventory.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("Session Prediction History Log")
    if not st.session_state.history.empty:
        st.dataframe(st.session_state.history, use_container_width=True)
        
        history_csv = st.session_state.history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download History CSV",
            history_csv,
            file_name="ev_prediction_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No predictions recorded in this session yet.")

with tab3:
    st.subheader("🔬 Research Benchmark Dataset & Batch Prediction")
    st.markdown("Test the machine learning model against a structured research dataset or upload custom telemetry batch records for bulk range inference.")
    
    col_res1, col_res2 = st.columns([1, 1])
    
    with col_res1:
        use_sample_data = st.checkbox("Load Built-in EV Research Test Dataset", value=True)
        
    with col_res2:
        uploaded_research_file = st.file_uploader("Or Upload Custom Research CSV", type=["csv"])

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
        st.markdown("#### Preview of Research Dataset")
        st.dataframe(research_df.head(5), use_container_width=True)
        
        required_cols = ["battery_capacity_kwh", "charge_percent", "extra_load_kg", "speed_kmph", "ac_on"]
        missing_cols = [col for col in required_cols if col not in research_df.columns]
        
        if missing_cols:
            st.error(f"Research dataset is missing required feature columns: {missing_cols}")
        else:
            if st.button("🚀 Run Batch Research Prediction", type="primary"):
                features_for_pred = research_df[required_cols].copy()
                for col in required_cols:
                    features_for_pred[col] = pd.to_numeric(features_for_pred[col])
                
                predictions = model.predict(features_for_pred)
                research_df["predicted_range_km"] = np.round(predictions, 2)
                
                st.success("Batch prediction completed successfully across all research records!")
                st.dataframe(research_df, use_container_width=True)
                
                if "actual_range_km" in research_df.columns:
                    st.markdown("### 📊 Research Model Evaluation Metrics")
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
                        title="Model Validation: Actual vs. Predicted Range (Research Benchmark)"
                    )
                    st.plotly_chart(fig_eval, use_container_width=True)
                
                research_csv = research_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Batch Prediction Results CSV",
                    research_csv,
                    file_name="ev_research_predictions_output.csv",
                    mime="text/csv"
                )
