import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

st.set_page_config(page_title="EV Intelligence Pro — Fleet Dashboard", page_icon="⚡", layout="wide")

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
    
    # Range formula baseline + noise
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
st.title("⚡ EV Intelligence Pro — Fleet Dashboard")
st.markdown("Company-level multi-vehicle EV monitoring, AI range prediction, and session analytics.")
st.markdown("---")

# --- Sidebar: Fleet Management & Controls ---
st.sidebar.header("🚗 Fleet Management")

# Add new vehicle form
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

# Get details of selected vehicle
selected_row = st.session_state.fleet[st.session_state.fleet["vehicle_id"] == selected_vehicle].iloc[0]

st.sidebar.markdown("---")
st.sidebar.info(f"**Selected Info:**\n- Model: {selected_row['model']}\n- Capacity: {selected_row['battery_kwh']} kWh\n- Status: {selected_row['status']}")

# --- KPI Cards ---
col1, col2, col3, col4 = st.columns(4)
total_vehicles = len(st.session_state.fleet)

avg_range = st.session_state.history["predicted_range_km"].mean() if not st.session_state.history.empty else 350.0
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
            "battery_capacity_kwh": [battery_cap],
            "charge_percent": [charge_pct],
            "extra_load_kg": [extra_load],
            "speed_kmph": [speed],
            "ac_on": [ac_val]
        })
        
        prediction = model.predict(input_data)[0]
        
        st.success(f"**Estimated Range for {selected_vehicle}: {prediction:.1f} km**")
        
        if charge_pct < 20:
            st.warning("⚠️ Low battery warning! Consider scheduling a recharge soon.")
            
        # Save to session history
        new_history_row = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "vehicle_id": selected_vehicle,
            "battery_kwh": battery_cap,
            "charge_percent": charge_pct,
            "extra_load_kg": extra_load,
            "speed_kmph": speed,
            "ac_on": ac_status,
            "predicted_range_km": round(prediction, 2)
        }])
        st.session_state.history = pd.concat([new_history_row, st.session_state.history], ignore_index=True)

with right_col:
    st.subheader("📊 Fleet Overview & Analytics")
    
    if not st.session_state.history.empty:
        fig = px.scatter(
            st.session_state.history, 
            x="speed_kmph", 
            y="predicted_range_km", 
            color="vehicle_id",
            size="extra_load_kg",
            hover_data=["charge_percent", "ac_on"],
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

# --- Fleet Register & History Tables ---
tab1, tab2 = st.tabs(["🚗 Registered Fleet", "📈 Prediction History"])

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
