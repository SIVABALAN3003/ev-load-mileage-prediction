
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="EV Mileage Prediction",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 EV Load-Based Mileage Prediction")
st.caption("Electric Vehicle Range Intelligence")

st.info(
    "Educational prototype. Results are estimates, "
    "not actual vehicle measurements."
)

st.sidebar.header("Vehicle Settings")

battery = st.sidebar.number_input(
    "Battery Capacity (kWh)",
    min_value=0.1,
    value=40.0
)

soc = st.sidebar.slider(
    "Battery Charge (%)", 0, 100, 90
)

usable = st.sidebar.slider(
    "Usable Battery (%)", 10, 100, 90
)

base = st.sidebar.number_input(
    "Base Consumption (Wh/km)",
    min_value=1.0,
    value=150.0
)

load = st.sidebar.number_input(
    "Extra Load (kg)",
    min_value=0.0,
    value=150.0
)

penalty = st.sidebar.slider(
    "Assumed consumption increase per 100 kg (%)",
    0.0, 10.0, 2.0, 0.5
)

# Approximate model assumption
adjusted = base * (
    1 + (load / 100) * penalty / 100
)

energy = battery * soc / 100 * usable / 100

range_km = energy * 1000 / adjusted

base_range = energy * 1000 / base

col1, col2, col3 = st.columns(3)

col1.metric(
    "Available Energy",
    f"{energy:.2f} kWh"
)

col2.metric(
    "Estimated Consumption",
    f"{adjusted:.1f} Wh/km"
)

col3.metric(
    "Estimated Remaining Range",
    f"{range_km:.1f} km",
    delta=f"{range_km-base_range:.1f} km"
)

st.divider()

st.subheader("Mileage Comparison")

df = pd.DataFrame({
    "Scenario": ["Without Extra Load", "With Extra Load"],
    "Estimated Range (km)": [base_range, range_km]
})

fig = px.bar(
    df,
    x="Scenario",
    y="Estimated Range (km)",
    color="Scenario",
    title="Estimated EV Driving Range"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Prediction Results")
st.dataframe(df, use_container_width=True)

st.download_button(
    "Download Results CSV",
    df.to_csv(index=False).encode("utf-8"),
    "ev_prediction_results.csv",
    "text/csv"
)

st.caption(
    "Model assumption only. Calibrate with measured EV "
    "data before drawing conclusions. Not for vehicle safety."
)
