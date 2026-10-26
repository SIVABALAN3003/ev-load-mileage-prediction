
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="EV Intelligence",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ EV Intelligence")
st.subheader("EV Load-Based Mileage Prediction")
st.write("Predict electric vehicle range based on battery, load and speed.")

st.sidebar.header("Vehicle Details")

battery = st.sidebar.slider("Battery Capacity (kWh)", 10, 100, 50)
charge = st.sidebar.slider("Battery Charge (%)", 0, 100, 80)
load = st.sidebar.slider("Extra Load (kg)", 0, 500, 100)
speed = st.sidebar.slider("Speed (km/h)", 10, 120, 60)
ac = st.sidebar.checkbox("Air Conditioning ON")

road = st.sidebar.selectbox(
    "Road Type",
    ["City", "Mixed", "Highway"]
)

consumption = 160 + speed * 0.35 + load * 0.04

if ac:
    consumption += 15

if road == "Highway":
    consumption *= 1.10
elif road == "City":
    consumption *= 1.05

energy = battery * charge / 100 * 0.90
estimated_range = energy * 1000 / consumption if consumption > 0 else 0

st.markdown("### 📊 Prediction Results")

col1, col2, col3 = st.columns(3)

col1.metric("Estimated Range", f"{estimated_range:.1f} km")
col2.metric("Energy Available", f"{energy:.1f} kWh")
col3.metric("Energy Consumption", f"{consumption:.1f} Wh/km")

st.success("Prediction completed!")

st.markdown("### 📈 Range Analysis")

labels = ["Estimated Range", "Remaining Range"]
values = [estimated_range, max(0, estimated_range * 0.2)]

chart_data = pd.DataFrame({
    "Category": labels,
    "Range (km)": values
})

fig = px.bar(
    chart_data,
    x="Category",
    y="Range (km)",
    title="Estimated EV Range",
    color="Category"
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    "This is an educational estimate, not a validated vehicle "
    "model. Actual range depends on battery condition, weather, "
    "driving conditions and vehicle efficiency."
)
