
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EV Intelligence", page_icon="⚡", layout="wide")

st.title("⚡ EV Intelligence Dashboard")
st.caption("EV Load-Based Mileage Prediction")

if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.header("Vehicle Details")

battery = st.sidebar.slider("Battery Capacity (kWh)", 10, 100, 50)
charge = st.sidebar.slider("Battery Charge (%)", 0, 100, 80)
load = st.sidebar.slider("Extra Load (kg)", 0, 500, 100)
speed = st.sidebar.slider("Speed (km/h)", 10, 120, 60)
ac = st.sidebar.checkbox("Air Conditioning ON")
road = st.sidebar.selectbox("Road Type", ["City", "Mixed", "Highway"])

consumption = 160 + speed * 0.35 + load * 0.04
if ac:
    consumption += 15
if road == "Highway":
    consumption *= 1.10
elif road == "City":
    consumption *= 1.05

energy = battery * charge / 100 * 0.90
estimated_range = energy * 1000 / consumption if consumption > 0 else 0

if st.button("Save Prediction", type="primary"):
    st.session_state.history.append({
        "Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Battery (kWh)": battery,
        "Charge (%)": charge,
        "Load (kg)": load,
        "Speed (km/h)": speed,
        "Road": road,
        "Consumption (Wh/km)": round(consumption, 1),
        "Estimated Range (km)": round(estimated_range, 1)
    })
    st.success("Prediction saved!")

a, b, c = st.columns(3)
a.metric("Estimated Range", f"{estimated_range:.1f} km")
b.metric("Available Energy", f"{energy:.1f} kWh")
c.metric("Consumption", f"{consumption:.1f} Wh/km")

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Dataset", "📈 Analytics", "🕘 History"])

with tab1:
    data = pd.DataFrame([
        {"Battery (kWh)": 40, "Load (kg)": 50, "Speed (km/h)": 40},
        {"Battery (kWh)": 50, "Load (kg)": 100, "Speed (km/h)": 60},
        {"Battery (kWh)": 60, "Load (kg)": 200, "Speed (km/h)": 70},
        {"Battery (kWh)": 75, "Load (kg)": 300, "Speed (km/h)": 80},
        {"Battery (kWh)": 90, "Load (kg)": 400, "Speed (km/h)": 90}
    ])
    st.dataframe(data, use_container_width=True)
    st.download_button(
        "Download Sample Dataset",
        data.to_csv(index=False),
        file_name="ev_dataset.csv",
        mime="text/csv"
    )

with tab2:
    chart = pd.DataFrame({
        "Metric": ["Estimated Range", "Energy Available"],
        "Value": [estimated_range, energy]
    })
    fig = px.bar(chart, x="Metric", y="Value", color="Metric",
                 title="EV Prediction Analytics")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        st.download_button(
            "Download Prediction History",
            history_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No saved predictions yet. Click Save Prediction first.")

st.caption("Educational estimates only — not validated vehicle performance data.")
