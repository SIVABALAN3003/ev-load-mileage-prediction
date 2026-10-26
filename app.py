
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="EV Intelligence AI", page_icon="⚡", layout="wide")

st.title("⚡ EV Intelligence — AI Range Predictor")
st.write("Machine-learning prediction using a generated demonstration dataset.")

# Generate demonstration training data
@st.cache_data
def create_dataset():
    rng = np.random.default_rng(42)
    n = 500

    battery = rng.uniform(20, 100, n)
    charge = rng.uniform(10, 100, n)
    load = rng.uniform(0, 500, n)
    speed = rng.uniform(20, 120, n)
    ac = rng.integers(0, 2, n)

    consumption = 160 + speed * 0.35 + load * 0.04 + ac * 15
    energy = battery * charge / 100 * 0.90
    target_range = energy * 1000 / consumption
    target_range += rng.normal(0, 5, n)

    return pd.DataFrame({
        "Battery": battery,
        "Charge": charge,
        "Load": load,
        "Speed": speed,
        "AC": ac,
        "Range": target_range
    })

data = create_dataset()

features = ["Battery", "Charge", "Load", "Speed", "AC"]
X = data[features]
y = data["Range"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

st.subheader("Model Performance")
a, b = st.columns(2)
a.metric("Mean Absolute Error", f"{mae:.2f} km")
b.metric("R² Score", f"{r2:.3f}")

st.caption(
    "These metrics measure performance on synthetic demonstration data, "
    "not on real-world EV measurements."
)

st.divider()
st.sidebar.header("Vehicle Inputs")

battery = st.sidebar.slider("Battery Capacity (kWh)", 20, 100, 50)
charge = st.sidebar.slider("Battery Charge (%)", 10, 100, 80)
load = st.sidebar.slider("Extra Load (kg)", 0, 500, 100)
speed = st.sidebar.slider("Speed (km/h)", 20, 120, 60)
ac_on = st.sidebar.checkbox("Air Conditioning ON")

ac = 1 if ac_on else 0

vehicle = pd.DataFrame([{
    "Battery": battery,
    "Charge": charge,
    "Load": load,
    "Speed": speed,
    "AC": ac
}])

predicted_range = max(0, float(model.predict(vehicle)[0]))

st.subheader("Your EV Prediction")
st.metric("AI Estimated Range", f"{predicted_range:.1f} km")

st.subheader("Feature Importance")
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

fig = px.bar(
    importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Which Inputs Influence the Model?"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Training Dataset")
st.dataframe(data.head(20), use_container_width=True)

st.download_button(
    "Download Training Dataset",
    data.to_csv(index=False),
    file_name="ev_training_dataset.csv",
    mime="text/csv"
)

result = vehicle.copy()
result["Predicted_Range_km"] = predicted_range

st.download_button(
    "Download Your Prediction",
    result.to_csv(index=False),
    file_name="ev_prediction.csv",
    mime="text/csv"
)
