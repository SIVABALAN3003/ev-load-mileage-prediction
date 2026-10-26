
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="EV Intelligence",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 EV Intelligence")
st.subheader("EV Load-Based Mileage Prediction")
st.write("Upload your EV dataset or use the demo dataset.")

# Create demo dataset
@st.cache_data
def create_demo_data():
    np.random.seed(42)
    n = 500

    battery = np.random.uniform(20, 100, n)
    charge = np.random.uniform(20, 100, n)
    load = np.random.uniform(0, 500, n)
    speed = np.random.uniform(20, 100, n)
    ac = np.random.randint(0, 2, n)

    range_km = (
        battery * 2.0
        * (charge / 100)
        - load * 0.025
        - speed * 0.15
        - ac * 5
        + np.random.normal(0, 3, n)
    )

    data = pd.DataFrame({
        "battery_capacity_kwh": battery,
        "charge_percent": charge,
        "extra_load_kg": load,
        "speed_kmph": speed,
        "ac_on": ac,
        "range_km": np.maximum(range_km, 0)
    })

    return data

uploaded_file = st.file_uploader(
    "Upload CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    df = create_demo_data()
    st.info("Using demo data. Upload your own CSV to replace it.")

# Required columns
features = [
    "battery_capacity_kwh",
    "charge_percent",
    "extra_load_kg",
    "speed_kmph",
    "ac_on"
]
target = "range_km"

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

missing = [c for c in features + [target] if c not in df.columns]

if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.write("Use these exact column names:")
    st.code(", ".join(features + [target]))
    st.stop()

# Clean data
for column in features + [target]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features + [target])

if len(df) < 10:
    st.error("Please provide at least 10 valid dataset rows.")
    st.stop()

if df[target].nunique() < 2:
    st.error("The range_km column must contain different values.")
    st.stop()

X = df[features]
y = df[target]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions) if len(y_test) > 1 else 0

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("Valid Rows", len(df))
col2.metric("Mean Absolute Error", f"{mae:.2f} km")
col3.metric("R² Score", f"{r2:.3f}")

st.caption(
    "These metrics use a held-out portion of the uploaded dataset. "
    "Results depend on data quality and representativeness."
)

# Prediction controls
st.subheader("Predict EV Range")

c1, c2, c3 = st.columns(3)

with c1:
    battery = st.number_input(
        "Battery capacity (kWh)", 1.0, 300.0, 60.0
    )
    charge = st.slider("Battery charge (%)", 0, 100, 80)

with c2:
    load = st.number_input(
        "Extra load (kg)", 0.0, 5000.0, 100.0
    )
    speed = st.number_input(
        "Average speed (km/h)", 1.0, 200.0, 60.0
    )

with c3:
    ac = st.selectbox("Air conditioner", ["Off", "On"])
    ac_value = 1 if ac == "On" else 0

if st.button("Predict Range", type="primary"):
    input_data = pd.DataFrame([{
        "battery_capacity_kwh": battery,
        "charge_percent": charge,
        "extra_load_kg": load,
        "speed_kmph": speed,
        "ac_on": ac_value
    }])

    result = max(0, float(model.predict(input_data)[0]))

    st.success(f"Predicted EV range: {result:.2f} km")
    st.caption(
        "This is an ML estimate, not a guaranteed driving range. "
        "Accuracy depends on the training dataset."
    )

# Download cleaned data
st.subheader("Download Dataset")
st.download_button(
    "Download Cleaned CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="ev_dataset_cleaned.csv",
    mime="text/csv"
)
