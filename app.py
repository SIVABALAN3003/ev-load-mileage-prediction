import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

st.set_page_config(page_title="EV Intelligence Pro", page_icon="🔋", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(145deg,#07111f,#101d32); color:#edf4fb;}
.block-container {padding-top:1.2rem; max-width:1500px;}
h1,h2,h3 {color:#69e6c2;}
[data-testid="stMetric"] {background:#12243a;border:1px solid #263d57;padding:16px;border-radius:14px;}
[data-testid="stMetricLabel"] {color:#a9bfd4;}
[data-testid="stMetricValue"] {color:#f5fbff;}
.stButton button,.stDownloadButton button {border-radius:9px;font-weight:650;}
</style>
""", unsafe_allow_html=True)

FEATURES = ["battery_capacity_kwh", "charge_percent", "extra_load_kg", "speed_kmph", "ac_on"]
TARGET = "range_km"

@st.cache_data
def make_demo_data():
    rng = np.random.default_rng(42)
    n = 500
    battery = rng.uniform(30, 100, n)
    charge = rng.uniform(20, 100, n)
    load = rng.uniform(0, 500, n)
    speed = rng.uniform(20, 100, n)
    ac = rng.integers(0, 2, n)
    range_km = battery * 2 * charge / 100 - load * 0.025 - speed * 0.15 - ac * 5 + rng.normal(0, 3, n)
    return pd.DataFrame({
        "battery_capacity_kwh": battery,
        "charge_percent": charge,
        "extra_load_kg": load,
        "speed_kmph": speed,
        "ac_on": ac,
        "range_km": np.maximum(range_km, 0)
    })

if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

# Header
h1, h2 = st.columns([3, 1])
with h1:
    st.title("🔋 EV INTELLIGENCE PRO")
    st.caption("ELECTRIC VEHICLE RANGE • ENERGY ANALYTICS • MACHINE LEARNING")
with h2:
    st.markdown("**SYSTEM TIME**")
    st.write(datetime.now().strftime("%d %b %Y · %I:%M %p"))
    st.success("Prediction engine ready")

st.divider()

# Dataset upload
u1, u2 = st.columns([2, 1])
with u1:
    st.subheader("Fleet intelligence dashboard")
    st.write("One-screen EV range prediction and dataset analytics.")
with u2:
    uploaded_file = st.file_uploader("Upload EV CSV dataset", type=["csv"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        source_label = "Uploaded dataset"
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    raw_df = make_demo_data()
    source_label = "Synthetic demo dataset"

missing = [col for col in FEATURES + [TARGET] if col not in raw_df.columns]
if missing:
    st.error("CSV is missing required columns: " + ", ".join(missing))
    st.info("Required CSV columns:")
    st.code(", ".join(FEATURES + [TARGET]))
    st.stop()

df = raw_df.copy()
for col in FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET])

if len(df) < 10:
    st.error("Please provide at least 10 valid rows in the CSV.")
    st.stop()
if df[TARGET].nunique() < 2:
    st.error("The range_km column must contain at least two different values.")
    st.stop()

# Train model
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
test_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, test_pred)
r2 = r2_score(y_test, test_pred) if len(y_test) > 1 else 0.0

# KPI cards
k1, k2, k3, k4 = st.columns(4)
k1.metric("Dataset records", f"{len(df):,}")
k2.metric("Average recorded range", f"{df[TARGET].mean():.1f} km")
k3.metric("Model MAE", f"{mae:.2f} km")
k4.metric("Model R²", f"{r2:.3f}")
st.caption(f"DATA SOURCE: {source_label} • Synthetic data is for demonstration only.")
st.divider()

# Main content
left, right = st.columns([1, 1.35], gap="large")

with left:
    st.subheader("⚡ Range prediction")
    with st.container(border=True):
        vehicle_id = st.text_input("Vehicle ID", value="EV-001")
        c1, c2 = st.columns(2)
        with c1:
            battery = st.number_input("Battery capacity (kWh)", min_value=1.0, max_value=300.0, value=60.0, step=1.0)
            charge = st.slider("Battery charge (%)", 0, 100, 80)
            load = st.number_input("Additional load (kg)", min_value=0.0, max_value=5000.0, value=100.0, step=10.0)
        with c2:
            speed = st.number_input("Average speed (km/h)", min_value=1.0, max_value=200.0, value=60.0, step=1.0)
            ac = st.selectbox("Air conditioning", ["Off", "On"])
        predict_clicked = st.button("⚡ Predict range", type="primary", use_container_width=True)

    if predict_clicked:
        input_data = pd.DataFrame([{
            "battery_capacity_kwh": battery,
            "charge_percent": charge,
            "extra_load_kg": load,
            "speed_kmph": speed,
            "ac_on": 1 if ac == "On" else 0
        }])
        prediction = max(0.0, float(model.predict(input_data)[0]))
        record = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Vehicle ID": vehicle_id.strip() or "EV-001",
            "Battery (kWh)": battery,
            "Charge (%)": charge,
            "Load (kg)": load,
            "Speed (km/h)": speed,
            "AC": ac,
            "Predicted Range (km)": round(prediction, 2)
        }
        st.session_state["last_prediction"] = record
        st.session_state["prediction_history"].insert(0, record)

    last = st.session_state["last_prediction"]
    if last:
        st.markdown("#### Latest prediction")
        st.metric(f"Estimated range · {last['Vehicle ID']}", f"{last['Predicted Range (km)']:.2f} km")
        st.progress(int(last["Charge (%)"]))
        st.caption(f"Battery charge input: {last['Charge (%)']}%")

with right:
    st.subheader("📊 Energy analytics")
    tab1, tab2 = st.tabs(["Load vs range", "Speed vs range"])
    with tab1:
        fig = px.scatter(df, x="extra_load_kg", y="range_km", color="charge_percent",
                         labels={"extra_load_kg": "Additional load (kg)", "range_km": "Recorded range (km)", "charge_percent": "Charge (%)"},
                         template="plotly_dark", opacity=0.8)
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig = px.scatter(df, x="speed_kmph", y="range_km", color="extra_load_kg",
                         labels={"speed_kmph": "Speed (km/h)", "range_km": "Recorded range (km)", "extra_load_kg": "Load (kg)"},
                         template="plotly_dark", opacity=0.8)
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    imp = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance")
    fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h",
                     title="Model feature importance", template="plotly_dark")
    fig_imp.update_layout(height=280, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig_imp, use_container_width=True)

# History and dataset are also on the same page
st.divider()
b1, b2 = st.columns([1.2, 1], gap="large")
with b1:
    st.subheader("🕘 Recent prediction history")
    history = st.session_state["prediction_history"]
    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(history_df.head(10), use_container_width=True, hide_index=True)
        st.download_button("Download history CSV", history_df.to_csv(index=False).encode("utf-8"),
                           file_name="ev_prediction_history.csv", mime="text/csv")
    else:
        st.info("Predictions will appear here after you click Predict range.")
with b2:
    st.subheader("🗂️ Dataset preview")
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)
    st.download_button("Download active dataset CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="ev_active_dataset.csv", mime="text/csv")

st.divider()
st.caption("EV Intelligence Pro • Single-page dashboard • Educational prototype. Validate predictions with real EV measurements before real-world use.")

