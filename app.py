
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

st.set_page_config(
    page_title="EV Intelligence Pro",
    page_icon="🔋",
    layout="wide"
)

FEATURES = [
    "battery_capacity_kwh",
    "charge_percent",
    "extra_load_kg",
    "speed_kmph",
    "ac_on"
]
TARGET = "range_km"

# ---------- DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #101d32);
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: #0b1423;
}
[data-testid="stMetric"] {
    background: #15243a;
    padding: 18px;
    border: 1px solid #29415e;
    border-radius: 14px;
}
h1, h2, h3 {
    color: #65e6c2;
}
.stButton button {
    border-radius: 9px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------- DATA ----------
@st.cache_data
def make_demo_data():
    np.random.seed(42)
    n = 500
    battery = np.random.uniform(30, 100, n)
    charge = np.random.uniform(20, 100, n)
    load = np.random.uniform(0, 500, n)
    speed = np.random.uniform(20, 100, n)
    ac = np.random.randint(0, 2, n)

    # Synthetic values for demonstration only
    distance = (
        battery * 2 * charge / 100
        - load * 0.025
        - speed * 0.15
        - ac * 5
        + np.random.normal(0, 3, n)
    )

    return pd.DataFrame({
        "battery_capacity_kwh": battery,
        "charge_percent": charge,
        "extra_load_kg": load,
        "speed_kmph": speed,
        "ac_on": ac,
        "range_km": np.maximum(distance, 0)
    })

if "history" not in st.session_state:
    st.session_state.history = []

# ---------- SIDEBAR ----------
st.sidebar.title("🔋 EV INTELLIGENCE")
st.sidebar.caption("EV Analytics Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Range Prediction",
        "Analytics",
        "Prediction History",
        "Dataset"
    ]
)

uploaded = st.sidebar.file_uploader(
    "Upload EV CSV",
    type=["csv"]
)

if uploaded:
    try:
        raw_df = pd.read_csv(uploaded)
        st.sidebar.success("CSV uploaded")
    except Exception:
        st.sidebar.error("Unable to read this CSV.")
        st.stop()
else:
    raw_df = make_demo_data()
    st.sidebar.info("Demo dataset active")

missing = [c for c in FEATURES + [TARGET] if c not in raw_df.columns]
if missing:
    st.error("Required columns missing: " + ", ".join(missing))
    st.code(", ".join(FEATURES + [TARGET]))
    st.stop()

df = raw_df.copy()
for col in FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES + [TARGET])

if len(df) < 10 or df[TARGET].nunique() < 2:
    st.error("Provide at least 10 valid rows with different range values.")
    st.stop()

# ---------- MACHINE LEARNING ----------
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, test_pred)
r2 = r2_score(y_test, test_pred)

# ---------- HEADER ----------
st.title("EV Intelligence Pro")
st.caption(
    "Electric Vehicle Range Prediction | Machine Learning | Energy Analytics"
)

if page == "Overview":
    st.subheader("Platform Overview")

    a, b, c, d = st.columns(4)
    a.metric("Valid Records", f"{len(df):,}")
    b.metric("Average Range", f"{df[TARGET].mean():.1f} km")
    c.metric("Model MAE", f"{mae:.1f} km")
    d.metric("Model R²", f"{r2:.3f}")

    st.markdown("### EV Range Distribution")
    fig = px.histogram(
        df, x=TARGET, nbins=25,
        title="Range Distribution",
        labels={TARGET: "Range (km)"},
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### System Status")
    st.write("**Prediction engine:** Ready")
    st.write("**Dataset:** Demo data" if uploaded is None
             else "**Dataset:** Uploaded CSV")
    st.write("**Live vehicle connection:** Not configured")

elif page == "Range Prediction":
    st.subheader("Machine Learning Range Prediction")

    left, right = st.columns(2)

    with left:
        battery = st.slider(
            "Battery capacity (kWh)", 10.0, 150.0, 60.0
        )
        charge = st.slider("Battery charge (%)", 0, 100, 80)
        load = st.slider("Additional load (kg)", 0, 1000, 100)

    with right:
        speed = st.slider("Average speed (km/h)", 10, 150, 60)
        ac = st.selectbox("Air conditioning", ["Off", "On"])
        st.info("Prediction uses the currently loaded dataset.")

    if st.button("⚡ Predict EV Range", type="primary"):
        input_df = pd.DataFrame([{
            "battery_capacity_kwh": battery,
            "charge_percent": charge,
            "extra_load_kg": load,
            "speed_kmph": speed,
            "ac_on": 1 if ac == "On" else 0
        }])

        prediction = max(0, float(model.predict(input_df)[0]))

        st.success(f"Estimated driving range: {prediction:.2f} km")

        st.session_state.history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Battery (kWh)": battery,
            "Charge (%)": charge,
            "Load (kg)": load,
            "Speed (km/h)": speed,
            "AC": ac,
            "Predicted Range (km)": round(prediction, 2)
        })

        st.caption(
            "Estimate only. Real range depends on the vehicle, road, "
            "weather, battery condition, and training data."
        )

elif page == "Analytics":
    st.subheader("EV Performance Analytics")

    fig1 = px.scatter(
        df, x="extra_load_kg", y=TARGET,
        color="speed_kmph",
        title="Additional Load vs Range",
        labels={
            "extra_load_kg": "Additional Load (kg)",
            TARGET: "Range (km)",
            "speed_kmph": "Speed (km/h)"
        },
        template="plotly_dark"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(
        x=FEATURES,
        y=model.feature_importances_,
        labels={"x": "Input Feature", "y": "Importance"},
        title="Model Feature Importance",
        template="plotly_dark"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.metric("Mean Absolute Error", f"{mae:.2f} km")
    st.metric("R² Score", f"{r2:.3f}")

elif page == "Prediction History":
    st.subheader("Prediction History")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        st.download_button(
            "Download Prediction History",
            history_df.to_csv(index=False).encode("utf-8"),
            file_name="ev_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions saved in this session yet.")

elif page == "Dataset":
    st.subheader("Dataset Explorer")
    st.write(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.download_button(
        "Download Cleaned Dataset",
        df.to_csv(index=False).encode("utf-8"),
        file_name="ev_dataset.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "EV Intelligence Pro | Educational prototype | "
    "Demo data is synthetic, not validated vehicle performance data."
)
