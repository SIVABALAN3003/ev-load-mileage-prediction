
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

try:
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

st.set_page_config(
    page_title="EV Intelligence | Mileage AI",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

.stApp {
    background: radial-gradient(circle at 12% 0%, #183354 0%, #0a1220 45%, #080d17 100%);
    color: #eaf1ff;
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0c1525;
    border-right: 1px solid #263651;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #192941, #0e1828);
    border: 1px solid #263651;
    border-radius: 16px;
    padding: 18px;
}
[data-testid="stMetricLabel"] {
    color: #9aa9c2;
}
[data-testid="stMetricValue"] {
    color: #f2f7ff;
}
.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    border: 1px solid #3a536f;
    font-weight: 700;
    min-height: 2.7rem;
}
.stButton > button[kind="primary"] {
    background: #55e6a5;
    color: #07150f;
    border: 0;
}
.hero {
    padding: 26px 28px;
    border: 1px solid #2b405d;
    border-radius: 22px;
    background: linear-gradient(120deg, #1a314d, #0e1a2c);
    margin-bottom: 18px;
}
.eyebrow {
    color: #55e6a5;
    text-transform: uppercase;
    letter-spacing: .16em;
    font-size: .75rem;
    font-weight: 700;
}
.hero h1 {
    font-size: 2.45rem;
    margin: .25rem 0 .45rem 0;
}
.hero p, .small-note {
    color: #9aa9c2;
}
hr {
    border-color: #263651;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DEMO DATA ----------------
@st.cache_data
def demo_dataset():
    return pd.DataFrame([
        [40,90,0,40,0,"City",145,248.28],
        [40,90,100,50,0,"Mixed",150,216.00],
        [40,80,200,60,1,"Highway",180,160.00],
        [50,90,150,50,1,"Mixed",165,245.45],
        [35,80,250,40,1,"City",175,144.00],
        [45,90,50,60,0,"Highway",160,227.81],
        [30,75,300,70,1,"City",190,106.58],
        [60,95,100,55,0,"Mixed",155,330.97],
        [50,85,400,80,1,"Highway",205,186.59],
        [40,70,200,45,0,"City",160,157.50],
        [35,90,0,35,0,"Mixed",140,202.50],
        [60,80,350,65,1,"Highway",195,221.54],
        [45,75,100,50,0,"City",150,202.50],
        [50,95,250,70,1,"Mixed",180,237.50],
        [40,85,300,55,1,"Highway",185,165.41],
        [30,90,50,40,0,"City",145,167.59],
        [60,90,200,60,0,"Mixed",160,303.75],
        [45,80,350,75,1,"Highway",200,162.00],
        [50,70,150,45,0,"Mixed",155,203.23],
        [35,85,200,55,1,"City",170,157.94],
    ], columns=[
        "Battery_Capacity_kWh",
        "Battery_Charge_pct",
        "Extra_Load_kg",
        "Speed_kmh",
        "AC_On",
        "Road_Type",
        "Consumption_Wh_km",
        "Range_km",
    ])

def numeric_or_default(df, column, default):
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series([default] * len(df), index=df.index)

def estimate_range(capacity, charge, usable, consumption,
                   load, speed, ac_on, road):
    # Illustrative assumptions, not validated vehicle physics.
    factor = 1 + max(load, 0) * 0.0002
    factor *= 1.08 if ac_on else 1.0
    factor *= 1.05 if road == "Highway" and speed > 90 else 1.0

    adjusted_consumption = consumption * factor
    usable_energy = capacity * charge / 100 * usable / 100

    estimated_range = (
        usable_energy * 1000 / adjusted_consumption
        if adjusted_consumption > 0 else 0
    )

    return usable_energy, adjusted_consumption, estimated_range

# ---------------- SESSION STATE ----------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "trained_model" not in st.session_state:
    st.session_state.trained_model = None

if "model_features" not in st.session_state:
    st.session_state.model_features = None

if "model_target" not in st.session_state:
    st.session_state.model_target = None

if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = None

if "model_test_results" not in st.session_state:
    st.session_state.model_test_results = None

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🔋 EV INTELLIGENCE")
    st.caption("Range • Energy • Load Analytics")
    st.markdown("---")

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Range Simulator",
            "Dataset Lab",
            "AI Model Studio",
            "Prediction History",
        ],
    )

    st.markdown("---")
    st.markdown("**DATA SOURCE**")
    st.caption("Upload measured data for meaningful analysis.")

    uploaded_file = st.file_uploader(
        "Upload EV CSV dataset",
        type=["csv"],
    )

    st.markdown("---")
    st.caption("EV Intelligence Prototype v2.0")

# ---------------- LOAD DATA ----------------
df = None
data_source = "Synthetic demo data"

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        data_source = f"Uploaded: {uploaded_file.name}"
    except Exception as error:
        st.error(f"Could not read CSV: {error}")

if df is None:
    df = demo_dataset()

# ---------------- HEADER ----------------
st.markdown(f"""
<div class="hero">
    <div class="eyebrow">
        EV ANALYTICS PLATFORM · {datetime.now().strftime('%d %b %Y')}
    </div>
    <h1>Electric Range Intelligence</h1>
    <p>
        Understand how battery charge, vehicle load and driving
        conditions influence estimated EV range.
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# 1. OVERVIEW DASHBOARD
# ==================================================
if page == "Overview":
    st.subheader("Fleet snapshot")

    c1, c2, c3, c4 = st.columns(4)

    charge = numeric_or_default(df, "Battery_Charge_pct", 0).dropna()
    load = numeric_or_default(df, "Extra_Load_kg", 0).dropna()
    ranges = numeric_or_default(df, "Range_km", np.nan).dropna()

    c1.metric("Dataset records", f"{len(df):,}")
    c2.metric(
        "Average charge",
        f"{charge.mean():.1f}%" if len(charge) else "N/A",
    )
    c3.metric(
        "Average load",
        f"{load.mean():.0f} kg" if len(load) else "N/A",
    )
    c4.metric(
        "Average range",
        f"{ranges.mean():.1f} km" if len(ranges) else "N/A",
    )

    st.caption(f"Data source: {data_source}")

    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("### Range vs. extra load")

        if {"Extra_Load_kg", "Range_km"}.issubset(df.columns):
            chart_df = df.copy()
            chart_df["Extra_Load_kg"] = pd.to_numeric(
                chart_df["Extra_Load_kg"], errors="coerce"
            )
            chart_df["Range_km"] = pd.to_numeric(
                chart_df["Range_km"], errors="coerce"
            )
            chart_df = chart_df.dropna(
                subset=["Extra_Load_kg", "Range_km"]
            )

            if len(chart_df):
                fig = px.scatter(
                    chart_df,
                    x="Extra_Load_kg",
                    y="Range_km",
                    color="Road_Type" if "Road_Type" in chart_df else None,
                    template="plotly_dark",
                    labels={
                        "Extra_Load_kg": "Extra load (kg)",
                        "Range_km": "Range (km)",
                    },
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid load/range values found.")
        else:
            st.info("Dataset needs Extra_Load_kg and Range_km columns.")

    with right:
        st.markdown("### Energy consumption")

        if "Consumption_Wh_km" in df.columns:
            chart_df = df.copy()
            chart_df["Consumption_Wh_km"] = pd.to_numeric(
                chart_df["Consumption_Wh_km"], errors="coerce"
            )
            chart_df = chart_df.dropna(subset=["Consumption_Wh_km"])

            if len(chart_df):
                fig = px.histogram(
                    chart_df,
                    x="Consumption_Wh_km",
                    nbins=12,
                    template="plotly_dark",
                    labels={"Consumption_Wh_km": "Consumption (Wh/km)"},
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid consumption values found.")
        else:
            st.info("Add Consumption_Wh_km to enable this chart.")

    st.markdown("### Recent dataset rows")
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)

    st.download_button(
        "Download current dataset",
        df.to_csv(index=False).encode("utf-8"),
        "ev_dataset_export.csv",
        "text/csv",
    )

# ==================================================
# 2. RANGE SIMULATOR
# ==================================================
elif page == "Range Simulator":
    st.subheader("Range & energy simulator")
    st.write(
        "Change the values to estimate remaining range. "
        "The adjustment factors are illustrative assumptions."
    )

    with st.form("range_form"):
        a, b, c = st.columns(3)

        with a:
            capacity = st.number_input(
                "Battery capacity (kWh)",
                min_value=1.0,
                max_value=250.0,
                value=40.0,
            )
            charge = st.slider("Battery charge (%)", 0, 100, 90)
            usable = st.slider("Usable battery (%)", 10, 100, 90)

        with b:
            extra_load = st.number_input(
                "Extra load (kg)",
                min_value=0,
                max_value=2000,
                value=100,
                step=10,
            )
            speed = st.slider("Average speed (km/h)", 5, 160, 50)
            ac_on = st.toggle("Air conditioning ON", value=False)

        with c:
            consumption = st.number_input(
                "Baseline consumption (Wh/km)",
                min_value=20,
                max_value=1000,
                value=150,
                step=5,
            )
            road = st.selectbox("Road type", ["City", "Mixed", "Highway"])

        calculate = st.form_submit_button(
            "Calculate estimated range",
            type="primary",
            use_container_width=True,
        )

    if calculate:
        energy, adjusted, distance = estimate_range(
            capacity, charge, usable, consumption,
            extra_load, speed, ac_on, road,
        )

        st.session_state.prediction_history.insert(0, {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Battery_kWh": capacity,
            "Charge_pct": charge,
            "Load_kg": extra_load,
            "Speed_kmh": speed,
            "AC_On": ac_on,
            "Road_Type": road,
            "Adjusted_Wh_km": round(adjusted, 2),
            "Estimated_Range_km": round(distance, 2),
        })

        x1, x2, x3 = st.columns(3)
        x1.metric("Usable energy", f"{energy:.2f} kWh")
        x2.metric("Adjusted consumption", f"{adjusted:.1f} Wh/km")
        x3.metric("Estimated range", f"{distance:.1f} km")

        st.progress(charge / 100, text=f"Battery charge: {charge}%")
        st.info(
            "Estimate only. This is not a live vehicle reading or "
            "a validated vehicle-specific prediction."
        )

# ==================================================
# 3. DATASET LAB
# ==================================================
elif page == "Dataset Lab":
    st.subheader("Dataset lab")
    st.write(f"**Data source:** {data_source}")

    search_text = st.text_input(
        "Filter rows by text",
        placeholder="Example: City or Highway",
    )

    view_df = df.copy()

    if search_text.strip():
        mask = view_df.astype(str).apply(
            lambda col: col.str.contains(
                search_text, case=False, na=False
            )
        ).any(axis=1)
        view_df = view_df[mask]

    st.dataframe(view_df, use_container_width=True, hide_index=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Rows", len(df))
    m2.metric("Columns", len(df.columns))
    m3.metric("Missing cells", int(df.isna().sum().sum()))

    with st.expander("Column types and missing values"):
        profile = pd.DataFrame({
            "Column": df.columns,
            "Data type": [str(df[col].dtype) for col in df.columns],
            "Missing values": [
                int(df[col].isna().sum()) for col in df.columns
            ],
            "Unique values": [
                int(df[col].nunique(dropna=True)) for col in df.columns
            ],
        })
        st.dataframe(profile, use_container_width=True, hide_index=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        selected_col = st.selectbox(
            "Choose a numeric column",
            numeric_cols,
        )

        fig = px.histogram(
            df,
            x=selected_col,
            nbins=20,
            template="plotly_dark",
            title=f"Distribution of {selected_col}",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

    st.download_button(
        "Download filtered CSV",
        view_df.to_csv(index=False).encode("utf-8"),
        "filtered_ev_dataset.csv",
        "text/csv",
    )

    quality_report = pd.DataFrame({
        "Column": df.columns,
        "Missing": [int(df[col].isna().sum()) for col in df.columns],
        "Unique": [int(df[col].nunique(dropna=True)) for col in df.columns],
    })

    st.download_button(
        "Download data quality report",
        quality_report.to_csv(index=False).encode("utf-8"),
        "ev_data_quality_report.csv",
        "text/csv",
    )

# ==================================================
# 4. AI MODEL STUDIO
# ==================================================
elif page == "AI Model Studio":
    st.subheader("AI model studio")
    st.write(
        "Train a Random Forest regression model using your dataset. "
        "Use measured EV trip data for meaningful results."
    )

    if not SKLEARN_AVAILABLE:
        st.error(
            "scikit-learn is missing. Add scikit-learn to "
            "requirements.txt and redeploy."
        )

    else:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            st.error("Upload a CSV containing numeric target data.")

        else:
            default_target = (
                "Range_km" if "Range_km" in numeric_cols
                else numeric_cols[-1]
            )

            target = st.selectbox(
                "Prediction target",
                numeric_cols,
                index=numeric_cols.index(default_target),
            )

            possible_features = [
                col for col in df.columns if col != target
            ]

            preferred = [
                "Battery_Capacity_kWh",
                "Battery_Charge_pct",
                "Extra_Load_kg",
                "Speed_kmh",
                "AC_On",
                "Road_Type",
            ]

            default_features = [
                col for col in preferred if col in possible_features
            ]

            features = st.multiselect(
                "Input features",
                possible_features,
                default=default_features or possible_features[:5],
            )

            test_percent = st.slider(
                "Test data (%)", 10, 40, 20, 5
            )

            if st.button(
                "Train machine learning model",
                type="primary",
                use_container_width=True,
            ):
                if not features:
                    st.error("Choose at least one input feature.")

                else:
                    train_df = df[features + [target]].copy()
                    train_df[target] = pd.to_numeric(
                        train_df[target], errors="coerce"
                    )
                    train_df = train_df.dropna(subset=[target])

                    if len(train_df) < 6:
                        st.error(
                            "Not enough valid rows. Upload a larger dataset."
                        )

                    else:
                        X = train_df[features]
                        y = train_df[target]

                        numeric_features = X.select_dtypes(
                            include=np.number
                        ).columns.tolist()

                        categorical_features = [
                            col for col in features
                            if col not in numeric_features
                        ]

                        transformers = []

                        if numeric_features:
                            transformers.append((
                                "num",
                                Pipeline([
                                    ("imputer", SimpleImputer(strategy="median"))
                                ]),
                                numeric_features,
                            ))

                        if categorical_features:
                            transformers.append((
                                "cat",
                                Pipeline([
                                    ("imputer", SimpleImputer(
                                        strategy="most_frequent"
                                    )),
                                    ("encoder", OneHotEncoder(
                                        handle_unknown="ignore"
                                    )),
                                ]),
                                categorical_features,
                            ))

                        preprocessor = ColumnTransformer(transformers)

                        model = Pipeline([
                            ("preprocessor", preprocessor),
                            ("regressor", RandomForestRegressor(
                                n_estimators=200,
                                random_state=42,
                            )),
                        ])

                        test_count = max(
                            2, int(round(len(train_df) * test_percent / 100))
                        )
                        test_count = min(test_count, len(train_df) - 2)

                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=test_count,
                            random_state=42,
                        )

                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)

                        mae = mean_absolute_error(y_test, predictions)
                        r2 = (
                            r2_score(y_test, predictions)
                            if len(y_test) > 1 else float("nan")
                        )

                        st.session_state.trained_model = model
                        st.session_state.model_features = features
                        st.session_state.model_target = target
                        st.session_state.model_metrics = {
                            "MAE": float(mae),
                            "R2": float(r2),
                            "Train rows": len(X_train),
                            "Test rows": len(X_test),
                        }

                        st.session_state.model_test_results = pd.DataFrame({
                            "Actual": y_test.to_numpy(),
                            "Predicted": predictions,
                            "Absolute_Error": np.abs(
                                y_test.to_numpy() - predictions
                            ),
                        })

            if (
                st.session_state.trained_model is not None
                and st.session_state.model_metrics is not None
            ):
                metrics = st.session_state.model_metrics

                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Mean absolute error", f"{metrics['MAE']:.2f}")
                q2.metric(
                    "R² score",
                    f"{metrics['R2']:.3f}"
                    if np.isfinite(metrics["R2"]) else "N/A",
                )
                q3.metric("Training rows", metrics["Train rows"])
                q4.metric("Test rows", metrics["Test rows"])

                st.caption(
                    f"Target: {st.session_state.model_target} | "
                    f"Features: {', '.join(st.session_state.model_features)}"
                )

                result_df = st.session_state.model_test_results

                if result_df is not None:
                    fig = px.scatter(
                        result_df,
                        x="Actual",
                        y="Predicted",
                        template="plotly_dark",
                        title="Actual vs predicted values",
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(
                        result_df, use_container_width=True, hide_index=True
                    )

                    st.download_button(
                        "Download model test results",
                        result_df.to_csv(index=False).encode("utf-8"),
                        "ev_model_test_results.csv",
                        "text/csv",
                    )

                st.markdown("### Predict a new case")

                with st.form("new_case_form"):
                    new_values = {}
                    cols = st.columns(
                        min(3, len(st.session_state.model_features))
                    )

                    for i, feature in enumerate(
                        st.session_state.model_features
                    ):
                        col = cols[i % len(cols)]

                        if feature in df.select_dtypes(
                            include=np.number
                        ).columns:
                            median = pd.to_numeric(
                                df[feature], errors="coerce"
                            ).median()

                            if not np.isfinite(median):
                                median = 0.0

                            new_values[feature] = col.number_input(
                                feature,
                                value=float(median),
                                key=f"new_{feature}",
                            )

                        else:
                            options = (
                                df[feature].dropna().astype(str).unique().tolist()
                            )
                            if not options:
                                options = ["Unknown"]

                            new_values[feature] = col.selectbox(
                                feature,
                                options,
                                key=f"new_{feature}",
                            )

                    predict_clicked = st.form_submit_button(
                        "Predict target",
                        type="primary",
                    )

                if predict_clicked:
                    new_X = pd.DataFrame(
                        [new_values],
                        columns=st.session_state.model_features,
                    )

                    prediction = float(
                        st.session_state.trained_model.predict(new_X)[0]
                    )

                    st.success(
                        f"Predicted {st.session_state.model_target}: "
                        f"{prediction:.2f}"
                    )

                    pred_export = pd.DataFrame([{
                        "Timestamp": datetime.now().isoformat(timespec="seconds"),
                        **new_values,
                        "Predicted_" + st.session_state.model_target: prediction,
                    }])

                    st.download_button(
                        "Download this prediction",
                        pred_export.to_csv(index=False).encode("utf-8"),
                        "ev_single_prediction.csv",
                        "text/csv",
                        key="download_single_prediction",
                    )

        st.warning(
            "The bundled demo dataset is synthetic and small. "
            "Its metrics are only for testing the workflow, not real-world "
            "accuracy. The trained model is kept in the current session."
        )

# ==================================================
# 5. PREDICTION HISTORY
# ==================================================
elif page == "Prediction History":
    st.subheader("Prediction history")
    history = st.session_state.prediction_history

    if history:
        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download prediction history",
            history_df.to_csv(index=False).encode("utf-8"),
            "ev_prediction_history.csv",
            "text/csv",
        )

        if st.button("Clear history"):
            st.session_state.prediction_history = []
            st.rerun()

    else:
        st.info(
            "No range simulations saved yet. "
            "Open Range Simulator and calculate an estimate."
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    '<p class="small-note">EV Intelligence is an educational prototype. '
    'Demo values are synthetic and range calculations are illustrative, '
    'not a substitute for measured vehicle data.</p>',
    unsafe_allow_html=True,
)
