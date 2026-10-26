
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="EV Intelligence | Mileage AI",
    page_icon="🔋",
    layout="wide"
)

# ==========================================
# CUSTOM DESIGN
# ==========================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.2);
}
div[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.2);
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HELPERS
# ==========================================
def safe_number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_column_names(df):
    df = df.copy()
    df.columns = [
        str(c).strip() for c in df.columns
    ]
    return df


# ==========================================
# HEADER
# ==========================================
st.title("🔋 EV Intelligence")
st.markdown(
    "### Load-Based Mileage Prediction & Analytics"
)
st.caption(
    "EV Range Intelligence | Data Analytics | Machine Learning"
)

st.info(
    "Educational project dashboard. Estimates and ML "
    "predictions are not guaranteed real-world vehicle range."
)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("⚡ EV Intelligence")
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Range Estimator",
        "Dataset Analysis",
        "AI Model",
        "Reports"
    ]
)

st.sidebar.divider()
st.sidebar.caption("EV Load Mileage Prediction System")

# ==========================================
# SESSION STATE
# ==========================================
if "ev_model" not in st.session_state:
    st.session_state.ev_model = None

if "ev_features" not in st.session_state:
    st.session_state.ev_features = []

if "ev_target" not in st.session_state:
    st.session_state.ev_target = None

if "ev_metrics" not in st.session_state:
    st.session_state.ev_metrics = None

if "ev_report" not in st.session_state:
    st.session_state.ev_report = None

# ==========================================
# DATA UPLOAD
# ==========================================
st.sidebar.subheader("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload EV CSV Dataset",
    type=["csv"],
    key="ev_csv_upload"
)

df = None
data_error = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df = clean_column_names(df)

        if df.empty:
            data_error = "The uploaded CSV has no rows."
        elif len(df.columns) < 1:
            data_error = "No columns found in the CSV."

    except Exception as e:
        data_error = f"Could not read CSV: {e}"

if data_error:
    st.error(data_error)
    df = None

# ==========================================
# PAGE: DASHBOARD
# ==========================================
if page == "Dashboard":

    st.subheader("Executive Dashboard")

    st.write(
        "Monitor EV range estimates, battery energy, "
        "extra load, and uploaded dataset information."
    )

    c1, c2, c3, c4 = st.columns(4)

    battery = 40.0
    soc = 90
    usable = 90
    base = 150.0
    load = 150.0
    penalty = 2.0

    energy = battery * soc / 100 * usable / 100
    adjusted = base * (
        1 + (load / 100) * penalty / 100
    )
    estimated_range = energy * 1000 / adjusted
    normal_range = energy * 1000 / base

    c1.metric("Battery Capacity", "40 kWh")
    c2.metric("Battery Charge", "90%")
    c3.metric("Extra Load", "150 kg")
    c4.metric(
        "Estimated Range",
        f"{estimated_range:.1f} km"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("#### Range Overview")

        range_df = pd.DataFrame({
            "Scenario": [
                "Without Extra Load",
                "With Extra Load"
            ],
            "Range (km)": [
                normal_range,
                estimated_range
            ]
        })

        fig = px.bar(
            range_df,
            x="Scenario",
            y="Range (km)",
            color="Scenario",
            text_auto=".1f",
            title="Illustrative Range Comparison"
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:
        st.markdown("#### Energy Overview")

        energy_df = pd.DataFrame({
            "Energy": [
                "Available",
                "Not Used in This Estimate"
            ],
            "kWh": [
                energy,
                battery - energy
            ]
        })

        fig2 = px.pie(
            energy_df,
            names="Energy",
            values="kWh",
            hole=0.55,
            title="Battery Energy Illustration"
        )
        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.divider()
    st.markdown("#### Dataset Status")

    if df is not None:
        a, b, c = st.columns(3)
        a.metric("Dataset Rows", len(df))
        b.metric("Dataset Columns", len(df.columns))
        c.metric(
            "Missing Values",
            int(df.isna().sum().sum())
        )

        st.success("EV dataset loaded successfully.")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning(
            "No dataset uploaded. Upload your EV CSV "
            "from the sidebar to activate dataset analysis."
        )

    st.caption(
        "Default figures are illustrative. The dashboard "
        "does not represent actual vehicle telemetry."
    )

# ==========================================
# PAGE: RANGE ESTIMATOR
# ==========================================
elif page == "Range Estimator":

    st.subheader("🚗 Load-Based Range Estimator")

    st.write(
        "Adjust the values to see how battery energy, "
        "consumption, and extra load affect estimated range."
    )

    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Vehicle Parameters")

        battery = st.number_input(
            "Battery Capacity (kWh)",
            min_value=0.1,
            max_value=500.0,
            value=40.0,
            step=1.0
        )

        soc = st.slider(
            "Battery Charge (%)",
            0, 100, 90
        )

        usable = st.slider(
            "Usable Battery Energy (%)",
            10, 100, 90
        )

        base = st.number_input(
            "Baseline Consumption (Wh/km)",
            min_value=1.0,
            max_value=2000.0,
            value=150.0,
            step=5.0
        )

        load = st.number_input(
            "Additional Load (kg)",
            min_value=0.0,
            max_value=5000.0,
            value=150.0,
            step=10.0
        )

        penalty = st.slider(
            "Assumed Consumption Increase per 100 kg (%)",
            0.0, 10.0, 2.0, 0.5
        )

    energy = (
        battery * soc / 100 * usable / 100
    )

    adjusted = base * (
        1 + (load / 100) * penalty / 100
    )

    normal_range = energy * 1000 / base
    estimated_range = energy * 1000 / adjusted

    with right:
        st.markdown("#### Estimation Results")

        st.metric(
            "Available Battery Energy",
            f"{energy:.2f} kWh"
        )

        st.metric(
            "Estimated Consumption",
            f"{adjusted:.1f} Wh/km"
        )

        st.metric(
            "Estimated Driving Range",
            f"{estimated_range:.1f} km",
            delta=f"{estimated_range-normal_range:.1f} km"
        )

        if normal_range > 0:
            reduction = (
                (normal_range-estimated_range)
                / normal_range * 100
            )
        else:
            reduction = 0

        st.metric(
            "Estimated Range Reduction",
            f"{reduction:.2f}%"
        )

        st.caption(
            "The assumed load penalty is user-defined, "
            "not a validated vehicle-specific value."
        )

    st.divider()

    loads = np.arange(0, 1001, 50)

    curve = pd.DataFrame({
        "Extra Load (kg)": loads,
        "Estimated Range (km)": [
            energy * 1000 / (
                base * (
                    1 + (x / 100) * penalty / 100
                )
            )
            for x in loads
        ]
    })

    fig = px.line(
        curve,
        x="Extra Load (kg)",
        y="Estimated Range (km)",
        markers=True,
        title="Estimated Range vs Additional Load"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Range Curve",
        curve.to_csv(index=False).encode("utf-8"),
        file_name="ev_range_curve.csv",
        mime="text/csv"
    )

# ==========================================
# PAGE: DATASET ANALYSIS
# ==========================================
elif page == "Dataset Analysis":

    st.subheader("📊 Dataset Analytics")

    if df is None:
        st.warning(
            "Upload an EV CSV dataset using the sidebar first."
        )
    else:
        st.success("Dataset is ready.")

        a, b, c = st.columns(3)

        a.metric("Rows", len(df))
        b.metric("Columns", len(df.columns))
        c.metric(
            "Missing Cells",
            int(df.isna().sum().sum())
        )

        st.markdown("#### Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)

        st.markdown("#### Column Information")

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [
                str(df[col].dtype) for col in df.columns
            ],
            "Missing Values": [
                int(df[col].isna().sum())
                for col in df.columns
            ],
            "Unique Values": [
                int(df[col].nunique(dropna=True))
                for col in df.columns
            ]
        })

        st.dataframe(
            info_df,
            use_container_width=True
        )

        st.markdown("#### Missing Value Analysis")

        missing_df = (
            df.isna().sum()
            .reset_index()
        )
        missing_df.columns = ["Column", "Missing Values"]

        missing_df = missing_df[
            missing_df["Missing Values"] > 0
        ]

        if not missing_df.empty:
            fig = px.bar(
                missing_df,
                x="Column",
                y="Missing Values",
                title="Missing Values by Column"
            )
            st.plotly_chart(
                fig,
                use_container_width=True
            )
        else:
            st.success("No missing values found.")

        st.markdown("#### Explore a Column")

        selected_col = st.selectbox(
            "Choose a column",
            df.columns
        )

        selected_series = df[selected_col]

        if pd.api.types.is_numeric_dtype(
            selected_series
        ):
            st.write(
                selected_series.describe().to_frame(
                    "Statistics"
                )
            )

            fig = px.histogram(
                df,
                x=selected_col,
                title=f"Distribution of {selected_col}"
            )
            st.plotly_chart(
                fig,
                use_container_width=True
            )
        else:
            top_values = (
                selected_series.fillna("Missing")
                .astype(str)
                .value_counts()
                .head(20)
                .reset_index()
            )
            top_values.columns = ["Value", "Count"]

            fig = px.bar(
                top_values,
                x="Value",
                y="Count",
                title=f"Top values in {selected_col}"
            )
            st.plotly_chart(
                fig,
                use_container_width=True
            )

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        st.markdown("#### Feature Relationship")

        if len(numeric_cols) >= 2:
            x_col = st.selectbox(
                "X axis",
                numeric_cols,
                key="analysis_x"
            )

            y_options = [
                col for col in numeric_cols
                if col != x_col
            ]

            if y_options:
                y_col = st.selectbox(
                    "Y axis",
                    y_options,
                    key="analysis_y"
                )

                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}"
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True
                )
        else:
            st.info(
                "At least two numeric columns are needed "
                "for a scatter plot."
            )

        st.download_button(
            "⬇️ Download Uploaded Dataset",
            df.to_csv(index=False).encode("utf-8"),
            file_name="ev_dataset_copy.csv",
            mime="text/csv"
        )

# ==========================================
# PAGE: AI MODEL
# ==========================================
elif page == "AI Model":

    st.subheader("🤖 EV Mileage Machine Learning")

    st.write(
        "Train a Random Forest regression model using "
        "your uploaded dataset."
    )

    st.warning(
        "Choose a numeric target such as range_km or "
        "mileage_km. A meaningful model requires "
        "representative, measured data."
    )

    if df is None:
        st.info(
            "Upload your CSV dataset using the sidebar."
        )
    else:
        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if not numeric_cols:
            st.error(
                "No numeric target columns found."
            )
        else:
            target = st.selectbox(
                "Select Prediction Target",
                numeric_cols
            )

            candidate_features = [
                c for c in df.columns
                if c != target
            ]

            features = st.multiselect(
                "Select Input Features",
                candidate_features,
                default=[
                    c for c in candidate_features
                    if pd.api.types.is_numeric_dtype(df[c])
                ][:8]
            )

            st.caption(
                "Avoid ID columns, future information, "
                "and columns calculated directly from "
                "the target. Such leakage can make scores "
                "misleading."
            )

            if st.button(
                "🚀 Train Machine Learning Model",
                type="primary"
            ):
                if not features:
                    st.error(
                        "Select at least one feature."
                    )
                else:
                    try:
                        data = df[
                            features + [target]
                        ].copy()

                        data[target] = pd.to_numeric(
                            data[target],
                            errors="coerce"
                        )

                        data = data.replace(
                            [np.inf, -np.inf],
                            np.nan
                        )

                        data = data.dropna(
                            subset=[target]
                        )

                        # Remove empty feature columns
                        features = [
                            col for col in features
                            if not data[col].isna().all()
                        ]

                        if not features:
                            raise ValueError(
                                "No usable feature columns."
                            )

                        data = data[
                            features + [target]
                        ]

                        # Need enough rows for a useful split
                        if len(data) < 10:
                            raise ValueError(
                                "At least 10 usable rows "
                                "are required for this demo."
                            )

                        X = data[features]
                        y = data[target]

                        numeric_features = (
                            X.select_dtypes(
                                include=np.number
                            ).columns.tolist()
                        )

                        categorical_features = [
                            col for col in features
                            if col not in numeric_features
                        ]

                        transformers = []

                        if numeric_features:
                            numeric_pipeline = Pipeline([
                                (
                                    "imputer",
                                    SimpleImputer(
                                        strategy="median"
                                    )
                                )
                            ])

                            transformers.append((
                                "numeric",
                                numeric_pipeline,
                                numeric_features
                            ))

                        if categorical_features:
                            categorical_pipeline = Pipeline([
                                (
                                    "imputer",
                                    SimpleImputer(
                                        strategy="most_frequent"
                                    )
                                ),
                                (
                                    "encoder",
                                    OneHotEncoder(
                                        handle_unknown="ignore"
                                    )
                                )
                            ])

                            transformers.append((
                                "categorical",
                                categorical_pipeline,
                                categorical_features
                            ))

                        preprocessor = ColumnTransformer(
                            transformers=transformers
                        )

                        model = Pipeline([
                            (
                                "preprocessing",
                                preprocessor
                            ),
                            (
                                "regressor",
                                RandomForestRegressor(
                                    n_estimators=200,
                                    random_state=42,
                                    n_jobs=-1
                                )
                            )
                        ])

                        X_train, X_test, y_train, y_test = (
                            train_test_split(
                                X,
                                y,
                                test_size=0.2,
                                random_state=42
                            )
                        )

                        model.fit(X_train, y_train)

                        predictions = model.predict(
                            X_test
                        )

                        mae = mean_absolute_error(
                            y_test,
                            predictions
                        )

                        r2 = r2_score(
                            y_test,
                            predictions
                        )

                        report = pd.DataFrame({
                            "Actual": y_test.to_numpy(),
                            "Predicted": predictions
                        })

                        st.session_state.ev_model = model
                        st.session_state.ev_features = features
                        st.session_state.ev_target = target
                        st.session_state.ev_metrics = {
                            "mae": mae,
                            "r2": r2,
                            "rows": len(data),
                            "train_rows": len(X_train),
                            "test_rows": len(X_test)
                        }
                        st.session_state.ev_report = report

                        st.success(
                            "Model trained successfully!"
                        )

                    except Exception as e:
                        st.error(
                            f"Training error: {e}"
                        )

            if st.session_state.ev_model is not None:
                if (
                    st.session_state.ev_target in numeric_cols
                ):
                    st.divider()
                    st.markdown("#### Model Performance")

                    metrics = st.session_state.ev_metrics

                    m1, m2, m3 = st.columns(3)

                    m1.metric(
                        "Mean Absolute Error",
                        f"{metrics['mae']:.3f}"
                    )

                    m2.metric(
                        "R² Score",
                        f"{metrics['r2']:.3f}"
                    )

                    m3.metric(
                        "Test Rows",
                        metrics["test_rows"]
                    )

                    st.caption(
                        f"Training rows: {metrics['train_rows']} | "
                        f"Usable rows: {metrics['rows']}"
                    )

                    st.caption(
                        "MAE is expressed in the target's units. "
                        "R² may be negative. A high score alone "
                        "does not prove generalization."
                    )

                    report = st.session_state.ev_report

                    fig = px.scatter(
                        report,
                        x="Actual",
                        y="Predicted",
                        title="Actual vs Predicted"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.dataframe(
                        report.head(100),
                        use_container_width=True
                    )

                    st.download_button(
                        "⬇️ Download Model Test Results",
                        report.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="ev_model_test_results.csv",
                        mime="text/csv"
                    )

                    st.divider()
                    st.markdown("#### Predict a New Sample")

                    st.write(
                        "Enter input values for the trained "
                        "model's selected features."
                    )

                    model_features = (
                        st.session_state.ev_features
                    )

                    input_values = {}

                    with st.form("ev_prediction_form"):
                        for feature in model_features:
                            series = df[feature]

                            if pd.api.types.is_numeric_dtype(
                                series
                            ):
                                valid = pd.to_numeric(
                                    series,
                                    errors="coerce"
                                ).dropna()

                                default_value = (
                                    float(valid.median())
                                    if not valid.empty
                                    else 0.0
                                )

                                input_values[feature] = (
                                    st.number_input(
                                        feature,
                                        value=default_value,
                                        key=f"pred_{feature}"
                                    )
                                )
                            else:
                                choices = (
                                    series.dropna()
                                    .astype(str)
                                    .unique()
                                    .tolist()
                                )

                                choices = choices[:100]

                                if choices:
                                    input_values[feature] = (
                                        st.selectbox(
                                            feature,
                                            choices,
                                            key=f"pred_{feature}"
                                        )
                                    )
                                else:
                                    input_values[feature] = (
                                        st.text_input(
                                            feature,
                                            key=f"pred_{feature}"
                                        )
                                    )

                        submitted = st.form_submit_button(
                            "Predict"
                        )

                    if submitted:
                        try:
                            sample = pd.DataFrame(
                                [input_values]
                            )

                            prediction = (
                                st.session_state.ev_model
                                .predict(sample)[0]
                            )

                            st.success(
                                f"Predicted {st.session_state.ev_target}: "
                                f"{prediction:.3f}"
                            )

                            st.caption(
                                "Prediction is based on the "
                                "trained dataset model."
                            )

                        except Exception as e:
                            st.error(
                                f"Prediction error: {e}"
                            )

# ==========================================
# PAGE: REPORTS
# ==========================================
elif page == "Reports":

    st.subheader("📥 Reports Center")

    st.write(
        "Download available results and dataset reports."
    )

    if df is not None:
        st.markdown("#### Dataset Report")

        st.download_button(
            "Download Dataset CSV",
            df.to_csv(index=False).encode("utf-8"),
            "ev_dataset_report.csv",
            "text/csv"
        )

        summary = df.describe(
            include="all"
        ).transpose().reset_index()

        st.download_button(
            "Download Dataset Statistics",
            summary.to_csv(index=False).encode("utf-8"),
            "ev_dataset_statistics.csv",
            "text/csv"
        )

    if st.session_state.ev_report is not None:
        st.markdown("#### AI Model Report")

        st.download_button(
            "Download AI Test Predictions",
            st.session_state.ev_report.to_csv(
                index=False
            ).encode("utf-8"),
            "ev_ai_predictions.csv",
            "text/csv"
        )

    st.markdown("#### Range Estimate Report")

    battery = st.number_input(
        "Report Battery Capacity (kWh)",
        min_value=0.1,
        value=40.0,
        key="report_battery"
    )

    charge = st.slider(
        "Report Battery Charge (%)",
        0, 100, 90,
        key="report_charge"
    )

    consumption = st.number_input(
        "Report Consumption (Wh/km)",
        min_value=1.0,
        value=150.0,
        key="report_consumption"
    )

    report_energy = battery * charge / 100
    report_range = (
        report_energy * 1000 / consumption
    )

    range_report = pd.DataFrame({
        "Battery Capacity (kWh)": [battery],
        "Charge (%)": [charge],
        "Available Energy (kWh)": [report_energy],
        "Consumption (Wh/km)": [consumption],
        "Estimated Range (km)": [report_range]
    })

    st.dataframe(
        range_report,
        use_container_width=True
    )

    st.download_button(
        "Download Range Estimate",
        range_report.to_csv(
            index=False
        ).encode("utf-8"),
        "ev_range_estimate.csv",
        "text/csv"
    )

    st.caption(
        "All report calculations are estimates. "
        "Validate against measured vehicle data."
    )

# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption(
    "EV Intelligence | Educational Prototype | "
    "Load-Based Mileage Prediction"
)
