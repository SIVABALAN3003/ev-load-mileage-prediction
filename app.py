            with col_a:
                test_pct = st.slider("Test split (%)", 10, 40, 20, 5)
            with col_b:
                tree_count = st.select_slider("Number of trees", options=[50, 100, 200, 300], value=100)
            if st.button("Train / refresh model", type="primary", use_container_width=True):
                if not features:
                    st.error("Choose at least one input feature.")
                else:
                    work = df[features + [target]].copy()
                    work[target] = pd.to_numeric(work[target], errors="coerce")
                    work = work.dropna(subset=[target])
                    if len(work) < 8:
                        st.error("At least eight rows with a valid numeric target are needed.")
                    else:
                        X = work[features]
                        y = work[target]
                        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
                        categorical_features = [col for col in features if col not in numeric_features]
                        transformers = []
                        if numeric_features:
                            transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features))
                        if categorical_features:
                            transformers.append(("cat", Pipeline([
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                            ]), categorical_features))
                        if not transformers:
                            st.error("No usable features were found.")
                        else:
                            test_rows = max(2, int(round(len(work) * test_pct / 100)))
                            test_rows = min(test_rows, len(work) - 2)
                            try:
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_rows, random_state=42)
                                model = Pipeline([
                                    ("prep", ColumnTransformer(transformers)),
                                    ("rf", RandomForestRegressor(n_estimators=tree_count, random_state=42, n_jobs=-1)),
                                ])
                                model.fit(X_train, y_train)
                                predicted = model.predict(X_test)
                                mae = mean_absolute_error(y_test, predicted)
                                r2 = r2_score(y_test, predicted) if len(y_test) > 1 else np.nan
                                st.session_state.trained_model = model
                                st.session_state.model_features = features
                                st.session_state.model_target = target
                                st.session_state.model_metrics = {"MAE": float(mae), "R2": float(r2), "Train rows": len(X_train), "Test rows": len(X_test), "Trees": tree_count}
                                st.session_state.model_test_results = pd.DataFrame({
                                    "Actual": y_test.to_numpy(),
                                    "Predicted": predicted,
                                    "Absolute_Error": np.abs(y_test.to_numpy() - predicted),
                                })
                                st.success("Model trained successfully.")
                            except Exception as exc:
                                st.error(f"Model training failed: {exc}")
            metrics = st.session_state.model_metrics
            results = st.session_state.model_test_results
            if metrics is not None and results is not None and st.session_state.trained_model is not None:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean absolute error", f"{metrics['MAE']:.3f}")
                c2.metric("R² score", f"{metrics['R2']:.3f}" if np.isfinite(metrics["R2"]) else "N/A")
                c3.metric("Training rows", metrics["Train rows"])
                c4.metric("Trees", metrics["Trees"])
                st.caption(f"Target: {st.session_state.model_target} · Features: {', '.join(st.session_state.model_features)}")
                left, right = st.columns(2)
                with left:
                    fig = px.scatter(results, x="Actual", y="Predicted", title="Actual vs predicted")
                    low = min(results["Actual"].min(), results["Predicted"].min())
                    high = max(results["Actual"].max(), results["Predicted"].max())
                    fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(dash="dash", color="#62f5bd"))
                    st.plotly_chart(style_chart(fig, 360), use_container_width=True)
                with right:
                    fig = px.bar(results.reset_index(), x="index", y="Absolute_Error", title="Absolute error per test row")
                    st.plotly_chart(style_chart(fig, 360), use_container_width=True)
                st.dataframe(results, use_container_width=True, hide_index=True)
                st.download_button("Download model evaluation CSV", results.to_csv(index=False).encode("utf-8"), "ev_model_evaluation.csv", "text/csv")
                st.markdown("#### Predict a new case")
                with st.form("new_case_form"):
                    new_case = {}
                    form_cols = st.columns(min(3, max(1, len(st.session_state.model_features))))
                    for i, feature in enumerate(st.session_state.model_features):
                        cell = form_cols[i % len(form_cols)]
                        if feature in df.select_dtypes(include=np.number).columns:
                            median = pd.to_numeric(df[feature], errors="coerce").median()
                            median = float(median) if pd.notna(median) else 0.0
                            new_case[feature] = cell.number_input(feature, value=median, key=f"predict_{feature}")
                        else:
                            options = df[feature].dropna().astype(str).unique().tolist()
                            if not options:
                                options = ["Unknown"]
                            new_case[feature] = cell.selectbox(feature, options, key=f"predict_{feature}")
                    predict_clicked = st.form_submit_button("Predict this case", type="primary")
                if predict_clicked:
                    try:
                        one_row = pd.DataFrame([new_case], columns=st.session_state.model_features)
                        prediction = float(st.session_state.trained_model.predict(one_row)[0])
                        st.success(f"Predicted {st.session_state.model_target}: {prediction:.2f}")
                        result_row = {**new_case, "Predicted_value": prediction, "Target": st.session_state.model_target}
                        st.download_button("Download this prediction", pd.DataFrame([result_row]).to_csv(index=False).encode("utf-8"), "ev_single_prediction.csv", "text/csv")
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
        st.warning("Use representative measured EV trips and independent test data before making real-world claims.")

elif page == "Prediction History":
    st.subheader("Prediction history")
    history = st.session_state.prediction_history
    if history:
        history_df = pd.DataFrame(history)
        st.metric("Saved simulations this session", len(history_df))
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        if "Estimated_Range_km" in history_df.columns:
            history_chart = history_df.iloc[::-1].copy()
            fig = px.line(history_chart, x="Time", y="Estimated_Range_km", markers=True, title="Saved range estimates over time")
            st.plotly_chart(style_chart(fig, 330), use_container_width=True)
        st.download_button("Export prediction history", history_df.to_csv(index=False).encode("utf-8"), "ev_prediction_history.csv", "text/csv")
        if st.button("Clear session history"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No simulations saved yet. Open Range Simulator and calculate an estimate.")

st.markdown("---")
st.markdown('<p class="small-note">EV Intelligence Pro is an educational prototype. Demo values and range calculations are illustrative. The app does not read live vehicle telemetry and must not be used for safety-critical battery decisions.</p>', unsafe_allow_html=True)
