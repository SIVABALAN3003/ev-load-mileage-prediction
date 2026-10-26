            "extra_load_kg": load,
            "speed_kmph": speed,
            "ac_on": 1 if ac == "On" else 0
        }])
        prediction = max(0.0, float(model.predict(input_data)[0]))
        st.session_state["last_prediction"] = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Vehicle ID": vehicle_id.strip() or "EV-001",
            "Battery (kWh)": battery,
            "Charge (%)": charge,
            "Load (kg)": load,
            "Speed (km/h)": speed,
            "AC": ac,
            "Predicted Range (km)": round(prediction, 2)
        }
        if "prediction_history" not in st.session_state:
            st.session_state["prediction_history"] = []
        st.session_state["prediction_history"].insert(
            0, st.session_state["last_prediction"]
        )

    if "last_prediction" in st.session_state:
        last = st.session_state["last_prediction"]
        st.markdown("#### Latest prediction")
        st.metric(
            f"Estimated range · {last['Vehicle ID']}",
            f"{last['Predicted Range (km)']:.2f} km"
        )
        st.progress(int(last["Charge (%)"]))
        st.caption(f"Battery charge input: {last['Charge (%)']}%")

with right:
    st.markdown("### 📊 Energy analytics")
    chart_tab1, chart_tab2 = st.tabs(["Load vs range", "Speed vs range"])
    with chart_tab1:
        fig_load = px.scatter(
            df, x="extra_load_kg", y="range_km", color="charge_percent",
            labels={
                "extra_load_kg": "Additional load (kg)",
                "range_km": "Recorded range (km)",
                "charge_percent": "Charge (%)"
            },
            template="plotly_dark", opacity=0.8
        )
        fig_load.update_layout(
            height=350, margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_load, use_container_width=True)
    with chart_tab2:
        fig_speed = px.scatter(
            df, x="speed_kmph", y="range_km", color="extra_load_kg",
            labels={
                "speed_kmph": "Speed (km/h)",
                "range_km": "Recorded range (km)",
                "extra_load_kg": "Load (kg)"
            },
            template="plotly_dark", opacity=0.8
        )
        fig_speed.update_layout(
            height=350, margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_speed, use_container_width=True)

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig_importance = px.bar(
        importance_df, x="Importance", y="Feature", orientation="h",
        title="What influences the model prediction?",
        template="plotly_dark"
    )
    fig_importance.update_layout(
        height=280, margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_importance, use_container_width=True)

# -------------------- HISTORY AND DATA --------------------
st.divider()
bottom_left, bottom_right = st.columns([1.2, 1], gap="large")

with bottom_left:
    st.markdown("### 🕘 Recent prediction history")
    history = st.session_state.get("prediction_history", [])
    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(history_df.head(10), use_container_width=True, hide_index=True)
        st.download_button(
            "Download prediction history CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="ev_prediction_history.csv",
            mime="text/csv"
        )
    else:
        st.info("Your predictions will appear here after you click Predict range.")
        st.caption("History is kept for this browser session only.")

with bottom_right:
    st.markdown("### 🗂️ Dataset preview")
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)
    st.download_button(
        "Download active dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="ev_active_dataset.csv",
        mime="text/csv"
    )

st.divider()
st.markdown(
    "<div class='small-note'>EV Intelligence Pro · Single-page analytics workspace · "
    "Educational prototype. Synthetic demo data is not real-world vehicle evidence. "
    "Use verified measured data before relying on predictions.</div>",
    unsafe_allow_html=True
)
