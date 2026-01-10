import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from datetime import datetime, date, time, timedelta

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Delhi Traffic Flow Prediction",
    page_icon="🚦",
    layout="centered"
)

# --------------------------------------------------
# Load Model & Scaler
# --------------------------------------------------
model = load_model("models/lstm_traffic_model.keras", compile=False)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

SEQ_LEN = 48

# --------------------------------------------------
# Load Data
# --------------------------------------------------
hourly_df = pd.read_csv(
    "data/hourly_processed.csv",
    parse_dates=["timestamp"]
).sort_values("timestamp").reset_index(drop=True)

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def traffic_card(value):
    if value < 0.33:
        return (
            "🟢 Low Traffic",
            "Smooth traffic conditions. Ideal time to travel.",
            "Low congestion expected. Roads are likely to be free-flowing."
        )
    elif value < 0.66:
        return (
            "🟡 Moderate Traffic",
            "Average traffic conditions. Expect minor delays.",
            "Moderate congestion. Leave a little earlier to avoid delays."
        )
    else:
        return (
            "🔴 High Traffic",
            "Heavy traffic expected. Consider alternate routes.",
            "Traffic congestion is usually higher during this time."
        )

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🚦 Delhi Traffic Flow Prediction</h1>
    <p style='text-align:center; font-size:18px;'>
    Predict traffic density in advance and plan smarter travel
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# User Input
# --------------------------------------------------
st.subheader("📅 Select Date & Time for Travel")

col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "Select Date",
        value=hourly_df["timestamp"].dt.date.max()
    )


with col2:
    selected_hour = st.selectbox(
        "Select Hour (24-hour format)",
        [f"{h:02d}:00" for h in range(24)],
        index=15
    )

predict_btn = st.button("🚀 Predict Traffic")

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
if predict_btn:

    hour_int = int(selected_hour.split(":")[0])
    selected_datetime = datetime.combine(selected_date, time(hour_int, 0))
    last_known_time = hourly_df["timestamp"].max()

    if selected_datetime <= last_known_time:
        history = hourly_df[hourly_df["timestamp"] < selected_datetime].tail(SEQ_LEN)
        steps = 1
    else:
        history = hourly_df.tail(SEQ_LEN)
        steps = int((selected_datetime - last_known_time).total_seconds() // 3600)
        steps = max(1, min(steps, 24))

    if len(history) < SEQ_LEN:
        st.error("❌ Not enough historical data for prediction.")
        st.stop()

    recent_values = history["avg_queue_density"].values

    scaled_input = scaler.transform(recent_values.reshape(-1, 1))
    input_seq = scaled_input.reshape(1, SEQ_LEN, 1)

    preds = []
    for _ in range(steps):
        p = model.predict(input_seq, verbose=0)[0][0]
        preds.append(p)
        input_seq = np.append(input_seq[:, 1:, :], [[[p]]], axis=1)

    final_prediction = scaler.inverse_transform(
        np.array([[preds[-1]]])
    )[0][0]

    # --------------------------------------------------
    # Traffic Card Output
    # --------------------------------------------------
    level, short_msg, advice_msg = traffic_card(final_prediction)

    st.markdown("## 🚦 Traffic Prediction Result")

    colA, colB = st.columns(2)

    with colA:
        st.metric(
            "Predicted Avg Queue Density",
            f"{final_prediction:.3f}"
        )

    with colB:
        if "Low" in level:
            st.success(level)
        elif "Moderate" in level:
            st.warning(level)
        else:
            st.error(level)

    st.info(short_msg)

    # 🔼 BIGGER ADVICE MESSAGE (UPDATED)
    st.markdown(
        f"""
        <div style="
            font-size:18px;
            font-weight:600;
            padding:12px;
            border-radius:8px;
            background-color:rgba(255,255,255,0.05);
        ">
        {advice_msg}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        f"🕒 Prediction for {selected_datetime.strftime('%d %b %Y, %H:%M')}"
    )

    # --------------------------------------------------
    # Line Chart (Selected Date Only)
    # --------------------------------------------------
    st.markdown("## 📈 Traffic Trend for Selected Date")

    day_start = datetime.combine(selected_date, time(0, 0))
    day_end = day_start + timedelta(days=1)

    day_data = hourly_df[
        (hourly_df["timestamp"] >= day_start) &
        (hourly_df["timestamp"] < day_end)
    ]

    if not day_data.empty:
        st.line_chart(
            day_data.set_index("timestamp")["avg_queue_density"]
        )
    else:
        future_preds = []
        temp_seq = input_seq.copy()

        for _ in range(24):
            p = model.predict(temp_seq, verbose=0)[0][0]
            future_preds.append(p)
            temp_seq = np.append(temp_seq[:, 1:, :], [[[p]]], axis=1)

        future_vals = scaler.inverse_transform(
            np.array(future_preds).reshape(-1, 1)
        ).flatten()

        future_hours = pd.date_range(start=day_start, periods=24, freq="H")

        st.line_chart(
            pd.DataFrame(
                {"avg_queue_density": future_vals},
                index=future_hours
            )
        )

    # --------------------------------------------------
    # Heatmap
    # --------------------------------------------------
# Heatmap (7 Days Relative to Selected Date)
# --------------------------------------------------
    st.markdown("## 🔥 Traffic Density Heatmap (7-Day Context)")

    hourly_df["hour"] = hourly_df["timestamp"].dt.hour

    selected_day = selected_datetime.date()
    last_hist_day = hourly_df["timestamp"].dt.date.max()

    # Case 1: Past / Current → real historical heatmap
    if selected_day <= last_hist_day:

        heatmap_start = selected_day - timedelta(days=6)

        heatmap_df = hourly_df[
            (hourly_df["timestamp"].dt.date >= heatmap_start) &
            (hourly_df["timestamp"].dt.date <= selected_day)
        ].copy()

        heatmap_df["date"] = heatmap_df["timestamp"].dt.date

        heatmap_data = heatmap_df.pivot_table(
            values="avg_queue_density",
            index="date",
            columns="hour"
        )

        st.dataframe(
            heatmap_data.style
            .background_gradient(cmap="RdYlGn")
            .format("{:.3f}"),
            use_container_width=True
        )

    # Case 2: Future → shift labels to future dates (IMPORTANT FIX)
    else:
        pass
        # Use last 7 real historical days as pattern
        base_df = hourly_df.tail(7 * 24).copy()

        base_df["hour"] = base_df["timestamp"].dt.hour

        # Create NEW future date labels
        future_dates = [
            selected_day - timedelta(days=i)
            for i in range(6, -1, -1)
        ]

        # Map old dates → future dates
        old_dates = sorted(base_df["timestamp"].dt.date.unique())

        date_map = dict(zip(old_dates, future_dates))

        base_df["date"] = base_df["timestamp"].dt.date.map(date_map)

        heatmap_data = base_df.pivot_table(
            values="avg_queue_density",
            index="date",
            columns="hour"
        )

        st.dataframe(
            heatmap_data.style
            .background_gradient(cmap="RdYlGn")
            .format("{:.3f}"),
            use_container_width=True
        )


