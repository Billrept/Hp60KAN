
import streamlit as st
from datetime import datetime, timedelta
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hp60_kan.predict_numpy import Hp60PredictorNumpy

st.set_page_config(page_title="Hp60 Predictor", layout="wide")

@st.cache_resource
def load_model():
    """Load model once and cache it."""
    predictor = Hp60PredictorNumpy()
    predictor.load()
    return predictor

predictor = load_model()
fig = go.Figure()

st.header("Hp60 Prediction Over Time Visualization")

st.markdown("\tOn this page, users can input the AP60 index and a time range to predict weather conditions. " \
                "The results from different predictive models can then be compared to better understand their behavior and accuracy.")
st.space(50)


graph_col, datetime_col = st.columns([3, 1])

with datetime_col:
    container = st.container(border=True)

    with container:
        st.subheader("Input Parameters")
        st.space(10)

        ap60 = st.slider("AP60 Index Value", 0, 400, 45, help="AP60 is a geomagnetic activity index used to predict HP60")
        st.space(10)

        date_col, hour_col = st.columns([3, 2])

        with date_col:
            start_date = st.date_input("From Date", datetime(2024, 5, 15), key="start_date")
            st.space(10)
            end_date = st.date_input("To Date", datetime(2024, 5, 16), key="end_date")

        with hour_col:
            start_hour = st.number_input("Hour (UTC)", 0, 23, 12, key="start_hour")
            st.space(10)
            end_hour = st.number_input("Hour (UTC)", 0, 23, 12, key="end_hour")

        start_dt = datetime(start_date.year, start_date.month, start_date.day, start_hour)
        end_dt = datetime(end_date.year, end_date.month, end_date.day, end_hour)

        times = pd.date_range(start=start_dt, end=end_dt, freq="h")
        st.space(20)

        if st.button("Predict", type="primary", width="stretch"):
            if end_dt > start_dt and len(times) <= 168:
                try:
                    prediction_dict = []

                    for t in times:
                        prediction = round(predictor.predict(ap60=ap60, dt=t), 2)
                        prediction_dict.append({"datetime": t, "hp60": prediction})

                    prediction_df = pd.DataFrame(prediction_dict)
                    fig = px.line(
                        prediction_df,
                        x="datetime",
                        y="hp60",
                    )
                    fig.update_layout(
                        xaxis_title="Datetime (UTC)",
                        yaxis_title="HP60 Value",
                        hovermode="x unified"
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("End datetime must be after start datetime and the range should not exceed 7 days.")

        st.space(20)


with graph_col:
    kan_container = st.container(border=True)
    with kan_container:
        st.subheader("KAN model Hp60 Predictions")
        try:
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            print(e)
        
        st.subheader("Explanation of Results")
        st.markdown("...........")

    st.divider()
    other_container = st.container(border=True)
    with other_container:
        st.subheader("Other model Hp60 Predictions")
        st.info("Other models' predictions will be displayed here for comparison in future updates.")
        st.subheader("Explanation of Results")
        st.markdown("...........")