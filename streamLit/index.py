import streamlit as st
from datetime import datetime
import sys
from pathlib import Path

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

st.title("🌍 Hp60 Geomagnetic Index Predictor")
st.markdown("Predict Hp60 values from ap60 index and datetime")

# Load model
predictor = load_model()

# Footer
st.divider()
st.caption("Built with hp60_kan API • Data: Hp60 Geomagnetic Index")