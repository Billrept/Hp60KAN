import streamlit as st
from datetime import datetime
import sys
from pathlib import Path

# Add Hp60KAN folder to Python path
sys.path.insert(0, str(Path(__file__).parent / "Hp60KAN"))

from hp60_kan import Hp60Predictor

# Load trained model
predictor = Hp60Predictor()

# Predict Hp60 from ap60 and datetime
hp60 = predictor.predict(ap60=45, dt=datetime(2024, 5, 15, 12, 0))
print(f"Predicted Hp60: {hp60:.2f}")

# Predict for storm conditions
hp60_storm = predictor.predict(ap60=300, dt=datetime(2024, 5, 11, 2, 0))
print(f"Storm Hp60: {hp60_storm:.2f}")  # Will predict ~9-10