"""
VA Woods – multipage Streamlit app entry point.

Pages are auto-discovered from viz/pages/. This file provides the shared
sidebar controls and landing page content.
"""

import streamlit as st

st.set_page_config(page_title="VA Woods", layout="wide")

st.title("Virginia Timberland Soil Fungus Composition Tool")
st.markdown(
    "This tool helps identify actively harvested timber parcels by biomass of Loblolly Pine, manage sampling of healthy woodlands with similar environmental characteristics, and generate predictions of healthy soil fungal communities at any point in Virginia.\n"
    "The proposed workflow is as follows:\n"
    "- **Target** – Identify actively harvested timber parcels and rank them by biomass of Loblolly Pine. Parcels of interest can be marked as targets for further investigation in the next stages.\n"
    "- **Sample** – Visualize environmental variables used to predict fungal communities, import fungal community samples (OTU tables) from healthy woodlands, and plan future sampling by targeting regions with environmental conditions not covered by the current samples.\n"
    "- **Predict** – Train a gradient boosting model on the environmental variables and healthy fungal community samples, and use it to predict expected fungal communities for any point in Virginia.\n"
)
