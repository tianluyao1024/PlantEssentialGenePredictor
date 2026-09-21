"""Compatibility launcher for the NAR-oriented default PlantEGP interface."""

# Existing deployment commands may still run this file. The complete public UI
# lives in app.py so a standard Streamlit launch and source review see one entry
# point.
from app import *  # noqa: F401,F403
