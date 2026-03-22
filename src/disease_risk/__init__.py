"""
src/disease_risk/__init__.py
----------------------------
Package entry point for the disease risk prediction module.

Main component used by the app:
  - XGBRiskPredictor  : Combined XGBoost + KDE predictor.
                        Loaded by api_server.py at startup.
                        Call .predict(lat, lon, age, gender) to get risk %.

Utility components (used internally):
  - DISEASES                  : tuple — ("dengue", "malaria", "typhoid")
  - load_cases_from_directory : reads all .xlsx files → returns case records

Usage:
    from src.disease_risk import XGBRiskPredictor
    from src.disease_risk.xgb_predictor import XGBRiskPredictor
"""

from .engine import DISEASES, load_cases_from_directory
from .xgb_predictor import XGBRiskPredictor

__all__ = ["DISEASES", "load_cases_from_directory", "XGBRiskPredictor"]
