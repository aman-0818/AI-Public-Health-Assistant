"""
src/disease_risk/xgb_predictor.py
----------------------------------
Combined XGBoost + KDE disease risk predictor.

HOW IT WORKS (Simple Story):
  Imagine two experts working together:

  Expert 1 — XGBoost Model:
    "Given this person's age, gender, location, and current season —
     what disease pattern does this MATCH from my training data?"
    → Gives probability based on learned patterns.

  Expert 2 — KDE (Kernel Density Estimation):
    "How many real cases actually happened near this EXACT location
     historically? How dense was the disease activity here?"
    → Gives a score based on actual historical case density.

  Final Answer = XGBoost probability × KDE density score
    → Then normalize so all 3 diseases sum to 100%

  This way:
    - Same city, different area → different results (thanks to KDE)
    - Age/season matters → different results (thanks to XGBoost)
    - Both together → more accurate and location-specific prediction

REQUIRES (run train_ml_model.py first):
  - ml_output/disease_xgb_model.json
  - ml_output/model_meta.json
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import xgboost as xgb
import numpy as np

from .engine import load_cases_from_directory, _haversine_km, DISEASES


class XGBRiskPredictor:
    """
    Combined XGBoost + KDE disease risk predictor.

    Usage:
        predictor = XGBRiskPredictor.load(
            model_dir   = Path("ml_output"),
            dataset_dir = Path("dataset")
        )
        result = predictor.predict(latitude=13.08, longitude=80.27, age=30, gender="male")
    """

    # KDE bandwidth in km — controls how far each case's "influence" spreads.
    # 15km means a case 15km away has ~60% influence, 30km away has ~13%.
    KDE_BANDWIDTH_KM = 15.0

    # Only consider cases within this radius for KDE calculation (performance cutoff).
    # Cases beyond 50km have negligible influence (<4%) so we skip them.
    KDE_MAX_RADIUS_KM = 50.0

    # Training data coverage — Tamil Nadu, India bounding box.
    # Predictions outside this region are unreliable (no training data).
    TRAINING_REGION = {
        "name":    "Tamil Nadu, India",
        "lat_min": 8.0,
        "lat_max": 13.6,
        "lon_min": 76.2,
        "lon_max": 80.4,
    }

    def __init__(self):
        self._model = None              # XGBoost classifier
        self._classes: List[str] = []   # e.g. ['Dengue', 'Malaria', 'Negative', 'Typhoid']
        self._meta: Dict = {}           # Model metadata (accuracy, trained_at, etc.)
        self._cases: List[Dict] = []    # Raw cases for KDE + nearby_cases calculation

    # ──────────────────────────────────────────────────────────────
    # Loading
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, model_dir: Path, dataset_dir: Path) -> "XGBRiskPredictor":
        """
        Load the XGBoost model + metadata + raw case data.

        Args:
            model_dir   : folder with disease_xgb_model.json and model_meta.json
            dataset_dir : folder with .xlsx files (used for KDE + nearby_cases)

        Raises:
            FileNotFoundError if XGBoost model file is missing (run train_ml_model.py)
        """
        predictor = cls()

        # Load the trained XGBoost model
        model_path = model_dir / "disease_xgb_model.json"
        if not model_path.exists():
            raise FileNotFoundError(
                f"XGBoost model not found at: {model_path}\n"
                "Please run train_ml_model.py first to generate the model."
            )
        predictor._model = xgb.XGBClassifier()
        predictor._model.load_model(str(model_path))

        # Load model metadata — class names, accuracy, etc.
        meta_path = model_dir / "model_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                predictor._meta = json.load(f)
            predictor._classes = predictor._meta.get("classes", [])

        # Load raw Excel case data for KDE scoring and nearby_cases display
        if dataset_dir.exists():
            try:
                cases, _ = load_cases_from_directory(dataset_dir)
                predictor._cases = cases
                print(f"  Loaded {len(cases)} raw cases for KDE + spatial context.")
            except Exception as e:
                print(f"  Warning: Could not load raw cases ({e}). KDE will be skipped.")
                predictor._cases = []

        return predictor

    # ──────────────────────────────────────────────────────────────
    # Feature building for XGBoost
    # ──────────────────────────────────────────────────────────────

    def _build_features(
        self,
        latitude: float,
        longitude: float,
        age: Optional[float],
        gender: Optional[str],
    ) -> np.ndarray:
        """
        Build the 6-feature vector for XGBoost.
        Order must match training: [latitude, longitude, age, gender, month, season]
        """
        # Use current month — seasonal patterns matter (dengue spikes in monsoon)
        now = datetime.now()
        month = now.month
        # Season: 1=Summer(Mar-May), 2=Monsoon(Jun-Sep), 3=Winter(Oct-Feb)
        season = 1 if month in [3, 4, 5] else 2 if month in [6, 7, 8, 9] else 3

        # Age: default to 30 if not provided
        age_val = float(age) if age is not None else 30.0

        # Gender encoding: male=1.0, female=0.0, unknown/other=0.5
        g = (gender or "").strip().lower()
        gender_val = 1.0 if g in ("male", "m") else 0.0 if g in ("female", "f") else 0.5

        return np.array([[latitude, longitude, age_val, gender_val, month, season]])

    # ──────────────────────────────────────────────────────────────
    # KDE — Kernel Density Estimation
    # ──────────────────────────────────────────────────────────────

    def _kde_scores(self, latitude: float, longitude: float) -> Dict[str, float]:
        """
        Calculate Gaussian KDE score for each disease at the given location.

        For each disease, we sum up the Gaussian kernel contributions from
        all nearby cases. Cases closer to the user's location contribute more,
        cases farther away contribute less.

        Gaussian kernel: weight = exp(-0.5 * (distance / bandwidth)^2)

        Example:
          Case 5km away  → weight = exp(-0.5 * (5/15)^2)  = 0.945  (strong influence)
          Case 15km away → weight = exp(-0.5 * (15/15)^2) = 0.607  (moderate)
          Case 30km away → weight = exp(-0.5 * (30/15)^2) = 0.135  (weak)
          Case 50km away → weight = exp(-0.5 * (50/15)^2) = 0.034  (negligible)

        Returns:
          {dengue: score, malaria: score, typhoid: score}
          A score of 0.0 means no cases nearby.
        """
        scores = {d: 0.0 for d in DISEASES}

        if not self._cases:
            return scores

        # Performance optimization: use bounding box to skip far-away cases
        # 1 degree latitude ≈ 111 km, so KDE_MAX_RADIUS_KM/111 gives degree cutoff
        deg_cutoff = self.KDE_MAX_RADIUS_KM / 111.0

        for case in self._cases:
            # Quick bounding box check — much faster than haversine for all cases
            if abs(case["latitude"] - latitude) > deg_cutoff:
                continue
            if abs(case["longitude"] - longitude) > deg_cutoff:
                continue

            disease = case.get("disease", "")
            if disease not in DISEASES:
                continue

            # Precise distance using haversine formula
            dist_km = _haversine_km(latitude, longitude, case["latitude"], case["longitude"])
            if dist_km > self.KDE_MAX_RADIUS_KM:
                continue

            # Gaussian kernel weight — closer = higher weight
            weight = math.exp(-0.5 * (dist_km / self.KDE_BANDWIDTH_KM) ** 2)
            scores[disease] += weight

        return scores

    # ──────────────────────────────────────────────────────────────
    # Nearby cases count (for frontend display)
    # ──────────────────────────────────────────────────────────────

    def _count_nearby_cases(
        self, latitude: float, longitude: float, radius_km: float = 25.0
    ) -> Dict[str, int]:
        """
        Count actual historical cases within radius_km of the given location.
        Used for the 'Nearby cases (25km)' chip in the frontend.
        """
        counts = {d: 0 for d in DISEASES}
        deg_cutoff = radius_km / 111.0

        for case in self._cases:
            if abs(case["latitude"] - latitude) > deg_cutoff:
                continue
            if abs(case["longitude"] - longitude) > deg_cutoff:
                continue
            dist = _haversine_km(latitude, longitude, case["latitude"], case["longitude"])
            if dist <= radius_km:
                disease = case.get("disease", "")
                if disease in counts:
                    counts[disease] += 1
        return counts

    def _density_level(self, nearby_total: int) -> str:
        """Map total nearby case count to a human-readable density label."""
        if nearby_total >= 500:
            return "very_high"
        elif nearby_total >= 100:
            return "high"
        elif nearby_total >= 20:
            return "medium"
        else:
            return "low"

    def _is_in_training_region(self, latitude: float, longitude: float) -> bool:
        """
        Check if the given location falls within the training data region (Tamil Nadu).
        Predictions outside this region are unreliable — no historical cases exist there.
        """
        r = self.TRAINING_REGION
        return (
            r["lat_min"] <= latitude  <= r["lat_max"] and
            r["lon_min"] <= longitude <= r["lon_max"]
        )

    # ──────────────────────────────────────────────────────────────
    # Main prediction — XGBoost × KDE combined
    # ──────────────────────────────────────────────────────────────

    def predict(
        self,
        latitude: float,
        longitude: float,
        age: Optional[float] = None,
        gender: Optional[str] = None,
    ) -> Dict:
        """
        Predict disease risk using XGBoost + KDE combined approach.

        Step 1: XGBoost gives probabilities based on features (age, gender, location, season)
        Step 2: KDE gives density scores based on historical case locations
        Step 3: Multiply both → normalize → final risk %

        If no cases are found nearby (KDE scores all zero), falls back to XGBoost only.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call XGBRiskPredictor.load() first.")

        # ── Out-of-region check ─────────────────────────────────────
        # If location is outside Tamil Nadu, we have no training data for it.
        # Return a clear warning instead of a misleading prediction.
        if not self._is_in_training_region(latitude, longitude):
            return {
                "balanced_risk": {d: round(1.0 / len(DISEASES), 6) for d in DISEASES},
                "raw_case_distribution": {d: round(1.0 / len(DISEASES), 6) for d in DISEASES},
                "nearby_cases_25km": {d: 0 for d in DISEASES},
                "location_density_level": "unknown",
                "out_of_region": True,
                "out_of_region_message": (
                    f"Your location ({latitude:.4f}, {longitude:.4f}) is outside the "
                    f"training data region ({self.TRAINING_REGION['name']}). "
                    "No historical case data is available for this area. "
                    "Risk percentages shown are not reliable for your location."
                ),
                "metadata": {
                    "model_type": "XGBoost + KDE",
                    "model_accuracy": self._meta.get("accuracy", "N/A"),
                    "kde_used": False,
                    "in_training_region": False,
                },
                "disclaimer": (
                    "This model is trained on Tamil Nadu, India data only. "
                    "Your location is outside this region — predictions are not valid here. "
                    "Not a medical diagnosis."
                ),
            }

        # ── Step 1: XGBoost probabilities ──────────────────────────
        features = self._build_features(latitude, longitude, age, gender)
        proba = self._model.predict_proba(features)[0]

        # Extract disease probabilities (exclude 'Negative' class)
        xgb_scores = {}
        for i, cls_name in enumerate(self._classes):
            if cls_name.lower() in DISEASES:
                xgb_scores[cls_name.lower()] = float(proba[i])

        # ── Step 2: KDE density scores ─────────────────────────────
        kde_scores = self._kde_scores(latitude, longitude)
        kde_total = sum(kde_scores.values())

        # ── Step 3: Combine XGBoost × KDE ──────────────────────────
        if kde_total > 0:
            # Both experts agree → multiply their scores
            combined = {
                d: xgb_scores.get(d, 0.0) * kde_scores.get(d, 0.0)
                for d in DISEASES
            }
        else:
            # No cases nearby → use XGBoost alone (new area or no historical data)
            combined = xgb_scores

        # Normalize so all 3 diseases sum to 1.0
        total = sum(combined.values())
        if total > 0:
            balanced_risk = {d: round(combined[d] / total, 6) for d in DISEASES}
        else:
            balanced_risk = {d: round(1.0 / len(DISEASES), 6) for d in DISEASES}

        # ── Step 4: Nearby cases count (for display) ───────────────
        nearby_counts = self._count_nearby_cases(latitude, longitude, radius_km=25.0)
        nearby_total = sum(nearby_counts.values())

        return {
            "balanced_risk": balanced_risk,
            "raw_case_distribution": balanced_risk,
            "nearby_cases_25km": nearby_counts,
            "location_density_level": self._density_level(nearby_total),
            "metadata": {
                "model_type": "XGBoost + KDE",
                "model_accuracy": self._meta.get("accuracy", "N/A"),
                "f1_score": self._meta.get("f1_score", "N/A"),
                "best_iteration": self._meta.get("best_iteration", "N/A"),
                "trained_at": self._meta.get("trained_at", "N/A"),
                "kde_bandwidth_km": self.KDE_BANDWIDTH_KM,
                "kde_used": kde_total > 0,
            },
            "disclaimer": (
                "Risk is estimated using XGBoost (92.81% accuracy) combined with "
                "KDE on historical case data from Tamil Nadu, India (2021). "
                "This reflects relative disease activity in your area — "
                "not absolute probability of infection. Not a medical diagnosis."
            ),
        }

    # ──────────────────────────────────────────────────────────────
    # Health check compatibility
    # ──────────────────────────────────────────────────────────────

    @property
    def training_summary(self) -> Dict:
        """Used by api_server.py GET /health endpoint."""
        return {
            "total_cases": len(self._cases),
            "model_type": "XGBoost + KDE",
            "model_accuracy": self._meta.get("accuracy", "N/A"),
        }
