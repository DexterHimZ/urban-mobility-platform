"""
Mode classification backends (pluggable).

Hybrid strategy, per the agreed design:

    - RuleModeClassifier : transparent, label-free, config-driven thresholds.
                           Produces mode + confidence today. Mirrors how
                           stay detection works (no training data needed).
    - MLModeClassifier   : same interface, consumes the exact feature table the
                           rule engine emits, so a supervised model drops in once
                           labeled trips exist — no downstream changes.

The classifier consumes the feature frame from TripFeatureExtractor and returns
three columns: mode_class, mode, mode_confidence.

Taxonomy
--------
mode_class : "public" | "private" | "unknown"
mode       : metro | local_train | bus            (public)
             car | two_wheeler | auto | hcv | lcv | bicycle   (private)
             unknown
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

PUBLIC_MODES = {"metro", "local_train", "bus", "tram", "ferry"}
PRIVATE_MODES = {"car", "two_wheeler", "auto", "hcv", "lcv", "bicycle"}


# First-cut default thresholds (km/h etc). Calibrated for telecom trips; every
# value is overridable from config so tuning needs no code change.
DEFAULT_RULES: Dict = {
    "public": {
        "min_align_score": 0.4,        # GTFS/transit coverage to call it public
        # Fallback speed/stop signatures when no feed match is available:
        "rail_min_avg_kmh": 40,        # local train: fast, long stop spacing
        "metro_min_avg_kmh": 25,       # metro: grade-separated, medium speed
        "bus_max_avg_kmh": 25,         # bus: slow, frequent stops on-road
    },
    # Plausible average-speed range (km/h) that gates each mode as a candidate.
    "private_speed_bands_kmh": {
        "bicycle": [0, 18],
        "auto": [8, 35],
        "two_wheeler": [15, 50],
        "car": [20, 100],
        "lcv": [25, 65],
        "hcv": [15, 55],
    },
    # Representative cruising speed (km/h) per mode. Selection picks the candidate
    # whose typical speed is nearest the trip's avg speed -- principled and stable,
    # unlike a band-width tie-break. Provisional; calibrate on real trips.
    "private_typical_speed_kmh": {
        "bicycle": 12,
        "auto": 22,
        "two_wheeler": 32,
        "car": 42,
        "lcv": 48,
        "hcv": 38,
    },
    # car vs two_wheeler overlap resolver (supervisor's explicit concern):
    "car_vs_tw": {
        "tw_accel_std_min": 0.7,       # TW: more stop-and-go, higher accel variance
        "tw_road_classes": ["residential", "service", "living_street"],
        "car_road_classes": ["primary", "secondary", "trunk", "motorway"],
    },
}


class BaseModeClassifier(ABC):
    """Interface for mode classifiers."""

    @abstractmethod
    def classify(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            features: output of TripFeatureExtractor.extract().

        Returns:
            DataFrame indexed like ``features`` with columns
            [mode_class, mode, mode_confidence].
        """


class RuleModeClassifier(BaseModeClassifier):
    """
    Transparent rule/threshold classifier.

    Two stages:
      A) public vs private  - transit alignment first, speed/stop signature as
         fallback when no feed is available.
      B) sub-mode           - route_type for public; speed bands + the car-vs-TW
         tie-breaker for private.

    Skeleton status: stage structure, config wiring, taxonomy, and confidence
    plumbing are final. Threshold values are first-cut and to be calibrated on
    `iitb new sample` during Phase 1; the car/TW resolver activates fully once
    accel_std and road_class features are populated from observations/OSM.
    """

    def __init__(self, rules: Optional[Dict] = None):
        self.rules = _deep_merge(DEFAULT_RULES, rules or {})

    def classify(self, features: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=features.index)
        results = [self._classify_one(row) for _, row in features.iterrows()]
        out["mode_class"] = [r[0] for r in results]
        out["mode"] = [r[1] for r in results]
        out["mode_confidence"] = [r[2] for r in results]
        return out

    # Stage A + B for a single trip -> (mode_class, mode, confidence)
    def _classify_one(self, f: pd.Series) -> tuple:
        align = float(f.get("gtfs_align_score", 0.0) or 0.0)
        route_type = f.get("transit_route_type")
        avg = f.get("avg_speed_kmh")

        if avg is None or (isinstance(avg, float) and np.isnan(avg)):
            return ("unknown", "unknown", 0.0)

        # --- Stage A: public vs private ---
        is_public = align >= self.rules["public"]["min_align_score"]

        if is_public:
            mode = self._public_submode(route_type, avg, f)
            # Confidence scales with alignment strength.
            conf = min(1.0, 0.5 + 0.5 * align)
            return ("public", mode, round(conf, 3))

        # --- Stage B: private sub-mode ---
        mode, conf = self._private_submode(avg, f)
        return ("private", mode, round(conf, 3))

    def _public_submode(
        self, route_type: Optional[str], avg: float, f: pd.Series
    ) -> str:
        # Prefer the feed's route_type when the match gave one.
        if route_type in ("metro", "local_train", "bus", "tram"):
            return "local_train" if route_type == "tram" else route_type

        # Fallback: infer from speed signature.
        r = self.rules["public"]
        if avg >= r["rail_min_avg_kmh"]:
            return "local_train"
        if avg >= r["metro_min_avg_kmh"]:
            return "metro"
        return "bus"

    def _private_submode(self, avg: float, f: pd.Series) -> tuple:
        bands = self.rules["private_speed_bands_kmh"]
        typical = self.rules["private_typical_speed_kmh"]
        candidates = [m for m, (lo, hi) in bands.items() if lo <= avg <= hi]

        if not candidates:
            # Outside every band: above all -> highway-capable; below all -> cycle.
            return ("car", 0.3) if avg > 30 else ("bicycle", 0.3)

        # Rank candidates by closeness of their typical speed to this trip.
        ranked = sorted(candidates, key=lambda m: abs(avg - typical[m]))
        base = ranked[0]

        # The car/two_wheeler overlap the supervisor flagged: when the best
        # speed-match is car or TW *and* both are plausible at this speed, defer
        # to the dedicated resolver (road class -> accel variance). At speeds
        # where auto/bicycle/lcv/hcv is the nearest match, base stays put, so
        # those modes are never masked.
        pair = {"car", "two_wheeler"}
        if base in pair and pair.issubset(candidates):
            base = self._resolve_car_vs_tw(f)

        # Confidence decays with the number of overlapping candidates: more modes
        # plausible at this speed -> more ambiguous -> lower confidence. Mean speed
        # alone cannot cleanly separate these; richer features raise this later.
        conf = max(0.30, 0.75 - 0.10 * (len(candidates) - 1))
        return (base, round(conf, 2))

    def _resolve_car_vs_tw(self, f: pd.Series) -> str:
        """
        Break the car vs two-wheeler tie using features beyond mean speed.

        Uses (in priority order): road class, then acceleration variance
        (two-wheelers show more stop-and-go / higher accel_std in urban traffic).
        Falls back to 'car' when neither feature is available (conservative,
        since car is the higher-VMT default).
        """
        cfg = self.rules["car_vs_tw"]
        road = f.get("road_class")
        if road in cfg["tw_road_classes"]:
            return "two_wheeler"
        if road in cfg["car_road_classes"]:
            return "car"

        accel_std = f.get("accel_std")
        if accel_std is not None and not (
            isinstance(accel_std, float) and np.isnan(accel_std)
        ):
            return "two_wheeler" if accel_std >= cfg["tw_accel_std_min"] else "car"

        return "car"


class MLModeClassifier(BaseModeClassifier):
    """
    Supervised classifier (interface ready, model not yet trained).

    Consumes the identical feature frame the rule engine uses, so switching
    ``mode_detection.classifier: ml`` in config activates it with no other
    changes once labeled trips exist.
    """

    FEATURE_COLUMNS = [
        "avg_speed_kmh",
        "max_speed_kmh",
        "p85_speed_kmh",
        "speed_std_kmh",
        "accel_std",
        "stop_count",
        "stop_time_ratio",
        "straightness",
        "distance_km",
        "gtfs_align_score",
    ]

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model = None  # TODO(phase2): load a fitted sklearn pipeline

    def classify(self, features: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise NotImplementedError(
                "MLModeClassifier has no trained model yet. Use classifier='rule' "
                "until labeled trips are available. Feature table is ML-ready."
            )
        # TODO(phase2): X = features[self.FEATURE_COLUMNS]; predict + proba.


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into a copy of base."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
