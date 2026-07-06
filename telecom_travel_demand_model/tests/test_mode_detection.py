"""
Unit tests for the mode_detection module.

Run with:
    cd telecom_travel_demand_model
    ../venv/bin/python -m pytest tests/test_mode_detection.py -v
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the telecom project root importable so `from src...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mode_detection import ModeDetector, RuleModeClassifier, TripFeatureExtractor
from src.mode_detection.transit_feed import GTFS_ROUTE_TYPE_MAP, TransitFeed
from src.utils.config import Config


def _trips():
    """A small trip table shaped like TripGenerator output."""
    return pd.DataFrame(
        [
            # ~26 km/h : car/tw/auto overlap zone
            {"trip_id": "u1_T1", "user_id": "u1", "origin_lat": 19.00, "origin_lon": 72.80,
             "dest_lat": 19.05, "dest_lon": 72.85, "distance_m": 6500, "duration_s": 900},
            # ~9 km/h : bicycle
            {"trip_id": "u2_T1", "user_id": "u2", "origin_lat": 19.00, "origin_lon": 72.80,
             "dest_lat": 19.01, "dest_lon": 72.81, "distance_m": 1500, "duration_s": 600},
            # ~70 km/h : must be car, NOT lcv (regression on the width tie-break)
            {"trip_id": "u3_T1", "user_id": "u3", "origin_lat": 19.00, "origin_lon": 72.80,
             "dest_lat": 19.30, "dest_lon": 72.95, "distance_m": 35000, "duration_s": 1800},
        ]
    )


def _enabled_config():
    cfg = Config()
    cfg.set("mode_detection.enabled", True)
    return cfg


# --------------------------------------------------------------------------- #
# ModeDetector integration
# --------------------------------------------------------------------------- #
class TestModeDetector:
    def test_disabled_is_noop(self):
        """Default (disabled) leaves modes 'unknown' and never drops trips."""
        cfg = Config()  # enabled defaults to false
        out = ModeDetector(cfg).detect(_trips())
        assert len(out) == 3
        assert set(["mode_class", "mode", "mode_confidence"]).issubset(out.columns)
        assert (out["mode"] == "unknown").all()

    def test_enabled_assigns_modes(self):
        out = ModeDetector(_enabled_config()).detect(_trips())
        assert (out["mode"] != "unknown").all()
        assert (out["mode_class"].isin(["public", "private"])).all()
        assert out["mode_confidence"].between(0.0, 1.0).all()

    def test_empty_trips_guarded(self):
        out = ModeDetector(_enabled_config()).detect(pd.DataFrame())
        assert list(out.columns) == ["mode_class", "mode", "mode_confidence"]
        assert len(out) == 0

    def test_feature_table_emitted(self):
        d = ModeDetector(_enabled_config())
        d.detect(_trips())
        ft = d.get_feature_table()
        assert ft is not None and len(ft) == 3
        for col in ["avg_speed_kmh", "gtfs_align_score", "accel_std"]:
            assert col in ft.columns


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
class TestFeatureExtractor:
    def test_avg_speed_math(self):
        feats = TripFeatureExtractor().extract(_trips())
        # 6500 m / 900 s * 3.6 = 26.0 km/h
        assert feats.loc[0, "avg_speed_kmh"] == pytest.approx(26.0, abs=0.1)

    def test_missing_columns_do_not_crash(self):
        """The 'works for any data' path: absent distance/duration -> NaN, no crash."""
        trips = pd.DataFrame([{"trip_id": "x", "user_id": "u"}])
        feats = TripFeatureExtractor().extract(trips)
        assert np.isnan(feats.loc[0, "avg_speed_kmh"])

    def test_zero_duration_no_div_by_zero(self):
        trips = _trips().copy()
        trips.loc[0, "duration_s"] = 0
        feats = TripFeatureExtractor().extract(trips)
        assert np.isnan(feats.loc[0, "avg_speed_kmh"])

    def test_speed_profile_handles_nan_segment(self):
        """Regression: a zero-dt segment yields a NaN speed; accel must still align."""
        base = datetime(2026, 1, 1, 8, 0, 0)
        pts = pd.DataFrame(
            {
                "latitude": [19.00, 19.001, 19.001, 19.004],
                "longitude": [72.80, 72.801, 72.801, 72.804],
                # duplicate timestamp -> a zero-dt segment -> NaN speed in the middle
                "timestamp": [base, base + timedelta(seconds=30), base + timedelta(seconds=30),
                              base + timedelta(seconds=90)],
            }
        )
        prof = TripFeatureExtractor()._speed_profile(pts)  # must not raise
        assert prof["n_obs"] == 4
        assert prof["max_speed_kmh"] >= 0
        assert 0.0 <= prof["straightness"] <= 1.0


# --------------------------------------------------------------------------- #
# Rule classifier: sub-mode selection + car/TW resolver
# --------------------------------------------------------------------------- #
class TestRuleClassifier:
    def _feat(self, **kw):
        row = {"avg_speed_kmh": kw.get("avg"), "gtfs_align_score": kw.get("align", 0.0),
               "transit_route_type": kw.get("rtype"), "accel_std": kw.get("accel_std"),
               "road_class": kw.get("road_class")}
        return pd.DataFrame([row])

    def test_highway_speed_is_car_not_lcv(self):
        """70 km/h should resolve to car; the old width tie-break gave lcv."""
        out = RuleModeClassifier().classify(self._feat(avg=70))
        assert out.loc[0, "mode"] == "car"

    def test_low_speed_is_bicycle(self):
        out = RuleModeClassifier().classify(self._feat(avg=9))
        assert out.loc[0, "mode"] == "bicycle"

    def test_mid_low_speed_is_auto(self):
        """~22 km/h: auto is the nearest typical speed, not masked by car/TW."""
        out = RuleModeClassifier().classify(self._feat(avg=22))
        assert out.loc[0, "mode"] == "auto"

    def test_car_vs_tw_road_class(self):
        c = RuleModeClassifier()
        # residential road -> two_wheeler; arterial -> car (avg in overlap zone)
        tw = c.classify(self._feat(avg=35, road_class="residential"))
        car = c.classify(self._feat(avg=35, road_class="primary"))
        assert tw.loc[0, "mode"] == "two_wheeler"
        assert car.loc[0, "mode"] == "car"

    def test_car_vs_tw_accel(self):
        c = RuleModeClassifier()
        tw = c.classify(self._feat(avg=35, accel_std=1.2))   # jerky -> TW
        car = c.classify(self._feat(avg=35, accel_std=0.2))  # smooth -> car
        assert tw.loc[0, "mode"] == "two_wheeler"
        assert car.loc[0, "mode"] == "car"

    def test_public_from_gtfs_alignment(self):
        out = RuleModeClassifier().classify(
            self._feat(avg=45, align=0.8, rtype="metro")
        )
        assert out.loc[0, "mode_class"] == "public"
        assert out.loc[0, "mode"] == "metro"

    def test_nan_speed_is_unknown(self):
        out = RuleModeClassifier().classify(self._feat(avg=np.nan))
        assert out.loc[0, "mode"] == "unknown"

    def test_confidence_decays_with_ambiguity(self):
        c = RuleModeClassifier()
        # 30 km/h sits in many bands (ambiguous) -> lower conf than a clean 9 km/h
        amb = c.classify(self._feat(avg=30)).loc[0, "mode_confidence"]
        clean = c.classify(self._feat(avg=9)).loc[0, "mode_confidence"]
        assert amb <= clean


class TestTransitFeed:
    def test_route_type_map(self):
        assert GTFS_ROUTE_TYPE_MAP[1] == "metro"
        assert GTFS_ROUTE_TYPE_MAP[2] == "local_train"
        assert GTFS_ROUTE_TYPE_MAP[3] == "bus"

    def test_none_feed_from_config(self):
        assert TransitFeed.from_config({"type": "none"}) is None
