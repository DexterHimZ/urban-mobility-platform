"""
Mode-detection demo — run it yourself.

Usage:
    cd telecom_travel_demand_model

    # 1) Built-in synthetic trips (no data needed):
    ../venv/bin/python examples/mode_detection_demo.py

    # 2) Your own trip table (must have distance_m, duration_s, and ideally
    #    origin_lat/lon, dest_lat/lon columns — e.g. a pipeline trips.csv):
    ../venv/bin/python examples/mode_detection_demo.py path/to/trips.csv

What it shows:
    - the mode assigned to each trip + confidence
    - the full per-trip feature table the classifier used (ML-ready)
    - the mode distribution

Tweak thresholds live in config/config.yaml under `mode_detection:` and re-run
to see the effect — no code changes needed.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mode_detection import ModeDetector
from src.utils.config import Config


def _sample_trips() -> pd.DataFrame:
    """Trips spanning the speed spectrum so every mode branch is exercised."""
    return pd.DataFrame(
        [
            {"trip_id": "T_bike", "user_id": "u1", "origin_lat": 19.00, "origin_lon": 72.80,
             "dest_lat": 19.01, "dest_lon": 72.81, "distance_m": 1500, "duration_s": 600},    # ~9 km/h
            {"trip_id": "T_auto", "user_id": "u2", "origin_lat": 19.00, "origin_lon": 72.80,
             "dest_lat": 19.03, "dest_lon": 72.83, "distance_m": 4000, "duration_s": 650},     # ~22 km/h
            {"trip_id": "T_mix", "user_id": "u3", "origin_lat": 19.00, "origin_lon": 72.80,
             "dest_lat": 19.05, "dest_lon": 72.85, "distance_m": 6500, "duration_s": 900},     # ~26 km/h
            {"trip_id": "T_urban", "user_id": "u4", "origin_lat": 19.00, "origin_lon": 72.80,
             "dest_lat": 19.10, "dest_lon": 72.88, "distance_m": 14000, "duration_s": 1400},   # ~36 km/h
            {"trip_id": "T_hwy", "user_id": "u5", "origin_lat": 19.00, "origin_lon": 72.80,
             "dest_lat": 19.30, "dest_lon": 72.95, "distance_m": 35000, "duration_s": 1800},   # ~70 km/h
        ]
    )


def main() -> None:
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Loading trips from {path}")
        trips = pd.read_csv(path)
    else:
        print("No file given — using built-in sample trips.\n")
        trips = _sample_trips()

    cfg = Config()
    cfg.set("mode_detection.enabled", True)  # demo turns it on (config default is off)

    detector = ModeDetector(cfg)
    result = detector.detect(trips)

    cols = [c for c in ["trip_id", "distance_m", "duration_s", "mode_class", "mode",
                        "mode_confidence"] if c in result.columns]
    print("\n=== Detected modes ===")
    print(result[cols].to_string(index=False))

    features = detector.get_feature_table()
    if features is not None:
        print("\n=== Feature table (what the classifier saw) ===")
        show = [c for c in ["avg_speed_kmh", "max_speed_kmh", "accel_std",
                            "stop_count", "gtfs_align_score", "transit_route_type",
                            "road_class"] if c in features.columns]
        print(features[show].round(2).to_string(index=False))

    print("\n=== Mode distribution ===")
    print(result["mode"].value_counts().to_string())


if __name__ == "__main__":
    main()
