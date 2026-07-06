# Mode Detection

Infers the **travel mode** of each trip — public vs private, and the sub-mode
within each — from trajectory-derived features plus optional GTFS transit
alignment.

It is built to mirror the **stay-point detection** design: a config-driven,
column-agnostic detector that "works for any data" (coarse telecom trips today,
GPS traces later) and plugs into the pipeline as one composable step.

```
mode_class : public | private | unknown
mode       : metro | local_train | bus                          (public)
             car | two_wheeler | auto | hcv | lcv | bicycle      (private)
             unknown
mode_confidence : 0.0 – 1.0
```

---

## Quick start

> The module is **gated off by default** (`mode_detection.enabled: false`), so
> the existing pipeline is unaffected until you turn it on.

### Run the tests

**Windows (PowerShell / cmd), from the project folder:**
```powershell
cd E:\urban_transit_tool\telecom_travel_demand_model
..\venv\Scripts\python.exe -m pytest tests\test_mode_detection.py -v
```

**Linux / WSL / macOS:**
```bash
cd telecom_travel_demand_model
../venv/bin/python -m pytest tests/test_mode_detection.py -v
```
Expected: **18 passed**.

### Run the demo

```powershell
# built-in sample trips (no data needed)
..\venv\Scripts\python.exe examples\mode_detection_demo.py

# your own trip table (needs at least distance_m, duration_s columns)
..\venv\Scripts\python.exe examples\mode_detection_demo.py ..\output_bengaluru_india\trips.csv
```

The demo prints each trip's assigned mode + confidence, the per-trip feature
table the classifier used, and the overall mode distribution.

### Use it in code

```python
from src.mode_detection import ModeDetector
from src.utils.config import Config

cfg = Config()
cfg.set("mode_detection.enabled", True)

detector = ModeDetector(cfg)
trips = detector.detect(trips_df, observations_df)   # adds mode columns
features = detector.get_feature_table()              # ML-ready feature frame
```

---

## How it works

Three composable, swappable pieces (mirroring how stay detection is built):

| Piece | File | Role |
|-------|------|------|
| `TripFeatureExtractor` | `features.py` | Per-trip features. Trip-level (speed/distance) always; speed-profile/accel/stop features when observations are supplied; GTFS alignment when a feed is supplied. |
| `TransitFeed` | `transit_feed.py` | Pluggable transit reference — `SyntheticTransitFeed` (OSM lines) now, `GTFSFeed` (real feed) later, selected by config. |
| classifier | `classifiers.py` | `RuleModeClassifier` (transparent, no labels) today; `MLModeClassifier` (same interface) once labeled trips exist. |
| `ModeDetector` | `mode_detector.py` | Orchestrator: composes the above, returns the enriched trip table. |

### Classification logic (rule backend)

1. **Public vs private** — a trip that aligns well with a transit line/route
   (GTFS/OSM) is *public*; otherwise *private*.
2. **Public sub-mode** — inherits the matched route's type (metro / local_train
   / bus); falls back to a speed signature when no feed match is available.
3. **Private sub-mode** — the candidate whose *typical cruising speed* is nearest
   the trip's average speed. Where **car and two-wheeler** overlap (the case
   flagged by the supervisor — they share urban speeds), a dedicated resolver
   breaks the tie using **road class**, then **acceleration variance**.
   Confidence decreases as more modes are plausible at a given speed.

---

## Configuration

All behaviour is driven from `config/config.yaml` under `mode_detection:` — tune
thresholds without touching code, then re-run the demo/tests.

```yaml
mode_detection:
  enabled: false            # turn the pipeline step on/off
  classifier: "rule"        # "rule" | "ml"
  stop_speed_kmh: 3.0
  transit_feed:
    type: "none"            # "none" | "synthetic" | "gtfs"
    match_radius_m: 150
    gtfs_dir: null          # point at a real GTFS feed to enable metro/bus/rail
  rules:
    private_speed_bands_kmh: { ... }    # plausible speed range per mode
    private_typical_speed_kmh: { ... }  # representative speed per mode
    car_vs_tw: { ... }                  # the car/two-wheeler tie-breaker
```

Full default thresholds live in `classifiers.py::DEFAULT_RULES`; the config
block overrides them.

---

## Design guarantees

- **Works for any data** — column names are configurable; GPS traces drop into
  the same `detect(trips, observations)` call with no rewrite.
- **Adaptable transit** — flip `transit_feed.type` `none → synthetic → gtfs`
  with one config key; no downstream changes.
- **Hybrid, ML-ready** — the rule engine emits the exact feature table a
  supervised model will consume, so `classifier: ml` activates later with no
  other changes.

---

## Status & roadmap

**Done:** module + interfaces, rule classifier, config, pipeline wiring
(`detect_modes` step after trip generation), 18 unit tests, demo.

**Next (fill-in):**
- Per-trip observation slicing → real max-speed / acceleration / stop features
  (currently trip-level only on coarse telecom data).
- Transit spatial-join in `SyntheticTransitFeed` / `GTFSFeed` (interfaces final,
  matching stubbed — marked `TODO(phase1)`).
- Threshold calibration on real trips; OSM road-class for HCV/LCV separation.
- Optional: `od_by_mode` matrices in the OD step.

> **Caveat:** on coarse telecom-only data, mean speed alone cannot fully
> separate car / two-wheeler / auto / HCV / LCV. That is expected — the
> observation-feature and road-class seams above are exactly what sharpen these
> once finer data (GPS / OSM / GTFS) is connected.
