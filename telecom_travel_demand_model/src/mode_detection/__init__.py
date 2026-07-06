"""
Transport mode detection module.

Infers the travel mode of each trip (public vs private, and sub-mode:
metro / local_train / bus / car / two_wheeler / auto / hcv / lcv / bicycle)
using trajectory-derived features plus optional GTFS transit alignment.

Designed to mirror the stay_detection module: a config-driven, column-agnostic
detector that works for any trajectory source (telecom trips today, GPS traces
later) without downstream rewrites.

Public API:
    ModeDetector        - orchestrator, mirrors StayPointDetector
    TripFeatureExtractor - per-trip feature engineering
    TransitFeed / SyntheticTransitFeed / GTFSFeed - pluggable transit reference
    RuleModeClassifier / MLModeClassifier - pluggable classification backends
"""

from src.mode_detection.classifiers import (
    BaseModeClassifier,
    MLModeClassifier,
    RuleModeClassifier,
)
from src.mode_detection.features import TripFeatureExtractor
from src.mode_detection.mode_detector import ModeDetector
from src.mode_detection.transit_feed import (
    GTFSFeed,
    SyntheticTransitFeed,
    TransitFeed,
)

__all__ = [
    "ModeDetector",
    "TripFeatureExtractor",
    "TransitFeed",
    "SyntheticTransitFeed",
    "GTFSFeed",
    "BaseModeClassifier",
    "RuleModeClassifier",
    "MLModeClassifier",
]
