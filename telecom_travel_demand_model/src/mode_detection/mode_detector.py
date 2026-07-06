"""
Transport Mode Detection.

Infers the travel mode of each trip, mirroring the StayPointDetector design:
a config-driven, column-agnostic orchestrator that composes three pluggable
pieces -- a feature extractor, an optional transit feed, and a classifier
backend -- and returns the trip table enriched with mode columns.

    detector = ModeDetector(config)
    trips = detector.detect(trips_df, observations_df)   # adds mode columns

Output columns added to trips:
    mode_class      : "public" | "private" | "unknown"
    mode            : metro/local_train/bus | car/two_wheeler/auto/hcv/lcv/bicycle
    mode_confidence : 0-1

Reference approach: trajectory-feature mode inference (speed/stop/acceleration
signatures) combined with GTFS alignment, in the spirit of Zheng et al. (2008)
"Understanding mobility based on GPS data" and standard GTFS map-matching.
"""

from typing import Optional

import pandas as pd

from src.mode_detection.classifiers import (
    BaseModeClassifier,
    MLModeClassifier,
    RuleModeClassifier,
)
from src.mode_detection.features import TripFeatureExtractor
from src.mode_detection.transit_feed import TransitFeed
from src.utils.config import Config, get_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

MODE_COLUMNS = ["mode_class", "mode", "mode_confidence"]


class ModeDetector:
    """
    Detect the travel mode of each trip.

    Args:
        config: Configuration object. Reads the ``mode_detection`` block:

            mode_detection:
              enabled: true
              classifier: "rule"            # "rule" | "ml"
              stop_speed_kmh: 3.0
              transit_feed:
                type: "none"                # "none" | "synthetic" | "gtfs"
                match_radius_m: 150
              rules: { ... }                # overrides classifiers.DEFAULT_RULES
              emit_feature_table: true      # keep the ML-ready feature frame
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        classifier: Optional[BaseModeClassifier] = None,
        transit_feed: Optional[TransitFeed] = None,
    ):
        self.config = config or get_config()
        cfg = self.config.mode_detection

        self.enabled = cfg.get("enabled", False)
        self.emit_feature_table = cfg.get("emit_feature_table", True)

        self.feature_extractor = TripFeatureExtractor(
            stop_speed_kmh=cfg.get("stop_speed_kmh", 3.0),
        )

        # Transit feed: explicit arg wins, else build from config.
        self.transit_feed = transit_feed or TransitFeed.from_config(
            cfg.get("transit_feed", {})
        )
        if self.transit_feed is not None:
            self.transit_feed.load()

        # Classifier backend: explicit arg wins, else pick from config.
        self.classifier = classifier or self._build_classifier(cfg)

        # Last feature table (for inspection / ML training export).
        self.features_: Optional[pd.DataFrame] = None

    def _build_classifier(self, cfg: dict) -> BaseModeClassifier:
        kind = cfg.get("classifier", "rule")
        if kind == "ml":
            return MLModeClassifier(model_path=cfg.get("model_path"))
        return RuleModeClassifier(rules=cfg.get("rules"))

    def detect(
        self,
        trips: pd.DataFrame,
        observations: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Enrich a trip table with mode columns.

        Args:
            trips: trip table from TripGenerator.
            observations: optional preprocessed point observations for richer
                speed/stop features (telecom-now works without them).

        Returns:
            The trips DataFrame with MODE_COLUMNS added. When disabled, returns
            trips unchanged (columns set to 'unknown'/NaN) so downstream code is
            never surprised by missing columns.
        """
        if trips is None or len(trips) == 0:
            logger.info("No trips to classify; skipping mode detection.")
            return self._with_empty_modes(trips)

        if not self.enabled:
            logger.info("Mode detection disabled (mode_detection.enabled=false).")
            return self._with_empty_modes(trips)

        logger.info(f"Detecting travel mode for {len(trips)} trips")

        features = self.feature_extractor.extract(
            trips, observations=observations, transit_feed=self.transit_feed
        )
        if self.emit_feature_table:
            self.features_ = features

        labels = self.classifier.classify(features)

        result = trips.copy()
        for col in MODE_COLUMNS:
            result[col] = labels[col].values

        self._log_distribution(result)
        return result

    @staticmethod
    def _with_empty_modes(trips: Optional[pd.DataFrame]) -> pd.DataFrame:
        if trips is None:
            return pd.DataFrame(columns=MODE_COLUMNS)
        result = trips.copy()
        result["mode_class"] = "unknown"
        result["mode"] = "unknown"
        result["mode_confidence"] = 0.0
        return result

    def _log_distribution(self, result: pd.DataFrame) -> None:
        dist = result["mode"].value_counts(dropna=False)
        logger.info("Mode distribution: " + ", ".join(f"{m}={n}" for m, n in dist.items()))
        mean_conf = result["mode_confidence"].mean()
        logger.info(f"Mean mode confidence: {mean_conf:.3f}")

    def get_feature_table(self) -> Optional[pd.DataFrame]:
        """Return the last computed feature table (ML-ready), if retained."""
        return self.features_
