"""
Per-trip feature engineering for mode detection.

Produces the feature vector the classifiers consume. Features degrade
gracefully with data quality:

    - From the trip row alone (always available):
        avg_speed_kmh, distance_km, duration_min
    - From intermediate observations (when a path is supplied):
        max_speed_kmh, p85_speed_kmh, speed_std_kmh,
        accel_std, stop_count, stop_time_ratio, straightness, n_obs
    - From a transit feed (when supplied):
        gtfs_align_score, transit_route_type
    - From OSM (hook, optional):
        road_class

This is the seam that makes the detector "work for any data": telecom trips
supply only the coarse features today; GPS traces later populate the richer
observation-based features through the same interface, with no downstream
changes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.mode_detection.transit_feed import TransitFeed
from src.utils.geo_utils import haversine_distance
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Column names that make the extractor source-agnostic (mirrors StayPointDetector).
DEFAULT_COLS = {
    "distance_m": "distance_m",
    "duration_s": "duration_s",
    "origin_lat": "origin_lat",
    "origin_lon": "origin_lon",
    "dest_lat": "dest_lat",
    "dest_lon": "dest_lon",
    "trip_id": "trip_id",
    "user_id": "user_id",
}

# Speed below which an observation is considered "stopped" (km/h).
STOP_SPEED_KMH = 3.0


class TripFeatureExtractor:
    """
    Build mode-detection features for each trip.

    Args:
        stop_speed_kmh: speed threshold for stop detection from observations.
        cols: optional column-name overrides for the trip table.
    """

    def __init__(
        self,
        stop_speed_kmh: float = STOP_SPEED_KMH,
        cols: Optional[Dict[str, str]] = None,
    ):
        self.stop_speed_kmh = stop_speed_kmh
        self.cols = {**DEFAULT_COLS, **(cols or {})}

    @staticmethod
    def _numeric_col(df: pd.DataFrame, name: str) -> pd.Series:
        """Return df[name] coerced to numeric, or an all-NaN Series if absent."""
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype="float64")

    def extract(
        self,
        trips: pd.DataFrame,
        observations: Optional[pd.DataFrame] = None,
        transit_feed: Optional[TransitFeed] = None,
    ) -> pd.DataFrame:
        """
        Compute features for all trips.

        Args:
            trips: trip table (one row per trip).
            observations: optional per-user point observations
                (imsi/user_id, timestamp, latitude, longitude) used to derive
                speed-profile and stop features.
            transit_feed: optional loaded TransitFeed for alignment scoring.

        Returns:
            DataFrame indexed like ``trips`` with the feature columns added.
        """
        c = self.cols
        feats = pd.DataFrame(index=trips.index)

        # --- Trip-level (always available) ---
        # Guard missing columns explicitly: trips.get(name) returns None when the
        # column is absent (the "works for any data" case with different schemas),
        # and pd.to_numeric(None) yields a scalar, not a Series -> downstream
        # .where()/division would break. Fall back to an all-NaN Series instead.
        dist_m = self._numeric_col(trips, c["distance_m"])
        dur_s = self._numeric_col(trips, c["duration_s"])

        feats["distance_km"] = dist_m / 1000.0
        feats["duration_min"] = dur_s / 60.0
        # Guard divide-by-zero; sub-minute trips get NaN avg speed.
        safe_dur = dur_s.where(dur_s > 0)
        feats["avg_speed_kmh"] = (dist_m / safe_dur) * 3.6

        # --- Observation-derived (optional, richer) ---
        obs_features = self._extract_observation_features(trips, observations)
        for col, series in obs_features.items():
            feats[col] = series

        # --- Transit alignment (optional) ---
        if transit_feed is not None and transit_feed.is_available:
            align, rtype = self._extract_transit_features(
                trips, observations, transit_feed
            )
            feats["gtfs_align_score"] = align
            feats["transit_route_type"] = rtype
        else:
            feats["gtfs_align_score"] = 0.0
            feats["transit_route_type"] = None

        # --- OSM road class (hook, not yet wired) ---
        feats["road_class"] = None  # TODO(phase1): fill from network_analysis/OSM

        return feats

    def _extract_observation_features(
        self,
        trips: pd.DataFrame,
        observations: Optional[pd.DataFrame],
    ) -> Dict[str, pd.Series]:
        """
        Derive speed-profile and stop features from intermediate observations.

        Skeleton: returns NaN/zero-filled columns of the right shape when no
        observations are supplied (the telecom-now case). When observations are
        available, per-trip point slicing + the speed profile below is filled in.
        """
        idx = trips.index
        empty = {
            "max_speed_kmh": pd.Series(np.nan, index=idx),
            "p85_speed_kmh": pd.Series(np.nan, index=idx),
            "speed_std_kmh": pd.Series(np.nan, index=idx),
            "accel_std": pd.Series(np.nan, index=idx),
            "stop_count": pd.Series(0, index=idx),
            "stop_time_ratio": pd.Series(np.nan, index=idx),
            "straightness": pd.Series(np.nan, index=idx),
            "n_obs": pd.Series(0, index=idx),
        }

        if observations is None or len(observations) == 0:
            return empty

        # TODO(phase1): slice observations per trip (by user + [departure,arrival]
        # window), call _speed_profile per trip, and populate the columns above.
        # Interface is final; the per-trip slicing loop is the remaining logic.
        logger.debug(
            "Observation features requested but per-trip slicing not yet "
            "implemented; returning trip-level features only."
        )
        return empty

    def _speed_profile(self, points: pd.DataFrame) -> Dict[str, float]:
        """
        Compute a speed/acceleration/stop profile from an ordered point path.

        points: DataFrame sorted by timestamp with 'latitude','longitude',
        'timestamp'. Returns the observation-derived feature scalars for one trip.
        """
        if points is None or len(points) < 2:
            return {}

        lat = points["latitude"].to_numpy(dtype="float64")
        lon = points["longitude"].to_numpy(dtype="float64")
        ts = pd.to_datetime(points["timestamp"]).astype("int64").to_numpy() / 1e9

        seg_dist = np.array(
            [
                haversine_distance(lat[i], lon[i], lat[i + 1], lon[i + 1])
                for i in range(len(lat) - 1)
            ]
        )
        seg_time = np.diff(ts)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Full-length segment speed array (NaN where dt <= 0). Keeping it
            # full-length (not filtered) is what lets acceleration align below.
            seg_speed_kmh = np.where(seg_time > 0, seg_dist / seg_time * 3.6, np.nan)

        valid = seg_speed_kmh[~np.isnan(seg_speed_kmh)]
        if valid.size == 0:
            return {}

        # Acceleration between consecutive segments (m/s^2). Compute on the
        # full-length speed array so np.diff(speed) and seg_time[1:] share a
        # length, THEN drop NaNs -- the previous version diffed the NaN-filtered
        # `valid`, whose length no longer matched seg_time[1:], raising whenever
        # any segment speed was NaN.
        speed_ms = seg_speed_kmh / 3.6
        with np.errstate(divide="ignore", invalid="ignore"):
            accel = np.diff(speed_ms) / np.where(seg_time[1:] > 0, seg_time[1:], np.nan)
        accel = accel[~np.isnan(accel)]

        path_len = float(np.nansum(seg_dist))
        crow = haversine_distance(lat[0], lon[0], lat[-1], lon[-1])

        is_stopped = seg_speed_kmh < self.stop_speed_kmh  # NaN compares False
        stopped_time = seg_time[is_stopped]
        total_time = float(np.nansum(seg_time))

        return {
            "max_speed_kmh": float(np.nanmax(valid)),
            "p85_speed_kmh": float(np.nanpercentile(valid, 85)),
            "speed_std_kmh": float(np.nanstd(valid)),
            "accel_std": float(np.nanstd(accel)) if accel.size else np.nan,
            "stop_count": int(np.sum(is_stopped)),
            "stop_time_ratio": (
                float(np.nansum(stopped_time) / total_time) if total_time > 0 else np.nan
            ),
            "straightness": (crow / path_len) if path_len > 0 else np.nan,
            "n_obs": int(len(points)),
        }

    def _extract_transit_features(
        self,
        trips: pd.DataFrame,
        observations: Optional[pd.DataFrame],
        transit_feed: TransitFeed,
    ) -> tuple:
        """Score each trip against the transit feed. Returns (align_series, route_type_series)."""
        c = self.cols
        scores: List[float] = []
        rtypes: List[Optional[str]] = []
        for _, row in trips.iterrows():
            match = transit_feed.match_trip(
                row.get(c["origin_lat"]),
                row.get(c["origin_lon"]),
                row.get(c["dest_lat"]),
                row.get(c["dest_lon"]),
                path=None,  # TODO(phase1): pass per-trip observation slice
            )
            scores.append(match.align_score)
            rtypes.append(match.route_type)
        return (
            pd.Series(scores, index=trips.index),
            pd.Series(rtypes, index=trips.index),
        )
