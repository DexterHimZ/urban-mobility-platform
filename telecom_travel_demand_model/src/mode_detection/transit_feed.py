"""
Transit reference feeds for mode detection.

Provides a pluggable ``TransitFeed`` interface so the public-transport branch of
mode detection can be driven by whatever transit reference is available:

    - SyntheticTransitFeed : OSM-derived / hand-authored rail & bus lines.
                             Used now, before a real GTFS feed exists.
    - GTFSFeed             : a real GTFS feed (stops.txt, routes.txt,
                             stop_times.txt, shapes.txt). Drops in via config
                             with no downstream changes.

A feed answers one question for the classifier: given a trip's geometry, how
well does it align with a transit line, and what route_type does the best match
have (metro / rail / bus)? That alignment score + route_type is what separates
public from private and, within public, metro vs local_train vs bus.

The point-to-line matching logic is intentionally thin here and delegates to the
existing GTFS/OSM matcher in
``src/data_fusion/fusion_algorithms/gps_gtfs_osm_fusion.py`` where practical, to
avoid duplicating spatial-join code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# GTFS route_type -> coarse transit class used by the classifier.
# https://gtfs.org/schedule/reference/#routestxt
GTFS_ROUTE_TYPE_MAP: Dict[int, str] = {
    0: "tram",           # Tram / light rail
    1: "metro",          # Subway / metro
    2: "local_train",    # Rail
    3: "bus",            # Bus
    4: "ferry",
    5: "cable_tram",
    6: "aerial",
    7: "funicular",
    11: "bus",           # Trolleybus
    12: "local_train",   # Monorail -> treat as rail-like
}


@dataclass
class TransitMatch:
    """Result of matching one trip against a transit feed."""

    align_score: float          # 0-1, higher = better geometric+temporal fit
    route_type: Optional[str]   # "metro" | "local_train" | "bus" | ... | None
    route_id: Optional[str] = None
    matched_stops: int = 0      # number of feed stops the trip passes near


class TransitFeed(ABC):
    """
    Abstract transit reference.

    Concrete feeds implement :meth:`match_trip`, returning a :class:`TransitMatch`
    describing how well a single trip aligns with any transit line in the feed.
    """

    @abstractmethod
    def load(self) -> "TransitFeed":
        """Load the underlying data. Returns self for chaining."""

    @abstractmethod
    def match_trip(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        path: Optional[pd.DataFrame] = None,
    ) -> TransitMatch:
        """
        Match a trip to the best-fitting transit line.

        Args:
            origin_lat/lon, dest_lat/lon: trip endpoints.
            path: optional DataFrame of intermediate observations (lat/lon/
                timestamp) for finer geometric matching. When absent, matching
                falls back to endpoint proximity only.

        Returns:
            TransitMatch. ``align_score == 0`` means no usable match.
        """

    @property
    def is_available(self) -> bool:
        """Whether this feed has usable data loaded."""
        return True

    @staticmethod
    def from_config(cfg: Dict) -> Optional["TransitFeed"]:
        """
        Build a transit feed from the ``mode_detection.transit_feed`` config block.

        Config shape::

            transit_feed:
              type: "synthetic" | "gtfs" | "none"
              gtfs_dir: "data/raw/gtfs"          # for type: gtfs
              synthetic_lines: "config/transit_lines.geojson"  # for type: synthetic
              match_radius_m: 150

        Returns None when type is "none" (public branch then relies on
        speed/stop heuristics alone).
        """
        feed_type = (cfg or {}).get("type", "none")
        radius = float((cfg or {}).get("match_radius_m", 150.0))

        if feed_type == "gtfs":
            return GTFSFeed(cfg.get("gtfs_dir"), match_radius_m=radius)
        if feed_type == "synthetic":
            return SyntheticTransitFeed(
                cfg.get("synthetic_lines"), match_radius_m=radius
            )
        return None


class SyntheticTransitFeed(TransitFeed):
    """
    Transit reference built from OSM-derived or hand-authored lines.

    Expected source: a GeoJSON/CSV of transit line geometries, each tagged with
    a ``route_type`` (metro / local_train / bus). This lets the public-transport
    branch work today, before a real GTFS feed is provided, and is swapped out
    for :class:`GTFSFeed` by changing one config key.

    NOTE (skeleton): geometry matching is stubbed. The interface, config wiring,
    and route_type mapping are final; the spatial-join implementation is the
    logic to be filled in Phase 1.
    """

    def __init__(
        self,
        source: Optional[Union[str, Path]] = None,
        match_radius_m: float = 150.0,
    ):
        self.source = Path(source) if source else None
        self.match_radius_m = match_radius_m
        self._lines: Optional[pd.DataFrame] = None

    def load(self) -> "SyntheticTransitFeed":
        if self.source and self.source.exists():
            logger.info(f"Loading synthetic transit lines from {self.source}")
            # TODO(phase1): parse GeoJSON/CSV lines into self._lines with columns
            # [route_id, route_type, geometry]. Left unloaded in skeleton.
        else:
            logger.warning(
                "SyntheticTransitFeed has no source file; public-transport "
                "matching will return no-match until transit lines are provided."
            )
        return self

    @property
    def is_available(self) -> bool:
        return self._lines is not None and len(self._lines) > 0

    def match_trip(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        path: Optional[pd.DataFrame] = None,
    ) -> TransitMatch:
        if not self.is_available:
            return TransitMatch(align_score=0.0, route_type=None)
        # TODO(phase1): buffer each line by match_radius_m, compute the fraction
        # of the trip path that falls within a line's buffer, pick the best line,
        # return its route_type and coverage fraction as align_score.
        return TransitMatch(align_score=0.0, route_type=None)


class GTFSFeed(TransitFeed):
    """
    Real GTFS feed reference (stops.txt, routes.txt, stop_times.txt, shapes.txt).

    Reuses the stop-matching approach from
    ``src/data_fusion/fusion_algorithms/gps_gtfs_osm_fusion.py``: a trip that
    passes near an ordered sequence of a route's stops, in order, is a strong
    public-transport match and inherits that route's route_type.

    NOTE (skeleton): file loading + matching are stubbed; the GTFS_ROUTE_TYPE_MAP
    and interface are final.
    """

    def __init__(
        self,
        gtfs_dir: Optional[Union[str, Path]] = None,
        match_radius_m: float = 150.0,
    ):
        self.gtfs_dir = Path(gtfs_dir) if gtfs_dir else None
        self.match_radius_m = match_radius_m
        self._stops: Optional[pd.DataFrame] = None
        self._routes: Optional[pd.DataFrame] = None
        self._stop_times: Optional[pd.DataFrame] = None

    def load(self) -> "GTFSFeed":
        if self.gtfs_dir and self.gtfs_dir.exists():
            logger.info(f"Loading GTFS feed from {self.gtfs_dir}")
            # TODO(phase1): read stops.txt, routes.txt, stop_times.txt; join
            # route_type onto stops via trips.txt; build a stop spatial index.
        else:
            logger.warning(
                "GTFSFeed directory missing; falling back to no-match. "
                "Set mode_detection.transit_feed.gtfs_dir to a real feed."
            )
        return self

    @property
    def is_available(self) -> bool:
        return self._stops is not None and len(self._stops) > 0

    def match_trip(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        path: Optional[pd.DataFrame] = None,
    ) -> TransitMatch:
        if not self.is_available:
            return TransitMatch(align_score=0.0, route_type=None)
        # TODO(phase1): find stops within match_radius_m of the path, group by
        # route, score routes by ordered-stop coverage, map route_type through
        # GTFS_ROUTE_TYPE_MAP, return the best.
        return TransitMatch(align_score=0.0, route_type=None)
