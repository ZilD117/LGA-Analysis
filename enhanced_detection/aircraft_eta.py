"""
Aircraft ETA — Compute time-to-threshold from ADS-B track data.

Loads the ADS-B CSV, interpolates aircraft state at any query time,
and computes distance/time to a runway point using great-circle math.
Sigma is derived from measured groundspeed variance in the track.
"""

import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

EARTH_RADIUS_M = 6_371_000.0
KTS_TO_MS = 0.514444

RW04_THRESHOLD = (40.76928, -73.884028)  # Rwy_01_001 — RW04 entry
RW04_TXY_D = (40.775511, -73.878895)     # Rwy_01_006 — Txy D crossing


@dataclass
class TrackPoint:
    timestamp: datetime
    lat: float
    lon: float
    alt_ft100: float
    groundspeed_kts: float
    heading: float
    source: str

    @property
    def epoch_s(self) -> float:
        return self.timestamp.timestamp()

    @property
    def alt_ft(self) -> float:
        return self.alt_ft100 * 100.0

    @property
    def groundspeed_ms(self) -> float:
        return self.groundspeed_kts * KTS_TO_MS


@dataclass
class AircraftState:
    timestamp: datetime
    lat: float
    lon: float
    alt_ft: float
    groundspeed_kts: float
    distance_to_point_m: float
    time_to_point_s: float
    sigma_s: float
    source: str

    def __repr__(self):
        return (
            f"AircraftState @ {self.timestamp.strftime('%H:%M:%S')}Z: "
            f"gs={self.groundspeed_kts:.0f}kts, "
            f"dist={self.distance_to_point_m:.0f}m, "
            f"ETA={self.time_to_point_s:.1f}s (σ={self.sigma_s:.1f}s)"
        )


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_track(csv_path: str) -> List[TrackPoint]:
    """Load ADS-B track from CSV."""
    points = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = datetime.strptime(row["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            points.append(TrackPoint(
                timestamp=ts,
                lat=float(row["latitude"]),
                lon=float(row["longitude"]),
                alt_ft100=float(row["altitude_ft100"]),
                groundspeed_kts=float(row["groundspeed_kts"]),
                heading=float(row["heading"]),
                source=row.get("source", ""),
            ))
    return points


def _extrapolate_forward(track: List[TrackPoint], query_epoch: float) -> TrackPoint:
    """
    Forward extrapolation from the most recent ADS-B report BEFORE query_time.

    A real-time system only has past data. We propagate position along the
    current heading at the current speed — the same dead-reckoning ATC radar
    systems use between sweeps.
    """
    if query_epoch <= track[0].epoch_s:
        return track[0]

    # Find the last point at or before query_epoch
    last = track[0]
    for pt in track:
        if pt.epoch_s <= query_epoch:
            last = pt
        else:
            break

    dt = query_epoch - last.epoch_s
    if dt < 0.1:
        return last

    gs_ms = last.groundspeed_kts * KTS_TO_MS
    hdg_rad = math.radians(last.heading)
    dlat = (gs_ms * dt * math.cos(hdg_rad)) / EARTH_RADIUS_M
    dlon = (gs_ms * dt * math.sin(hdg_rad)) / (
        EARTH_RADIUS_M * math.cos(math.radians(last.lat))
    )

    # Estimate altitude descent rate from the last two points
    alt_rate = 0.0
    idx = track.index(last) if last in track else -1
    if idx > 0:
        prev = track[idx - 1]
        dt_prev = last.epoch_s - prev.epoch_s
        if dt_prev > 0:
            alt_rate = (last.alt_ft100 - prev.alt_ft100) / dt_prev

    return TrackPoint(
        timestamp=datetime.fromtimestamp(query_epoch, tz=timezone.utc),
        lat=last.lat + math.degrees(dlat),
        lon=last.lon + math.degrees(dlon),
        alt_ft100=max(0, last.alt_ft100 + alt_rate * dt),
        groundspeed_kts=last.groundspeed_kts,
        heading=last.heading,
        source="extrapolated_forward",
    )


def compute_speed_sigma(track: List[TrackPoint], window_start_epoch: float, window_end_epoch: float) -> float:
    """Compute standard deviation of groundspeed (kts) within a past-only time window."""
    speeds = [
        p.groundspeed_kts for p in track
        if window_start_epoch <= p.epoch_s <= window_end_epoch
        and p.groundspeed_kts > 0
    ]
    if len(speeds) < 2:
        return 5.0  # fallback
    mean = sum(speeds) / len(speeds)
    var = sum((s - mean) ** 2 for s in speeds) / (len(speeds) - 1)
    return math.sqrt(var)


def get_aircraft_state(
    track: List[TrackPoint],
    query_time: datetime,
    target_point: Tuple[float, float] = RW04_TXY_D,
    speed_window_s: float = 120.0,
) -> AircraftState:
    """
    Compute aircraft state and ETA to a target point at a given time.
    Sigma is derived from measured groundspeed variance.
    """
    q_epoch = query_time.timestamp()
    pt = _extrapolate_forward(track, q_epoch)

    dist_m = haversine(pt.lat, pt.lon, target_point[0], target_point[1])
    gs_ms = pt.groundspeed_ms

    if gs_ms > 1.0:
        eta_s = dist_m / gs_ms
    else:
        eta_s = float("inf")

    # Derive sigma from actual ADS-B groundspeed variance
    speed_sigma_kts = compute_speed_sigma(track, q_epoch - speed_window_s, q_epoch)
    if gs_ms > 1.0:
        speed_sigma_ms = speed_sigma_kts * KTS_TO_MS
        sigma_s = dist_m * speed_sigma_ms / (gs_ms ** 2)
    else:
        sigma_s = 5.0

    return AircraftState(
        timestamp=pt.timestamp,
        lat=pt.lat,
        lon=pt.lon,
        alt_ft=pt.alt_ft,
        groundspeed_kts=pt.groundspeed_kts,
        distance_to_point_m=dist_m,
        time_to_point_s=eta_s,
        sigma_s=sigma_s,
        source=pt.source,
    )


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "..", "surface_data", "lga_case_study", "flight_8646_track.csv")
    track = load_track(csv_path)
    print(f"Loaded {len(track)} ADS-B positions")
    print(f"  First: {track[0].timestamp} — Last: {track[-1].timestamp}")
    print(f"  RW04 Threshold: {RW04_THRESHOLD}")
    print(f"  Txy D crossing: {RW04_TXY_D}\n")

    conflict_time = datetime(2026, 3, 23, 3, 37, 1, tzinfo=timezone.utc)
    state = get_aircraft_state(track, conflict_time, RW04_TXY_D)
    print(f"At conflict clearance time (03:37:01Z):")
    print(f"  {state}")
    print(f"  Distance to Txy D: {state.distance_to_point_m:.0f} m")
    print(f"  ETA to Txy D: {state.time_to_point_s:.1f} s  (σ = {state.sigma_s:.1f} s)")

    request_time = datetime(2026, 3, 23, 3, 36, 57, tzinfo=timezone.utc)
    state2 = get_aircraft_state(track, request_time, RW04_TXY_D)
    print(f"\nAt truck request time (03:36:57Z):")
    print(f"  {state2}")
