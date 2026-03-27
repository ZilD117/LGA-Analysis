"""
Synthetic ADS-B Track Generator — Build realistic TrackPoints from airport layout nodes.

Given a path through an airport's node-link graph and segment traversal times,
generates interpolated position reports at configurable intervals. Used for
backtesting the detection system on incidents where real ADS-B data is unavailable.
"""

import csv
import math
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from enhanced_detection.aircraft_eta import TrackPoint, haversine, EARTH_RADIUS_M

LAYOUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "taxigen", "Airport Layouts")


def load_airport_nodes(icao: str) -> Dict[str, Tuple[float, float]]:
    """Load node ID -> (lat, lon) mapping from the FACET airport layout CSV."""
    csv_path = os.path.join(LAYOUTS_DIR, f"{icao}_Nodes_Def.csv")
    nodes = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes[row["id"]] = (float(row["lat"]), float(row["lon"]))
    return nodes


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute initial bearing (degrees) from point 1 to point 2."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def generate_track(
    nodes: Dict[str, Tuple[float, float]],
    path: List[str],
    segment_times: List[float],
    start_time: datetime,
    report_interval_s: float = 5.0,
    altitude_ft: float = 0.0,
    source: str = "synthetic",
) -> List[TrackPoint]:
    """
    Generate synthetic ADS-B track points along a node path.

    For each segment (node_i -> node_{i+1}), interpolates positions at
    report_interval_s intervals. Groundspeed is computed from segment
    distance / time. Heading is the bearing between segment endpoints.
    """
    if len(segment_times) != len(path) - 1:
        raise ValueError(f"Need {len(path)-1} segment times for {len(path)} nodes, got {len(segment_times)}")

    points: List[TrackPoint] = []
    current_epoch = start_time.timestamp()

    for seg_idx in range(len(path) - 1):
        n1, n2 = path[seg_idx], path[seg_idx + 1]
        lat1, lon1 = nodes[n1]
        lat2, lon2 = nodes[n2]
        seg_time = segment_times[seg_idx]
        seg_dist = haversine(lat1, lon1, lat2, lon2)
        gs_kts = (seg_dist / max(seg_time, 0.1)) / 0.514444
        hdg = bearing(lat1, lon1, lat2, lon2)

        n_reports = max(1, int(seg_time / report_interval_s))
        for i in range(n_reports):
            frac = i / n_reports
            t = current_epoch + frac * seg_time
            lat = lat1 + frac * (lat2 - lat1)
            lon = lon1 + frac * (lon2 - lon1)
            points.append(TrackPoint(
                timestamp=datetime.fromtimestamp(t, tz=timezone.utc),
                lat=lat,
                lon=lon,
                alt_ft100=altitude_ft / 100.0,
                groundspeed_kts=gs_kts,
                heading=hdg,
                source=source,
            ))

        current_epoch += seg_time

    # Final point at path end
    last_lat, last_lon = nodes[path[-1]]
    points.append(TrackPoint(
        timestamp=datetime.fromtimestamp(current_epoch, tz=timezone.utc),
        lat=last_lat,
        lon=last_lon,
        alt_ft100=altitude_ft / 100.0,
        groundspeed_kts=0.0,
        heading=points[-1].heading if points else 0.0,
        source=source,
    ))

    return points


def write_track_csv(points: List[TrackPoint], output_path: str):
    """Write track points to CSV in the same format as flight_8646_track.csv."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "latitude", "longitude", "altitude_ft100",
            "groundspeed_kts", "heading", "update_type", "source",
        ])
        for pt in points:
            writer.writerow([
                pt.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                f"{pt.lat:.6f}",
                f"{pt.lon:.6f}",
                f"{pt.alt_ft100:.1f}",
                f"{pt.groundspeed_kts:.1f}",
                f"{pt.heading:.1f}",
                "A",
                pt.source,
            ])
