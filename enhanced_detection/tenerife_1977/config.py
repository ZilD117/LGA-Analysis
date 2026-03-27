"""
Tenerife 1977 — KLM 4805 vs Pan Am Clipper 1736.

On 27 March 1977, KLM Flight 4805 (Boeing 747) began its takeoff roll on
Runway 12/30 at Los Rodeos Airport (GCXO) while Pan Am Flight 1736 (Boeing 747)
was still taxiing on the same runway in fog. 583 people were killed — the
deadliest accident in aviation history.

KLM backtracked the full runway and began takeoff without clearance (read back
"we're now at takeoff" was ambiguous). Pan Am had not yet exited at taxiway C3/C4.
"""

from datetime import datetime, timezone

from enhanced_detection.backtest_runner import IncidentConfig, EntityConfig
from enhanced_detection.clearance_parser import ClearanceType

# Node coordinates from GCXO_Nodes_Def.csv
RWY_12_001 = (28.487889, -16.357273)  # RW12 threshold
RWY_12_002 = (28.485553, -16.350255)
RWY_12_003 = (28.483932, -16.345360)
RWY_12_004 = (28.482443, -16.340874)
RWY_12_005 = (28.481667, -16.338548)  # Collision point
RWY_12_006 = (28.477426, -16.325730)  # RW30 threshold
TXY_C0_001 = (28.488761, -16.356464)
TXY_C0_002 = (28.488325, -16.357476)

INCIDENT = IncidentConfig(
    name="Tenerife 1977 — KLM 4805 vs Clipper 1736",
    airport_icao="GCXO",
    entity_a=EntityConfig(
        name="KLM 4805",
        path=[
            "Rwy_12_001", "Rwy_12_002", "Rwy_12_003", "Rwy_12_004",
            "Rwy_12_005", "Rwy_12_006", "Rwy_12_005",
        ],
        segment_times=[76, 53, 49, 35, 190, 54],
        clearance_type=ClearanceType.TAKEOFF,
        altitude_ft=0.0,
    ),
    entity_b=EntityConfig(
        name="Clipper 1736",
        path=[
            "Txy_C0_001", "Txy_C0_002", "Rwy_12_001", "Rwy_12_002",
            "Rwy_12_003", "Rwy_12_004", "Rwy_12_005",
        ],
        segment_times=[121, 57, 96, 67, 62, 54],
        clearance_type=ClearanceType.TAXI,
        altitude_ft=0.0,
    ),
    conflict_node="Rwy_12_005",
    crossing_entry=RWY_12_004,
    crossing_midpoint=RWY_12_005,
    crossing_exit=RWY_12_006,
    collision_offset_s=41.0,  # ~41s from KLM brake release to impact
    start_time=datetime(1977, 3, 27, 16, 58, 15, tzinfo=timezone.utc),
    rc_km=0.032,
    entity_a_wingspan_m=59.6,  # Boeing 747-200
    entity_b_width_m=59.6,     # Boeing 747-100
    t_max_sec=500.0,
    mean_speed_kmh=25.0,
    sigma_speed_kmh=8.0,
)
