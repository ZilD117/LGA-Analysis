"""
Haneda 2024 — Japan Airlines Flight 516 vs Coast Guard JA722A.

On 2 January 2024, JAL 516 (Airbus A350) was landing on Runway 34R at Tokyo
Haneda when it collided with a Japan Coast Guard DHC-8 (JA722A) that had entered
the runway without clearance. The DHC-8 crew misinterpreted "number 1, taxi to
holding point C5" as clearance to enter the runway.

All 379 on JAL 516 survived; 5 of 6 on JA722A were killed.
"""

import os
from datetime import datetime, timezone

from enhanced_detection.backtest_runner import IncidentConfig, EntityConfig
from enhanced_detection.clearance_parser import ClearanceType

# Node coordinates from HND_Nodes_Def.csv
RWY_03_001 = (35.5398282, 139.8050352)  # RW34R threshold
RWY_03_005 = (35.5483909, 139.7989829)
RWY_03_006 = (35.5508438, 139.7972341)  # Conflict point (C5 intersection)
RWY_03_007 = (35.5533053, 139.7954638)
TXY_C5_C5B = (35.5486411, 139.7974712)

JAL516_TRACK = os.path.join(os.path.dirname(__file__), "jal516_track.csv")

INCIDENT = IncidentConfig(
    name="Haneda 2024 — JAL 516 vs JA722A",
    airport_icao="HND",
    entity_a=EntityConfig(
        name="JAL 516",
        path=[
            "Rwy_03_001", "Rwy_03_002", "Rwy_03_003", "Rwy_03_004",
            "Rwy_03_005", "Rwy_03_006", "Rwy_03_007", "Rwy_03_008",
            "Rwy_03_009", "Rwy_03_010", "Rwy_03_011",
        ],
        segment_times=[7.2, 7.2, 7.2, 7.2, 7.2, 7.2, 7.2, 7.2, 7.2, 7.2],
        clearance_type=ClearanceType.LANDING,
        altitude_ft=200.0,
        track_csv=JAL516_TRACK,
    ),
    entity_b=EntityConfig(
        name="JA722A",
        path=[
            "Txy_C5_C5B", "Rwy_03_006", "Rwy_03_007", "Rwy_03_008",
            "Rwy_03_009", "Rwy_03_010", "Rwy_03_011",
        ],
        segment_times=[56.0, 18.0, 18.0, 18.0, 18.0, 18.0],
        clearance_type=ClearanceType.TAXI,
        altitude_ft=0.0,
    ),
    conflict_node="Rwy_03_006",
    crossing_entry=TXY_C5_C5B,
    crossing_midpoint=RWY_03_006,
    crossing_exit=RWY_03_007,
    collision_offset_s=144.0,
    start_time=datetime(2024, 1, 2, 8, 44, 56, tzinfo=timezone.utc),
    rc_km=0.032,
    entity_a_wingspan_m=64.75,  # A350-900
    entity_b_width_m=28.4,      # DHC-8-300
    t_max_sec=160.0,
    mean_speed_kmh=35.0,
    sigma_speed_kmh=8.0,
)
