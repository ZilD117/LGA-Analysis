"""
KATL 2024 — Endeavor 5526 vs Delta 295 taxiway collision.

On 10 September 2024, Endeavor Air Flight 5526 (CRJ-900) and Delta Air Lines
Flight 295 (Airbus A350) collided on Taxiway Echo at Atlanta Hartsfield-Jackson.
The CRJ-900's tail was clipped by the A350's wingtip. No injuries.

This is a TAXIWAY collision — both entities are ground-level, moving on the
same taxiway segment. The detection challenge is recognizing converging paths.
"""

from datetime import datetime, timezone

from enhanced_detection.backtest_runner import IncidentConfig, EntityConfig
from enhanced_detection.clearance_parser import ClearanceType

# Node coordinates from KATL_Nodes_Def.csv
TXY_3N_002 = (33.644102, -84.433922)
TXY_F_005 = (33.644602, -84.433956)
TXY_F_104 = (33.644597, -84.434257)
TXY_E_002 = (33.645243, -84.438198)
TXY_E_003 = (33.645422, -84.436893)
TXY_E_004 = (33.645422, -84.434251)  # Conflict point
TXY_E_005 = (33.645432, -84.430993)
TXY_E_104 = (33.645427, -84.432820)

INCIDENT = IncidentConfig(
    name="KATL 2024 — Endeavor 5526 vs Delta 295",
    airport_icao="KATL",
    entity_a=EntityConfig(
        name="Endeavor 5526",
        path=["Txy_3N_002", "Txy_F_005", "Txy_F_104", "Txy_E_004", "Txy_E_003", "Txy_E_002"],
        segment_times=[14, 7, 20, 52, 27],
        clearance_type=ClearanceType.TAXI,
        altitude_ft=0.0,
    ),
    entity_b=EntityConfig(
        name="Delta 295",
        path=["Txy_E_005", "Txy_E_104", "Txy_E_004", "Txy_E_003", "Txy_E_002"],
        segment_times=[30, 24, 40, 26],
        clearance_type=ClearanceType.TAXI,
        altitude_ft=0.0,
    ),
    conflict_node="Txy_E_004",
    crossing_entry=TXY_E_005,
    crossing_midpoint=TXY_E_004,
    crossing_exit=TXY_E_003,
    collision_offset_s=41.0,
    start_time=datetime(2024, 9, 10, 14, 0, 0, tzinfo=timezone.utc),
    rc_km=0.035,
    entity_a_wingspan_m=26.0,   # CRJ-900
    entity_b_width_m=64.75,     # A350-900
    t_max_sec=130.0,
    mean_speed_kmh=15.0,
    sigma_speed_kmh=5.0,
)
