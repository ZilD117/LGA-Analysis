"""
Clearances extracted from KATL transcript (voice_data/test_file/delta_taxiway_collision_atl.txt).

Both aircraft were cleared onto Taxiway Echo. The controller told Delta 295
to "follow the RJ" (Endeavor 5526) but the A350's wingtip caught the CRJ's tail
when both were on Echo near the Foxtrot intersection.
"""

from enhanced_detection.clearance_parser import Clearance, ClearanceType

# Timestamps relative to config start_time (14:00:00 UTC)
# Both aircraft begin taxiing at roughly the same time.
# The audio starts ~14s before the first clearance.
CLEARANCES = [
    Clearance(
        timestamp_s=14.0,
        utc="14:00:14",
        entity="Delta 295",
        clearance_type=ClearanceType.TAXI,
        taxiway="G",
        raw_text="Delta 295 heavy Atlanta ground runway 8R taxi golf short of foxtrot",
    ),
    Clearance(
        timestamp_s=57.0,
        utc="14:00:57",
        entity="Endeavor 5526",
        clearance_type=ClearanceType.TAXI,
        taxiway="E",
        raw_text="Endeavor 5526 Atlanta grounds runway 8R taxi via echo",
    ),
    Clearance(
        timestamp_s=63.0,
        utc="14:01:03",
        entity="Delta 295",
        clearance_type=ClearanceType.TAXI,
        taxiway="E",
        raw_text="Delta 295 heavy at fox 3 follow the RJ joining Echo monitor tower",
    ),
    Clearance(
        timestamp_s=74.0,
        utc="14:01:14",
        entity="Endeavor 5526",
        clearance_type=ClearanceType.TAXI,
        taxiway="E",
        raw_text="Endeavor 5526 this heavy Airbus will wait for you monitor tower",
    ),
]
