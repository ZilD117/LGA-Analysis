"""
Clearances extracted from Haneda ATC transcript (voice_data/test_file/henada_accident.txt).

The critical ambiguity: JA722A was told "number 1, taxi to holding point C5"
which the crew interpreted as clearance to enter the runway. The system should
detect that JA722A was given TAXI (not TAKEOFF) while JAL 516 was cleared to land.
"""

from enhanced_detection.clearance_parser import Clearance, ClearanceType

# Timestamps relative to config start_time (17:44:56 UTC — JAL 516 cleared to land)
CLEARANCES = [
    Clearance(
        timestamp_s=0.0,
        utc="17:44:56",
        entity="JAL 516",
        clearance_type=ClearanceType.LANDING,
        runway="34R",
        raw_text="Japan Air 516, runway 34R, cleared to land, wind 310 at 8",
    ),
    Clearance(
        timestamp_s=15.0,
        utc="17:45:11",
        entity="JA722A",
        clearance_type=ClearanceType.TAXI,
        runway="34R",
        taxiway="C5",
        raw_text="JA722A, Tokyo Tower, number 1, taxi to holding point C5",
    ),
    Clearance(
        timestamp_s=44.0,
        utc="17:45:40",
        entity="JAL 179",
        clearance_type=ClearanceType.TAXI,
        taxiway="C1",
        raw_text="Japan Air 179, number 3, taxi to holding point C1",
    ),
    Clearance(
        timestamp_s=60.0,
        utc="17:45:56",
        entity="JAL 166",
        clearance_type=ClearanceType.LANDING,
        runway="34R",
        raw_text="Japan Air 166, number 2, runway 34R, continue approach",
    ),
]
