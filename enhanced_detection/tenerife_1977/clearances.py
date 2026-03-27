"""
Clearances extracted from Tenerife CVR transcript (voice_data/test_file/tenerife_head_to_head.txt).

The critical sequence: KLM received ATC route clearance (not takeoff clearance)
but read back "we're now at takeoff." The tower said "stand by for takeoff, I will
call you" — but this was stepped on by Pan Am's simultaneous transmission. KLM
crew began the takeoff roll without clearance.
"""

from enhanced_detection.clearance_parser import Clearance, ClearanceType

# Timestamps relative to config start_time (16:58:15 UTC)
# Both entities start their paths at this time. KLM backtracks the full runway
# (76+53+49+35+190 = 403s) then begins takeoff run (54s to collision point).
# Clipper 1736 enters from Txy C0 and taxis slowly on the runway.
#
# Key event: KLM begins takeoff roll at ~T+403s (17:04:58)
# Collision at ~T+457s (17:05:52)
CLEARANCES = [
    Clearance(
        timestamp_s=55.0,
        utc="16:59:10",
        entity="KLM 4805",
        clearance_type=ClearanceType.TAXI,
        runway="12",
        raw_text="KLM 4805 is now on the runway (backtracking for RW30 takeoff)",
    ),
    Clearance(
        timestamp_s=233.0,
        utc="17:02:08",
        entity="Clipper 1736",
        clearance_type=ClearanceType.TAXI,
        runway="12",
        raw_text="Clipper 1736 taxi into the runway, leave third to your left",
    ),
    Clearance(
        timestamp_s=403.0,
        utc="17:05:00",
        entity="KLM 4805",
        clearance_type=ClearanceType.TAKEOFF,
        runway="12",
        raw_text="We're now at takeoff (KLM begins takeoff roll without clearance)",
    ),
    Clearance(
        timestamp_s=415.0,
        utc="17:05:12",
        entity="Clipper 1736",
        clearance_type=ClearanceType.TAXI,
        runway="12",
        raw_text="We are still taxiing down the runway, the Clipper 1736",
    ),
]
