"""
Runway State Tracker — Layer 1: Pure-logic conflict detection.

Maintains a dict of active clearances per runway and emits ConflictEvent
whenever two incompatible clearances are simultaneously active.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from enhanced_detection.clearance_parser import (
    Clearance, ClearanceType, parse_clearances,
)

INCOMPATIBLE_PAIRS = {
    frozenset({ClearanceType.LANDING, ClearanceType.CROSSING}),
    frozenset({ClearanceType.LANDING, ClearanceType.TAKEOFF}),
    frozenset({ClearanceType.TAKEOFF, ClearanceType.CROSSING}),
}

DEFAULT_EXPIRY_S = 180.0


@dataclass
class ConflictEvent:
    timestamp_s: float
    utc: str
    runway: str
    clearance_a: Clearance
    clearance_b: Clearance
    description: str

    def __repr__(self):
        return (
            f"CONFLICT @ {self.utc} on RW{self.runway}: "
            f"{self.clearance_a.entity} ({self.clearance_a.clearance_type.value}) "
            f"vs {self.clearance_b.entity} ({self.clearance_b.clearance_type.value}) "
            f"— {self.description}"
        )


@dataclass
class _ActiveClearance:
    clearance: Clearance
    expires_at_s: float


class RunwayStateTracker:
    """Feed clearances chronologically; emits conflicts when incompatible ops coexist."""

    def __init__(self, expiry_s: float = DEFAULT_EXPIRY_S):
        self.expiry_s = expiry_s
        self._active: Dict[str, List[_ActiveClearance]] = {}
        self.conflicts: List[ConflictEvent] = []

    def _prune(self, runway: str, now_s: float):
        if runway in self._active:
            self._active[runway] = [
                a for a in self._active[runway] if a.expires_at_s > now_s
            ]

    def _cancel_entity(self, entity: str, runway: Optional[str], now_s: float):
        """Remove all active clearances for entity (STOP / GO_AROUND cancels prior ops)."""
        for rwy in (self._active if runway is None else [runway]):
            if rwy in self._active:
                self._active[rwy] = [
                    a for a in self._active[rwy]
                    if a.clearance.entity != entity
                ]

    def feed(self, clearance: Clearance) -> Optional[ConflictEvent]:
        """Process a clearance. Returns a ConflictEvent if one is detected."""
        t = clearance.timestamp_s

        if clearance.clearance_type in (ClearanceType.STOP, ClearanceType.GO_AROUND):
            self._cancel_entity(clearance.entity, clearance.runway, t)
            return None

        if clearance.clearance_type in (ClearanceType.HOLD_SHORT, ClearanceType.OTHER):
            return None

        rwy = clearance.runway
        if rwy is None:
            return None

        self._prune(rwy, t)

        conflict = None
        for ac in self._active.get(rwy, []):
            pair = frozenset({ac.clearance.clearance_type, clearance.clearance_type})
            if pair in INCOMPATIBLE_PAIRS:
                desc = (
                    f"{ac.clearance.entity} was cleared to "
                    f"{ac.clearance.clearance_type.value.lower()} "
                    f"{int(t - ac.clearance.timestamp_s)}s earlier; "
                    f"now {clearance.entity} cleared to "
                    f"{clearance.clearance_type.value.lower()} same runway"
                )
                conflict = ConflictEvent(
                    timestamp_s=t,
                    utc=clearance.utc,
                    runway=rwy,
                    clearance_a=ac.clearance,
                    clearance_b=clearance,
                    description=desc,
                )
                self.conflicts.append(conflict)
                break

        if rwy not in self._active:
            self._active[rwy] = []
        self._active[rwy].append(_ActiveClearance(
            clearance=clearance,
            expires_at_s=t + self.expiry_s,
        ))

        return conflict


def run_tracker(transcript_path: str) -> Tuple[List[Clearance], List[ConflictEvent]]:
    """Parse transcript and run conflict detection. Returns (clearances, conflicts)."""
    clearances = parse_clearances(transcript_path)
    tracker = RunwayStateTracker()
    for c in clearances:
        tracker.feed(c)
    return clearances, tracker.conflicts


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "voice_data", "lga_case_study", "transcript.txt")
    clearances, conflicts = run_tracker(path)
    print(f"Processed {len(clearances)} clearances.\n")
    if conflicts:
        print(f"{'!'*60}")
        print(f"  DETECTED {len(conflicts)} CONFLICT(S):")
        print(f"{'!'*60}")
        for c in conflicts:
            print(f"\n  {c}")
            print(f"    Clearance A: {c.clearance_a}")
            print(f"    Clearance B: {c.clearance_b}")
    else:
        print("No conflicts detected.")
