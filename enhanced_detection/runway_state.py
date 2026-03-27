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
    frozenset({ClearanceType.LINE_UP_WAIT, ClearanceType.LANDING}),
    frozenset({ClearanceType.LINE_UP_WAIT, ClearanceType.CROSSING}),
    frozenset({ClearanceType.LANDING, ClearanceType.TAXI}),
    frozenset({ClearanceType.TAKEOFF, ClearanceType.TAXI}),
}

# Runways that are the same physical surface (opposite directions)
RUNWAY_ALIASES = {
    # KLGA
    "04": "04", "22": "04",
    "13": "13", "31": "13",
    # HND (Tokyo Haneda)
    "34R": "34R", "16L": "34R",
    "34L": "34L", "16R": "34L",
    "05": "05", "23": "05",
    # GCXO (Tenerife North)
    "12": "12", "30": "12",
    # KATL
    "8R": "8R", "26L": "8R",
    "8L": "8L", "26R": "8L",
    "9R": "9R", "27L": "9R",
    "9L": "9L", "27R": "9L",
    "10": "10", "28": "10",
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

    @staticmethod
    def _canonical_runway(rwy: str) -> str:
        """Normalize runway IDs so opposite-direction runways map to the same key."""
        return RUNWAY_ALIASES.get(rwy, rwy)

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

        rwy = self._canonical_runway(rwy)
        self._prune(rwy, t)

        # Same-entity re-clearance: update rather than duplicate
        if rwy in self._active:
            self._active[rwy] = [
                a for a in self._active[rwy]
                if a.clearance.entity != clearance.entity
            ]

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
