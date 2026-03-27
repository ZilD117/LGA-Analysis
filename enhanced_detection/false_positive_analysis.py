#!/usr/bin/env python3
"""
False Positive Analysis — Measure the conflict detection false alarm rate
on non-incident ATC transcripts.

Runs the clearance parser + runway state tracker on all available transcripts
and counts how many spurious conflicts are detected. This is the key metric
for real-world viability: a system that alerts on every other transmission
is useless regardless of its true positive rate.

The train_file transcripts are plain text (no Whisper timestamps), so we
parse them with a lightweight adapter that assigns sequential timestamps.
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from enhanced_detection.clearance_parser import (
    Clearance, ClearanceType, _classify, _extract_callsign,
    _extract_runway, _extract_taxiway, _apply_corrections,
)
from enhanced_detection.runway_state import RunwayStateTracker

VOICE_DIR = os.path.join(os.path.dirname(__file__), "..", "voice_data")
TRAIN_DIR = os.path.join(VOICE_DIR, "train_file")
TEST_DIR = os.path.join(VOICE_DIR, "test_file")
VAL_DIR = os.path.join(VOICE_DIR, "val_file")

# Transcripts with known runway incursion or collision — TRUE conflicts expected
KNOWN_INCIDENT_FILES = {
    "deltaclippedKATL",           # KATL taxiway collision
    "henada_accident",            # Haneda runway collision
    "tenerife_head_to_head",      # Tenerife disaster
    "ACA759",                     # SFO near-miss (go-around)
    "ac759-sfo",                  # Same incident, different file
    "wingclipKMSP",               # KMSP wing clip
    "SWA1643 OFF RUNWAY",         # Southwest runway excursion
    "Southwest SKID OFF",         # Southwest skid
    "accessroadKBTV",             # KBTV access road incursion
}


def _is_known_incident(filename: str) -> bool:
    for prefix in KNOWN_INCIDENT_FILES:
        if prefix.lower() in filename.lower():
            return True
    return False


def parse_plain_transcript(filepath: str):
    """
    Parse a plain-text transcript (no Whisper timestamps) into clearances.

    Treats each non-empty line as a separate transmission, assigned a
    sequential timestamp (1 second per line).
    """
    clearances = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return []

    t = 0.0
    for line in lines:
        text = line.strip()
        if not text or len(text) < 5:
            continue

        ctype = _classify(text)
        if ctype == ClearanceType.OTHER:
            t += 1.0
            continue

        entity = _extract_callsign(text) or "UNKNOWN"
        runway = _extract_runway(text)
        taxiway = _extract_taxiway(text) if ctype == ClearanceType.CROSSING else None

        c = Clearance(
            timestamp_s=t,
            utc=f"T+{t:.0f}s",
            entity=entity,
            clearance_type=ctype,
            runway=runway,
            taxiway=taxiway,
            raw_text=text,
        )
        clearances.append(_apply_corrections(c))
        t += 1.0

    return clearances


def analyze_transcript(filepath: str):
    """Run conflict detection on a single transcript. Returns (clearances, conflicts)."""
    clearances = parse_plain_transcript(filepath)
    tracker = RunwayStateTracker()
    for c in clearances:
        tracker.feed(c)
    return clearances, tracker.conflicts


def run_analysis():
    """Analyze all available transcripts and report false positive rate."""
    all_dirs = []
    if os.path.isdir(TRAIN_DIR):
        all_dirs.append(("train", TRAIN_DIR))
    if os.path.isdir(TEST_DIR):
        all_dirs.append(("test", TEST_DIR))
    if os.path.isdir(VAL_DIR):
        all_dirs.append(("val", VAL_DIR))
        txts_dir = os.path.join(VAL_DIR, "txts")
        if os.path.isdir(txts_dir):
            all_dirs.append(("val/txts", txts_dir))

    results = []
    total_clearances = 0
    total_conflicts = 0
    incident_conflicts = 0
    non_incident_conflicts = 0
    non_incident_files = 0
    incident_files = 0

    print("=" * 80)
    print("  FALSE POSITIVE ANALYSIS — Conflict Detection on ATC Transcripts")
    print("=" * 80)

    for label, dirpath in all_dirs:
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(dirpath, fname)
            if not os.path.isfile(fpath):
                continue

            clearances, conflicts = analyze_transcript(fpath)
            is_incident = _is_known_incident(fname)

            if is_incident:
                incident_files += 1
                incident_conflicts += len(conflicts)
            else:
                non_incident_files += 1
                non_incident_conflicts += len(conflicts)

            total_clearances += len(clearances)
            total_conflicts += len(conflicts)

            status = "INCIDENT" if is_incident else "normal"
            conflict_str = f"{len(conflicts)} conflict(s)" if conflicts else "clean"

            results.append({
                "file": fname,
                "label": label,
                "is_incident": is_incident,
                "clearances": len(clearances),
                "conflicts": len(conflicts),
                "conflict_details": conflicts,
            })

            flag = " *** " if conflicts and not is_incident else "     "
            print(f"{flag}[{label:>8}] {fname:<60} {len(clearances):>3} clr  {conflict_str:<20}  ({status})")

    print(f"\n{'=' * 80}")
    print(f"  SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n  Total transcripts analyzed: {len(results)}")
    print(f"  Total clearances parsed:    {total_clearances}")
    print(f"  Total conflicts detected:   {total_conflicts}")
    print(f"\n  Known incident files:       {incident_files}")
    print(f"  Conflicts in incidents:     {incident_conflicts}")
    print(f"\n  Non-incident files:         {non_incident_files}")
    print(f"  FALSE POSITIVES:            {non_incident_conflicts}")

    if non_incident_files > 0:
        fp_rate = non_incident_conflicts / non_incident_files
        print(f"  FP rate (per transcript):   {fp_rate:.2f}")
    if total_clearances > 0:
        fp_per_clearance = non_incident_conflicts / total_clearances
        print(f"  FP rate (per clearance):    {fp_per_clearance:.4f} ({fp_per_clearance*100:.2f}%)")

    # Show details of false positives
    fps = [r for r in results if r["conflicts"] and not r["is_incident"]]
    if fps:
        print(f"\n{'─' * 80}")
        print(f"  FALSE POSITIVE DETAILS")
        print(f"{'─' * 80}")
        for r in fps:
            print(f"\n  {r['file']}:")
            for c in r["conflict_details"]:
                print(f"    {c.description}")
                print(f"      {c.clearance_a.entity} ({c.clearance_a.clearance_type.value}) vs "
                      f"{c.clearance_b.entity} ({c.clearance_b.clearance_type.value}) on {c.runway}")

    # Show details of true positives (incidents that were detected)
    tps = [r for r in results if r["conflicts"] and r["is_incident"]]
    if tps:
        print(f"\n{'─' * 80}")
        print(f"  TRUE POSITIVE DETAILS (known incidents)")
        print(f"{'─' * 80}")
        for r in tps:
            print(f"\n  {r['file']}: {r['conflicts']} conflict(s) detected ✓")

    # Missed incidents
    missed = [r for r in results if not r["conflicts"] and r["is_incident"]]
    if missed:
        print(f"\n{'─' * 80}")
        print(f"  MISSED INCIDENTS (false negatives)")
        print(f"{'─' * 80}")
        for r in missed:
            print(f"\n  {r['file']}: {r['clearances']} clearances parsed, 0 conflicts ✗")

    return results


if __name__ == "__main__":
    run_analysis()
