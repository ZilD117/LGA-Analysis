#!/usr/bin/env python3
"""
Improved False Positive Analysis — Methodologically sound evaluation.

Three evaluation modes:
  1. LGA pre-incident: real Whisper-transcribed tower audio, first 27 minutes
     of normal operations before the incident (most credible)
  2. LiveATC-format transcripts: files with standard ATC phraseology
  3. Narrative transcripts: incident summaries (less credible, reported separately)
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from enhanced_detection.clearance_parser import (
    Clearance, ClearanceType, _classify, _extract_callsign,
    _extract_runway, _extract_taxiway, _apply_corrections,
    parse_clearances,
)
from enhanced_detection.runway_state import RunwayStateTracker
from workload_analysis import parse_transcript

VOICE_DIR = os.path.join(os.path.dirname(__file__), "..", "voice_data")
TRAIN_DIR = os.path.join(VOICE_DIR, "train_file")
TEST_DIR = os.path.join(VOICE_DIR, "test_file")
VAL_DIR = os.path.join(VOICE_DIR, "val_file")
LGA_TRANSCRIPT = os.path.join(VOICE_DIR, "lga_case_study", "transcript.txt")

INCIDENT_CUTOFF_S = 400.0  # ~03:36:40 UTC, before any incident-related comms

# LiveATC-sourced files have timestamps or airport codes in the name
LIVEATC_PATTERNS = re.compile(
    r"(K[A-Z]{3}|KLGA|KBOS|KATL|KSFO|KRNO|KVNY|KGSO|KIAD|KSNA|KOMA|KBTV|KMSN|KSYR)"
    r"|-\d{4}Z"
    r"|Tower|Gnd|App|Del"
    r"|TWR|GND",
    re.I
)

KNOWN_INCIDENT_PREFIXES = {
    "deltaclippedKATL", "henada_accident", "tenerife_head_to_head",
    "ACA759", "ac759-sfo", "wingclipKMSP", "SWA1643 OFF RUNWAY",
    "Southwest SKID OFF", "accessroadKBTV",
}


def _is_incident(fname):
    for prefix in KNOWN_INCIDENT_PREFIXES:
        if prefix.lower() in fname.lower():
            return True
    return False


def _is_liveatc_format(fname):
    return bool(LIVEATC_PATTERNS.search(fname))


def _parse_plain(filepath):
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
        c = Clearance(timestamp_s=t, utc=f"T+{t:.0f}s", entity=entity,
                      clearance_type=ctype, runway=runway, taxiway=taxiway, raw_text=text)
        clearances.append(_apply_corrections(c))
        t += 1.0
    return clearances


def analyze_lga_pre_incident():
    """Run conflict detection on the LGA transcript BEFORE the incident."""
    print("=" * 70)
    print("  MODE 1: LGA Pre-Incident (Real Whisper Audio)")
    print("=" * 70)

    segments = parse_transcript(LGA_TRANSCRIPT)
    pre_incident = [s for s in segments if s["start_s"] < INCIDENT_CUTOFF_S]

    clearances = []
    for seg in pre_incident:
        text = seg["text"]
        ctype = _classify(text)
        if ctype == ClearanceType.OTHER:
            continue
        entity = _extract_callsign(text) or "UNKNOWN"
        runway = _extract_runway(text)
        taxiway = _extract_taxiway(text) if ctype == ClearanceType.CROSSING else None
        c = Clearance(timestamp_s=seg["start_s"], utc=seg["utc_str"], entity=entity,
                      clearance_type=ctype, runway=runway, taxiway=taxiway, raw_text=text)
        clearances.append(_apply_corrections(c))

    tracker = RunwayStateTracker()
    for c in clearances:
        tracker.feed(c)

    duration_min = INCIDENT_CUTOFF_S / 60.0
    n_segments = len(pre_incident)
    n_clearances = len(clearances)
    n_conflicts = len(tracker.conflicts)

    print(f"\n  Audio duration: {duration_min:.1f} minutes of normal tower ops")
    print(f"  Segments transcribed: {n_segments}")
    print(f"  Clearances parsed: {n_clearances}")
    print(f"  Conflicts detected: {n_conflicts}")
    if n_clearances > 0:
        print(f"  FP rate: {n_conflicts}/{n_clearances} = "
              f"{n_conflicts/n_clearances*100:.2f}% per clearance")

    if tracker.conflicts:
        print(f"\n  False positive details:")
        for ev in tracker.conflicts:
            print(f"    {ev.utc}: {ev.clearance_a.entity} ({ev.clearance_a.clearance_type.value}) "
                  f"vs {ev.clearance_b.entity} ({ev.clearance_b.clearance_type.value}) "
                  f"on RW{ev.runway}")
            print(f"      A: '{ev.clearance_a.raw_text}'")
            print(f"      B: '{ev.clearance_b.raw_text}'")

    return {
        "type": "lga_pre_incident",
        "duration_min": duration_min,
        "segments": n_segments,
        "clearances": n_clearances,
        "conflicts": n_conflicts,
        "fp_rate": n_conflicts / max(n_clearances, 1),
    }


def analyze_external_transcripts():
    """Analyze all external transcripts, partitioned by type."""
    liveatc_results = {"files": 0, "clearances": 0, "conflicts": 0, "incident_files": 0}
    narrative_results = {"files": 0, "clearances": 0, "conflicts": 0, "incident_files": 0}

    all_dirs = []
    if os.path.isdir(TRAIN_DIR):
        all_dirs.append(("train", TRAIN_DIR))
    if os.path.isdir(TEST_DIR):
        all_dirs.append(("test", TEST_DIR))
    if os.path.isdir(VAL_DIR):
        txts_dir = os.path.join(VAL_DIR, "txts")
        if os.path.isdir(txts_dir):
            all_dirs.append(("val/txts", txts_dir))

    fp_details_liveatc = []
    fp_details_narrative = []

    for label, dirpath in all_dirs:
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(dirpath, fname)
            if not os.path.isfile(fpath):
                continue

            clearances = _parse_plain(fpath)
            tracker = RunwayStateTracker()
            for c in clearances:
                tracker.feed(c)

            is_incident = _is_incident(fname)
            is_liveatc = _is_liveatc_format(fname)
            bucket = liveatc_results if is_liveatc else narrative_results
            detail_bucket = fp_details_liveatc if is_liveatc else fp_details_narrative

            bucket["files"] += 1
            bucket["clearances"] += len(clearances)
            bucket["conflicts"] += len(tracker.conflicts)
            if is_incident:
                bucket["incident_files"] += 1

            if tracker.conflicts and not is_incident:
                for ev in tracker.conflicts:
                    detail_bucket.append((fname, ev))

    # Print LiveATC-format results
    print(f"\n{'=' * 70}")
    print(f"  MODE 2: LiveATC-Format Transcripts (Standard ATC Phraseology)")
    print(f"{'=' * 70}")
    la = liveatc_results
    non_inc_files = la["files"] - la["incident_files"]
    print(f"\n  Total files: {la['files']} ({la['incident_files']} incident, "
          f"{non_inc_files} normal)")
    print(f"  Total clearances: {la['clearances']}")
    print(f"  Conflicts: {la['conflicts']}")
    if la["clearances"] > 0:
        print(f"  FP rate: {la['conflicts']}/{la['clearances']} = "
              f"{la['conflicts']/la['clearances']*100:.2f}% per clearance")
    if fp_details_liveatc:
        print(f"\n  False positive details:")
        for fname, ev in fp_details_liveatc:
            print(f"    [{fname}] {ev.clearance_a.entity} ({ev.clearance_a.clearance_type.value}) "
                  f"vs {ev.clearance_b.entity} ({ev.clearance_b.clearance_type.value})")

    # Print narrative results
    print(f"\n{'=' * 70}")
    print(f"  MODE 3: Narrative Transcripts (Incident Summaries, Less Credible)")
    print(f"{'=' * 70}")
    na = narrative_results
    non_inc_files_n = na["files"] - na["incident_files"]
    print(f"\n  Total files: {na['files']} ({na['incident_files']} incident, "
          f"{non_inc_files_n} normal)")
    print(f"  Total clearances: {na['clearances']}")
    print(f"  Conflicts: {na['conflicts']}")
    if na["clearances"] > 0:
        print(f"  FP rate: {na['conflicts']}/{na['clearances']} = "
              f"{na['conflicts']/na['clearances']*100:.2f}% per clearance")

    return liveatc_results, narrative_results


def print_summary(lga, liveatc, narrative):
    print(f"\n{'=' * 70}")
    print(f"  COMBINED RESULTS")
    print(f"{'=' * 70}")

    print(f"\n  {'Data Source':<40} {'Clr':>5} {'FP':>4} {'Rate':>8}")
    print(f"  {'-'*40} {'-'*5} {'-'*4} {'-'*8}")
    print(f"  {'LGA real audio (pre-incident, 6.7min)':<40} "
          f"{lga['clearances']:>5} {lga['conflicts']:>4} "
          f"{lga['fp_rate']*100:>7.2f}%")

    la_rate = liveatc["conflicts"] / max(liveatc["clearances"], 1)
    print(f"  {'LiveATC-format transcripts':<40} "
          f"{liveatc['clearances']:>5} {liveatc['conflicts']:>4} "
          f"{la_rate*100:>7.2f}%")

    na_rate = narrative["conflicts"] / max(narrative["clearances"], 1)
    print(f"  {'Narrative transcripts (less credible)':<40} "
          f"{narrative['clearances']:>5} {narrative['conflicts']:>4} "
          f"{na_rate*100:>7.2f}%")

    total_clr = lga["clearances"] + liveatc["clearances"]
    total_fp = lga["conflicts"] + liveatc["conflicts"]
    combined_rate = total_fp / max(total_clr, 1)
    print(f"  {'-'*40} {'-'*5} {'-'*4} {'-'*8}")
    print(f"  {'Combined (excl. narrative)':<40} "
          f"{total_clr:>5} {total_fp:>4} {combined_rate*100:>7.2f}%")

    print(f"\n  Note: narrative transcripts excluded from headline FP rate")
    print(f"  because they do not represent real ATC phraseology.")


if __name__ == "__main__":
    lga = analyze_lga_pre_incident()
    liveatc, narrative = analyze_external_transcripts()
    print_summary(lga, liveatc, narrative)
