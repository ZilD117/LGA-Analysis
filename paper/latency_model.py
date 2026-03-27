#!/usr/bin/env python3
"""
ASR Latency Model — Honest end-to-end latency analysis for the paper.

Whisper (and any batch ASR) introduces unavoidable latency:
  1. Utterance must complete before it can be transcribed
  2. Whisper inference time (~1-3s for "small" model on GPU, ~5-8s on CPU)
  3. NER parsing (negligible: <1ms)

This script computes latency-corrected detection timelines for the LGA
case study using the real transcript timestamps.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from enhanced_detection.clearance_parser import parse_clearances, ClearanceType
from enhanced_detection.runway_state import RunwayStateTracker

TRANSCRIPT = os.path.join(os.path.dirname(__file__), "..", "voice_data",
                          "lga_case_study", "transcript.txt")

# Whisper inference latency for the "small" model
# Published benchmarks: ~0.5-1.0x real-time on GPU, ~3-5x on CPU
# For a 3-second utterance:
#   GPU: 0.5-1.0x * 3s = 1.5-3.0s
#   CPU: 3-5x * 3s = 9-15s
# We model three scenarios.
WHISPER_LATENCY_SCENARIOS = {
    "GPU (optimistic)": {"rtf": 0.5, "overhead_s": 0.3},
    "GPU (typical)":    {"rtf": 1.0, "overhead_s": 0.5},
    "CPU (laptop)":     {"rtf": 3.5, "overhead_s": 1.0},
}

COLLISION_OFFSET_S = 448.0  # T+448s = 03:37:28 UTC (collision)
CONTROLLER_FIRST_CORRECT_STOP_S = 437.02  # T+437s = "Stop, truck 1"


def compute_latency_corrected_timeline():
    clearances = parse_clearances(TRANSCRIPT)
    tracker = RunwayStateTracker()

    conflict_event = None
    trigger_clearance = None
    all_conflicts = []
    for c in clearances:
        ev = tracker.feed(c)
        if ev:
            all_conflicts.append((ev, c))
            entities = {ev.clearance_a.entity, ev.clearance_b.entity}
            if ("Jazz 8646" in entities and
                ("Truck 1" in entities or "UNKNOWN" in entities) and
                c.clearance_type == ClearanceType.CROSSING):
                if not conflict_event:
                    conflict_event = ev
                    trigger_clearance = c

    if not conflict_event and all_conflicts:
        for ev, c in all_conflicts:
            if c.clearance_type == ClearanceType.CROSSING and c.runway in ("04", "4"):
                conflict_event = ev
                trigger_clearance = c
                break

    if not conflict_event:
        print("ERROR: Could not find conflict event")
        return

    # Use the actual conflict-triggering clearance's timing
    # From the transcript the issuance is [421.06s - 423.70s]
    utterance_start = trigger_clearance.timestamp_s
    # Estimate utterance end from raw_text length (~2.6s typical ATC utterance)
    utterance_end = utterance_start + 2.64
    utterance_duration = utterance_end - utterance_start

    print("=" * 70)
    print("  ASR LATENCY MODEL — Honest Detection Timeline")
    print("=" * 70)

    print(f"\n  Conflict trigger: '{trigger_clearance.raw_text}'")
    print(f"  Utterance window: T+{utterance_start:.1f}s to T+{utterance_end:.1f}s "
          f"(duration: {utterance_duration:.1f}s)")
    print(f"  Collision at: T+{COLLISION_OFFSET_S:.0f}s")
    print(f"  Controller correct STOP at: T+{CONTROLLER_FIRST_CORRECT_STOP_S:.0f}s")

    results = {}

    print(f"\n  {'Scenario':<22} {'ASR Latency':>12} {'Detection':>12} "
          f"{'Lead Time':>12} {'vs Controller':>14}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12} {'-'*14}")

    for name, params in WHISPER_LATENCY_SCENARIOS.items():
        asr_latency = utterance_duration * params["rtf"] + params["overhead_s"]
        detection_time = utterance_end + asr_latency
        lead_time = COLLISION_OFFSET_S - detection_time
        controller_lead = CONTROLLER_FIRST_CORRECT_STOP_S - utterance_start
        advantage = lead_time - (COLLISION_OFFSET_S - CONTROLLER_FIRST_CORRECT_STOP_S)

        results[name] = {
            "asr_latency_s": asr_latency,
            "detection_time_s": detection_time,
            "lead_time_s": lead_time,
            "advantage_s": advantage,
        }

        print(f"  {name:<22} {asr_latency:>10.1f}s  T+{detection_time:>7.1f}s  "
              f"{lead_time:>10.1f}s  {advantage:>+12.1f}s")

    print(f"\n  Controller's timeline:")
    print(f"    Clearance issued:           T+{utterance_start:.0f}s")
    print(f"    First STOP (wrong target):  T+{428.82:.0f}s (+{428.82 - utterance_start:.0f}s)")
    print(f"    First correct STOP:         T+{CONTROLLER_FIRST_CORRECT_STOP_S:.0f}s "
          f"(+{CONTROLLER_FIRST_CORRECT_STOP_S - utterance_start:.0f}s)")
    print(f"    Collision:                  T+{COLLISION_OFFSET_S:.0f}s "
          f"(+{COLLISION_OFFSET_S - utterance_start:.0f}s)")
    print(f"    Controller lead time:       {COLLISION_OFFSET_S - CONTROLLER_FIRST_CORRECT_STOP_S:.0f}s "
          "(too late — truck already on runway)")

    # Compute the honest "system + controller action" timeline
    print(f"\n  System + controller action pipeline:")
    print(f"    {'Component':<35} {'Duration':>10}")
    print(f"    {'-'*35} {'-'*10}")

    gpu_typical = results["GPU (typical)"]
    steps = [
        ("Utterance completes", utterance_duration),
        ("Whisper inference (GPU typical)", utterance_duration * 1.0 + 0.5),
        ("NER + conflict detection", 0.001),
        ("Alert displayed to controller", 0.5),
        ("Controller processes alert", 2.0),
        ("Controller transmits STOP", 1.0),
    ]
    cumulative = utterance_start
    for step_name, duration in steps:
        cumulative += duration
        remaining = COLLISION_OFFSET_S - cumulative
        print(f"    {step_name:<35} {duration:>8.1f}s  (T+{cumulative:.1f}s, "
              f"{remaining:.0f}s before collision)")

    total_pipeline = cumulative - utterance_start
    print(f"\n    Total pipeline: {total_pipeline:.1f}s")
    print(f"    Honest lead time: {COLLISION_OFFSET_S - cumulative:.1f}s")
    print(f"    Truck travel at T+{cumulative:.0f}s: ~{(cumulative - 424.48) * 3.75:.0f}m "
          f"of 86m to runway")

    return results


if __name__ == "__main__":
    compute_latency_corrected_timeline()
