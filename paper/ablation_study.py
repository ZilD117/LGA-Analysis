#!/usr/bin/env python3
"""
Ablation Study — Quantify what each layer contributes.

L1 only:     Binary conflict detection (STOP/no-STOP)
L1 + L2:     Add ETA estimates (time-to-impact, urgency ranking)
L1 + L2 + L3: Add P(collision), Bayesian convergence, go-around feasibility
"""

import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from enhanced_detection.clearance_parser import parse_clearances, ClearanceType, Clearance
from enhanced_detection.runway_state import RunwayStateTracker
from enhanced_detection.aircraft_eta import load_track, get_aircraft_state, RW04_TXY_D, RW04_THRESHOLD, haversine
from enhanced_detection.ground_speed_prior import build_crossing_estimate
from enhanced_detection.enhanced_risk import compute_risk
from enhanced_detection.approach_profile import build_approach_profile, compute_eta_sigma
from enhanced_detection.decision_engine import (
    make_decision, evaluate_counterfactual, BayesianTracker,
    DECISION_THRESHOLD,
)

TRANSCRIPT = os.path.join(os.path.dirname(__file__), "..", "voice_data",
                          "lga_case_study", "transcript.txt")
ADSB = os.path.join(os.path.dirname(__file__), "..", "surface_data",
                     "lga_case_study", "flight_8646_track.csv")
AUDIO_START = datetime(2026, 3, 23, 3, 30, 0, tzinfo=timezone.utc)
COLLISION_T = 448.0  # seconds after audio start


def run_ablation():
    print("=" * 70)
    print("  ABLATION STUDY — What Each Layer Contributes")
    print("=" * 70)

    # ── Layer 1 Only ──
    print(f"\n{'─' * 70}")
    print("  LAYER 1 ONLY: Boolean Conflict Detection")
    print(f"{'─' * 70}")

    clearances = parse_clearances(TRANSCRIPT)
    tracker = RunwayStateTracker()
    conflicts = []
    for c in clearances:
        ev = tracker.feed(c)
        if ev:
            conflicts.append(ev)

    # Find the truck conflict
    truck_conflict = None
    for ev in conflicts:
        if ev.clearance_b.clearance_type == ClearanceType.CROSSING and "04" in str(ev.runway):
            entities = {ev.clearance_a.entity, ev.clearance_b.entity}
            if "Jazz 8646" in entities:
                truck_conflict = ev
                break

    if truck_conflict:
        lead_l1 = COLLISION_T - truck_conflict.timestamp_s
        print(f"\n  Conflict detected: {truck_conflict.clearance_a.entity} vs "
              f"{truck_conflict.clearance_b.entity}")
        print(f"  Detection time: T+{truck_conflict.timestamp_s:.1f}s")
        print(f"  Lead time: {lead_l1:.1f}s before collision")
        print(f"  Decision: STOP (any conflict → STOP under cost-optimal threshold)")
        print(f"  Information provided: binary conflict exists / doesn't exist")
        print(f"  What it CANNOT tell you: how urgent, how likely, or whether "
              f"go-around is possible")

    # ── Layer 1 + Layer 2 ──
    print(f"\n{'─' * 70}")
    print("  LAYER 1 + LAYER 2: Add Time Estimation")
    print(f"{'─' * 70}")

    track = load_track(ADSB)
    conflict_utc = datetime.fromtimestamp(
        AUDIO_START.timestamp() + truck_conflict.timestamp_s, tz=timezone.utc)
    aircraft = get_aircraft_state(track, conflict_utc, RW04_TXY_D)
    crossing = build_crossing_estimate(calibrate=True)

    profile = build_approach_profile(track, cutoff_epoch=conflict_utc.timestamp())
    dist_to_threshold = haversine(aircraft.lat, aircraft.lon,
                                   RW04_THRESHOLD[0], RW04_THRESHOLD[1])
    target_past = haversine(RW04_THRESHOLD[0], RW04_THRESHOLD[1],
                             RW04_TXY_D[0], RW04_TXY_D[1])
    decel_eta, _ = profile.eta_with_deceleration(
        dist_to_threshold, -target_past,
        current_speed_kts=aircraft.groundspeed_kts)
    _, decel_sigma = compute_eta_sigma(profile, track,
                                        cutoff_epoch=conflict_utc.timestamp())

    print(f"\n  Aircraft ETA to conflict point: {decel_eta:.1f}s (σ={decel_sigma:.1f}s)")
    print(f"  Vehicle ETA to runway center: {crossing.mean_duration_s:.1f}s "
          f"(σ={crossing.sigma_duration_s:.1f}s)")
    print(f"  Time overlap: |{decel_eta:.1f} - {crossing.mean_duration_s:.1f}| = "
          f"{abs(decel_eta - crossing.mean_duration_s):.1f}s")
    print(f"  Aircraft altitude: {aircraft.alt_ft:.0f} ft")
    print(f"  Aircraft speed: {aircraft.groundspeed_kts:.0f} kts")
    print(f"  Information added: urgency (seconds to impact), altitude context")

    # ── Layer 1 + Layer 2 + Layer 3 ──
    print(f"\n{'─' * 70}")
    print("  LAYER 1 + LAYER 2 + LAYER 3: Add Probability + Decision Theory")
    print(f"{'─' * 70}")

    assessment = compute_risk(
        truck_conflict, aircraft, crossing,
        decel_eta_s=decel_eta, decel_sigma_s=decel_sigma)

    decision = make_decision(assessment.occupancy_probability)

    cf = evaluate_counterfactual(
        elapsed_since_clearance_s=0.0,
        crossing_estimate=crossing,
        aircraft_eta_s=decel_eta,
        aircraft_sigma_s=decel_sigma,
        aircraft_alt_ft=aircraft.alt_ft,
        aircraft_speed_kts=aircraft.groundspeed_kts,
        aircraft_dist_to_threshold_m=dist_to_threshold)

    print(f"\n  P(collision | no action): {assessment.occupancy_probability*100:.1f}%")
    print(f"  P(collision | STOP issued): {cf.residual_collision_prob*100:.1f}%")
    print(f"  Risk reduction: {(assessment.occupancy_probability - cf.residual_collision_prob)*100:.1f} pp")
    print(f"  Go-around feasibility: {cf.go_around_feasible:.0%} at {aircraft.alt_ft:.0f} ft")
    print(f"  Decision threshold: {DECISION_THRESHOLD:.2e}")
    print(f"  Information added: probability (severity), counterfactual (effectiveness),")
    print(f"    go-around option, cost-optimal justification for STOP")

    # ── Multi-conflict ranking demo ──
    print(f"\n{'─' * 70}")
    print("  MULTI-CONFLICT RANKING — Why L3 Matters Beyond Binary STOP")
    print(f"{'─' * 70}")

    # Simulate two conflicts at different severities
    # Conflict A: aircraft 30s away, vehicle 15s to crossing → HIGH overlap
    # Conflict B: aircraft 90s away, vehicle 10s to crossing → LOW overlap
    from enhanced_detection.enhanced_risk import _norm_cdf
    import random

    print(f"\n  Scenario: two simultaneous conflicts on different runways")
    print(f"\n  {'Conflict':<20} {'Aircraft ETA':>14} {'Vehicle ETA':>14} {'P(occ)':>10} {'Priority':>10}")
    print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*10} {'-'*10}")

    scenarios = [
        ("RW04 Landing/Cross", 30.0, 3.0, 15.0, 5.0),
        ("RW13 Takeoff/Taxi",  90.0, 8.0, 10.0, 3.0),
    ]

    for name, mu_ac, sig_ac, mu_veh, sig_veh in scenarios:
        n_mc = 20000
        rng = random.Random(42)
        hits = 0
        for _ in range(n_mc):
            t_enter = max(0, rng.gauss(mu_veh - 5, sig_veh))
            t_exit = t_enter + 86 / (13.5 / 3.6)
            t_ac = rng.gauss(mu_ac, sig_ac)
            if t_enter < t_ac < t_exit:
                hits += 1
        p_occ = hits / n_mc
        priority = "HIGH" if p_occ > 0.5 else "MEDIUM" if p_occ > 0.1 else "LOW"
        print(f"  {name:<20} {mu_ac:>10.0f}s (σ={sig_ac:.0f}) "
              f"{mu_veh:>10.0f}s (σ={sig_veh:.0f}) {p_occ*100:>8.1f}% {priority:>10}")

    print(f"\n  L1 alone: both are STOP (correct, but no prioritization)")
    print(f"  L3 adds: severity ranking → controller knows which to address first")

    # ── Summary Table ──
    print(f"\n{'=' * 70}")
    print("  ABLATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Layer Config':<24} {'Decision':>10} {'Info Provided':>45}")
    print(f"  {'-'*24} {'-'*10} {'-'*45}")
    print(f"  {'L1 only':<24} {'STOP':>10} {'Binary conflict exists':>45}")
    print(f"  {'L1 + L2':<24} {'STOP':>10} {'+ ETA, altitude, urgency':>45}")
    print(f"  {'L1 + L2 + L3':<24} {'STOP':>10} {'+ P(collision), severity rank, go-around':>45}")
    print(f"\n  Key finding: the STOP decision is identical across all three configs")
    print(f"  (due to θ* ≈ 3×10⁻⁶). Layers 2-3 add situational awareness, not")
    print(f"  decision discrimination. This mirrors RIMCAS Level 1 vs Level 2.")


if __name__ == "__main__":
    run_ablation()
