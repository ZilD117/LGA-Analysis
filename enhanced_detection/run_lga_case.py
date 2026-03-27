#!/usr/bin/env python3
"""
End-to-end demonstration of the Enhanced Conflict Detection System
on the 2026 LaGuardia Airport Runway Collision (Jazz 8646 vs Truck 1).

Runs all three layers and produces:
1. Console output showing the detection timeline
2. A comparison plot: old model (assumed σ=5) vs new model (calibrated σ)
"""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from enhanced_detection.clearance_parser import parse_clearances, ClearanceType
from enhanced_detection.runway_state import RunwayStateTracker, ConflictEvent
from enhanced_detection.aircraft_eta import load_track, get_aircraft_state, RW04_TXY_D, RW04_THRESHOLD, haversine
from enhanced_detection.ground_speed_prior import build_crossing_estimate
from enhanced_detection.enhanced_risk import compute_risk, compute_old_model_risk
from enhanced_detection.approach_profile import build_approach_profile, compute_eta_sigma
from enhanced_detection.decision_engine import (
    make_decision, evaluate_counterfactual, BayesianTracker,
    C_FALSE_ALARM_S, C_MISSED_DETECTION, DECISION_THRESHOLD,
)

TRANSCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "voice_data", "lga_case_study", "transcript.txt")
ADSB_PATH = os.path.join(os.path.dirname(__file__), "..", "surface_data", "lga_case_study", "flight_8646_track.csv")
AUDIO_START_UTC = datetime(2026, 3, 23, 3, 30, 0, tzinfo=timezone.utc)
COLLISION_UTC = datetime(2026, 3, 23, 3, 37, 28, tzinfo=timezone.utc)


def utc_from_offset(offset_s: float) -> datetime:
    return datetime.fromtimestamp(AUDIO_START_UTC.timestamp() + offset_s, tz=timezone.utc)


def find_fatal_conflict(conflicts):
    """Find the conflict involving Truck 1 and Jazz 8646 on RW04."""
    for c in conflicts:
        entities = {c.clearance_a.entity, c.clearance_b.entity}
        if "Truck 1" in entities and "Jazz 8646" in entities:
            return c
    # Fallback: first conflict involving Truck 1
    for c in conflicts:
        if c.clearance_b.entity == "Truck 1" or c.clearance_a.entity == "Truck 1":
            return c
    return conflicts[0] if conflicts else None


def run():
    print("=" * 70)
    print("  ENHANCED CONFLICT DETECTION SYSTEM — LGA CASE STUDY")
    print("  Air Canada Express 8646 vs Port Authority Fire Truck")
    print("=" * 70)

    # ── Layer 1: Clearance Parsing & Conflict Detection ──
    print("\n" + "─" * 70)
    print("  LAYER 1: Clearance Logic (Boolean Conflict Detection)")
    print("─" * 70)

    clearances = parse_clearances(TRANSCRIPT_PATH)
    tracker = RunwayStateTracker()

    print(f"\n  Parsed {len(clearances)} clearances from transcript.\n")

    key_types = {ClearanceType.LANDING, ClearanceType.CROSSING, ClearanceType.TAKEOFF}
    for c in clearances:
        conflict = tracker.feed(c)
        if c.clearance_type in key_types or conflict:
            flag = "  ◀ CONFLICT" if conflict else ""
            print(f"    {c}{flag}")
            if conflict:
                print(f"        ⚠ {conflict.description}")

    fatal = find_fatal_conflict(tracker.conflicts)
    if not fatal:
        print("\n  ERROR: Could not find the fatal conflict. Exiting.")
        return

    print(f"\n  ✓ Fatal conflict detected at {fatal.utc}")
    print(f"    {fatal.clearance_a.entity} LANDING vs {fatal.clearance_b.entity} CROSSING")

    # ── Layer 2: Time Estimation ──
    print("\n" + "─" * 70)
    print("  LAYER 2: Time Estimation (ADS-B + Ground Speed Prior)")
    print("─" * 70)

    track = load_track(ADSB_PATH)
    conflict_utc = utc_from_offset(fatal.timestamp_s)

    aircraft = get_aircraft_state(track, conflict_utc, RW04_TXY_D)
    crossing = build_crossing_estimate(calibrate=True)

    # Build approach speed profile CAUSALLY: only data before conflict time
    cutoff = conflict_utc.timestamp()
    profile = build_approach_profile(track, cutoff_epoch=cutoff)
    dist_to_threshold = haversine(aircraft.lat, aircraft.lon, RW04_THRESHOLD[0], RW04_THRESHOLD[1])
    target_past_threshold = haversine(RW04_THRESHOLD[0], RW04_THRESHOLD[1], RW04_TXY_D[0], RW04_TXY_D[1])
    decel_eta, speed_at_crossing = profile.eta_with_deceleration(
        dist_to_threshold, -target_past_threshold,
        current_speed_kts=aircraft.groundspeed_kts,
    )
    decel_bias, decel_sigma = compute_eta_sigma(profile, track, cutoff_epoch=cutoff)

    print(f"\n  Aircraft (Jazz 8646) at conflict time:")
    print(f"    Position: ({aircraft.lat:.5f}, {aircraft.lon:.5f})")
    print(f"    Altitude: {aircraft.alt_ft:.0f} ft")
    print(f"    Groundspeed: {aircraft.groundspeed_kts:.0f} kts")
    print(f"    Distance to Txy D: {aircraft.distance_to_point_m:.0f} m")
    print(f"    Constant-speed ETA: {aircraft.time_to_point_s:.1f} s (σ={aircraft.sigma_s:.1f}s)")
    print(f"    Decel-aware ETA:    {decel_eta:.1f} s (σ={decel_sigma:.1f}s, bias={decel_bias:+.1f}s)")
    print(f"    Speed at crossing:  {speed_at_crossing:.0f} kts (after rollout deceleration)")

    print(f"\n  Approach speed profile ({len(profile.profile)} points):")
    print(f"    From {profile.profile[0].groundspeed_kts:.0f} kts @ {profile.profile[0].distance_to_threshold_m:.0f}m "
          f"→ {profile.profile[-1].groundspeed_kts:.0f} kts @ {profile.profile[-1].distance_to_threshold_m:.0f}m")
    print(f"    Touchdown speed: {profile.touchdown_speed_kts:.0f} kts")
    print(f"    Deceleration accounts for +{decel_eta - aircraft.time_to_point_s:.1f}s vs constant-speed")

    print(f"\n  Vehicle (Truck 1) crossing estimate:")
    print(f"    Crossing distance: {crossing.crossing_dist_m:.0f} m")
    print(f"    Speed prior: {crossing.mean_speed_kmh:.0f} ± {crossing.sigma_speed_kmh:.0f} km/h")
    print(f"    Reaction delay: {crossing.reaction_delay_s:.0f} s")
    print(f"    Time to runway center: {crossing.mean_duration_s:.1f} s")
    print(f"    σ (from speed prior): {crossing.sigma_duration_s:.1f} s")

    overlap_const = abs(aircraft.time_to_point_s - crossing.mean_duration_s)
    overlap_decel = abs(decel_eta - crossing.mean_duration_s)
    print(f"\n  ⚠ Temporal overlap (constant-speed): {overlap_const:.1f} s")
    print(f"  ⚠ Temporal overlap (decel-aware):    {overlap_decel:.1f} s")
    if overlap_decel < 10:
        print(f"    → CRITICAL: Both entities reach crossing within {overlap_decel:.1f}s of each other")

    # ── Layer 3: Calibrated Risk ──
    print("\n" + "─" * 70)
    print("  LAYER 3: Calibrated Risk Model")
    print("─" * 70)

    assessment_const = compute_risk(fatal, aircraft, crossing)
    assessment_decel = compute_risk(
        fatal, aircraft, crossing,
        decel_eta_s=decel_eta, decel_sigma_s=decel_sigma,
    )
    print(f"\n{assessment_decel.summary()}")

    # ── Comparison: Three Models ──
    print("\n" + "─" * 70)
    print("  COMPARISON: Old Model vs Constant-Speed vs Deceleration-Aware")
    print("─" * 70)

    old_tg, old_rpn, old_rfw, old_cpn, old_cfw = compute_old_model_risk(
        mu_aircraft=aircraft.time_to_point_s,
        mu_vehicle=crossing.mean_duration_s,
        sigma=5.0,
        v1_kmh=aircraft.groundspeed_kts * 1.852,
        v2_kmh=crossing.mean_speed_kmh,
    )

    print(f"\n  {'Model':<32} {'Aircraft σ':>12} {'Aircraft ETA':>14} {'FW Risk':>10} {'Occupancy':>12}")
    print(f"  {'-'*32} {'-'*12} {'-'*14} {'-'*10} {'-'*12}")
    print(f"  {'Old (σ=5s assumed)':<32} {'5.0 s':>12} {aircraft.time_to_point_s:>11.1f} s  {max(old_cfw)*100:>8.2f}% {'N/A':>12}")
    print(f"  {'Enhanced (constant-speed)':<32} {aircraft.sigma_s:>9.1f} s  {aircraft.time_to_point_s:>11.1f} s  {max(assessment_const.cum_fw)*100:>8.2f}% {assessment_const.occupancy_probability*100:>10.1f}%")
    print(f"  {'Enhanced (decel-aware)':<32} {decel_sigma:>9.1f} s  {decel_eta:>11.1f} s  {max(assessment_decel.cum_fw)*100:>8.2f}% {assessment_decel.occupancy_probability*100:>10.1f}%")

    print(f"\n  KEY INSIGHT: The FW model measures point-arrival temporal coincidence.")
    print(f"  But what physically matters is whether the aircraft arrives while the")
    print(f"  vehicle is still ON the runway (occupancy window: T+{assessment_decel.vehicle_enter_s:.0f}s to T+{assessment_decel.vehicle_exit_s:.0f}s).")
    print(f"  The aircraft arrives at T+{decel_eta:.1f}s — well inside this window.")
    print(f"  Occupancy probability: {assessment_decel.occupancy_probability*100:.1f}% → {assessment_decel.alert_level}")

    # ── Decision Framework (3A) ──
    print(f"\n" + "─" * 70)
    print(f"  DECISION FRAMEWORK (Cost-Optimal Threshold)")
    print(f"─" * 70)
    print(f"\n  C_false_alarm   = {C_FALSE_ALARM_S:.0f}s delay    (cost of unnecessary STOP)")
    print(f"  C_missed_detect = {C_MISSED_DETECTION:.0e}    (cost of collision)")
    print(f"  Optimal threshold = C_FA / (C_FA + C_MD) = {DECISION_THRESHOLD:.2e}")
    print(f"  → Any non-negligible P(collision) triggers STOP")

    decision = make_decision(assessment_decel.occupancy_probability)
    print(f"\n  {decision}")

    # ── Counterfactual (3C) ──
    print(f"\n" + "─" * 70)
    print(f"  COUNTERFACTUAL: 'What if we issue STOP right now?'")
    print(f"─" * 70)

    dist_to_threshold = haversine(aircraft.lat, aircraft.lon, RW04_THRESHOLD[0], RW04_THRESHOLD[1])
    cf = evaluate_counterfactual(
        elapsed_since_clearance_s=0.0,
        crossing_estimate=crossing,
        aircraft_eta_s=decel_eta,
        aircraft_sigma_s=decel_sigma,
        aircraft_alt_ft=aircraft.alt_ft,
        aircraft_speed_kts=aircraft.groundspeed_kts,
        aircraft_dist_to_threshold_m=dist_to_threshold,
    )
    print(f"\n  Vehicle on runway: {cf.vehicle_on_runway}")
    print(f"  Can clear before aircraft: {cf.can_clear_in_time}")
    print(f"  Truck stopping distance: {cf.stopping_distance_m:.1f} m")
    print(f"  Time to stop (reaction + braking): {cf.time_to_stop_s:.1f} s")
    print(f"  Go-around feasibility: {cf.go_around_feasible:.0%} at {cf.aircraft_alt_ft:.0f}ft")
    print(f"  P(collision | no action): {assessment_decel.occupancy_probability*100:.1f}%")
    print(f"  P(collision | STOP issued): {cf.residual_collision_prob*100:.1f}%")
    print(f"  → {cf.recommendation}")

    # ── Bayesian Updating (3B) ──
    print(f"\n" + "─" * 70)
    print(f"  SEQUENTIAL BAYESIAN UPDATING (every 5s ADS-B)")
    print(f"─" * 70)

    bt = BayesianTracker(target_point=RW04_TXY_D, crossing_estimate=crossing)
    conflict_epoch = conflict_utc.timestamp()
    update_times = [conflict_epoch + dt for dt in range(0, 26, 5)]
    print(f"\n  {'T+Δ':>6}  {'Dist (m)':>10}  {'ETA (s)':>8}  {'σ (s)':>8}  {'P(occ)':>10}  {'Decision':>15}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*15}")
    for t_epoch in update_times:
        state = bt.update(track, t_epoch)
        dt = t_epoch - conflict_epoch
        act = "STOP" if state.decision and state.decision.should_stop else "monitor"
        print(f"  {dt:>5.0f}s  {state.distance_m:>10.0f}  {state.mu_eta_s:>7.1f}s  {state.sigma_eta_s:>7.1f}s  {state.occupancy_prob*100:>9.1f}%  {act:>15}")

    # ── Prevention Timeline ──
    collision_offset = (COLLISION_UTC - AUDIO_START_UTC).total_seconds()
    seconds_before = collision_offset - fatal.timestamp_s

    print(f"\n" + "─" * 70)
    print(f"  PREVENTION TIMELINE")
    print(f"─" * 70)
    print(f"\n  Crossing clearance issued:  {fatal.utc} (T+{fatal.timestamp_s:.0f}s)")
    print(f"  Collision occurred:          03:37:28 (T+{collision_offset:.0f}s)")
    print(f"  Time between:                {seconds_before:.0f} seconds")
    print(f"\n  Layer 1 CONFLICT alert fires at: T+{fatal.timestamp_s:.0f}s (INSTANT)")
    print(f"  → {seconds_before:.0f} seconds before collision")
    print(f"  → Enough time for controller to issue STOP command")
    print(f"     (Actual stop was issued at T+438s, only ~10s before impact)")

    print(f"\n{'='*70}")
    print(f"  CONCLUSION: Enhanced system would have detected the conflict")
    print(f"  {seconds_before:.0f} seconds before the collision — purely from voice-based")
    print(f"  clearance parsing, without requiring any vehicle transponder.")
    print(f"{'='*70}")

    # ── Generate comparison plot ──
    try:
        _generate_plot(assessment_decel, assessment_const, old_tg, old_cfw, old_rfw)
    except ImportError:
        print("\n  [matplotlib not available — skipping plot generation]")

    return assessment_decel


def _generate_plot(assessment_decel, assessment_const, old_tg, old_cfw, old_rfw):
    """Generate comparison plot: FW models + occupancy model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11), sharex=True,
                                          gridspec_kw={"height_ratios": [1, 1, 1.2]})
    fig.suptitle("Three-Model Risk Comparison — LGA Case Study",
                 fontsize=14, fontweight="bold", y=0.98)

    t_decel = assessment_decel.time_grid
    t_const = assessment_const.time_grid
    t_old = old_tg
    collision_t = (COLLISION_UTC - AUDIO_START_UTC).total_seconds() - 421.06

    # ── Panel 1: Instantaneous FW risk ──
    ax1.plot(t_decel, assessment_decel.r_fw, "#00CFE3", linewidth=2.5,
             label=f"Decel-aware (σ={assessment_decel.aircraft_sigma_s:.1f}s, "
                   f"ETA={assessment_decel.aircraft_eta_s:.1f}s)")
    ax1.plot(t_const, assessment_const.r_fw, "#55EF88", linewidth=2, linestyle="-.",
             label=f"Constant-speed (σ={assessment_const.aircraft_sigma_s:.1f}s, "
                   f"ETA={assessment_const.aircraft_eta_s:.1f}s)")
    ax1.plot(t_old, old_rfw, "#D4993D", linewidth=2, linestyle="--",
             label="Original (σ=5.0s both)")
    ax1.set_ylabel("Instantaneous Risk (FW)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Fenton-Wilkinson Point-Coincidence Risk", fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Cumulative FW risk ──
    ax2.plot(t_decel, assessment_decel.cum_fw, "#00CFE3", linewidth=2.5,
             label=f"Decel-aware: {max(assessment_decel.cum_fw)*100:.2f}%")
    ax2.plot(t_const, assessment_const.cum_fw, "#55EF88", linewidth=2, linestyle="-.",
             label=f"Constant-speed: {max(assessment_const.cum_fw)*100:.2f}%")
    ax2.plot(t_old, old_cfw, "#D4993D", linewidth=2, linestyle="--",
             label=f"Original: {max(old_cfw)*100:.2f}%")
    ax2.set_ylabel("Cumulative Risk (FW)")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_title("Cumulative FW Probability (point-arrival overlap)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Occupancy model ──
    mu_a = assessment_decel.aircraft_eta_s
    sig_a = assessment_decel.aircraft_sigma_s
    v_enter = assessment_decel.vehicle_enter_s
    v_exit = assessment_decel.vehicle_exit_s

    t_arr = np.array(t_decel)
    from scipy.stats import norm as scipy_norm
    aircraft_pdf = scipy_norm.pdf(t_arr, loc=mu_a, scale=sig_a)

    # Vehicle occupancy bar
    ax3.axvspan(v_enter, v_exit, alpha=0.18, color="#E13131",
                label=f"Vehicle on runway (T+{v_enter:.0f}s → T+{v_exit:.0f}s)")
    ax3.axvline(v_enter, color="#E13131", linewidth=1.5, linestyle="--", alpha=0.6)
    ax3.axvline(v_exit, color="#E13131", linewidth=1.5, linestyle="--", alpha=0.6)

    # Aircraft arrival PDF
    ax3.plot(t_arr, aircraft_pdf / max(aircraft_pdf), "#00CFE3", linewidth=2.5,
             label=f"Aircraft arrival PDF (μ={mu_a:.1f}s, σ={sig_a:.1f}s)")
    ax3.fill_between(t_arr, 0, aircraft_pdf / max(aircraft_pdf),
                      where=(t_arr >= v_enter) & (t_arr <= v_exit),
                      alpha=0.35, color="#00CFE3")

    # Occupancy probability annotation
    occ_prob = assessment_decel.occupancy_probability
    ax3.text(mu_a, 0.55,
             f"P(aircraft arrives while\nvehicle on runway)\n= {occ_prob*100:.1f}%",
             ha="center", va="center", fontsize=11, fontweight="bold",
             color="#00CFE3",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#00CFE3", alpha=0.9))

    ax3.set_ylabel("Normalized Density")
    ax3.set_xlabel("Time after crossing clearance (seconds)")
    ax3.set_title("Occupancy Model — P(aircraft arrives during vehicle runway occupancy)", fontsize=11)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.set_ylim(-0.05, 1.15)
    ax3.grid(True, alpha=0.3)

    # Collision line on all panels
    if 0 < collision_t < 60:
        for ax in (ax1, ax2, ax3):
            ax.axvline(collision_t, color="#E13131", alpha=0.5, linewidth=1.5, linestyle="-.")
        ax3.text(collision_t + 0.8, 0.95, f"Collision\nT+{collision_t:.0f}s",
                 color="#E13131", fontsize=8, va="top")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "lga_enhanced_vs_original.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved to: {out_path}")
    plt.close()


if __name__ == "__main__":
    run()
