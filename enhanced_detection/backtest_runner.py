"""
Backtest Runner — Generic incident pipeline for the enhanced detection system.

Accepts a standardized IncidentConfig and runs all three layers + decision engine.
Used by per-incident run_case.py scripts.
"""

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from enhanced_detection.aircraft_eta import (
    TrackPoint, AircraftState, haversine, get_aircraft_state, load_track, KTS_TO_MS,
)
from enhanced_detection.clearance_parser import Clearance, ClearanceType
from enhanced_detection.runway_state import RunwayStateTracker, ConflictEvent
from enhanced_detection.ground_speed_prior import CrossingEstimate, build_crossing_estimate
from enhanced_detection.enhanced_risk import compute_risk, compute_old_model_risk
from enhanced_detection.decision_engine import (
    make_decision, evaluate_counterfactual, BayesianTracker,
    C_FALSE_ALARM_S, C_MISSED_DETECTION, DECISION_THRESHOLD,
)
from enhanced_detection.synthetic_adsb import load_airport_nodes, generate_track


@dataclass
class EntityConfig:
    name: str
    path: List[str]
    segment_times: List[float]
    clearance_type: ClearanceType
    altitude_ft: float = 0.0
    track_csv: Optional[str] = None


@dataclass
class IncidentConfig:
    name: str
    airport_icao: str
    entity_a: EntityConfig
    entity_b: EntityConfig
    conflict_node: str
    crossing_entry: Tuple[float, float] = (0.0, 0.0)
    crossing_midpoint: Tuple[float, float] = (0.0, 0.0)
    crossing_exit: Tuple[float, float] = (0.0, 0.0)
    collision_offset_s: float = 0.0
    clearances: List[Clearance] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    rc_km: float = 0.03
    entity_a_wingspan_m: float = 40.0
    entity_b_width_m: float = 40.0
    t_max_sec: float = 120.0
    mean_speed_kmh: float = 20.0
    sigma_speed_kmh: float = 6.0


@dataclass
class BacktestResult:
    config: IncidentConfig
    conflicts: List[ConflictEvent]
    fatal_conflict: Optional[ConflictEvent]
    aircraft_state: Optional[AircraftState]
    crossing_estimate: Optional[CrossingEstimate]
    occupancy_prob: float = 0.0
    alert_level: str = ""
    eta_a_s: float = 0.0
    eta_b_s: float = 0.0
    detection_lead_s: float = 0.0
    decision_stop: bool = False
    counterfactual_prob: float = 0.0
    bayesian_history: list = field(default_factory=list)


def run_backtest(config: IncidentConfig) -> BacktestResult:
    """Run the full detection pipeline on an incident."""
    print("=" * 70)
    print(f"  BACKTEST: {config.name}")
    print(f"  {config.entity_a.name} vs {config.entity_b.name}")
    print("=" * 70)

    nodes = load_airport_nodes(config.airport_icao)

    # Load real ADS-B tracks when available, otherwise generate synthetic
    if config.entity_a.track_csv:
        track_a = load_track(config.entity_a.track_csv)
        print(f"  Loaded real ADS-B: {config.entity_a.name} — {len(track_a)} positions")
    else:
        track_a = generate_track(
            nodes, config.entity_a.path, config.entity_a.segment_times,
            config.start_time, altitude_ft=config.entity_a.altitude_ft,
            source="synthetic",
        )
    if config.entity_b.track_csv:
        track_b = load_track(config.entity_b.track_csv)
        print(f"  Loaded real ADS-B: {config.entity_b.name} — {len(track_b)} positions")
    else:
        track_b = generate_track(
            nodes, config.entity_b.path, config.entity_b.segment_times,
            config.start_time, altitude_ft=config.entity_b.altitude_ft,
            source="synthetic",
        )

    # ── Layer 1: Conflict Detection ──
    print(f"\n{'─' * 70}")
    print(f"  LAYER 1: Clearance Logic")
    print(f"{'─' * 70}")

    tracker = RunwayStateTracker()
    for c in config.clearances:
        conflict = tracker.feed(c)
        if conflict:
            print(f"    CONFLICT @ {c.utc}: {conflict.description}")

    fatal = _find_conflict(tracker.conflicts, config.entity_a.name, config.entity_b.name)
    if not fatal and tracker.conflicts:
        fatal = tracker.conflicts[0]

    if fatal:
        print(f"\n  Fatal conflict detected: {fatal.clearance_a.entity} vs {fatal.clearance_b.entity}")
        print(f"    at {fatal.utc} on surface {fatal.runway}")
    else:
        print(f"\n  No conflict detected from clearances.")
        print(f"  (Proceeding with synthetic conflict for risk analysis)")
        fatal = _make_synthetic_conflict(config)

    # ── Layer 2: Time Estimation ──
    print(f"\n{'─' * 70}")
    print(f"  LAYER 2: Time Estimation")
    print(f"{'─' * 70}")

    conflict_node_coords = nodes.get(config.conflict_node, (0, 0))
    conflict_epoch = config.start_time.timestamp() + fatal.timestamp_s

    state_a = get_aircraft_state(
        track_a,
        datetime.fromtimestamp(conflict_epoch, tz=timezone.utc),
        conflict_node_coords,
    )

    state_b = get_aircraft_state(
        track_b,
        datetime.fromtimestamp(conflict_epoch, tz=timezone.utc),
        conflict_node_coords,
    )

    print(f"\n  {config.entity_a.name}:")
    print(f"    Distance to conflict: {state_a.distance_to_point_m:.0f} m")
    print(f"    ETA: {state_a.time_to_point_s:.1f} s (σ={state_a.sigma_s:.1f}s)")
    print(f"    Speed: {state_a.groundspeed_kts:.0f} kts")

    print(f"\n  {config.entity_b.name}:")
    print(f"    Distance to conflict: {state_b.distance_to_point_m:.0f} m")
    print(f"    ETA: {state_b.time_to_point_s:.1f} s (σ={state_b.sigma_s:.1f}s)")
    print(f"    Speed: {state_b.groundspeed_kts:.0f} kts")

    # Build crossing estimate for entity B
    crossing = build_crossing_estimate(
        mean_speed_kmh=config.mean_speed_kmh,
        sigma_speed_kmh=config.sigma_speed_kmh,
        reaction_delay_s=0.0,
        sigma_reaction_s=0.5,
        calibrate=False,
        entry=config.crossing_entry,
        midpoint=config.crossing_midpoint,
        exit_pt=config.crossing_exit,
    )

    print(f"\n  Crossing geometry:")
    print(f"    Distance: {crossing.crossing_dist_m:.0f} m")
    print(f"    Speed prior: {crossing.mean_speed_kmh:.1f} ± {crossing.sigma_speed_kmh:.1f} km/h")

    # ── Layer 3: Risk Assessment ──
    print(f"\n{'─' * 70}")
    print(f"  LAYER 3: Risk Assessment")
    print(f"{'─' * 70}")

    assessment = compute_risk(
        fatal, state_a, crossing,
        rc_km=config.rc_km,
        decel_eta_s=state_a.time_to_point_s,
        decel_sigma_s=state_a.sigma_s,
        t_max_sec=config.t_max_sec,
    )
    print(f"\n{assessment.summary()}")

    # ── Decision Framework ──
    print(f"\n{'─' * 70}")
    print(f"  DECISION FRAMEWORK")
    print(f"{'─' * 70}")

    decision = make_decision(assessment.occupancy_probability)
    print(f"\n  {decision}")

    # ── Counterfactual ──
    print(f"\n{'─' * 70}")
    print(f"  COUNTERFACTUAL: What if STOP issued at detection?")
    print(f"{'─' * 70}")

    from enhanced_detection.aircraft_eta import haversine as _hav
    _dist_thresh = _hav(state_a.lat, state_a.lon, conflict_node_coords[0], conflict_node_coords[1])
    cf = evaluate_counterfactual(
        elapsed_since_clearance_s=0.0,
        crossing_estimate=crossing,
        aircraft_eta_s=state_a.time_to_point_s,
        aircraft_sigma_s=state_a.sigma_s,
        aircraft_alt_ft=state_a.alt_ft,
        aircraft_speed_kts=state_a.groundspeed_kts,
        aircraft_dist_to_threshold_m=_dist_thresh,
    )
    print(f"\n  P(collision | no action): {assessment.occupancy_probability*100:.1f}%")
    print(f"  P(collision | STOP):      {cf.residual_collision_prob*100:.1f}%")
    print(f"  Go-around: {cf.go_around_feasible:.0%} at {cf.aircraft_alt_ft:.0f}ft")
    print(f"  Recommendation: {cf.recommendation}")

    # ── Bayesian Updates ──
    print(f"\n{'─' * 70}")
    print(f"  BAYESIAN UPDATING")
    print(f"{'─' * 70}")

    bt = BayesianTracker(target_point=conflict_node_coords, crossing_estimate=crossing)
    update_interval = 10.0 if config.collision_offset_s > 60 else 5.0
    n_updates = min(16, int(config.collision_offset_s / update_interval) + 1)
    bay_history = []

    print(f"\n  {'T+Δ':>6}  {'Dist':>8}  {'ETA':>8}  {'σ':>6}  {'P(occ)':>8}  {'Decision':>10}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*10}")

    for i in range(n_updates):
        dt = i * update_interval
        t_epoch = conflict_epoch + dt
        bs = bt.update(track_a, t_epoch)
        act = "STOP" if bs.decision and bs.decision.should_stop else "monitor"
        print(f"  {dt:>5.0f}s  {bs.distance_m:>7.0f}m  {bs.mu_eta_s:>6.1f}s  {bs.sigma_eta_s:>5.1f}s  {bs.occupancy_prob*100:>7.1f}%  {act:>10}")
        bay_history.append(bs)

    # ── Prevention Timeline ──
    print(f"\n{'─' * 70}")
    print(f"  PREVENTION TIMELINE")
    print(f"{'─' * 70}")

    det_lead = config.collision_offset_s
    print(f"\n  Conflict detected:  T+0s")
    print(f"  Collision occurs:   T+{det_lead:.0f}s")
    print(f"  Detection lead:     {det_lead:.0f} seconds")
    if det_lead > 10:
        print(f"  → Sufficient time for STOP command")
    else:
        print(f"  → Very tight window — immediate automated response needed")

    print(f"\n{'=' * 70}")

    return BacktestResult(
        config=config,
        conflicts=tracker.conflicts,
        fatal_conflict=fatal,
        aircraft_state=state_a,
        crossing_estimate=crossing,
        occupancy_prob=assessment.occupancy_probability,
        alert_level=assessment.alert_level,
        eta_a_s=state_a.time_to_point_s,
        eta_b_s=state_b.time_to_point_s,
        detection_lead_s=det_lead,
        decision_stop=decision.should_stop,
        counterfactual_prob=cf.residual_collision_prob,
        bayesian_history=bay_history,
    )


def generate_comparison_plot(result: BacktestResult, output_path: str):
    """Generate a risk comparison plot for the backtest result."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [matplotlib not available — skipping plot]")
        return

    assessment = compute_risk(
        result.fatal_conflict,
        result.aircraft_state,
        result.crossing_estimate,
        rc_km=result.config.rc_km,
        decel_eta_s=result.eta_a_s,
        decel_sigma_s=result.aircraft_state.sigma_s,
        t_max_sec=result.config.t_max_sec,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Risk Analysis — {result.config.name}", fontsize=14, fontweight="bold")

    t = assessment.time_grid

    # Panel 1: Instantaneous FW risk
    ax1.plot(t, assessment.r_fw, "#00CFE3", linewidth=2.5,
             label=f"FW risk (σ_a={result.aircraft_state.sigma_s:.1f}s)")
    ax1.set_ylabel("Instantaneous Risk (FW)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title("Fenton-Wilkinson Point-Coincidence Risk", fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Cumulative FW risk
    ax2.plot(t, assessment.cum_fw, "#00CFE3", linewidth=2.5,
             label=f"Cumulative FW: {max(assessment.cum_fw)*100:.2f}%")
    ax2.set_ylabel("Cumulative Risk")
    ax2.set_xlabel("Time after conflict detection (seconds)")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_title(
        f"Cumulative Risk — Occupancy P={assessment.occupancy_probability*100:.1f}%",
        fontsize=11,
    )
    ax2.grid(True, alpha=0.3)

    # Collision marker
    col_t = result.config.collision_offset_s
    if 0 < col_t < result.config.t_max_sec:
        for ax in (ax1, ax2):
            ax.axvline(col_t, color="#E13131", linewidth=1.5, linestyle="-.", alpha=0.6)
        ax2.text(col_t + 1, max(assessment.cum_fw) * 0.5,
                 f"Collision\nT+{col_t:.0f}s", color="#E13131", fontsize=9)

    # Bayesian overlay on panel 2
    if result.bayesian_history:
        bay_t = [i * 5 for i in range(len(result.bayesian_history))]
        bay_p = [bs.occupancy_prob for bs in result.bayesian_history]
        ax2_twin = ax2.twinx()
        ax2_twin.plot(bay_t, bay_p, "#55EF88", linewidth=2, linestyle="--",
                      marker="o", markersize=4, label="Bayesian P(occ)")
        ax2_twin.set_ylabel("Occupancy Probability", color="#55EF88")
        ax2_twin.set_ylim(-0.05, 1.1)
        ax2_twin.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved to: {output_path}")


def _find_conflict(
    conflicts: List[ConflictEvent], name_a: str, name_b: str,
) -> Optional[ConflictEvent]:
    """Find a conflict involving both named entities."""
    for c in conflicts:
        entities = {c.clearance_a.entity, c.clearance_b.entity}
        if name_a in entities and name_b in entities:
            return c
    for c in conflicts:
        entities = {c.clearance_a.entity, c.clearance_b.entity}
        if name_a in entities or name_b in entities:
            return c
    return None


def _make_synthetic_conflict(config: IncidentConfig) -> ConflictEvent:
    """Create a synthetic conflict event when clearance parsing doesn't produce one."""
    t = config.collision_offset_s * 0.5
    clearance_a = Clearance(
        timestamp_s=0.0,
        utc=config.start_time.strftime("%H:%M:%S"),
        entity=config.entity_a.name,
        clearance_type=config.entity_a.clearance_type,
        runway=config.conflict_node[:6] if "Rwy" in config.conflict_node else None,
    )
    clearance_b = Clearance(
        timestamp_s=t,
        utc="",
        entity=config.entity_b.name,
        clearance_type=config.entity_b.clearance_type,
        runway=config.conflict_node[:6] if "Rwy" in config.conflict_node else None,
    )
    return ConflictEvent(
        timestamp_s=t,
        utc=f"T+{t:.0f}s",
        runway=config.conflict_node,
        clearance_a=clearance_a,
        clearance_b=clearance_b,
        description=f"{config.entity_a.name} vs {config.entity_b.name} at {config.conflict_node}",
    )
