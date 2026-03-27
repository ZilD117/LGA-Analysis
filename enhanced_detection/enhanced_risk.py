"""
Enhanced Risk Model — Calibrated risk combining all three layers.

Layer 1 (Boolean): Clearance conflict from runway_state
Layer 2 (Time):    Aircraft ETA from ADS-B + Vehicle crossing estimate from speed prior
Layer 3 (Risk):    Probability of co-location using Gaussian arrival distributions

Uses Petri-Net (PN) and Fenton-Wilkinson (FW) formulas from the original model,
but with measured / empirically-grounded sigmas instead of assumed values.

Occupancy model uses correlated Monte Carlo: vehicle enter and exit times share
the same speed draw, eliminating the independence assumption of the previous model.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from enhanced_detection.aircraft_eta import AircraftState
from enhanced_detection.ground_speed_prior import CrossingEstimate
from enhanced_detection.runway_state import ConflictEvent

SEC_PER_HOUR = 3600.0
KMH_TO_MS = 1.0 / 3.6


@dataclass
class RiskAssessment:
    conflict: ConflictEvent
    aircraft_state: AircraftState
    crossing_estimate: CrossingEstimate
    time_grid: List[float]
    r_pn: List[float]
    r_fw: List[float]
    cum_pn: List[float]
    cum_fw: List[float]
    alert_level: str
    time_to_conflict_s: float
    peak_risk_time_s: float
    aircraft_eta_s: float = 0.0
    aircraft_sigma_s: float = 0.0
    decel_aware: bool = False
    occupancy_probability: float = 0.0
    vehicle_enter_s: float = 0.0
    vehicle_exit_s: float = 0.0

    def summary(self) -> str:
        max_cum = max(self.cum_fw) if self.cum_fw else 0
        eta_label = "decel-aware ADS-B" if self.decel_aware else "constant-speed ADS-B"
        lines = [
            f"{'='*60}",
            f"RISK ASSESSMENT — {self.conflict.runway}",
            f"{'='*60}",
            f"  Conflict: {self.conflict.clearance_a.entity} ({self.conflict.clearance_a.clearance_type.value}) "
            f"vs {self.conflict.clearance_b.entity} ({self.conflict.clearance_b.clearance_type.value})",
            f"  Detected at: {self.conflict.utc}",
            f"",
            f"  Aircraft: {self.aircraft_state.groundspeed_kts:.0f} kts, "
            f"{self.aircraft_state.distance_to_point_m:.0f}m away",
            f"  Aircraft ETA to crossing: {self.aircraft_eta_s:.1f}s "
            f"(σ={self.aircraft_sigma_s:.1f}s) — {eta_label}",
            f"",
            f"  Vehicle crossing distance: {self.crossing_estimate.crossing_dist_m:.0f}m",
            f"  Vehicle time to runway center: {self.crossing_estimate.mean_duration_s:.1f}s "
            f"(σ={self.crossing_estimate.sigma_duration_s:.1f}s) — from speed prior",
            f"  Vehicle on runway: T+{self.vehicle_enter_s:.1f}s to T+{self.vehicle_exit_s:.1f}s "
            f"({self.vehicle_exit_s - self.vehicle_enter_s:.0f}s occupancy)",
            f"",
            f"  FW cumulative probability:    {max_cum:.4f} ({max_cum*100:.2f}%)",
            f"  Occupancy collision prob:     {self.occupancy_probability:.4f} ({self.occupancy_probability*100:.2f}%)",
            f"  Alert level: {self.alert_level}",
            f"  Time to conflict: {self.time_to_conflict_s:.1f}s",
        ]
        return "\n".join(lines)


def _norm_pdf(x: float, mu: float, sigma: float) -> float:
    """Gaussian probability density function."""
    if sigma < 1e-9:
        return 1.0 if abs(x - mu) < 0.5 else 0.0
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))


def _norm_cdf(x: float, mu: float, sigma: float) -> float:
    """Gaussian cumulative distribution function."""
    if sigma < 1e-9:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2))))


CRJ900_WINGSPAN_M = 26.0
TRUCK_WIDTH_M = 3.0
TRUCK_LENGTH_M = 10.0

RC_KM = (CRJ900_WINGSPAN_M / 2 + TRUCK_WIDTH_M / 2) / 1000.0  # ~0.0145 km


def compute_risk(
    conflict: ConflictEvent,
    aircraft_state: AircraftState,
    crossing_estimate: CrossingEstimate,
    epsilon_sec: float = None,
    rc_km: float = None,
    dt_sec: float = 0.5,
    t_max_sec: float = 60.0,
    decel_eta_s: float = None,
    decel_sigma_s: float = None,
) -> RiskAssessment:
    """
    Compute calibrated risk for a conflict event.

    Time is relative to the crossing clearance moment.
    Aircraft Gaussian: N(mu=aircraft_ETA, sigma=sigma_adsb)
    Vehicle Gaussian:  N(mu=crossing_estimate.mean_duration, sigma=crossing_estimate.sigma_duration)

    Physical constants are derived from aircraft geometry:
      rc_km: collision capture radius = (wingspan/2 + truck_width/2)
      epsilon_sec: time the aircraft takes to traverse the truck's length
    """
    if rc_km is None:
        rc_km = RC_KM
    if epsilon_sec is None:
        gs_ms = aircraft_state.groundspeed_kts * 0.514444
        epsilon_sec = TRUCK_LENGTH_M / max(gs_ms, 1.0)
    use_decel = decel_eta_s is not None
    mu_aircraft = decel_eta_s if use_decel else aircraft_state.time_to_point_s
    sigma_aircraft = decel_sigma_s if decel_sigma_s is not None else aircraft_state.sigma_s

    mu_vehicle = crossing_estimate.mean_duration_s
    sigma_vehicle = crossing_estimate.sigma_duration_s

    n_steps = int(t_max_sec / dt_sec) + 1
    time_grid = [i * dt_sec for i in range(n_steps)]

    r_pn = []
    r_fw = []

    # FW inverse-speed factor: use aircraft and vehicle speeds
    v_aircraft_kmh = aircraft_state.groundspeed_kts * 1.852
    v_vehicle_kmh = crossing_estimate.mean_speed_kmh
    e_inv_sec_per_km = 0.5 * (
        SEC_PER_HOUR / max(v_aircraft_kmh, 1e-3)
        + SEC_PER_HOUR / max(v_vehicle_kmh, 1e-3)
    )

    for t in time_grid:
        f_a = _norm_pdf(t, mu_aircraft, sigma_aircraft)
        f_v = _norm_pdf(t, mu_vehicle, sigma_vehicle)

        r_pn.append(2.0 * epsilon_sec * f_a * f_v)
        r_fw.append(2.0 * rc_km * e_inv_sec_per_km * f_a * f_v)

    cum_pn = []
    cum_fw = []
    acc_pn = 0.0
    acc_fw = 0.0
    for rp, rf in zip(r_pn, r_fw):
        acc_pn += rp * dt_sec
        acc_fw += rf * dt_sec
        cum_pn.append(acc_pn)
        cum_fw.append(acc_fw)

    peak_idx = r_fw.index(max(r_fw))
    peak_time = time_grid[peak_idx]

    # ── Occupancy-based collision probability (correlated Monte Carlo) ──
    #
    # For each Monte Carlo sample we draw ONE speed for the vehicle, which
    # determines BOTH T_enter and T_exit (they are correlated through speed).
    #   T_enter = reaction_delay_draw
    #   T_exit  = T_enter + crossing_dist / speed_draw
    #   T_aircraft ~ N(mu_aircraft, sigma_aircraft)
    #
    # P(collision) = E[ P(T_enter < T_aircraft < T_exit) ]
    #              ≈ (1/N) * sum( 1{T_enter < T_aircraft_draw < T_exit} )

    mean_speed_ms = crossing_estimate.mean_speed_kmh / 3.6
    sigma_speed_ms = crossing_estimate.sigma_speed_kmh / 3.6
    full_crossing_s = crossing_estimate.crossing_dist_m / max(mean_speed_ms, 0.1)

    vehicle_enter_s = crossing_estimate.reaction_delay_s
    vehicle_exit_s = crossing_estimate.reaction_delay_s + full_crossing_s

    use_lognormal = crossing_estimate.lognormal and crossing_estimate.sigma_ln > 0
    mu_ln = crossing_estimate.mu_ln
    sigma_ln = crossing_estimate.sigma_ln

    N_MC = 50_000
    rng = random.Random(42)
    hits = 0
    for _ in range(N_MC):
        reaction_draw = max(0.0, rng.gauss(
            crossing_estimate.reaction_delay_s,
            crossing_estimate.sigma_reaction_s,
        ))

        if use_lognormal:
            speed_draw_kmh = math.exp(rng.gauss(mu_ln, sigma_ln))
            speed_draw_ms = speed_draw_kmh / 3.6
        else:
            speed_draw_ms = max(0.5, rng.gauss(mean_speed_ms, sigma_speed_ms))

        t_enter = reaction_draw
        t_exit = t_enter + crossing_estimate.crossing_dist_m / speed_draw_ms

        t_aircraft = rng.gauss(mu_aircraft, max(sigma_aircraft, 0.01))
        if t_enter < t_aircraft < t_exit:
            hits += 1

    occupancy_prob = hits / N_MC

    if occupancy_prob > 0.8:
        alert_level = "CRITICAL"
    elif occupancy_prob > 0.5:
        alert_level = "HIGH"
    elif occupancy_prob > 0.2:
        alert_level = "ELEVATED"
    else:
        alert_level = "LOW"

    time_to_conflict = min(mu_aircraft, mu_vehicle)

    return RiskAssessment(
        conflict=conflict,
        aircraft_state=aircraft_state,
        crossing_estimate=crossing_estimate,
        time_grid=time_grid,
        r_pn=r_pn,
        r_fw=r_fw,
        cum_pn=cum_pn,
        cum_fw=cum_fw,
        alert_level=alert_level,
        time_to_conflict_s=time_to_conflict,
        peak_risk_time_s=peak_time,
        aircraft_eta_s=mu_aircraft,
        aircraft_sigma_s=sigma_aircraft,
        decel_aware=use_decel,
        occupancy_probability=occupancy_prob,
        vehicle_enter_s=vehicle_enter_s,
        vehicle_exit_s=vehicle_exit_s,
    )


def compute_old_model_risk(
    mu_aircraft: float,
    mu_vehicle: float,
    sigma: float = 5.0,
    epsilon_sec: float = 1.0,
    rc_km: float = 0.075,
    v1_kmh: float = 250.0,
    v2_kmh: float = 20.0,
    dt_sec: float = 0.5,
    t_max_sec: float = 60.0,
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """Compute risk with the OLD model (assumed sigma=5 for both entities)."""
    n_steps = int(t_max_sec / dt_sec) + 1
    time_grid = [i * dt_sec for i in range(n_steps)]

    e_inv = 0.5 * (SEC_PER_HOUR / max(v1_kmh, 1e-3) + SEC_PER_HOUR / max(v2_kmh, 1e-3))

    r_pn = []
    r_fw = []
    for t in time_grid:
        f_a = _norm_pdf(t, mu_aircraft, sigma)
        f_v = _norm_pdf(t, mu_vehicle, sigma)
        r_pn.append(2.0 * epsilon_sec * f_a * f_v)
        r_fw.append(2.0 * rc_km * e_inv * f_a * f_v)

    cum_pn = []
    cum_fw = []
    acc_pn = 0.0
    acc_fw = 0.0
    for rp, rf in zip(r_pn, r_fw):
        acc_pn += rp * dt_sec
        acc_fw += rf * dt_sec
        cum_pn.append(acc_pn)
        cum_fw.append(acc_fw)

    return time_grid, r_pn, r_fw, cum_pn, cum_fw
