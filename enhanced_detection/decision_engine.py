"""
Decision Engine — Turns risk estimates into actionable decisions.

Three capabilities:
  1. Cost-based decision threshold  (3A)
  2. Sequential Bayesian updating   (3B)
  3. Counterfactual "what if we act" (3C)
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from enhanced_detection.aircraft_eta import (
    AircraftState, TrackPoint, haversine, KTS_TO_MS,
    get_aircraft_state, _extrapolate_forward,
)
from enhanced_detection.ground_speed_prior import CrossingEstimate


# ── 3A: Cost-based decision framework ─────────────────────────────

C_FALSE_ALARM_S = 30.0    # cost of a false STOP: ~30 s runway delay
C_MISSED_DETECTION = 1e7  # cost of a collision: lives + aircraft + closure

DECISION_THRESHOLD = C_FALSE_ALARM_S / (C_FALSE_ALARM_S + C_MISSED_DETECTION)


@dataclass
class Decision:
    should_stop: bool
    occupancy_prob: float
    threshold: float
    margin: str
    reason: str

    def __repr__(self):
        action = "ISSUE STOP" if self.should_stop else "MONITOR"
        return (
            f"{action}: P(collision)={self.occupancy_prob:.4f} "
            f"{'>' if self.should_stop else '<='} "
            f"threshold={self.threshold:.2e} — {self.reason}"
        )


def make_decision(
    occupancy_prob: float,
    counterfactual_prob: Optional[float] = None,
    threshold: float = DECISION_THRESHOLD,
) -> Decision:
    """
    Decide whether to issue STOP based on cost-ratio threshold.

    Since C_MD >> C_FA, the threshold is extremely low (~3e-6).
    Any non-negligible occupancy probability triggers STOP.
    If a counterfactual is available, also verify that acting helps.
    """
    exceeds = occupancy_prob > threshold

    if counterfactual_prob is not None and counterfactual_prob >= occupancy_prob:
        return Decision(
            should_stop=False,
            occupancy_prob=occupancy_prob,
            threshold=threshold,
            margin="N/A",
            reason="STOP would not reduce risk (vehicle already on runway, cannot stop in time)",
        )

    if exceeds:
        return Decision(
            should_stop=True,
            occupancy_prob=occupancy_prob,
            threshold=threshold,
            margin=f"{occupancy_prob / max(threshold, 1e-12):.0f}x above threshold",
            reason="P(collision) exceeds cost-optimal threshold; issue STOP immediately",
        )

    return Decision(
        should_stop=False,
        occupancy_prob=occupancy_prob,
        threshold=threshold,
        margin=f"{occupancy_prob / max(threshold, 1e-12):.1f}x of threshold",
        reason="Risk below decision threshold; continue monitoring",
    )


# ── 3B: Sequential Bayesian updating ──────────────────────────────

@dataclass
class BayesianState:
    """Tracks aircraft state estimates that get refined with each ADS-B update."""
    mu_eta_s: float
    sigma_eta_s: float
    distance_m: float
    speed_kts: float
    timestamp_epoch: float
    occupancy_prob: float = 0.0
    decision: Optional[Decision] = None


@dataclass
class BayesianTracker:
    """
    Sequential risk estimator: re-computes risk each time a new ADS-B
    report arrives, with uncertainty shrinking as the aircraft gets closer.
    """
    target_point: Tuple[float, float]
    crossing_estimate: CrossingEstimate
    history: List[BayesianState] = field(default_factory=list)

    def update(
        self,
        track: List[TrackPoint],
        query_epoch: float,
        decel_eta_fn=None,
    ) -> BayesianState:
        """
        Process a new ADS-B timestamp, recompute ETA and risk.

        decel_eta_fn: optional callable(dist_m, speed_kts) -> (eta_s, sigma_s)
                      for deceleration-aware ETA. If None, uses constant-speed.
        """
        from datetime import datetime, timezone
        query_dt = datetime.fromtimestamp(query_epoch, tz=timezone.utc)
        state = get_aircraft_state(track, query_dt, self.target_point)

        if decel_eta_fn is not None:
            eta_s, sigma_s = decel_eta_fn(state.distance_to_point_m, state.groundspeed_kts)
        else:
            eta_s = state.time_to_point_s
            sigma_s = state.sigma_s

        # Bayesian shrinkage: if we have a prior, fuse it with the new measurement
        if self.history:
            prior = self.history[-1]
            dt_elapsed = query_epoch - prior.timestamp_epoch
            prior_mu_now = prior.mu_eta_s - dt_elapsed
            prior_sigma_now = prior.sigma_eta_s

            if prior_sigma_now > 0.01 and sigma_s > 0.01:
                # Kalman-style fusion of prior prediction and new measurement
                k = prior_sigma_now ** 2 / (prior_sigma_now ** 2 + sigma_s ** 2)
                fused_mu = prior_mu_now + k * (eta_s - prior_mu_now)
                fused_sigma = math.sqrt((1 - k) * prior_sigma_now ** 2)
            else:
                fused_mu = eta_s
                fused_sigma = sigma_s
        else:
            fused_mu = eta_s
            fused_sigma = sigma_s

        occ_prob = _mc_occupancy(
            fused_mu, fused_sigma, self.crossing_estimate, query_epoch,
        )

        decision = make_decision(occ_prob)

        bs = BayesianState(
            mu_eta_s=fused_mu,
            sigma_eta_s=fused_sigma,
            distance_m=state.distance_to_point_m,
            speed_kts=state.groundspeed_kts,
            timestamp_epoch=query_epoch,
            occupancy_prob=occ_prob,
            decision=decision,
        )
        self.history.append(bs)
        return bs


def _mc_occupancy(
    mu_aircraft: float,
    sigma_aircraft: float,
    ce: CrossingEstimate,
    query_epoch: float,
    n_samples: int = 20_000,
) -> float:
    """Quick Monte Carlo occupancy probability for Bayesian updates."""
    mean_speed_ms = ce.mean_speed_kmh / 3.6
    use_lognormal = ce.lognormal and ce.sigma_ln > 0
    rng = random.Random(int(query_epoch * 1000) % (2**31))
    hits = 0
    for _ in range(n_samples):
        reaction = max(0.0, rng.gauss(ce.reaction_delay_s, ce.sigma_reaction_s))
        if use_lognormal:
            speed = math.exp(rng.gauss(ce.mu_ln, ce.sigma_ln)) / 3.6
        else:
            speed = max(0.5, rng.gauss(mean_speed_ms, ce.sigma_speed_kmh / 3.6))
        t_enter = reaction
        t_exit = t_enter + ce.crossing_dist_m / speed
        t_ac = rng.gauss(mu_aircraft, max(sigma_aircraft, 0.01))
        if t_enter < t_ac < t_exit:
            hits += 1
    return hits / n_samples


# ── 3C: Counterfactual — "What if we issue STOP right now?" ───────

TRUCK_DECEL_MS2 = 3.0     # emergency braking decel for a fire truck
TRUCK_REACTION_S = 1.5    # driver reaction time to radio STOP command

# Go-around parameters (transport category aircraft)
GA_PILOT_REACTION_S = 3.0   # pilot processing tower "go around" command
GA_TOGA_ACCEL_S = 2.0        # spool-up time for TOGA thrust
GA_CLIMB_GRADIENT = 0.032    # 3.2% min climb gradient (FAR 25.121)
GA_MIN_ALTITUDE_FT = 50.0    # below this, go-around is physically impossible
GA_DECISION_HEIGHT_FT = 200.0  # CAT I ILS decision altitude


def go_around_feasibility(
    altitude_ft: float,
    groundspeed_kts: float,
    distance_to_threshold_m: float,
) -> Tuple[float, float]:
    """
    Estimate go-around feasibility given current aircraft state.

    Returns (probability_success, time_needed_s).

    Physics: after tower command, pilot needs reaction time + spool-up time.
    During this period the aircraft continues descending on the glideslope
    (~3 degrees, ~700 ft/min descent rate at approach speed). If the altitude
    consumed during this delay drops the aircraft below ~50 ft, a go-around
    is no longer safely executable.

    Above decision height (200ft): high probability (~0.98)
    Between 50-200ft: decreasing linearly
    Below 50ft: near zero (committed to landing)
    """
    total_delay_s = GA_PILOT_REACTION_S + GA_TOGA_ACCEL_S  # ~5s

    descent_rate_fpm = groundspeed_kts * 100 * math.tan(math.radians(3.0)) if groundspeed_kts > 0 else 700
    descent_rate_fps = descent_rate_fpm / 60.0
    altitude_lost_ft = descent_rate_fps * total_delay_s

    residual_alt_ft = altitude_ft - altitude_lost_ft

    if residual_alt_ft <= 0 or altitude_ft < GA_MIN_ALTITUDE_FT:
        prob = 0.02  # not zero — edge-case saves happen
    elif altitude_ft >= GA_DECISION_HEIGHT_FT:
        prob = 0.98
    else:
        frac = (altitude_ft - GA_MIN_ALTITUDE_FT) / (GA_DECISION_HEIGHT_FT - GA_MIN_ALTITUDE_FT)
        prob = 0.02 + 0.96 * frac

    gs_ms = groundspeed_kts * KTS_TO_MS
    time_to_threshold = distance_to_threshold_m / max(gs_ms, 1.0)
    time_needed = total_delay_s + time_to_threshold * 0.3  # partial overshoot

    return prob, time_needed


@dataclass
class CounterfactualResult:
    """What happens if we issue STOP at the current moment."""
    vehicle_on_runway: bool
    can_clear_in_time: bool
    stopping_distance_m: float
    time_to_stop_s: float
    aircraft_eta_s: float
    residual_collision_prob: float
    recommendation: str
    go_around_feasible: float = 0.0
    go_around_time_s: float = 0.0
    aircraft_alt_ft: float = 0.0


def evaluate_counterfactual(
    elapsed_since_clearance_s: float,
    crossing_estimate: CrossingEstimate,
    aircraft_eta_s: float,
    aircraft_sigma_s: float,
    aircraft_alt_ft: float = 0.0,
    aircraft_speed_kts: float = 130.0,
    aircraft_dist_to_threshold_m: float = 0.0,
    n_samples: int = 20_000,
) -> CounterfactualResult:
    """
    Model the outcome of issuing STOP right now.

    Three scenarios:
      a) Vehicle hasn't entered runway yet → STOP prevents entry entirely
      b) Vehicle is on runway → compute if it can brake and clear before aircraft arrives
      c) Vehicle has already cleared → no conflict
    """
    mean_speed_ms = crossing_estimate.mean_speed_kmh / 3.6
    use_lognormal = crossing_estimate.lognormal and crossing_estimate.sigma_ln > 0

    rng = random.Random(99)
    collision_no_action = 0
    collision_with_stop = 0

    for _ in range(n_samples):
        reaction = max(0.0, rng.gauss(
            crossing_estimate.reaction_delay_s, crossing_estimate.sigma_reaction_s
        ))
        if use_lognormal:
            speed_ms = math.exp(rng.gauss(crossing_estimate.mu_ln, crossing_estimate.sigma_ln)) / 3.6
        else:
            speed_ms = max(0.5, rng.gauss(mean_speed_ms, crossing_estimate.sigma_speed_kmh / 3.6))

        t_enter = reaction
        t_exit = t_enter + crossing_estimate.crossing_dist_m / speed_ms
        t_ac = rng.gauss(aircraft_eta_s, max(aircraft_sigma_s, 0.01))

        # No-action scenario
        if t_enter < t_ac < t_exit:
            collision_no_action += 1

        # STOP scenario: driver hears STOP, reacts, then brakes
        t_stop_heard = elapsed_since_clearance_s
        if t_stop_heard < t_enter:
            # Vehicle hasn't entered runway — it simply doesn't enter
            pass  # no collision
        elif t_enter <= t_stop_heard <= t_exit:
            # Vehicle is on the runway, moving at speed_ms
            # Time on runway so far: t_stop_heard - t_enter
            dist_on_runway = speed_ms * (t_stop_heard - t_enter)

            # After hearing STOP: reaction time, then braking
            t_brake_start = t_stop_heard + TRUCK_REACTION_S
            # During reaction, vehicle keeps moving
            dist_during_reaction = speed_ms * TRUCK_REACTION_S
            # Braking distance: v²/(2a)
            stopping_dist = speed_ms ** 2 / (2 * TRUCK_DECEL_MS2)
            time_to_stop = speed_ms / TRUCK_DECEL_MS2

            total_dist_on_runway = dist_on_runway + dist_during_reaction + stopping_dist
            t_vehicle_stopped = t_brake_start + time_to_stop

            # Vehicle is still on runway from t_enter until max(t_vehicle_stopped, t_exit)
            # but now stationary. If aircraft arrives while vehicle is on runway:
            effective_exit = min(
                max(t_vehicle_stopped, t_enter),
                t_exit  # can't be worse than no-action
            )
            # Vehicle is still on runway between t_enter and effective_exit
            if t_enter < t_ac < effective_exit:
                # Check if vehicle has actually left the runway
                if total_dist_on_runway < crossing_estimate.crossing_dist_m:
                    collision_with_stop += 1
        # else: vehicle already cleared the runway

    p_no_action = collision_no_action / n_samples
    p_with_stop = collision_with_stop / n_samples

    # Deterministic estimates for the summary
    t_enter_mean = crossing_estimate.reaction_delay_s
    vehicle_on_runway = t_enter_mean < elapsed_since_clearance_s < (
        t_enter_mean + crossing_estimate.crossing_dist_m / max(mean_speed_ms, 0.1)
    )
    stopping_dist = mean_speed_ms ** 2 / (2 * TRUCK_DECEL_MS2)
    time_to_stop = mean_speed_ms / TRUCK_DECEL_MS2 + TRUCK_REACTION_S

    can_clear = (not vehicle_on_runway) or (time_to_stop + elapsed_since_clearance_s < aircraft_eta_s)

    ga_prob, ga_time = go_around_feasibility(
        aircraft_alt_ft, aircraft_speed_kts, aircraft_dist_to_threshold_m,
    )

    if not vehicle_on_runway and elapsed_since_clearance_s < t_enter_mean:
        rec = "STOP NOW — vehicle has not entered runway, conflict fully preventable"
    elif can_clear:
        rec = "STOP NOW — vehicle can brake and clear before aircraft arrival"
    elif ga_prob > 0.5:
        rec = (
            f"STOP + GO AROUND — vehicle cannot clear in time; "
            f"go-around feasible (P={ga_prob:.0%} at {aircraft_alt_ft:.0f}ft)"
        )
    elif ga_prob > 0.1:
        rec = (
            f"STOP + GO AROUND (MARGINAL) — go-around probability only "
            f"{ga_prob:.0%} at {aircraft_alt_ft:.0f}ft; dual action critical"
        )
    else:
        rec = (
            f"STOP (CRITICAL) — vehicle cannot clear AND go-around infeasible "
            f"(P={ga_prob:.0%} at {aircraft_alt_ft:.0f}ft); collision likely"
        )

    return CounterfactualResult(
        vehicle_on_runway=vehicle_on_runway,
        can_clear_in_time=can_clear,
        stopping_distance_m=stopping_dist,
        time_to_stop_s=time_to_stop,
        aircraft_eta_s=aircraft_eta_s,
        residual_collision_prob=p_with_stop,
        recommendation=rec,
        go_around_feasible=ga_prob,
        go_around_time_s=ga_time,
        aircraft_alt_ft=aircraft_alt_ft,
    )
