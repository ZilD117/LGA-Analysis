"""
Approach Speed Profile — Build empirical deceleration model from ADS-B data.

Aircraft decelerate significantly on final approach. Using constant speed to
estimate ETA introduces bias. This module builds a speed-vs-distance-to-threshold
profile from ADS-B track data, then uses numerical integration for more accurate
ETA prediction.

KEY DESIGN DECISION (causal / real-time safe):
  In a production system, the profile is built from HISTORICAL fleet data
  (past approaches to the same runway), NOT from the incident aircraft's own
  future track.  For this PoC we simulate that by building the profile from
  all available approach data that occurred BEFORE the query time.  The sigma
  calibration likewise uses only past segments.
"""

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from enhanced_detection.aircraft_eta import (
    TrackPoint, load_track, haversine, RW04_THRESHOLD, RW04_TXY_D, KTS_TO_MS,
)

ROLLOUT_DECEL_MS2 = -1.5  # typical braking deceleration, m/s²
TOUCHDOWN_SPEED_KTS = 130.0  # CRJ-900 Vref ≈ 130 kts


@dataclass
class SpeedDistPoint:
    distance_to_threshold_m: float
    groundspeed_kts: float
    altitude_ft: float
    timestamp_s: float  # epoch

    @property
    def groundspeed_ms(self):
        return self.groundspeed_kts * KTS_TO_MS


@dataclass
class ApproachProfile:
    """Empirical speed-vs-distance profile for an approach to a runway."""
    profile: List[SpeedDistPoint]  # sorted by distance descending (far → near)
    runway_heading: float
    touchdown_speed_kts: float
    rollout_decel_ms2: float

    def speed_at_distance(self, dist_m: float) -> float:
        """Interpolate groundspeed (kts) at a given distance to threshold."""
        if not self.profile:
            return TOUCHDOWN_SPEED_KTS

        if dist_m >= self.profile[0].distance_to_threshold_m:
            return self.profile[0].groundspeed_kts

        # On-runway rollout: decelerate from touchdown speed
        if dist_m <= 0:
            return max(0, self.touchdown_speed_kts)

        for i in range(len(self.profile) - 1):
            d0 = self.profile[i].distance_to_threshold_m
            d1 = self.profile[i + 1].distance_to_threshold_m
            if d1 <= dist_m <= d0:
                frac = (d0 - dist_m) / (d0 - d1) if (d0 - d1) > 0 else 0
                s0 = self.profile[i].groundspeed_kts
                s1 = self.profile[i + 1].groundspeed_kts
                return s0 + frac * (s1 - s0)

        return self.profile[-1].groundspeed_kts

    def eta_with_deceleration(
        self,
        current_dist_m: float,
        target_dist_m: float,
        current_speed_kts: Optional[float] = None,
        ds_m: float = 10.0,
    ) -> Tuple[float, float]:
        """
        Compute ETA from current_dist to target_dist using the speed profile.

        Returns (eta_seconds, speed_at_target_kts).
        Uses numerical integration with step size ds_m.

        If current_dist > target_dist, the aircraft is approaching (distance decreasing).
        If target_dist < 0, the target is past the threshold on the runway.
        """
        if current_dist_m <= target_dist_m:
            return 0.0, self.speed_at_distance(current_dist_m)

        total_time = 0.0
        d = current_dist_m

        while d > target_dist_m:
            step = min(ds_m, d - target_dist_m)
            if d > 0:
                speed_kts = self.speed_at_distance(d)
                if current_speed_kts is not None and d == current_dist_m:
                    speed_kts = current_speed_kts
            else:
                # Rollout phase: kinematic deceleration
                dist_past_threshold = -d
                v_td_ms = self.touchdown_speed_kts * KTS_TO_MS
                v_sq = v_td_ms ** 2 + 2 * self.rollout_decel_ms2 * dist_past_threshold
                speed_ms = math.sqrt(max(v_sq, 0))
                speed_kts = speed_ms / KTS_TO_MS

            speed_ms = speed_kts * KTS_TO_MS
            if speed_ms < 0.5:
                # Aircraft has stopped
                return float("inf"), 0.0
            dt = step / speed_ms
            total_time += dt
            d -= step

        final_speed = self.speed_at_distance(target_dist_m)
        return total_time, final_speed


def build_approach_profile(
    track: List[TrackPoint],
    threshold: Tuple[float, float] = RW04_THRESHOLD,
    max_approach_dist_m: float = 50_000.0,
    cutoff_epoch: Optional[float] = None,
) -> ApproachProfile:
    """
    Build a speed-vs-distance profile from ADS-B track for the approach phase.

    CAUSAL CONSTRAINT: only uses data points with timestamp <= cutoff_epoch.
    In production, the profile would come from historical fleet data.  Here we
    simulate causal behaviour by discarding future data.

    Strategy: from causally available points, walk backwards from the most
    recent point keeping the contiguous segment where distance to threshold
    is monotonically increasing (aircraft continuously approaching).
    """
    real_pts = [pt for pt in track if pt.source in ("aeroapi_adsb", "")]
    if cutoff_epoch is not None:
        real_pts = [pt for pt in real_pts if pt.epoch_s <= cutoff_epoch]
    if not real_pts:
        return ApproachProfile([], 40.0, TOUCHDOWN_SPEED_KTS, ROLLOUT_DECEL_MS2)

    real_pts.sort(key=lambda p: p.epoch_s)

    dist_pts = []
    for pt in real_pts:
        d = haversine(pt.lat, pt.lon, threshold[0], threshold[1])
        dist_pts.append((d, pt))

    approach_segment = [dist_pts[-1]]
    for i in range(len(dist_pts) - 2, -1, -1):
        d_curr = dist_pts[i][0]
        d_prev = approach_segment[-1][0]
        if d_curr > d_prev and d_curr <= max_approach_dist_m:
            approach_segment.append(dist_pts[i])
        elif d_curr < d_prev:
            break

    approach_segment.reverse()

    points = []
    for d, pt in approach_segment:
        points.append(SpeedDistPoint(
            distance_to_threshold_m=d,
            groundspeed_kts=pt.groundspeed_kts,
            altitude_ft=pt.alt_ft,
            timestamp_s=pt.epoch_s,
        ))

    td_speed = TOUCHDOWN_SPEED_KTS
    if points:
        td_speed = points[-1].groundspeed_kts

    return ApproachProfile(
        profile=points,
        runway_heading=40.0,
        touchdown_speed_kts=td_speed,
        rollout_decel_ms2=ROLLOUT_DECEL_MS2,
    )


def compute_eta_sigma(
    profile: ApproachProfile,
    track: List[TrackPoint],
    threshold: Tuple[float, float] = RW04_THRESHOLD,
    cutoff_epoch: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute the prediction error (sigma) of the deceleration-aware ETA model.

    CAUSAL CONSTRAINT: only uses ADS-B points with timestamp <= cutoff_epoch.

    For each consecutive pair of real approach points, predict the traversal
    time using the speed profile and compare to actual elapsed time.
    """
    real_pts = [pt for pt in track if pt.source in ("aeroapi_adsb", "")]
    if cutoff_epoch is not None:
        real_pts = [pt for pt in real_pts if pt.epoch_s <= cutoff_epoch]
    real_pts.sort(key=lambda p: p.epoch_s)

    approach_pairs = []
    for pt in real_pts:
        d = haversine(pt.lat, pt.lon, threshold[0], threshold[1])
        if d < 20_000:
            approach_pairs.append((d, pt))

    filtered = [approach_pairs[0]] if approach_pairs else []
    for i in range(1, len(approach_pairs)):
        if approach_pairs[i][0] < filtered[-1][0]:
            filtered.append(approach_pairs[i])

    if len(filtered) < 3:
        return 0.0, 2.0

    errors = []
    for i in range(len(filtered) - 1):
        d_start, pt_start = filtered[i]
        d_end, pt_end = filtered[i + 1]

        actual_dt = pt_end.epoch_s - pt_start.epoch_s
        if actual_dt <= 0 or actual_dt > 30:
            continue

        dist_delta = d_start - d_end
        if dist_delta < 50:
            continue

        actual_rate_ms = dist_delta / actual_dt
        expected_rate_ms = pt_start.groundspeed_kts * KTS_TO_MS
        if actual_rate_ms < 0.5 * expected_rate_ms:
            continue

        predicted_dt, _ = profile.eta_with_deceleration(
            d_start, d_end,
            current_speed_kts=pt_start.groundspeed_kts,
        )
        if predicted_dt < float("inf"):
            errors.append(predicted_dt - actual_dt)

    if len(errors) < 2:
        return 0.0, 2.0

    mean_err = sum(errors) / len(errors)
    var = sum((e - mean_err) ** 2 for e in errors) / (len(errors) - 1)
    per_segment_sigma = math.sqrt(var)

    avg_segment_dist = sum(
        filtered[i][0] - filtered[i + 1][0] for i in range(len(filtered) - 1)
    ) / (len(filtered) - 1)

    return mean_err, per_segment_sigma


if __name__ == "__main__":
    from datetime import datetime, timezone
    from enhanced_detection.aircraft_eta import get_aircraft_state

    csv_path = os.path.join(os.path.dirname(__file__), "..", "surface_data", "lga_case_study", "flight_8646_track.csv")
    track = load_track(csv_path)

    conflict_utc = datetime(2026, 3, 23, 3, 37, 1, tzinfo=timezone.utc)
    cutoff = conflict_utc.timestamp()

    profile = build_approach_profile(track, cutoff_epoch=cutoff)

    print("=" * 60)
    print("  APPROACH SPEED PROFILE — Jazz 8646 (CRJ-900)")
    print(f"  (causal: only ADS-B data before {conflict_utc.strftime('%H:%M:%SZ')})")
    print("=" * 60)
    print(f"\n  {'Distance (m)':>14}  {'Speed (kts)':>12}  {'Alt (ft)':>10}")
    print(f"  {'-'*14}  {'-'*12}  {'-'*10}")
    for p in profile.profile:
        print(f"  {p.distance_to_threshold_m:>14.0f}  {p.groundspeed_kts:>12.0f}  {p.altitude_ft:>10.0f}")

    print(f"\n  Touchdown speed: {profile.touchdown_speed_kts:.0f} kts")
    print(f"  Rollout deceleration: {profile.rollout_decel_ms2:.1f} m/s²")

    state = get_aircraft_state(track, conflict_utc, RW04_TXY_D)

    dist_to_threshold = haversine(state.lat, state.lon, RW04_THRESHOLD[0], RW04_THRESHOLD[1])
    target_past_threshold = haversine(RW04_THRESHOLD[0], RW04_THRESHOLD[1], RW04_TXY_D[0], RW04_TXY_D[1])

    eta_decel, speed_at_target = profile.eta_with_deceleration(
        dist_to_threshold, -target_past_threshold,
        current_speed_kts=state.groundspeed_kts,
    )

    bias, sigma = compute_eta_sigma(profile, track, cutoff_epoch=cutoff)

    print(f"\n{'─'*60}")
    print(f"  ETA COMPARISON at conflict time (03:37:01Z)")
    print(f"{'─'*60}")
    print(f"  Current speed: {state.groundspeed_kts:.0f} kts")
    print(f"  Distance to threshold: {dist_to_threshold:.0f} m")
    print(f"  Distance threshold→Txy D: {target_past_threshold:.0f} m")
    print(f"  Total distance to Txy D: {state.distance_to_point_m:.0f} m")
    print(f"\n  Constant-speed ETA:     {state.time_to_point_s:.1f} s (σ = {state.sigma_s:.1f} s)")
    print(f"  Deceleration-aware ETA: {eta_decel:.1f} s (σ = {sigma:.1f} s, bias = {bias:.1f} s)")
    print(f"  Speed at Txy D crossing: {speed_at_target:.0f} kts")
    print(f"\n  Difference: +{eta_decel - state.time_to_point_s:.1f} s "
          f"(decel model predicts LATER arrival due to braking)")
