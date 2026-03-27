"""
Ground Speed Prior — Empirical vehicle crossing speed distribution.

Builds a vehicle speed prior from:
1. Real vehicle speed observations from KATL linktime data (actype='VEH')
2. Crossing distance measured from the airport node-link graph
3. A reaction delay after clearance is given

The linktime data contains 514 actual vehicle (VEH) speed observations from
ASDE-X surface surveillance across 7 days at KATL. The distribution is
log-normal (bounded below, right-skewed): median 12.7 km/h, mean 13.5 km/h.
"""

import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

from enhanced_detection.aircraft_eta import haversine

TXY_D_001 = (40.775617, -73.879901)
RWY_01_006 = (40.775511, -73.878895)
TXY_D_002 = (40.775426, -73.87791)

# Log-normal parameters for vehicle speed, fitted from 514 VEH observations
# at KATL (May 2023 linktime data). Speed is in km/h.
#   P(v) = LogNormal(mu_ln, sigma_ln)
#   Mean=13.5 km/h, Median=12.7 km/h, Mode=11.2 km/h, P5=6.3, P95=18.0
VEH_MU_LN = 2.5429
VEH_SIGMA_LN = 0.3547
VEH_MEAN_KMH = math.exp(VEH_MU_LN + VEH_SIGMA_LN ** 2 / 2)  # ~13.5
VEH_SIGMA_KMH = VEH_MEAN_KMH * math.sqrt(math.exp(VEH_SIGMA_LN ** 2) - 1)  # ~4.9


@dataclass
class CrossingEstimate:
    crossing_dist_m: float
    mean_speed_kmh: float
    sigma_speed_kmh: float
    reaction_delay_s: float
    sigma_reaction_s: float
    mean_duration_s: float
    sigma_duration_s: float
    lognormal: bool = False
    mu_ln: float = 0.0
    sigma_ln: float = 0.0

    def __repr__(self):
        dist_label = "log-normal VEH" if self.lognormal else "Gaussian"
        return (
            f"CrossingEstimate: dist={self.crossing_dist_m:.0f}m, "
            f"speed={self.mean_speed_kmh:.1f}±{self.sigma_speed_kmh:.1f} km/h ({dist_label}), "
            f"reaction={self.reaction_delay_s:.0f}±{self.sigma_reaction_s:.1f}s, "
            f"duration={self.mean_duration_s:.1f}±{self.sigma_duration_s:.1f}s"
        )


def calibrate_vehicle_speed(
    data_dir: str = None,
) -> Tuple[float, float, float, float]:
    """
    Calibrate vehicle speed prior from linktime data (actype='VEH').

    Returns (mean_kmh, sigma_kmh, mu_ln, sigma_ln) where:
      - mean_kmh, sigma_kmh: Gaussian-equivalent parameters
      - mu_ln, sigma_ln: log-normal parameters (more accurate distribution)

    Uses the VEH-typed observations from ASDE-X linktime data. These are
    real ground vehicle speeds (fire trucks, tugs, service vehicles) measured
    on the airport surface, not aircraft taxi speeds used as a proxy.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "linktime_data")

    speeds: List[float] = []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".csv"):
            continue
        try:
            with open(os.path.join(data_dir, fname), "r") as f:
                for row in csv.DictReader(f):
                    actype = row.get("actype", "").strip()
                    if actype == "VEH":
                        s = float(row["avg_speed"])
                        if s > 0.5:
                            speeds.append(s)
        except (ValueError, KeyError):
            continue

    if len(speeds) < 20:
        return VEH_MEAN_KMH, VEH_SIGMA_KMH, VEH_MU_LN, VEH_SIGMA_LN

    log_speeds = [math.log(s) for s in speeds]
    mu_ln = sum(log_speeds) / len(log_speeds)
    var_ln = sum((ls - mu_ln) ** 2 for ls in log_speeds) / (len(log_speeds) - 1)
    sigma_ln = math.sqrt(var_ln)

    mean_kmh = math.exp(mu_ln + sigma_ln ** 2 / 2)
    sigma_kmh = mean_kmh * math.sqrt(math.exp(sigma_ln ** 2) - 1)

    return mean_kmh, sigma_kmh, mu_ln, sigma_ln


def compute_crossing_distance(
    entry: Tuple[float, float] = TXY_D_001,
    midpoint: Tuple[float, float] = RWY_01_006,
    exit_pt: Tuple[float, float] = TXY_D_002,
) -> float:
    """Compute total crossing distance from taxiway entry through runway midpoint to exit."""
    d1 = haversine(entry[0], entry[1], midpoint[0], midpoint[1])
    d2 = haversine(midpoint[0], midpoint[1], exit_pt[0], exit_pt[1])
    return d1 + d2


def build_crossing_estimate(
    mean_speed_kmh: Optional[float] = None,
    sigma_speed_kmh: Optional[float] = None,
    reaction_delay_s: float = 3.0,
    sigma_reaction_s: float = 1.0,
    calibrate: bool = True,
    entry: Tuple[float, float] = TXY_D_001,
    midpoint: Tuple[float, float] = RWY_01_006,
    exit_pt: Tuple[float, float] = TXY_D_002,
) -> CrossingEstimate:
    """
    Build a crossing time estimate using physics and empirical priors.

    If calibrate=True and speed params are not overridden, calibrates from
    real vehicle (VEH) speed data in the linktime CSVs. The VEH distribution
    is log-normal: median 12.7, mean 13.5 km/h — measured from 514 actual
    ground vehicle observations at KATL via ASDE-X surface surveillance.

    Duration variance is propagated from speed uncertainty via:
      σ_t ≈ (d / v²) · σ_v   (first-order Taylor expansion)
    Combined with reaction delay uncertainty.
    """
    use_lognormal = False
    mu_ln = 0.0
    sigma_ln = 0.0

    if mean_speed_kmh is None or sigma_speed_kmh is None:
        if calibrate:
            cal_mean, cal_sigma, mu_ln, sigma_ln = calibrate_vehicle_speed()
            use_lognormal = True
        else:
            cal_mean, cal_sigma = VEH_MEAN_KMH, VEH_SIGMA_KMH
            mu_ln, sigma_ln = VEH_MU_LN, VEH_SIGMA_LN
            use_lognormal = True
        if mean_speed_kmh is None:
            mean_speed_kmh = cal_mean
        if sigma_speed_kmh is None:
            sigma_speed_kmh = cal_sigma

    crossing_dist_m = compute_crossing_distance(entry=entry, midpoint=midpoint, exit_pt=exit_pt)
    midpoint_dist_m = crossing_dist_m / 2.0

    mean_speed_ms = mean_speed_kmh / 3.6
    sigma_speed_ms = sigma_speed_kmh / 3.6

    mean_travel_s = midpoint_dist_m / mean_speed_ms
    sigma_travel_s = midpoint_dist_m * sigma_speed_ms / (mean_speed_ms ** 2)

    mean_duration_s = reaction_delay_s + mean_travel_s
    sigma_duration_s = math.sqrt(sigma_travel_s ** 2 + sigma_reaction_s ** 2)

    return CrossingEstimate(
        crossing_dist_m=crossing_dist_m,
        mean_speed_kmh=mean_speed_kmh,
        sigma_speed_kmh=sigma_speed_kmh,
        reaction_delay_s=reaction_delay_s,
        sigma_reaction_s=sigma_reaction_s,
        mean_duration_s=mean_duration_s,
        sigma_duration_s=sigma_duration_s,
        lognormal=use_lognormal,
        mu_ln=mu_ln,
        sigma_ln=sigma_ln,
    )


if __name__ == "__main__":
    dist = compute_crossing_distance()
    print(f"Runway crossing geometry (Txy D at KLGA):")
    print(f"  Entry (Txy_D_001):  {TXY_D_001}")
    print(f"  Midpoint (Rwy_01_006): {RWY_01_006}")
    print(f"  Exit  (Txy_D_002):  {TXY_D_002}")
    print(f"  Total crossing distance: {dist:.1f} m")
    d1 = haversine(*TXY_D_001, *RWY_01_006)
    d2 = haversine(*RWY_01_006, *TXY_D_002)
    print(f"    Entry → Midpoint: {d1:.1f} m")
    print(f"    Midpoint → Exit:  {d2:.1f} m")
    print()

    est = build_crossing_estimate()
    print(est)
    print(f"\n  Time for vehicle to reach runway midpoint after clearance:")
    print(f"    μ = {est.mean_duration_s:.1f} s  (reaction {est.reaction_delay_s:.0f}s + travel)")
    print(f"    σ = {est.sigma_duration_s:.1f} s")
    print(f"  This is the Gaussian(μ, σ) for vehicle presence on runway center.")
