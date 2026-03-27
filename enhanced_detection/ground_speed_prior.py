"""
Ground Speed Prior — Empirical vehicle crossing speed distribution.

Since we have no direct truck telemetry, we build a prior from:
1. Airport surface movement constraints (fire trucks typically 15-25 km/h)
2. Crossing distance measured from the airport node-link graph
3. A reaction delay after clearance is given
"""

import csv
import math
import os
from dataclasses import dataclass
from typing import Tuple

from enhanced_detection.aircraft_eta import haversine

TXY_D_001 = (40.775617, -73.879901)
RWY_01_006 = (40.775511, -73.878895)
TXY_D_002 = (40.775426, -73.87791)


@dataclass
class CrossingEstimate:
    crossing_dist_m: float
    mean_speed_kmh: float
    sigma_speed_kmh: float
    reaction_delay_s: float
    mean_duration_s: float
    sigma_duration_s: float

    def __repr__(self):
        return (
            f"CrossingEstimate: dist={self.crossing_dist_m:.0f}m, "
            f"speed={self.mean_speed_kmh:.1f}±{self.sigma_speed_kmh:.1f} km/h, "
            f"reaction={self.reaction_delay_s:.0f}s, "
            f"duration={self.mean_duration_s:.1f}±{self.sigma_duration_s:.1f}s"
        )


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
    mean_speed_kmh: float = 20.0,
    sigma_speed_kmh: float = 4.0,
    reaction_delay_s: float = 3.0,
    sigma_reaction_s: float = 1.0,
) -> CrossingEstimate:
    """
    Build a crossing time estimate using physics and empirical priors.

    Speed prior: airport fire trucks on taxiways typically 15-25 km/h.
    We use mean=20 km/h, sigma=4 km/h (covers 12-28 km/h at 2σ).

    Reaction delay: time between clearance and actual movement.
    Typically 2-5s for ground vehicles (radio processing + throttle).

    Duration variance is propagated from speed uncertainty via:
      σ_t ≈ (d / v²) · σ_v   (first-order Taylor expansion)
    Combined with reaction delay uncertainty.
    """
    crossing_dist_m = compute_crossing_distance()
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
        mean_duration_s=mean_duration_s,
        sigma_duration_s=sigma_duration_s,
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
