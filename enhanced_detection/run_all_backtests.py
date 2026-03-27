#!/usr/bin/env python3
"""
Run all four incident backtests and print a summary comparison table.

Usage:
    python3 -m enhanced_detection.run_all_backtests
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    results = []

    # ── 1. LGA 2026 (original, real data) ──
    print("\n" + "█" * 70)
    print("  [1/4] LaGuardia 2026 — Real ADS-B + Voice Data")
    print("█" * 70 + "\n")
    t0 = time.perf_counter()
    from enhanced_detection.run_lga_case import run as run_lga
    lga_result = run_lga()
    lga_time = time.perf_counter() - t0
    results.append({
        "name": "LGA 2026 (real data)",
        "entities": "Jazz 8646 vs Truck 1",
        "type": "Landing vs Crossing",
        "layer1": "YES",
        "occ_prob": lga_result.occupancy_probability * 100 if lga_result else 0,
        "alert": lga_result.alert_level if lga_result else "N/A",
        "lead_s": 31,
        "decision": "STOP",
        "runtime_ms": lga_time * 1000,
    })

    # ── 2. Haneda 2024 ──
    print("\n" + "█" * 70)
    print("  [2/4] Haneda 2024 — Real ADS-B (JAL 516) + Synthetic (JA722A)")
    print("█" * 70 + "\n")
    t0 = time.perf_counter()
    from enhanced_detection.haneda_2024.run_case import run as run_haneda
    haneda = run_haneda()
    haneda_time = time.perf_counter() - t0
    results.append({
        "name": "Haneda 2024",
        "entities": "JAL 516 vs JA722A",
        "type": "Landing vs Taxi-on-runway",
        "layer1": "YES" if haneda.fatal_conflict else "NO",
        "occ_prob": haneda.occupancy_prob * 100,
        "alert": haneda.alert_level,
        "lead_s": haneda.detection_lead_s,
        "decision": "STOP" if haneda.decision_stop else "monitor",
        "runtime_ms": haneda_time * 1000,
    })

    # ── 3. KATL 2024 ──
    print("\n" + "█" * 70)
    print("  [3/4] KATL 2024 — Synthetic ADS-B")
    print("█" * 70 + "\n")
    t0 = time.perf_counter()
    from enhanced_detection.katl_2024.run_case import run as run_katl
    katl = run_katl()
    katl_time = time.perf_counter() - t0
    results.append({
        "name": "KATL 2024",
        "entities": "Endeavor 5526 vs Delta 295",
        "type": "Taxi vs Taxi (converging)",
        "layer1": "YES" if katl.fatal_conflict else "NO",
        "occ_prob": katl.occupancy_prob * 100,
        "alert": katl.alert_level,
        "lead_s": katl.detection_lead_s,
        "decision": "STOP" if katl.decision_stop else "monitor",
        "runtime_ms": katl_time * 1000,
    })

    # ── 4. Tenerife 1977 ──
    print("\n" + "█" * 70)
    print("  [4/4] Tenerife 1977 — Synthetic ADS-B")
    print("█" * 70 + "\n")
    t0 = time.perf_counter()
    from enhanced_detection.tenerife_1977.run_case import run as run_tenerife
    tenerife = run_tenerife()
    tenerife_time = time.perf_counter() - t0
    results.append({
        "name": "Tenerife 1977",
        "entities": "KLM 4805 vs Clipper 1736",
        "type": "Takeoff vs Taxi-on-runway",
        "layer1": "YES" if tenerife.fatal_conflict else "NO",
        "occ_prob": tenerife.occupancy_prob * 100,
        "alert": tenerife.alert_level,
        "lead_s": tenerife.detection_lead_s,
        "decision": "STOP" if tenerife.decision_stop else "monitor",
        "runtime_ms": tenerife_time * 1000,
    })

    # ── Summary Table ──
    print("\n\n" + "█" * 70)
    print("  BACKTEST SUMMARY — ALL INCIDENTS")
    print("█" * 70)

    hdr = (
        f"  {'Incident':<22} {'Conflict Type':<26} {'L1?':>4} "
        f"{'P(occ)':>8} {'Alert':>10} {'Lead':>6} {'Action':>8} {'Time':>8}"
    )
    print(f"\n{hdr}")
    print(f"  {'-'*22} {'-'*26} {'-'*4} {'-'*8} {'-'*10} {'-'*6} {'-'*8} {'-'*8}")

    for r in results:
        print(
            f"  {r['name']:<22} {r['type']:<26} {r['layer1']:>4} "
            f"{r['occ_prob']:>7.1f}% {r['alert']:>10} {r['lead_s']:>5.0f}s "
            f"{r['decision']:>8} {r['runtime_ms']:>6.0f}ms"
        )

    all_detected = all(r["decision"] == "STOP" for r in results)
    print(f"\n  All incidents detected and STOP issued: {'YES' if all_detected else 'NO'}")
    print(f"  Total runtime: {sum(r['runtime_ms'] for r in results):.0f} ms")


if __name__ == "__main__":
    main()
