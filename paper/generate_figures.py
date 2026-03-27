#!/usr/bin/env python3
"""Generate publication-quality figures for the arXiv preprint paper."""

import math
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(PAPER_DIR, "figures")
REPO_ROOT = os.path.join(PAPER_DIR, "..")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ── Figure 1: System Architecture Diagram ──────────────────────────

def fig_architecture():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    colors = {
        "input": "#4A90D9",
        "layer1": "#E74C3C",
        "layer2": "#F39C12",
        "layer3": "#27AE60",
        "output": "#8E44AD",
    }

    def box(x, y, w, h, label, sublabel, color, fontsize=9):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="black", linewidth=1.2, alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + 0.12, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color="white")
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.18, sublabel,
                    ha="center", va="center", fontsize=7, color="white", style="italic")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))

    # Inputs
    box(0.3, 4.5, 2.0, 1.0, "ATC Audio", "Tower frequency", colors["input"])
    box(0.3, 2.8, 2.0, 1.0, "ADS-B Feed", "Aircraft positions", colors["input"])
    box(0.3, 1.1, 2.0, 1.0, "Airport Graph", "NASA FACET nodes", colors["input"])

    # Layer 1
    box(3.3, 4.0, 2.2, 1.2, "Layer 1", "Clearance Parser\n+ Conflict Detect", colors["layer1"])
    arrow(2.3, 5.0, 3.3, 4.8)

    # Layer 2
    box(3.3, 2.2, 2.2, 1.2, "Layer 2", "Decel ETA + VEH\nSpeed Prior", colors["layer2"])
    arrow(2.3, 3.3, 3.3, 2.8)
    arrow(2.3, 1.6, 3.3, 2.4)
    arrow(4.4, 4.0, 4.4, 3.4)

    # Layer 3
    box(6.5, 3.1, 2.2, 1.2, "Layer 3", "MC Occupancy +\nBayesian + GA", colors["layer3"])
    arrow(5.5, 4.6, 6.5, 3.9)
    arrow(5.5, 2.8, 6.5, 3.3)

    # Output
    box(6.5, 1.0, 2.2, 1.0, "Decision", "STOP / MONITOR", colors["output"])
    arrow(7.6, 3.1, 7.6, 2.0)

    ax.text(5.0, 0.3, "Three-Layer Conflict Detection Architecture",
            ha="center", va="center", fontsize=12, fontweight="bold")

    fig.savefig(os.path.join(FIG_DIR, "architecture.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "architecture.png"))
    plt.close(fig)
    print("  [1/3] architecture.pdf")


# ── Figure 2: VEH Speed Distribution ──────────────────────────────

def fig_veh_speed():
    sys.path.insert(0, REPO_ROOT)

    mu_ln = 2.5429
    sigma_ln = 0.3547

    rng = np.random.default_rng(42)
    samples = rng.lognormal(mu_ln, sigma_ln, 514)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.hist(samples, bins=30, density=True, alpha=0.6, color="#3498DB",
            edgecolor="white", linewidth=0.5, label="Simulated VEH data (n=514)")

    x = np.linspace(0.5, 40, 500)
    pdf = (1 / (x * sigma_ln * np.sqrt(2 * np.pi))) * \
          np.exp(-0.5 * ((np.log(x) - mu_ln) / sigma_ln) ** 2)
    ax.plot(x, pdf, "r-", linewidth=2, label=f"LogNormal($\\mu_{{\\ln}}$={mu_ln:.2f}, $\\sigma_{{\\ln}}$={sigma_ln:.2f})")

    median = np.exp(mu_ln)
    mean = np.exp(mu_ln + sigma_ln ** 2 / 2)
    ax.axvline(median, color="#E67E22", linestyle="--", linewidth=1.2, label=f"Median = {median:.1f} km/h")
    ax.axvline(mean, color="#8E44AD", linestyle=":", linewidth=1.2, label=f"Mean = {mean:.1f} km/h")

    p5 = np.exp(mu_ln - 1.645 * sigma_ln)
    p95 = np.exp(mu_ln + 1.645 * sigma_ln)
    ax.axvspan(p5, p95, alpha=0.08, color="green", label=f"5th–95th pctl: [{p5:.1f}, {p95:.1f}]")

    ax.set_xlabel("Ground Vehicle Speed (km/h)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Empirical Ground Vehicle Speed Distribution (ASDE-X)")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_xlim(0, 35)

    fig.savefig(os.path.join(FIG_DIR, "veh_speed_distribution.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "veh_speed_distribution.png"))
    plt.close(fig)
    print("  [2/3] veh_speed_distribution.pdf")


# ── Figure 3: Go-Around Feasibility Curve ─────────────────────────

def fig_goaround():
    h = np.linspace(0, 500, 1000)

    h_min = 50.0
    h_da = 200.0

    p = np.where(
        h >= h_da, 0.98,
        np.where(h <= h_min, 0.02,
                 0.02 + 0.96 * (h - h_min) / (h_da - h_min))
    )

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(h, p * 100, "b-", linewidth=2)
    ax.fill_between(h, p * 100, alpha=0.15, color="blue")

    ax.axvline(200, color="green", linestyle="--", linewidth=1, alpha=0.8)
    ax.text(205, 50, "CAT I DA\n(200 ft)", fontsize=8, color="green")

    ax.axvline(50, color="red", linestyle="--", linewidth=1, alpha=0.8)
    ax.text(55, 30, "Physical\nlimit (50 ft)", fontsize=8, color="red")

    ax.axhspan(0, 20, alpha=0.05, color="red")
    ax.axhspan(80, 100, alpha=0.05, color="green")

    ax.set_xlabel("Aircraft Altitude (ft AGL)")
    ax.set_ylabel("Go-Around Success Probability (%)")
    ax.set_title("Go-Around Feasibility Model")
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(FIG_DIR, "goaround_feasibility.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "goaround_feasibility.png"))
    plt.close(fig)
    print("  [3/3] goaround_feasibility.pdf")


# ── Figure 4: Sensitivity Heatmap ──────────────────────────────────

def fig_sensitivity():
    import random

    # Vehicle speed (rows) vs reaction delay (columns)
    speed_configs = [
        ("Slow (8)", False, 0, 0, 8.0, 3.0),
        ("KATL LN (13.5)", True, 2.543, 0.355, 13.5, 4.5),
        ("Literature (20)", False, 0, 0, 20.0, 4.0),
        ("Fast (25)", False, 0, 0, 25.0, 5.0),
        ("Emergency (30)", False, 0, 0, 30.0, 8.0),
    ]
    delays = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0]
    cross_dist = 169.0
    eta_ac = 27.8
    sigma_ac = 2.3

    matrix = np.zeros((len(speed_configs), len(delays)))
    for i, (name, use_ln, mln, sln, mkm, skm) in enumerate(speed_configs):
        for j, tau in enumerate(delays):
            rng = random.Random(42)
            hits = 0
            n = 20000
            for _ in range(n):
                reaction = max(0, rng.gauss(tau, 1.0))
                if use_ln:
                    speed_ms = math.exp(rng.gauss(mln, sln)) / 3.6
                else:
                    speed_ms = max(0.5, rng.gauss(mkm / 3.6, skm / 3.6))
                t_enter = reaction
                t_exit = t_enter + cross_dist / speed_ms
                t_ac = rng.gauss(eta_ac, sigma_ac)
                if t_enter < t_ac < t_exit:
                    hits += 1
            matrix[i, j] = hits / n * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(delays)))
    ax.set_xticklabels([f"{d:.1f}" for d in delays])
    ax.set_yticks(range(len(speed_configs)))
    ax.set_yticklabels([c[0] for c in speed_configs])
    ax.set_xlabel("Reaction Delay τ (s)")
    ax.set_ylabel("Vehicle Speed Distribution")
    ax.set_title("P(occupancy) Sensitivity: Speed × Reaction Delay")

    for i in range(len(speed_configs)):
        for j in range(len(delays)):
            val = matrix[i, j]
            color = "white" if val > 60 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, label="P(occupancy) %", shrink=0.8)

    fig.savefig(os.path.join(FIG_DIR, "sensitivity_heatmap.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "sensitivity_heatmap.png"))
    plt.close(fig)
    print("  [4/5] sensitivity_heatmap.pdf")


# ── Figure 5: Latency-Corrected Timeline ──────────────────────────

def fig_latency_timeline():
    fig, ax = plt.subplots(figsize=(7, 3.5))

    events = [
        (416.8, "Truck request\nutterance starts", "#3498DB"),
        (419.4, "Utterance\ncompletes", "#3498DB"),
        (422.6, "Whisper\ninference done", "#E67E22"),
        (423.1, "Alert\ndisplayed", "#E67E22"),
        (425.1, "Controller\nprocesses", "#9B59B6"),
        (426.1, "STOP\ntransmitted", "#27AE60"),
        (428.8, "Controller's\nactual STOP\n(wrong target)", "#E74C3C"),
        (437.0, "Controller's\ncorrect STOP", "#E74C3C"),
        (448.0, "COLLISION", "#C0392B"),
    ]

    t0 = 415
    ax.set_xlim(t0, 452)
    ax.set_ylim(-1.5, 2.5)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)

    for i, (t, label, color) in enumerate(events):
        side = 1 if i % 2 == 0 else -1
        ax.plot([t, t], [0, side * 0.4], color=color, linewidth=1.5)
        ax.plot(t, 0, "o", color=color, markersize=6, zorder=5)
        ax.text(t, side * 0.55, label, ha="center",
                va="bottom" if side > 0 else "top",
                fontsize=6.5, color=color, fontweight="bold")

    # Annotate lead times
    ax.annotate("", xy=(426.1, -1.1), xytext=(448.0, -1.1),
                arrowprops=dict(arrowstyle="<->", color="#27AE60", lw=1.5))
    ax.text(437, -1.3, "System: 21.9s lead", ha="center", fontsize=8,
            color="#27AE60", fontweight="bold")

    ax.annotate("", xy=(437.0, -0.7), xytext=(448.0, -0.7),
                arrowprops=dict(arrowstyle="<->", color="#E74C3C", lw=1.5))
    ax.text(442.5, -0.9, "Controller: 11s", ha="center", fontsize=8,
            color="#E74C3C", fontweight="bold")

    ax.set_xlabel("Time (seconds after audio start)")
    ax.set_title("Latency-Corrected Detection Timeline — LGA 2026")
    ax.get_yaxis().set_visible(False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    fig.savefig(os.path.join(FIG_DIR, "latency_timeline.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "latency_timeline.png"))
    plt.close(fig)
    print("  [5/5] latency_timeline.pdf")


# ── Copy existing backtest result PNGs ─────────────────────────────

def copy_existing_figures():
    copies = [
        ("enhanced_detection/haneda_2024/results.png", "haneda_results.png"),
        ("enhanced_detection/katl_2024/results.png", "katl_results.png"),
        ("enhanced_detection/tenerife_1977/results.png", "tenerife_results.png"),
        ("enhanced_detection/lga_enhanced_vs_original.png", "lga_enhanced_vs_original.png"),
        ("lga_risk_0.01_real.png", "lga_risk.png"),
        ("lga_risk_visualization_real.png", "lga_risk_visualization.png"),
        ("lga_blind_spot_analysis.png", "lga_blind_spot.png"),
        ("lga_prevention_analysis.png", "lga_prevention.png"),
    ]
    count = 0
    for src_rel, dst_name in copies:
        src = os.path.join(REPO_ROOT, src_rel)
        dst = os.path.join(FIG_DIR, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            count += 1
    print(f"  Copied {count} existing result PNGs to figures/")


if __name__ == "__main__":
    print("Generating publication figures...")
    fig_architecture()
    fig_veh_speed()
    fig_goaround()
    fig_sensitivity()
    fig_latency_timeline()
    copy_existing_figures()
    print("Done. Output in paper/figures/")
