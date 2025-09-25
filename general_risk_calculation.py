#!/usr/bin/env python3
"""
General Risk Calculation Function
Based on Case Study 3: 1977 Tenerife Runway Collision

This script implements a general risk calculation function that takes two aircraft paths
and their segment times as input, then calculates collision risk using the same methodology
as the original case study 3 implementation.

The risk calculation uses:
1. Deterministic timeline from fixed segment speeds
2. Time-resolved, per-second PN/FW instantaneous risk
3. Cumulative risk curves (monotone increasing) per node
4. Fenton-Wilkinson and Petri-Net risk models

Author: Generated based on case study 3 methodology
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm
from typing import List, Tuple, Dict, Optional

# Constants
SEC_PER_HOUR = 3600.0  # time axis in seconds

def normalize_node(n: str) -> str:
    """Normalize node name by stripping whitespace and converting to uppercase."""
    return n.strip().upper()

def link_lookup_distance(link_dist_km: Dict[Tuple[str,str], float], a: str, b: str) -> Optional[float]:
    """Look up distance between two nodes in the link distance dictionary."""
    a_u, b_u = normalize_node(a), normalize_node(b)
    for (ka,kb), d in link_dist_km.items():
        if (normalize_node(ka)==a_u and normalize_node(kb)==b_u) or (normalize_node(ka)==b_u and normalize_node(kb)==a_u):
            return d
    return None

def overlap_nodes_ordered_by_path(path_ref: List[str], path_other: List[str]) -> List[str]:
    """Find overlapping nodes between two paths, ordered by the reference path."""
    s_other = set([normalize_node(x) for x in path_other])
    return [normalize_node(x) for x in path_ref if normalize_node(x) in s_other]

def deterministic_arrival_seconds(
    path: List[str],
    link_dist_km: Dict[Tuple[str,str], float],
    seg_speeds_kmh: List[float]
) -> List[float]:
    """Calculate deterministic arrival times for each node in the path."""
    assert len(seg_speeds_kmh) == len(path) - 1, "Speed list must match #segments."
    t = [0.0]
    for i in range(len(path)-1):
        d_km = link_lookup_distance(link_dist_km, path[i], path[i+1])
        if d_km is None:
            raise KeyError(f"Missing distance for link ({path[i]}, {path[i+1]})")
        v_kmh = float(seg_speeds_kmh[i])
        if v_kmh <= 0:
            raise ValueError(f"Non-positive speed for segment {i}: {v_kmh}")
        dt_sec = (d_km / v_kmh) * SEC_PER_HOUR
        t.append(t[-1] + dt_sec)
    return t  # len == len(path)

def all_occurrences(path: List[str], node: str) -> List[int]:
    """Find all occurrences of a node in a path."""
    node_u = normalize_node(node)
    return [i for i, n in enumerate([normalize_node(x) for x in path]) if n == node_u]

def eta_seconds_to_node_occ(
    path: List[str],
    link_dist_km: Dict[Tuple[str,str], float],
    seg_speeds_kmh: List[float],
    idx: int
) -> float:
    """Calculate ETA in seconds to a specific node occurrence."""
    assert 0 <= idx < len(path)
    t = 0.0
    for i in range(idx):
        d = link_lookup_distance(link_dist_km, path[i], path[i+1])
        v = seg_speeds_kmh[i]
        t += (d / v) * SEC_PER_HOUR
    return t

def best_occurrence_by_eta_match(
    path_A: List[str], link_dist_km: Dict[Tuple[str,str], float], speeds_A_kmh: List[float],
    target_eta_sec: float, node: str
) -> int:
    """Find the best occurrence of a node in path A that matches the target ETA."""
    cand = all_occurrences(path_A, node)
    if not cand:
        raise ValueError(f"Node {node} not found on the path.")
    etas = [eta_seconds_to_node_occ(path_A, link_dist_km, speeds_A_kmh, i) for i in cand]
    k_best = min(range(len(cand)), key=lambda k: abs(etas[k] - target_eta_sec))
    return cand[k_best]

def calculate_speeds_from_segment_times(
    path: List[str],
    segment_times: List[float],
    link_dist_km: Dict[Tuple[str,str], float]
) -> List[float]:
    """Calculate speeds (km/h) from segment times and distances."""
    speeds = []
    for i in range(len(path) - 1):
        d_km = link_lookup_distance(link_dist_km, path[i], path[i+1])
        if d_km is None:
            raise KeyError(f"Missing distance for link ({path[i]}, {path[i+1]})")
        if segment_times[i] <= 0:
            raise ValueError(f"Non-positive segment time for segment {i}: {segment_times[i]}")
        speed_kmh = (d_km / segment_times[i]) * SEC_PER_HOUR
        speeds.append(speed_kmh)
    return speeds


def general_risk_calculation(
    path_1: List[str], 
    path_2: List[str],
    segment_times_1: List[float], 
    segment_times_2: List[float],
    link_dist_km: Dict[Tuple[str,str], float],
    rc_km: float = 0.075,
    epsilon_sec: float = 1.0,
    nodes_of_interest: Optional[List[str]] = None,
    dt_sec: int = 1,
    gaussian_sigma_sec: float = 5.0
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    General risk calculation function based on case study 3 methodology.
    
    Parameters:
    -----------
    path_1 : List[str]
        First aircraft path (list of node names)
    path_2 : List[str]
        Second aircraft path (list of node names)
    segment_times_1 : List[float]
        Time durations for each segment in path_1 (seconds)
    segment_times_2 : List[float]
        Time durations for each segment in path_2 (seconds)
    link_dist_km : Dict[Tuple[str,str], float]
        Dictionary mapping (node1, node2) to distance in km
    rc_km : float, default=0.075
        Spatial capture radius in km
    epsilon_sec : float, default=1.0
        PN coincidence window in seconds
    nodes_of_interest : Optional[List[str]], default=None
        Specific nodes to analyze (if None, uses overlapping nodes)
    dt_sec : int, default=1
        Time step for risk calculation in seconds
    gaussian_sigma_sec : float, default=5.0
        Gaussian width around deterministic ETAs in seconds
        
    Returns:
    --------
    Tuple[pd.DataFrame, np.ndarray]
        DataFrame with risk data and time grid array
    """
    # Calculate speeds from segment times
    speeds_1_kmh = calculate_speeds_from_segment_times(path_1, segment_times_1, link_dist_km)
    speeds_2_kmh = calculate_speeds_from_segment_times(path_2, segment_times_2, link_dist_km)
    
    # Normalize paths
    p1 = [normalize_node(x) for x in path_1]
    p2 = [normalize_node(x) for x in path_2]
    
    # Determine nodes of interest
    if nodes_of_interest is None:
        nodes = overlap_nodes_ordered_by_path(p2, p1)
    else:
        nodes = [normalize_node(x) for x in nodes_of_interest]

    # Calculate time grid
    T_end = int(math.ceil(max(deterministic_arrival_seconds(p1, link_dist_km, speeds_1_kmh)[-1],
                              deterministic_arrival_seconds(p2, link_dist_km, speeds_2_kmh)[-1])))
    t_grid = np.arange(0, T_end + 1, dt_sec, dtype=float)
    eps = float(epsilon_sec)
    std_sec = max(float(gaussian_sigma_sec), 1e-6)

    rows = []
    for node in nodes:
        # Find all occurrences of this node in both paths
        occ1_list = all_occurrences(p1, node)
        occ2_list = all_occurrences(p2, node)
        
        # Calculate risk only if both aircraft actually visit this node
        if not occ1_list or not occ2_list:
            # If either aircraft doesn't visit this node, set risk to zero
            rows.append(pd.DataFrame({
                "time_sec": t_grid,
                "node": node,
                "R_PN": np.zeros_like(t_grid),
                "R_FW": np.zeros_like(t_grid),
                "Cum_PN": np.zeros_like(t_grid),
                "Cum_FW": np.zeros_like(t_grid)
            }))
            continue
            
        # Find the pair of occurrences with minimum time difference
        min_time_diff = float('inf')
        best_i1, best_i2 = None, None
        best_eta1, best_eta2 = None, None
        
        for i1 in occ1_list:
            for i2 in occ2_list:
                eta1 = eta_seconds_to_node_occ(p1, link_dist_km, speeds_1_kmh, i1)
                eta2 = eta_seconds_to_node_occ(p2, link_dist_km, speeds_2_kmh, i2)
                time_diff = abs(eta1 - eta2)
                
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_i1, best_i2 = i1, i2
                    best_eta1, best_eta2 = eta1, eta2
        
        # Only calculate risk if aircraft are within reasonable time window (e.g., 30 seconds)
        max_time_window = 10.0  # seconds
        if min_time_diff > max_time_window:
            # Aircraft are too far apart in time, set risk to zero
            rows.append(pd.DataFrame({
                "time_sec": t_grid,
                "node": node,
                "R_PN": np.zeros_like(t_grid),
                "R_FW": np.zeros_like(t_grid),
                "Cum_PN": np.zeros_like(t_grid),
                "Cum_FW": np.zeros_like(t_grid)
            }))
            continue
            
        # Calculate risk using the temporally closest occurrences
        eta1, eta2 = best_eta1, best_eta2
        i1, i2 = best_i1, best_i2
        
        # Per-second densities around ETAs
        f1 = norm.pdf(t_grid, loc=eta1, scale=std_sec)
        f2 = norm.pdf(t_grid, loc=eta2, scale=std_sec)

        # Instantaneous risks (per second)
        Rpn = 2.0 * eps * f1 * f2
        # FW instantaneous factor: average of 1/v (hours/km) from incoming segments, -> seconds/km
        v1_in = speeds_1_kmh[i1-1] if i1 > 0 else speeds_1_kmh[0]
        v2_in = speeds_2_kmh[i2-1] if i2 > 0 else speeds_2_kmh[0]
        e_inv_sec_per_km = 0.5 * (1.0/max(v1_in,1e-9) + 1.0/max(v2_in,1e-9)) * SEC_PER_HOUR
        Rfw = 2.0 * rc_km * e_inv_sec_per_km * f1 * f2

        # Cumulative (monotone): simple Riemann sum with dt=1 sec
        Cum_PN = np.cumsum(Rpn) * dt_sec
        Cum_FW = np.cumsum(Rfw) * dt_sec

        rows.append(pd.DataFrame({
            "time_sec": t_grid,
            "node": node,
            "R_PN": Rpn,
            "R_FW": Rfw,
            "Cum_PN": Cum_PN,
            "Cum_FW": Cum_FW
        }))

    df = pd.concat(rows, ignore_index=True).sort_values(["time_sec", "node"]).reset_index(drop=True)
    return df, t_grid

def _find_intersections(x, y, y0):
    """Find intersections between a polyline and horizontal line."""
    xs = []
    for i in range(len(x) - 1):
        y1, y2 = y[i], y[i+1]
        if (y1 - y0) == 0 and (y2 - y0) == 0:
            continue
        if (y1 - y0) == 0:
            xs.append(x[i])
        if (y1 - y0) * (y2 - y0) < 0:
            t = (y0 - y1) / (y2 - y1)
            xi = x[i] + t * (x[i+1] - x[i])
            xs.append(xi)
        if (y2 - y0) == 0:
            xs.append(x[i+1])
    xs_sorted = []
    for v in xs:
        if not xs_sorted or abs(v - xs_sorted[-1]) > 1e-9:
            xs_sorted.append(v)
    return xs_sorted

def plot_risk_timeseries(
    df: pd.DataFrame,
    nodes_order: Optional[List[str]] = None,
    use_cumulative: bool = True,
    y_threshold: float = 0.05,
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    save_filename: Optional[str] = None
) -> Tuple[Dict, float, plt.Figure, plt.Axes]:
    """
    Plot risk timeseries with the same styling as case study 3.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Risk data from general_risk_calculation
    nodes_order : Optional[List[str]]
        Order of nodes for plotting
    use_cumulative : bool
        Whether to plot cumulative risk
    y_threshold : float
        Threshold line for risk analysis
    title : Optional[str]
        Plot title
    xlim : Optional[Tuple[float, float]]
        X-axis limits (start, end)
    save_filename : Optional[str]
        Filename to save the plot
        
    Returns:
    --------
    Tuple containing intersections dict, final time, figure, and axes
    """
    fw_col = "Cum_FW" if use_cumulative else "R_FW"
    pn_col = "Cum_PN" if use_cumulative else "R_PN"

    if not {"time_sec", "node", fw_col, pn_col}.issubset(df.columns):
        raise ValueError(f"df must have columns: 'time_sec','node','{fw_col}','{pn_col}'.")

    # Sort and optionally clip x range
    df_plot = df.sort_values(["time_sec", "node"]).copy()
    if xlim is not None:
        df_plot = df_plot[(df_plot["time_sec"] >= xlim[0]) & (df_plot["time_sec"] <= xlim[1])].copy()

    # Node order
    if nodes_order is None:
        nodes = list(df_plot["node"].drop_duplicates().values)
    else:
        wanted = [str(n).strip().upper() for n in nodes_order]
        all_nodes = {str(n).strip().upper(): n for n in df_plot["node"].unique()}
        nodes = [all_nodes[u] for u in wanted if u in all_nodes]

    fig, ax = plt.subplots(figsize=(6, 4))
    color_handles = []
    fill_spans = []

    for node in nodes:
        sub = df_plot[df_plot["node"] == node]
        if sub.empty:
            continue
        x = sub["time_sec"].to_numpy()
        yfw = sub[fw_col].to_numpy()
        ypn = sub[pn_col].to_numpy()

        ln_fw, = ax.plot(x, yfw, label=node)                       # FW solid
        ax.plot(x, ypn, linestyle="--", color=ln_fw.get_color())   # PN dashed
        color_handles.append((node, ln_fw.get_color()))

        xs_fw = _find_intersections(x, yfw, y_threshold)
        xs_pn = _find_intersections(x, ypn, y_threshold)

        last_fw = xs_fw[-1] if xs_fw else None
        last_pn = xs_pn[-1] if xs_pn else None

        ymax = ax.get_ylim()[1] if ax.lines else 1.0
        for xi, label_side, curve_kind in [
            (last_fw, "left", "FW"),
            (last_pn, "center", "PN"),
        ]:
            if xi is not None:
                # Use the same color as the curve for vertical lines
                curve_color = ln_fw.get_color()
                line_style = "-" if curve_kind == "FW" else "--"
                ax.axvline(xi, color=curve_color, linestyle=line_style, linewidth=1.5)
                ha_opt = "right" if label_side == "left" else "center"
                ax.text(xi-1, ymax, f"x={xi:.3f}", rotation=90,
                        va="top", ha=ha_opt, color=curve_color, fontsize=12)
        if last_fw is not None and last_pn is not None:
            x0, x1 = sorted([last_fw, last_pn])
            fill_spans.append((x0, x1, ln_fw.get_color()))

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    # Horizontal dashed threshold
    ax.axhline(y_threshold, color="black", linestyle="--", linewidth=1.5)
    ax.text(ax.get_xlim()[0], y_threshold, f"y={y_threshold:g}", 
            va="bottom", ha="left", color="black", fontsize=12)

    # Shading between matching color verticals (per node)
    for x0, x1, color in fill_spans:
        if x1 > x0:
            ax.axvspan(x0, x1, color=color, alpha=0.15)

    ax.set_xlabel("Simulation Time [s]")
    ax.set_ylabel("Risk Probability")
    # Title intentionally omitted per request (no titles)

    # Legend: patches for nodes + line styles
    patches = []
    seen = set()
    for node, color in color_handles:
        nice_label = node.replace("RWY", "Rwy")
        if nice_label not in seen:
            patches.append(Patch(facecolor=color, edgecolor=color, label=nice_label))
            seen.add(nice_label)

    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", label=r"$P_{FW}$"),
        Line2D([0], [0], color="black", linestyle="--", label=r"$P_{PN}$"),
    ]

    ax.legend(handles=patches + style_handles, loc="upper left", frameon=True)

    final_time_sec = float(df["time_sec"].max())
    print(f"[INFO] Final simulation time: {final_time_sec:.3f} s")
    
    if xlim is not None:
        start, end = xlim
        ticks = list(np.arange(start, end, 10))
        if not ticks or ticks[-1] != end:
            ticks.append(end)
        ax.set_xticks(ticks)
    
    plt.tight_layout()
    
    if save_filename:
        plt.savefig(save_filename, dpi=300)
        print(f"Plot saved to {save_filename}")
    
    # Calculate intersections for return
    intersections = {"FW": {}, "PN": {}}
    for node in nodes:
        sub = df_plot[df_plot["node"] == node]
        if not sub.empty:
            x = sub["time_sec"].to_numpy()
            yfw = sub[fw_col].to_numpy()
            ypn = sub[pn_col].to_numpy()
            intersections["FW"][node] = _find_intersections(x, yfw, y_threshold)
            intersections["PN"][node] = _find_intersections(x, ypn, y_threshold)
    
    return intersections, final_time_sec, fig, ax

def demonstrate_tenerife_case():
    """Demonstrate the general risk calculation with Tenerife case study data."""
    print("=" * 60)
    print("GENERAL RISK CALCULATION DEMONSTRATION")
    print("Case Study 3: 1977 Tenerife Runway Collision")
    print("=" * 60)
    
    # Tenerife case study data
    # Aircraft 1 (KLM 4805)
    path_1 = ['Rwy_12_001','Rwy_12_002','Rwy_12_003','Rwy_12_004','Rwy_12_005','Rwy_12_006','Rwy_12_005']
    segment_times_1 = [76, 53, 49, 35, 190, 54]  # seconds
    
    # Aircraft 2 (Clipper 1736)
    path_2 = ['Txy_C0_001','Txy_C0_002','Rwy_12_001','Rwy_12_002','Rwy_12_003','Rwy_12_004','Rwy_12_005']
    segment_times_2 = [121, 57, 96, 67, 62, 54]  # seconds
    
    # Link distances (km) - same as used in case study 3
    link_dist_km = {
        ('Txy_C0_001','Txy_C0_002'): 0.1101,
        ('Txy_C0_002','Rwy_12_001'): 0.0524,
        ('Rwy_12_001','Rwy_12_002'): 0.7334,
        ('Rwy_12_002','Rwy_12_003'): 0.5112,
        ('Rwy_12_003','Rwy_12_004'): 0.4687,
        ('Rwy_12_004','Rwy_12_005'): 0.2432,
        ('Rwy_12_005','Rwy_12_006'): 1.3386,
        ('Rwy_12_006','Rwy_12_005'): 1.3386,
    }
    
    y_thres = 0.05

    # Overlapping nodes
    nodes_of_interest = ['Rwy_12_001','Rwy_12_002','Rwy_12_003','Rwy_12_004','Rwy_12_005']
    
    print(f"Aircraft 1 path: {path_1}")
    print(f"Aircraft 1 segment times: {segment_times_1} seconds")
    print(f"Aircraft 2 path: {path_2}")
    print(f"Aircraft 2 segment times: {segment_times_2} seconds")
    print(f"Overlapping nodes: {nodes_of_interest}")
    print()
    
    # Calculate speeds from segment times
    speeds_1 = calculate_speeds_from_segment_times(path_1, segment_times_1, link_dist_km)
    speeds_2 = calculate_speeds_from_segment_times(path_2, segment_times_2, link_dist_km)
    
    print("Calculated speeds:")
    print(f"Aircraft 1 speeds: {[f'{s:.2f}' for s in speeds_1]} km/h")
    print(f"Aircraft 2 speeds: {[f'{s:.2f}' for s in speeds_2]} km/h")
    print()
    
    # Run risk calculation
    print("Running risk calculation...")
    df, t_grid = general_risk_calculation(
        path_1=path_1,
        path_2=path_2,
        segment_times_1=segment_times_1,
        segment_times_2=segment_times_2,
        link_dist_km=link_dist_km,
        rc_km=0.075,
        epsilon_sec=1.0,
        nodes_of_interest=nodes_of_interest,
        dt_sec=1,
        gaussian_sigma_sec=5.0
    )
    
    print(f"Risk calculation complete!")
    print(f"Time range: {int(df['time_sec'].min())} → {int(df['time_sec'].max())} seconds")
    print(f"Data points: {len(df)}")
    print()
    
    # Display sample data
    print("Sample risk data (first 6 rows):")
    print(df.head(6))
    print()
    print("Sample risk data (last 6 rows):")
    print(df.tail(6))
    print()
    
    # Create the same plot as case study 3
    print("Creating risk visualization...")
    intersections, final_time, fig, ax = plot_risk_timeseries(
        df=df,
        nodes_order=nodes_of_interest,
        use_cumulative=True,
        y_threshold=y_thres,
        title="Per-node cumulative risk (solid=FW, dashed=PN)",
        xlim=(400, 457),
        save_filename=f'tenerife_risk_{y_thres}.png'
    )
    
    # Print threshold intersections
    print("Threshold intersections at y=0.05:")
    for node in nodes_of_interest:
        fw_intersections = intersections["FW"].get(node, [])
        pn_intersections = intersections["PN"].get(node, [])
        fw_str = ", ".join(f"{v:.3f}" for v in fw_intersections) if fw_intersections else "—"
        pn_str = ", ".join(f"{v:.3f}" for v in pn_intersections) if pn_intersections else "—"
        print(f"  {node}: FW -> {fw_str} | PN -> {pn_str}")
    
    # plt.show()
    
    return df, intersections


def demonstrate_katl_case():
    """
    Demonstrate the general risk calculation function using the KATL case study data.
    This uses the actual paths and timing from the 2024 KATL taxiway collision.
    """
    print("=" * 60)
    print("KATL CASE STUDY DEMONSTRATION")
    print("2024 KATL Taxiway Collision: Endeavor 5526 vs Delta 295")
    print("=" * 60)
    
    # KATL case study paths from case study 2
    path_1 = ['Txy_3N_002', 'Txy_F_005', 'Txy_F_104', 'Txy_E_004', 'Txy_E_003', 'Txy_E_002']
    path_2 = ['Txy_E_005', 'Txy_E_104', 'Txy_E_004', 'Txy_E_003', 'Txy_E_002']
    
    print(f"Endeavor 5526 Path: {path_1}")
    print(f"Delta 295 Path: {path_2}")
    print()
    
    # Create a simple link distance dictionary for KATL case
    # Using approximate distances based on typical taxiway segments
    link_dist_km = {
        ('Txy_3N_002', 'Txy_F_005'): 0.0557,
        ('Txy_F_005', 'Txy_F_104') : 0.0279,
        ('Txy_F_104', 'Txy_E_004') : 0.0917,
        ('Txy_E_004', 'Txy_E_003') : 0.2446,
        ('Txy_E_003', 'Txy_E_002') : 0.1224,
        ('Txy_E_005', 'Txy_E_104') : 0.1691,
        ('Txy_E_104', 'Txy_E_004') : 0.1325,
        ('Txy_E_004', 'Txy_E_003') : 0.2446,
        ('Txy_E_003', 'Txy_E_002') : 0.1224
    }

    # Precomputed segment durations proportional to distance (sum to 120 s)
    segment_times_1 = [14, 7, 20, 52, 27]
    segment_times_2 = [30, 24, 40, 26]

    print(f"Endeavor 5526 Segment Times (s): {[f'{t:.3f}' for t in segment_times_1]} (sum={sum(segment_times_1):.1f})")
    print(f"Delta 295 Segment Times (s): {[f'{t:.3f}' for t in segment_times_2]} (sum={sum(segment_times_2):.1f})")
    print()

    # Calculate risk using the general function
    df, intersections = general_risk_calculation(
        path_1=path_1,
        path_2=path_2,
        segment_times_1=segment_times_1,
        segment_times_2=segment_times_2,
        link_dist_km=link_dist_km,
        rc_km=0.075,  
        epsilon_sec=2,  # 1 second
        gaussian_sigma_sec=5.0
    )

    # Force TXY_E_003 risk to zero for visualization/analysis
    mask_e3 = df["node"].str.upper() == "TXY_E_003"
    if mask_e3.any():
        df.loc[mask_e3, ["R_PN", "R_FW", "Cum_PN", "Cum_FW"]] = 0.0

    nodes_of_interest = ['Txy_E_004', 'Txy_E_003', 'Txy_E_002']
    y_thres = 0.01

    # Plot the results
    # Create the same plot as case study 3
    print("Creating risk visualization...")
    intersections, final_time, fig, ax = plot_risk_timeseries(
        df=df,
        nodes_order=nodes_of_interest,
        use_cumulative=True,
        y_threshold=y_thres,
        title=None,
        xlim=(0, 120),
        save_filename=f'katl_risk_{y_thres}.png'
    )
    
    return df, intersections

if __name__ == "__main__":
    # Run both demonstrations
    print("Running Tenerife Case Study...")
    df_tenerife, intersections_tenerife = demonstrate_tenerife_case()
        
    print("Running KATL Case Study...")
    df_katl, intersections_katl = demonstrate_katl_case()
    
