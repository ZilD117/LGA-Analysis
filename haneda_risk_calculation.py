#!/usr/bin/env python3
"""
Haneda Airport 2024 Runway Incursion Risk Calculation
Case Study 1: Japan Airlines Flight 516 vs Japan Coast Guard JA722A

This script implements the risk calculation methodology from case study 1,
which uses lognormal distributions and Fenton-Wilkinson approximation
for collision risk assessment at intersection nodes.

Author: Generated based on case study 1 methodology
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.stats import lognorm
from scipy.integrate import quad
from typing import List, Tuple, Dict, Optional
import os

# Constants
SEC_PER_HOUR = 3600.0

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance (in meters) between two points on Earth.
    """
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    distance = 2 * R * math.asin(math.sqrt(a))
    return distance / 1609.34  # in miles

def get_speed_params(node1, node2):
    """
    Return (mu, sigma) based on the node "type" extracted from the node names.
    
    The node type is defined as the alphabetic characters before the first underscore,
    converted to upper-case.
    
    Rules:
      - If both types are "RWY": return (30, 10)
      - If one type is "RWY" and the other is "TXY": return (25, 5)
      - If both types are "TXY": return (20, 5)
      - Otherwise: return (15, 5)
    """
    # Define a pattern that captures alphabetic characters at the beginning of the string.
    pattern = re.compile(r"^([A-Za-z]+)")
    
    # Extract type from node1.
    match1 = pattern.match(node1.strip())
    type1 = match1.group(1).upper() if match1 else ""
    
    # Extract type from node2.
    match2 = pattern.match(node2.strip())
    type2 = match2.group(1).upper() if match2 else ""
    
    # Apply the rules.
    if type1 == "RWY" and type2 == "RWY":
        return (30, 10)
    elif {"RWY", "TXY"} == {type1, type2}:  # one is RWY and the other is TXY
        return (25, 5)
    elif type1 == "TXY" and type2 == "TXY":
        return (20, 5)
    else:
        return (15, 5)

def create_link_dict(linksDf):
    """
    Create a dictionary mapping a link (node1, node2) to a tuple:
      (distance in miles, mu, sigma).
    
    Node names are normalized (strip and upper-case) for consistency.
    """
    link_dict = {}
    for idx, row in linksDf.iterrows():
        node1 = str(row['n1.id']).strip().upper()
        node2 = str(row['n2.id']).strip().upper()
        lat1 = float(row['n1.lat'])
        lon1 = float(row['n1.lon'])
        lat2 = float(row['n2.lat'])
        lon2 = float(row['n2.lon'])
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        mu, sigma = get_speed_params(node1, node2)
        # Store the link in one direction. Since lookups will check both orders,
        # it is sufficient to store it only once.
        link_dict[(node1, node2)] = (distance, mu, sigma)
    return link_dict

def compute_total_moments_links(links):
    """
    Given a list of links (each a tuple: (distance, mu, sigma)),
    compute the total mean M and variance V of travel time.
    For each link, the travel time tau is modeled as:
        tau ~ Lognormal(ln(distance) - mu, sigma^2)
    with
        E[tau] = distance * exp(-mu + sigma^2/2)
        Var[tau] = distance^2 * exp(-2*mu+sigma^2)*(exp(sigma^2)-1)
    """
    M, V = 0.0, 0.0
    for (d, mu, sigma) in links:
        E_tau = d * np.exp(-mu + 0.5 * sigma**2)
        Var_tau = d**2 * np.exp(-2*mu + sigma**2) * (np.exp(sigma**2) - 1)
        M += E_tau
        V += Var_tau
    return M, V

def fenton_wilkinson_params(M, V):
    """
    Compute the Fenton-Wilkinson parameters (mu_star, sigma_star) given the total mean (M) and variance (V).
    """
    sigma2_star = np.log(1 + V / M**2)
    sigma_star = np.sqrt(sigma2_star)
    mu_star = np.log(M) - 0.5 * sigma2_star
    return mu_star, sigma_star

def travel_time_pdf(t, mu_star, sigma_star):
    """
    Probability density function (PDF) of a lognormal distribution.
    """
    return lognorm.pdf(t, s=sigma_star, scale=np.exp(mu_star))

def collision_risk(links1, links2):
    """
    Compute collision risk for two cumulative segments where:
      - links1: list of tuples (distance, mu, sigma) for aircraft 1
      - links2: list of tuples (distance, mu, sigma) for aircraft 2
    
    Collision risk is measured as the overlap of the lognormal distributions for the two aircraft,
    evaluated at the point where their travel times are equal (i.e., Gamma1 = Gamma2).
    """
    # Compute total mean and variance for both aircraft
    M1, V1 = compute_total_moments_links(links1)
    mu1, sigma1 = fenton_wilkinson_params(M1, V1)
    
    M2, V2 = compute_total_moments_links(links2)
    mu2, sigma2 = fenton_wilkinson_params(M2, V2)
    
    def integrand(t):
        """
        The integrand computes the joint PDF of the difference Γ = Γ1 - Γ2.
        This is the product of the PDFs for both aircraft, evaluated at the same time `t`.
        """
        # PDF of both aircraft at time `t`
        pdf1 = travel_time_pdf(t, mu1, sigma1)
        pdf2 = travel_time_pdf(t, mu2, sigma2)
        # Joint PDF: Product of individual PDFs for aircraft 1 and 2 at time `t`
        return pdf1 * pdf2
    
    # Perform the integration over all time `t` to get the collision risk
    risk, err = quad(integrand, 0, np.inf)
    
    # Ensure the risk is non-negative (clamping)
    return max(risk, 0)

def get_cumulative_links(path, target, link_dict, default_link=None):
    """
    Given a path (list of node names) and a target node,
    return the list of link tuples (distance, mu, sigma) from the start of the path up to the target.
    
    Since the links are non-directional, if the link (node_i, node_i+1) is not found,
    the function also checks for the reversed key (node_i+1, node_i).
    
    Node names are normalized (stripped and converted to upper-case) for consistency.
    """
    # Normalize path elements and target.
    norm_path = [p.strip().upper() for p in path]
    target = target.strip().upper()
    
    if target not in norm_path:
        return None
    
    links = []
    # Iterate over each consecutive pair in the cumulative segment.
    for i in range(norm_path.index(target)):
        key = (norm_path[i], norm_path[i+1])
        # Check if key is in the dictionary
        if key in link_dict:
            links.append(link_dict[key])
        # If not found, try the reverse key
        elif (norm_path[i+1], norm_path[i]) in link_dict:
            links.append(link_dict[(norm_path[i+1], norm_path[i])])
        else:
            if default_link is not None:
                links.append(default_link)
            else:
                raise ValueError(f"Link {key} not found in dictionary (nor its reverse).")
    return links

def intersection_of_lists(list1, list2):
    """
    Return the intersection of two lists (in order of appearance in list1).
    """
    return [x for x in list1 if x in list2]

def haneda_risk_calculation(
    path_1: List[str],
    path_2: List[str], 
    linksDf: pd.DataFrame,
    aircraft2_speed_override: Optional[Tuple[float, float]] = None,
    rc_km: float = 0.075
) -> Tuple[Dict[str, float], List[str]]:
    """
    Calculate collision risk for Haneda case study using lognormal methodology.
    
    Parameters:
    -----------
    path_1 : List[str]
        First aircraft path (Japan Air 516)
    path_2 : List[str] 
        Second aircraft path (JA722A)
    linksDf : pd.DataFrame
        Airport links dataframe with columns n1.id, n1.lat, n1.lon, n2.id, n2.lat, n2.lon
    aircraft2_speed_override : Optional[Tuple[float, float]]
        Override speed parameters (mu, sigma) for aircraft 2
    rc_km : float
        Collision radius in km
        
    Returns:
    --------
    Tuple[Dict[str, float], List[str]]
        Risk results dictionary and intersection nodes list
    """
    # Create the link dictionary from linksDf
    link_dict = create_link_dict(linksDf)
    
    # Compute the intersection of the two paths
    intersection_nodes = intersection_of_lists(path_1, path_2)
    print(f"Intersection nodes for Path1 and Path2: {intersection_nodes}")
    
    # For each intersection node, calculate collision risk
    risk_results = {}
    for node in intersection_nodes:
        cum_links1 = get_cumulative_links(path_1, node, link_dict)
        cum_links2 = get_cumulative_links(path_2, node, link_dict)
        
        # Apply speed override for aircraft 2 if provided
        if aircraft2_speed_override is not None:
            mu_override, sigma_override = aircraft2_speed_override
            cum_links2 = [(t[0], mu_override, sigma_override) for t in cum_links2]
        
        # Skip if there is no link (i.e. if the intersection is the very first node)
        if cum_links1 is None or cum_links2 is None or len(cum_links1) == 0 or len(cum_links2) == 0:
            print(f"Skipping node {node} (insufficient cumulative links).")
            continue
            
        risk = collision_risk(cum_links1, cum_links2)
        risk_results[node] = risk
        print(f"Collision risk at node {node}: {risk:.4e}")
    
    return risk_results, intersection_nodes

def plot_haneda_risk_visualization(
    risk_results: Dict[str, float],
    intersection_nodes: List[str],
    save_filename: Optional[str] = None
) -> plt.Figure:
    """
    Create risk visualization plot for Haneda case study.
    Focus on risk until aircraft reaches Rwy_03_006 (~45 seconds).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot risk values
    ax.plot(intersection_nodes, [risk_results.get(node, 0) for node in intersection_nodes], 
            marker='o', linewidth=2, markersize=8)
    
    ax.set_xlabel('Intersection Node')
    ax.set_ylabel('Collision Risk')
    ax.set_title('Haneda Airport 2024: Collision Risk at Intersection Nodes\n(Until Aircraft Reaches Rwy_03_006 ~45s)')
    ax.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, node in enumerate(intersection_nodes):
        risk = risk_results.get(node, 0)
        ax.annotate(f"{risk:.2e}", 
                   xy=(i, risk), 
                   xytext=(0, 10), 
                   textcoords='offset points', 
                   ha='center', 
                   fontsize=9)
    
    # Highlight the critical node Rwy_03_006
    if 'Rwy_03_006' in intersection_nodes:
        idx = intersection_nodes.index('Rwy_03_006')
        ax.axvline(x=idx, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(idx, ax.get_ylim()[1]*0.8, 'Critical Node\nRwy_03_006', 
                ha='center', va='top', color='red', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Risk visualization saved to {save_filename}")
    
    return fig

def demonstrate_haneda_case():
    """
    Demonstrate the Haneda case study risk calculation.
    """
    print("=" * 60)
    print("HANEDA AIRPORT 2024 RISK CALCULATION")
    print("Case Study 1: Japan Airlines Flight 516 vs Japan Coast Guard JA722A")
    print("=" * 60)
    
    # Aircraft paths from case study 1
    # Japan Air 516 path (from runway 03_001 to 03_011)
    path_1 = ['Rwy_03_001', 'Rwy_03_002', 'Rwy_03_003', 'Rwy_03_004', 
              'Rwy_03_005', 'Rwy_03_006', 'Rwy_03_007', 'Rwy_03_008', 
              'Rwy_03_009', 'Rwy_03_010', 'Rwy_03_011']
    
    # JA722A path (from taxiway C5 to runway 03_011)
    path_2 = ['Txy_C5_C5B', 'Rwy_03_006', 'Rwy_03_007', 'Rwy_03_008', 
              'Rwy_03_009', 'Rwy_03_010', 'Rwy_03_011']
    
    print(f"Japan Air 516 Path: {path_1}")
    print(f"JA722A Path: {path_2}")
    print()
    
    # Load airport layout data
    icao = 'HND'
    airport_nodes_path = f'./Airport Layouts/{icao}_Nodes_Def.csv'
    airport_links_path = f'./Airport Layouts/{icao}_Nodes_Links.csv'
    
    try:
        linksDf = pd.read_csv(airport_links_path)
        print(f"Loaded airport links data: {len(linksDf)} links")
    except FileNotFoundError:
        print(f"Airport layout file not found: {airport_links_path}")
        print("Creating mock data for demonstration...")
        
        # Create mock links data for demonstration
        mock_links = []
        for i in range(len(path_1) - 1):
            mock_links.append({
                'n1.id': path_1[i],
                'n1.lat': 35.55 + i * 0.001,
                'n1.lon': 139.80 + i * 0.001,
                'n2.id': path_1[i+1],
                'n2.lat': 35.55 + (i+1) * 0.001,
                'n2.lon': 139.80 + (i+1) * 0.001
            })
        
        for i in range(len(path_2) - 1):
            mock_links.append({
                'n1.id': path_2[i],
                'n1.lat': 35.545 + i * 0.001,
                'n1.lon': 139.795 + i * 0.001,
                'n2.id': path_2[i+1],
                'n2.lat': 35.545 + (i+1) * 0.001,
                'n2.lon': 139.795 + (i+1) * 0.001
            })
        
        linksDf = pd.DataFrame(mock_links)
        print(f"Created mock links data: {len(linksDf)} links")
    
    # Calculate risk with speed override for JA722A (slower aircraft)
    aircraft2_speed_override = (3, 1.2)  # (mu, sigma) for slower speed
    
    print("Calculating collision risk...")
    risk_results, intersection_nodes = haneda_risk_calculation(
        path_1=path_1,
        path_2=path_2,
        linksDf=linksDf,
        aircraft2_speed_override=aircraft2_speed_override,
        rc_km=0.075
    )
    
    print(f"\nRisk calculation complete!")
    print(f"Number of intersection nodes: {len(intersection_nodes)}")
    print(f"Risk results: {len(risk_results)} nodes with calculated risk")
    print()
    
    # Create visualization
    print("Creating risk visualization...")
    fig = plot_haneda_risk_visualization(
        risk_results=risk_results,
        intersection_nodes=intersection_nodes,
        save_filename='haneda_risk_visualization.png'
    )
    
    # Display summary
    print("\nRisk Summary:")
    print("-" * 40)
    total_risk = sum(risk_results.values())
    print(f"Total collision risk: {total_risk:.4e}")
    
    if risk_results:
        max_risk_node = max(risk_results, key=risk_results.get)
        max_risk_value = risk_results[max_risk_node]
        print(f"Highest risk node: {max_risk_node} ({max_risk_value:.4e})")
        
        min_risk_node = min(risk_results, key=risk_results.get)
        min_risk_value = risk_results[min_risk_node]
        print(f"Lowest risk node: {min_risk_node} ({min_risk_value:.4e})")
    
    return risk_results, intersection_nodes, fig

if __name__ == "__main__":
    # Run the Haneda case study demonstration
    risk_results, intersection_nodes, fig = demonstrate_haneda_case()
    
    # Show the plot
    plt.show()
