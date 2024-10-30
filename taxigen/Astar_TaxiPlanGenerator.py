import pandas as pd
import numpy as np
from collections import defaultdict
import heapq
import os

def loadData(defFiles, linkFiles):
    graphs = {}
    nodePositions = {}

    for defFile, linkFile in zip(defFiles, linkFiles):
        nodesDf = pd.read_csv(defFile)
        nodes = nodesDf[['id', 'lat', 'lon']]

        linksDf = pd.read_csv(linkFile)
        links = linksDf[['n1.id', 'n2.id', 'n1.lat', 'n1.lon', 'n2.lat', 'n2.lon']]

        airportName = os.path.basename(defFile).split('_')[0]
        nodePositions[airportName] = {row['id']: (row['lat'], row['lon']) for _, row in nodes.iterrows()}

        graph = defaultdict(list)
        for _, row in links.iterrows():
            n1Id, n2Id = row['n1.id'], row['n2.id']
            n1Lat, n1Lon, n2Lat, n2Lon = row['n1.lat'], row['n1.lon'], row['n2.lat'], row['n2.lon']
            distance = haversine(n1Lat, n1Lon, n2Lat, n2Lon)
            graph[n1Id].append((n2Id, distance))
            graph[n2Id].append((n1Id, distance))

        graphs[airportName] = graph

    return graphs, nodePositions

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def aStar(graph, start, goal, nodePositions):
    def heuristic(n1, n2):
        lat1, lon1 = nodePositions[n1]
        lat2, lon2 = nodePositions[n2]
        return haversine(lat1, lon1, lat2, lon2)

    openList = []
    heapq.heappush(openList, (0, 0, [start]))

    gCost = {start: 0}

    while openList:
        _, currentCost, path = heapq.heappop(openList)
        currentNode = path[-1]

        if currentNode == goal:
            return path, currentCost

        for neighbor, weight in graph[currentNode]:
            tentativeGCost = currentCost + weight

            if neighbor not in gCost or tentativeGCost < gCost[neighbor]:
                gCost[neighbor] = tentativeGCost
                fCost = tentativeGCost + heuristic(neighbor, goal)
                heapq.heappush(openList, (fCost, tentativeGCost, path + [neighbor]))

    return None, float('inf')

if __name__ == "__main__":
    airports = [
        "AMS", "BOM", "CAI", "CDG", "DEL", "DME", "DXB", "FCO", "FRA", "HKG", "HND",
        "ICN", "IST", "JNB", "KABQ", "KATL", "KBDL", "KBHM", "KBNA", "KBOI", "KBOS",
        "KBTV", "KBUR", "KBWI", "KBZN", "KCHS", "KCLE", "KCLT", "KCRW", "KCVG", "KDAL",
        "KDCA", "KDEN", "KDFW", "KDSM", "KDTW", "KEWR", "KFAR", "KFLL", "KFSD", "KGYY",
        "KHPN", "KIAD", "KIAH", "KICT", "KILG", "KIND", "KISP", "KJAC", "KJAN", "KJAX",
        "KJFK", "KLAS", "KLAX", "KLEX", "KLGA", "KLGB", "KLIT", "KMCO", "KMDW", "KMEM",
        "KMHT", "KMIA", "KMKE", "KMSP", "KMSY", "KOAK", "KOKC", "KOMA", "KONT", "KORD",
        "KPBI", "KPDX", "KPHL", "KPHX", "KPIT", "KPVD", "KPWM", "KSAN", "KSAT", "KSDF",
        "KSEA", "KSFO", "KSJC", "KSLC", "KSNA", "KSTL", "KSWF", "KTEB", "KTPA", "KUL",
        "KVGT", "LHR", "MAD", "MEX", "PANC", "PEK", "PHNL", "PVG", "SIN", "SYD", "TLV",
        "TPE", "YYZ"
    ]

    defFiles = [f"Airport Layouts/{airport}_Nodes_Def.csv" for airport in airports]
    linkFiles = [f"Airport Layouts/{airport}_Nodes_Links.csv" for airport in airports]

    graphs, nodePositions = loadData(defFiles, linkFiles)

    airport = 'KATL'
    startNode = 'Gate_A_009'
    goalNode = 'Rwy_01_001'

    graph = graphs.get(airport)
    positions = nodePositions.get(airport)

    shortestPath, totalDistance = aStar(graph, startNode, goalNode, positions)

    if shortestPath:
        print(f"Shortest Path: {shortestPath}, Distance: {totalDistance:.2f} km")
    else:
        print("No path found.")
