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

def aStarMultiple(graph, start, goal, nodePositions, k=5):
    def heuristic(n1, n2):
        lat1, lon1 = nodePositions[n1]
        lat2, lon2 = nodePositions[n2]
        return haversine(lat1, lon1, lat2, lon2)

    openPaths = []
    heapq.heappush(openPaths, (0, [start]))

    kPaths = []
    pathCosts = {}

    while openPaths and len(kPaths) < k:
        currentF, currentPath = heapq.heappop(openPaths)
        currentNode = currentPath[-1]

        if currentNode == goal:
            totalDistance = sum(
                haversine(nodePositions[currentPath[i]][0], nodePositions[currentPath[i]][1],
                          nodePositions[currentPath[i + 1]][0], nodePositions[currentPath[i + 1]][1])
                for i in range(len(currentPath) - 1)
            )
            kPaths.append((currentPath, totalDistance))
            continue

        for neighbor, weight in graph[currentNode]:
            if neighbor not in currentPath:
                newPath = currentPath + [neighbor]
                gCost = currentF + weight
                fCost = gCost + heuristic(neighbor, goal)

                if tuple(newPath) not in pathCosts or pathCosts[tuple(newPath)] > gCost:
                    pathCosts[tuple(newPath)] = gCost
                    heapq.heappush(openPaths, (fCost, newPath))

    kPaths.sort(key=lambda x: x[1])
    return kPaths

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

    k = 10
    kShortestPaths = aStarMultiple(graph, startNode, goalNode, positions, k=k)

    for i, (path, distance) in enumerate(kShortestPaths, start=1):
        print(f"Path {i}: {path}, Distance: {distance:.2f} km")
