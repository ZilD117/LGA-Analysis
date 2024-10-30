import pandas as pd
import matplotlib.pyplot as plt

airport = 'KBOS'
filePath = 'Airport Layouts/' + airport + '_Nodes_Links.csv'
linksDf = pd.read_csv(filePath)

def plotAirportGraph(linksDf):
    nodes = pd.concat([
        linksDf[['n1.id', 'n1.lat', 'n1.lon']].rename(columns={'n1.id': 'id', 'n1.lat': 'lat', 'n1.lon': 'lon'}),
        linksDf[['n2.id', 'n2.lat', 'n2.lon']].rename(columns={'n2.id': 'id', 'n2.lat': 'lat', 'n2.lon': 'lon'})
    ]).drop_duplicates()

    nodePositions = {row['id']: (row['lat'], row['lon']) for _, row in nodes.iterrows()}

    plt.figure(figsize=(12, 12))

    for _, row in linksDf.iterrows():
        n1Pos = (row['n1.lat'], row['n1.lon'])
        n2Pos = (row['n2.lat'], row['n2.lon'])
        plt.plot([n1Pos[1], n2Pos[1]], [n1Pos[0], n2Pos[0]], color='gray', linestyle='-', linewidth=0.5)

    for nodeId, (lat, lon) in nodePositions.items():
        plt.scatter(lon, lat, color='blue', s=20, zorder=2)
        plt.text(lon, lat, nodeId, fontsize=8, ha='right', va='bottom')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(str(airport) + ' Layout')
    plt.grid(True)
    plt.show()

plotAirportGraph(linksDf)
