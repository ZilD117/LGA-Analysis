from pyspark.sql import SparkSession
from pyspark.sql.types import *
from sedona.register import SedonaRegistrator
from sedona.utils import SedonaKryoRegistrator, KryoSerializer
import pandas as pd
from sedona.spark import *

config = SedonaContext.builder(). \
    config('spark.jars.packages',
           'org.apache.sedona:sedona-spark-3.3_2.12:1.7.0,'
           'org.datasyslab:geotools-wrapper:1.7.0-28.5'). \
    config('spark.jars.repositories', 'https://artifacts.unidata.ucar.edu/repository/unidata-all'). \
    getOrCreate()
sedona = SedonaContext.create(config)

spark = SparkSession.\
        builder.\
        master("local[*]").\
        appName("Sector_IFF_Parser").\
        config("spark.serializer", KryoSerializer.getName).\
        config("spark.kryo.registrator", SedonaKryoRegistrator.getName) .\
        config("spark.driver.memory", "150g") .\
        config("spark.executor.memory", "150g") .\
        config("spark.executor.instances", "40") .\
        config("spark.jars.packages", "org.apache.sedona:sedona-python-adapter-3.0_2.12:1.0.0-incubating,org.datasyslab:geotools-wrapper:geotools-24.0") .\
        getOrCreate()
SedonaRegistrator.registerAll(spark)

sc = spark.sparkContext
def load_schema():
    myschema = StructType([
        StructField("recType", ShortType(), True),  # 1  //track point record type number
        StructField("recTime", StringType(), True),  # 2  //seconds since midnigght 1/1/70 UTC
        StructField("fltKey", LongType(), True),  # 3  //flight key
        StructField("bcnCode", IntegerType(), True),  # 4  //digit range from 0 to 7
        StructField("cid", IntegerType(), True),  # 5  //computer flight id
        StructField("Source", StringType(), True),  # 6  //source of the record
        StructField("msgType", StringType(), True),  # 7
        StructField("acId", StringType(), True),  # 8  //call sign
        StructField("recTypeCat", IntegerType(), True),  # 9
        StructField("lat", DoubleType(), True),  # 10
        StructField("lon", DoubleType(), True),  # 11
        StructField("alt", DoubleType(), True),  # 12  //in 100s of feet
        StructField("significance", ShortType(), True),  # 13 //digit range from 1 to 10
        StructField("latAcc", DoubleType(), True),  # 14
        StructField("lonAcc", DoubleType(), True),  # 15
        StructField("altAcc", DoubleType(), True),  # 16
        StructField("groundSpeed", IntegerType(), True),  # 17 //in knots
        StructField("course", DoubleType(), True),  # 18  //in degrees from true north
        StructField("rateOfClimb", DoubleType(), True),  # 19  //in feet per minute
        StructField("altQualifier", StringType(), True),  # 20  //Altitude qualifier (the “B4 character”)
        StructField("altIndicator", StringType(), True),  # 21  //Altitude indicator (the “C4 character”)
        StructField("trackPtStatus", StringType(), True),  # 22  //Track point status (e.g., ‘C’ for coast)
        StructField("leaderDir", IntegerType(), True),  # 23  //int 0-8 representing the direction of the leader line
        StructField("scratchPad", StringType(), True),  # 24
        StructField("msawInhibitInd", ShortType(), True),  # 25 // MSAW Inhibit Indicator (0=not inhibited, 1=inhibited)
        StructField("assignedAltString", StringType(), True),  # 26
        StructField("controllingFac", StringType(), True),  # 27
        StructField("controllingSec", StringType(), True),  # 28
        StructField("receivingFac", StringType(), True),  # 29
        StructField("receivingSec", StringType(), True),  # 30
        StructField("activeContr", IntegerType(), True),  # 31  // the active control number
        StructField("primaryContr", IntegerType(), True),
        # 32  //The primary(previous, controlling, or possible next)controller number
        StructField("kybrdSubset", StringType(), True),  # 33  //identifies a subset of controller keyboards
        StructField("kybrdSymbol", StringType(), True),  # 34  //identifies a keyboard within the keyboard subsets
        StructField("adsCode", IntegerType(), True),  # 35  //arrival departure status code
        StructField("opsType", StringType(), True),  # 36  //Operations type (O/E/A/D/I/U)from ARTS and ARTS 3A data
        StructField("airportCode", StringType(), True),  # 37
        StructField("trackNumber", IntegerType(), True),  # 38
        StructField("tptReturnType", StringType(), True),  # 39
        StructField("modeSCode", StringType(), True)  # 40
    ])
    return myschema

if __name__ == "__main__":

    iff_schema = load_schema()
    # The date of ASDE-X we are going to process
    #date = '20220101'
    # The path to the csv file
    data_path = "/Users/alexporcayo/Documents/python/IFF_ATL+ASDEX_20230506.csv"
    node_path = "/Users/alexporcayo/Documents/python/KATL_Nodes_Def.csv"
    links_path = "/Users/alexporcayo/Documents/python/KATL_Nodes_Links (1).csv"
    df = spark.read.csv(data_path, header=False, sep=",", schema=iff_schema)
    nodes_df = spark.read.csv(node_path, header=True, inferSchema=True)
    links_df = spark.read.csv(links_path, header=True, inferSchema=True)
    nodes_df.createOrReplaceTempView("nodes")
    links_df.createOrReplaceTempView("links")
    nodes_with_geom = spark.sql("""
        SELECT id, index, lon, lat, ST_Point(lon, lat) AS geom
        FROM nodes
    """)
    links_with_geom = spark.sql("""
        SELECT `n1.index`, `n2.index`
        FROM links
    """)
    nodes_pd = nodes_with_geom.toPandas()
    links_pd = links_with_geom.toPandas()
    import networkx as nx
    import matplotlib.pyplot as plt
    # Create a dictionary mapping each node id to its (lon, lat) coordinates
    node_positions = {row['index']: (row['lon'], row['lat']) for _, row in nodes_pd.iterrows()}
    # Build a graph from the links DataFrame
    G = nx.from_pandas_edgelist(links_pd, source='n1.index', target='n2.index')
    # Plot the graph using the node positions defined above
    nx.draw(G, pos=node_positions,
            with_labels=True,
            node_color='skyblue',
            edge_color='gray',
            node_size=500)
    plt.title("Nodes and Links Plot")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()