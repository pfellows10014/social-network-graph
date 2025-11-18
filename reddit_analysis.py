# File: twitter_analysis_loading_only.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import LongType
from graphframes import GraphFrame
from pyspark import StorageLevel
import sys
import os

# --- Configuration ---
# 1. Update this path to the location of your downloaded 'twitter_combined.txt'
FILE_PATH = "twitter_combined.txt" 
OUTPUT_DIR = "twitter_analysis_output"

# GLOBAL SPARK SESSION INITIALIZATION
# Requires correct configuration in .vscode/launch.json to load GraphFrames.
try:
    spark = SparkSession.builder \
        .appName("TwitterGraphAnalysisLoading") \
        .getOrCreate()
except Exception as e:
    print(f"Error initializing SparkSession: {e}")
    sys.exit(1)

SPARK_CHECKPOINT_DIR = "/tmp/spark-checkpoint" 
spark.sparkContext.setCheckpointDir(SPARK_CHECKPOINT_DIR)

# --- Core Functions (2.1 & 2.2) ---

def load_data_and_prepare_structure(file_path: str):
    """
    Loads edge data, parses it, and creates both the Edge and Vertex DataFrames 
    to prepare for GraphFrame construction.
    """
    print("\n--- 2.1 Loading Data & Parsing ---")

    # Load the raw edge data: Edges are (Source ID, Destination ID) separated by a space.
    edges_df = spark.read.csv(file_path, sep=" ", header=False).toDF("src", "dst")
    
    # Define parser: Cast IDs to LongType and handle missing data.
    edges_df = edges_df.withColumn("src", col("src").cast(LongType())) \
                       .withColumn("dst", col("dst").cast(LongType())) \
                       .na.drop()
    
    print(f"Total loaded and parsed edges: {edges_df.count()}")
    edges_df.printSchema()

    print("\n--- 2.2 Creating Vertex Structure ---")
    
    # Create the Vertices DataFrame (id column is mandatory)
    vertices_df = edges_df.select(col("src").alias("id")) \
        .union(edges_df.select(col("dst").alias("id"))) \
        .distinct()
    
    # Add a simple property (optional)
    vertices_df = vertices_df.withColumn("type", lit("user"))

    print(f"Unique vertices identified: {vertices_df.count()}")
    vertices_df.printSchema()
    
    # Return structures ready for GraphFrame(vertices_df, edges_df)
    return vertices_df, edges_df

def create_graph(vertices_df, edges_df) -> GraphFrame:
    """
    Step 2.2: Defines edge and vertex structure and creates a property graph.
    """
    print("\n--- 2.2 Creating GraphFrame ---")
    
    g = GraphFrame(vertices_df, edges_df)
    
    print(f"Graph created with {g.vertices.count()} vertices and {g.edges.count()} edges.")
    return g

def save_results(df, name):
    """Saves DataFrame results to a single CSV file in the output directory."""
    output_path = f"{OUTPUT_DIR}/{name}"
    df.coalesce(1).write.csv(output_path, mode="overwrite", header=True)
    print(f"Saved results to: {output_path}")

# --- Query Functions (2.3 a & b) ---
def find_top_outdegree(g: GraphFrame):
    """
    Query 2.3a: Find the top 5 nodes with the highest outdegree (most users followed).
    """
    print("\n--- 2.3a. Top 5 Nodes with Highest Outdegree ---")
    top_out_degree = g.outDegrees.withColumnRenamed("id", "user_id") \
        .orderBy(col("outDegree").desc()) \
        .limit(5)
    top_out_degree.show()
    save_results(top_out_degree, "top_5_outdegree")

def find_top_indegree(g: GraphFrame):
    """
    Query 2.3b: Find the top 5 nodes with the highest indegree (most followers).
    """
    print("\n--- 2.3b. Top 5 Nodes with Highest Indegree ---")
    top_in_degree = g.inDegrees.withColumnRenamed("id", "user_id") \
        .orderBy(col("inDegree").desc()) \
        .limit(5)
    top_in_degree.show()
    save_results(top_in_degree, "top_5_indegree")

# --- Query Functions (2.3 c, d, & e) ---

def calculate_pagerank(g: GraphFrame):
    """
    Query 2.3c: Calculate PageRank and output the top 5 nodes.
    """
    print("\n--- 2.3c. Top 5 Nodes with Highest PageRank ---")
    # Run PageRank: maxIter=5 is a standard setting for large networks.
    pr_results = g.pageRank(resetProbability=0.15, maxIter=5)

    top_pagerank = pr_results.vertices \
        .orderBy(col("pagerank").desc()) \
        .limit(5)

    top_pagerank.select("id", "pagerank").show(truncate=False)
    save_results(top_pagerank.select("id", "pagerank"), "top_5_pagerank")

def find_connected_components(g: GraphFrame):
    """
    Query 2.3d: Run the Connected Components algorithm and find the top 5 largest components.
    """
    print("\n--- 2.3d. Top 5 Largest Connected Components ---")
    # Note: Running Connected Components can be time-consuming.
    cc_results = g.connectedComponents()

    # Find the size of each component and rank the top 5
    top_components = cc_results.groupBy("component") \
        .count() \
        .withColumnRenamed("count", "componentSize") \
        .orderBy(col("componentSize").desc()) \
        .limit(5)

    top_components.show()
    save_results(top_components, "top_5_connected_components")

def calculate_triangle_counts(g: GraphFrame):
    """
    Query 2.3e: Run the Triangle Counts algorithm and output the top 5 vertices.
    """
    print("\n--- 2.3e. Top 5 Vertices with Largest Triangle Count ---")
    
    # Run triangleCount. The result (tc_results) is likely the DataFrame itself.
    tc_results = g.triangleCount(
        storage_level=StorageLevel.MEMORY_AND_DISK
    )

    # FIX: Remove the .vertices accessor
    top_triangle_count = tc_results \
        .orderBy(col("count").desc()) \
        .limit(5)

    top_triangle_count.select("id", col("count").alias("triangleCount")).show()
    save_results(top_triangle_count.select("id", col("count").alias("triangleCount")), "top_5_triangle_counts")

if __name__ == "__main__":
    
    # 1. Load Data and Prepare Structure
    vertices_df, edges_df = load_data_and_prepare_structure(FILE_PATH)

    # 2. Create Graph
    g = create_graph(vertices_df, edges_df)

    # 3. Run Queries
    find_top_outdegree(g)
    find_top_indegree(g)
    calculate_pagerank(g)
    find_connected_components(g)
    calculate_triangle_counts(g)

    # Final Cleanup
    spark.stop()