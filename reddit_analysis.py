# File: reddit_analysis.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, avg, count, split, element_at, to_timestamp
from pyspark.sql.types import LongType, StringType, DoubleType, StructType, StructField, TimestampType # Added TimestampType
from graphframes import GraphFrame
from pyspark import StorageLevel
import sys
import os

# --- Configuration & Global Setup ---
FILE_PATH_BODY = "soc-RedditHyperlinks-body.tsv"
FILE_PATH_TITLE = "soc-RedditHyperlinks-title.tsv"
OUTPUT_DIR = "reddit_analysis_output"

try:
    spark = SparkSession.builder.appName("RedditGraphAnalysis").getOrCreate()
except Exception as e:
    print(f"Error initializing SparkSession: {e}")
    sys.exit(1)

SPARK_CHECKPOINT_DIR = "/tmp/spark-reddit-checkpoint" 
spark.sparkContext.setCheckpointDir(SPARK_CHECKPOINT_DIR)

def save_results(df, name):
    output_path = f"{OUTPUT_DIR}/{name}"
    df.coalesce(1).write.csv(output_path, mode="overwrite", header=True)
    print(f"Saved results to: {output_path}")

def get_base_schema():
    """
    Defines the 6-field schema with all problematic numeric fields defined as StringType
    to prevent initial data loss.
    """
    return StructType([
        StructField("SOURCE_SUBREDDIT", StringType(), True),
        StructField("TARGET_SUBREDDIT", StringType(), True),
        StructField("POST_ID", StringType(), True),
        StructField("TIMESTAMP", StringType(), True),  # Loaded as String
        StructField("LINK_SENTIMENT", StringType(), True), # Loaded as String
        StructField("PROPERTIES", StringType(), True) 
    ])

def parse_and_enrich_data(spark, path_body, path_title):
    """Loads both files, unions them, and parses feature columns."""
    
    base_schema = get_base_schema()
    
    # 1. Load, Union, and Clean Raw Data
    df_body = spark.read.csv(path_body, sep="\t", header=True, schema=base_schema)
    df_title = spark.read.csv(path_title, sep="\t", header=True, schema=base_schema)
    combined_df = df_body.unionByName(df_title).na.drop()
    
    # 2. Parse Numeric Columns
    combined_df = combined_df.withColumn(
        "TIMESTAMP", 
        to_timestamp(col("TIMESTAMP"), "yyyy-MM-dd HH:mm:ss").cast(LongType())
    )

    # 3. Parse LINK_SENTIMENT
    combined_df = combined_df.withColumn("LINK_SENTIMENT", col("LINK_SENTIMENT").cast(LongType()))
                             
    # 4. Parse PROPERTIES Column
    df = combined_df.withColumn("prop_array", split(col("PROPERTIES"), ","))
    
    # Extract key features (VADER, LIWC I/We, Negemo/Anger) by 1-based index
    parsed_df = df.withColumn("vader_compound", element_at(col("prop_array"), 21).cast(DoubleType())) \
                  .withColumn("liwc_i", element_at(col("prop_array"), 25).cast(DoubleType())) \
                  .withColumn("liwc_we", element_at(col("prop_array"), 26).cast(DoubleType())) \
                  .withColumn("liwc_negemo", element_at(col("prop_array"), 50).cast(DoubleType())) \
                  .withColumn("liwc_anger", element_at(col("prop_array"), 52).cast(DoubleType()))
                  
    return parsed_df.drop("PROPERTIES", "prop_array")

def load_data_and_prepare_structure(spark, path_body, path_title):
    """Loads data, calculates vertex features, and prepares Vertices and Edges DataFrames."""
    print("\n--- Loading Data & Feature Engineering ---")
    
    parsed_df = parse_and_enrich_data(spark, path_body, path_title)

    # 1. Vertex Feature Calculation (Aggregation)
    vertex_features_df = parsed_df.groupBy("SOURCE_SUBREDDIT").agg(
        avg(col("vader_compound")).alias("avg_vader_score"),
        avg(col("liwc_we") - col("liwc_i")).alias("avg_social_clout"),
        count(col("POST_ID")).alias("activity_count")
    ).withColumnRenamed("SOURCE_SUBREDDIT", "id")

    # 2. Create Base Edge DataFrame
    edges_df = parsed_df.select(
        col("SOURCE_SUBREDDIT").alias("src"),
        col("TARGET_SUBREDDIT").alias("dst"),
        col("TIMESTAMP").alias("timestamp"),
        col("LINK_SENTIMENT").alias("sentiment")
    ).na.drop()
    
    # 3. Create Final Vertices DataFrame (Join features to all nodes)
    all_subreddits = edges_df.select(col("src").alias("id")) \
        .union(edges_df.select(col("dst").alias("id"))) \
        .distinct()
        
    final_vertices_df = all_subreddits.join(vertex_features_df, on="id", how="left")
    final_vertices_df = final_vertices_df.withColumn("type", lit("subreddit"))
    
    # Fill nulls for subreddits that only receive links
    final_vertices_df = final_vertices_df.na.fill({
        "avg_vader_score": 0.0,
        "avg_social_clout": 0.0,
        "activity_count": 0
    })

    print(f"Prepared {final_vertices_df.count()} vertices and {edges_df.count()} edges.")
    return final_vertices_df, edges_df

def create_graph(vertices_df, edges_df) -> GraphFrame:
    """Creates the property graph."""
    print("\n--- Creating GraphFrame ---")
    g = GraphFrame(vertices_df, edges_df)
    print(f"Graph created with {g.vertices.count()} vertices and {g.edges.count()} edges.")
    return g

# --- Query Functions (2.3 a-e) ---
def find_top_outdegree(g: GraphFrame):
    """Query: Find the top 5 nodes with the highest outdegree (Link Senders)."""
    print("\n--- Top 5 Nodes with Highest Outdegree (Link Senders) ---")
    top_out_degree = g.outDegrees.withColumnRenamed("id", "subreddit_id") \
        .orderBy(col("outDegree").desc()) \
        .limit(5)
    top_out_degree.show()
    save_results(top_out_degree, "top_5_outdegree")

def find_top_indegree(g: GraphFrame):
    """Query: Find the top 5 nodes with the highest indegree (Link Receivers)."""
    print("\n--- Top 5 Nodes with Highest Indegree (Link Receivers) ---")
    top_in_degree = g.inDegrees.withColumnRenamed("id", "subreddit_id") \
        .orderBy(col("inDegree").desc()) \
        .limit(5)
    top_in_degree.show()
    save_results(top_in_degree, "top_5_indegree")

def calculate_pagerank(g: GraphFrame):
    """Query: Calculate PageRank and output the top 5 nodes."""
    print("\n--- Top 5 Nodes with Highest PageRank ---")
    pr_results = g.pageRank(resetProbability=0.15, maxIter=5)
    top_pagerank = pr_results.vertices \
        .orderBy(col("pagerank").desc()) \
        .limit(5)
    top_pagerank.select("id", "pagerank", "avg_vader_score", "avg_social_clout").show(truncate=False)
    save_results(top_pagerank.select("id", "pagerank"), "top_5_pagerank")

def find_connected_components(g: GraphFrame):
    """Query: Run the Connected Components algorithm and find the top 5 largest components."""
    print("\n--- Top 5 Largest Connected Components ---")
    cc_results = g.connectedComponents()
    top_components = cc_results.groupBy("component") \
        .count() \
        .withColumnRenamed("count", "componentSize") \
        .orderBy(col("componentSize").desc()) \
        .limit(5)
    top_components.show()
    save_results(top_components, "top_5_connected_components")

def calculate_triangle_counts(g: GraphFrame):
    """Query: Run the Triangle Counts algorithm and output the top 5 vertices."""
    print("\n--- Top 5 Vertices with Largest Triangle Count ---")
    
    tc_results = g.triangleCount(
        storage_level=StorageLevel.MEMORY_AND_DISK
    )

    top_triangle_count = tc_results.select(col("id"), col("count").alias("triangleCount")) \
        .orderBy(col("triangleCount").desc()) \
        .limit(5)

    top_triangle_count.show()
    save_results(top_triangle_count, "top_5_triangle_counts")


# --- Main Execution Flow ---
if __name__ == "__main__":
    
    if not os.path.exists(FILE_PATH_BODY) or not os.path.exists(FILE_PATH_TITLE):
        print("ERROR: One or both input files not found. Please update FILE_PATH_BODY and FILE_PATH_TITLE.")
        spark.stop()
        sys.exit(1)
        
    # 1. Load Data and Prepare Structure (2.1)
    vertices_df, edges_df = load_data_and_prepare_structure(spark, FILE_PATH_BODY, FILE_PATH_TITLE)

    # 2. Create Graph (2.2)
    g = create_graph(vertices_df, edges_df)

    # 3. Run Queries
    find_top_outdegree(g)
    find_top_indegree(g)
    calculate_pagerank(g)
    find_connected_components(g)
    calculate_triangle_counts(g)

    # Final Cleanup
    spark.stop()
    print("\n--- All analysis steps complete. Results saved in 'reddit_analysis_output'. ---")