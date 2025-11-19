# Social Network Graph Analysis with PySpark and GraphFrames

This project performs social network graph analysis on Reddit hyperlink data using Apache Spark and GraphFrames. It processes large datasets of subreddit interactions to identify key subreddits based on various graph metrics.

## Features

-   **Data Ingestion:** Loads and parses Reddit hyperlink data from TSV files.
-   **Graph Construction:** Builds a property graph using GraphFrames, representing subreddits as vertices and hyperlinks as edges.
-   **Graph Algorithms:**
    -   **Out-degree:** Identifies subreddits that are the most frequent link senders.
    -   **In-degree:** Identifies subreddits that are the most frequent link receivers.
    -   **PageRank:** Calculates the influence of subreddits within the network.
    -   **Connected Components:** Finds groups of subreddits that are connected to each other.
    -   **Triangle Count:** Measures the number of triangles (a common motif in social networks) each subreddit participates in.
-   **Results Output:** Saves the top results of each analysis to CSV files for easy review.

## Setup

### Prerequisites

-   **Python 3.x:** Ensure Python is installed on your system.
-   **Java Development Kit (JDK):** Apache Spark requires Java. Ensure a compatible JDK (e.g., OpenJDK 8 or 11) is installed and `JAVA_HOME` is set.
-   **Apache Spark:** While `pyspark` will be installed, a local Spark installation might be beneficial for advanced configurations.

### 1. Create a Virtual Environment (Recommended)

It's good practice to use a virtual environment to manage project dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

Install `pyspark` and `graphframes` (which requires a specific Spark version and Scala version).

```bash
pip install pyspark
# GraphFrames is typically used with spark-submit and its package argument.
# No direct pip install for graphframes library itself, but pyspark is needed.
```

### 3. Data Files

Place the following data files in the root directory of this project:

-   `soc-RedditHyperlinks-body.tsv`
-   `soc-RedditHyperlinks-title.tsv`

These files contain the raw hyperlink data used for analysis.

## Usage

To run the analysis, you will use `spark-submit` to include the GraphFrames package.

```bash
spark-submit --packages graphframes:graphframes:0.8.2-spark3.0-s_2.12 reddit_analysis.py
```

**Note:** The `graphframes` package version (`0.8.2-spark3.0-s_2.12`) should match your installed Spark version and Scala version. If you encounter issues, please check the official GraphFrames documentation for the correct package string.

The script will perform the following analyses:
1.  Load and preprocess the Reddit hyperlink data.
2.  Construct a GraphFrame.
3.  Execute out-degree, in-degree, PageRank, Connected Components, and Triangle Count algorithms.
4.  Save the top 5 results for each analysis to CSV files.

## Output

All analysis results will be saved in the `reddit_analysis_output/` directory. The following CSV files will be generated:

-   `top_5_outdegree.csv`: Subreddits with the highest number of outgoing links.
-   `top_5_indegree.csv`: Subreddits with the highest number of incoming links.
-   `top_5_pagerank.csv`: Subreddits with the highest PageRank scores (most influential).
-   `top_5_connected_components.csv`: Information about the largest connected components in the graph.
-   `top_5_triangle_counts.csv`: Subreddits with the highest number of triangles they are part of.