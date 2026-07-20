import pandas as pd
import networkx as nx
import os
import time

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Base Directory ---
# Assuming the script is in the same directory as the data folders
BASE_DIR = r"C:\Users\Asus\Desktop\check_gene_overlap NetEM&XGDAG"

# --- INPUT ---
# The network file we created in the beginning of the project.
# User confirmed the name is "XGDAG_PPI.csv"
NETWORK_FILE = os.path.join(BASE_DIR, "XGDAG_PPI.txt")

# --- OUTPUT ---
# The file where the calculated features will be saved.
OUTPUT_FILE = os.path.join(BASE_DIR, "topological_features.csv")

# ==============================================================================
# CALCULATION FUNCTION
# ==============================================================================

def calculate_and_save_features():
    """
    Loads the PPI network, calculates topological features, and saves them to a CSV file.
    """
    print("="*80)
    print("Starting Topological Feature Calculation")
    print("="*80)

    # 1. Load the network
    if not os.path.exists(NETWORK_FILE):
        print(f"ERROR: Network file not found at: {NETWORK_FILE}")
        return

    print(f"1. Loading network from: {os.path.basename(NETWORK_FILE)}...")
    # Assuming the file has two columns with gene names and no header.
    # We specify the separator as a flexible whitespace regex to handle tabs or spaces.
    # We also explicitly set the dtype to string to avoid misinterpretation of gene names as numbers.
    edge_list = pd.read_csv(
        NETWORK_FILE,
        header=None,
        names=['gene1', 'gene2'],
        sep=r'\s+',  # Use regex for one or more whitespace characters (handles tabs/spaces)
        dtype=str    # Treat all columns as strings
    )
    
    # Convert all gene names to lower case to ensure consistency
    edge_list['gene1'] = edge_list['gene1'].str.lower()
    edge_list['gene2'] = edge_list['gene2'].str.lower()
    
    # 2. Create Graph
    print("2. Building graph with NetworkX...")
    G = nx.from_pandas_edgelist(edge_list, 'gene1', 'gene2')
    print(f"   Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # 3. Calculate Features
    # This can take some time, especially betweenness centrality on large graphs.
    
    start_time = time.time()
    print("\n3. Calculating Degree Centrality...")
    degree_centrality = nx.degree_centrality(G)
    print(f"   Done. ({time.time() - start_time:.2f}s)")

    start_time = time.time()
    print("4. Calculating Betweenness Centrality... (This may take a while)")
    betweenness_centrality = nx.betweenness_centrality(G)
    print(f"   Done. ({time.time() - start_time:.2f}s)")

    start_time = time.time()
    print("4. Calculating PageRank... (Fast alternative to Betweenness)")
    pagerank = nx.pagerank(G)
    print(f"   Done. ({time.time() - start_time:.2f}s)")

    start_time = time.time()
    print("5. Calculating Closeness Centrality...")
    closeness_centrality = nx.closeness_centrality(G)
    print(f"   Done. ({time.time() - start_time:.2f}s)")

    start_time = time.time()
    print("6. Calculating Clustering Coefficient...")
    clustering_coefficient = nx.clustering(G)
    print(f"   Done. ({time.time() - start_time:.2f}s)")

    # 4. Combine features into a DataFrame
    print("\n7. Combining all features into a single DataFrame...")
    features_df = pd.DataFrame({
        'degree_centrality': pd.Series(degree_centrality),
        'betweenness_centrality': pd.Series(betweenness_centrality),
        'pagerank': pd.Series(pagerank),
        'closeness_centrality': pd.Series(closeness_centrality),
        'clustering_coefficient': pd.Series(clustering_coefficient)
    })
    features_df.index.name = 'gene'
    
    # 5. Save to CSV
    print(f"8. Saving features to: {os.path.basename(OUTPUT_FILE)}...")
    features_df.to_csv(OUTPUT_FILE)

    print("\n" + "="*80)
    print("SUCCESS: Topological features have been calculated and saved.")
    print(f"Output file location: {OUTPUT_FILE}")
    print("="*80)


# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    calculate_and_save_features()