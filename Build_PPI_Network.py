from config import *
import pandas as pd
import networkx as nx
import os


# --- Configuration ---
BIOGRID_FILENAME = "BIOGRID-ALL-4.4.206.tab3.txt"
BIOGRID_FILE = PATH_TO_RAW_DATA + BIOGRID_FILENAME
OUTPUT_NETWORK = PATH_TO_PROCESSED_DATA + "BioGRID_HomoSapiens_LCC.txt"




# --- Step 1: Read and Filter Data ---
def create_ppi_network():
    """
    Reads the raw BioGRID data, filters it for Homo Sapiens,
    extracts the Largest Connected Component (LCC), and saves the final network.
    """
    print("--- Starting PPI Network Creation ---")

    # Check if the output file already exists to save time
    if os.path.exists(OUTPUT_NETWORK):
        print(f"Final network file already exists at: {OUTPUT_NETWORK}")
        print("Skipping creation process.")
        # Optional: Load and print stats of the existing graph
        final_graph = nx.read_edgelist(OUTPUT_NETWORK)
        print("\nStats of the existing final graph:")
        print(f"  - Number of nodes (genes): {final_graph.number_of_nodes()}")
        print(f"  - Number of edges (interactions): {final_graph.number_of_edges()}")
        return

    print(f"Reading data from text file: {BIOGRID_FILE}")
    
    try:
        use_cols = ['#BioGRID Interaction ID', 'Official Symbol Interactor A', 'Official Symbol Interactor B', 'Organism ID Interactor A', 'Organism ID Interactor B']
        df = pd.read_csv(BIOGRID_FILE, sep='\t', usecols=use_cols, low_memory=False)
    except FileNotFoundError:
        print(f"\nERROR: BioGRID txt file not found at '{BIOGRID_FILE}'")
        print("Please update the BIOGRID_FILE variable in the script with the correct path.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    print(f"Successfully loaded {len(df)} total interactions.")

    # --- Step 2: Filter for Homo Sapiens ---
    print("\nFiltering for Homo Sapiens (Organism ID: 9606)...")
    homo_sapiens_id = 9606
    df_human = df[(df['Organism ID Interactor A'] == homo_sapiens_id) & (df['Organism ID Interactor B'] == homo_sapiens_id)]
    print(f"Found {len(df_human)} interactions in Homo Sapiens.")

    # --- Step 3: Build Initial Graph ---
    print("\nBuilding the initial graph from filtered interactions...")
    # We use official gene symbols for node names
    edge_list = df_human[['Official Symbol Interactor A', 'Official Symbol Interactor B']].dropna()
    
    G_initial = nx.from_pandas_edgelist(edge_list, 'Official Symbol Interactor A', 'Official Symbol Interactor B')
    print("Initial graph stats:")
    print(f"  - Number of nodes (genes): {G_initial.number_of_nodes()}")
    print(f"  - Number of edges (interactions): {G_initial.number_of_edges()}")

    # --- Step 4: Extract Largest Connected Component (LCC) ---
    print("\nExtracting the Largest Connected Component (LCC)...")
    
    connected_components = sorted(nx.connected_components(G_initial), key=len, reverse=True)
    
    if not connected_components:
        print("Error: No connected components found in the graph.")
        return
        
    lcc_nodes = connected_components[0]
    G_lcc = G_initial.subgraph(lcc_nodes).copy()
    
    print("\nFinal Graph (LCC) stats:")
    print(f"  - Number of nodes (genes): {G_lcc.number_of_nodes()}")
    print(f"  - Number of edges (interactions): {G_lcc.number_of_edges()}")
    print("(These numbers should be close to the paper's ~19k nodes and ~678k edges)")

    # --- Step 5: Save the Final Network ---
    print(f"\nSaving the final network to: {OUTPUT_NETWORK}")
    nx.write_edgelist(G_lcc, OUTPUT_NETWORK, data=False)
    
    print("\n--- Process Complete! ---")

if __name__ == "__main__":
    create_ppi_network()