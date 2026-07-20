import networkx as nx
from node2vec import Node2Vec
import pandas as pd
import time
import multiprocessing

# ==============================================================================
# SETUP SELECTION - UNCOMMENT ONLY ONE LINE
# ==============================================================================
# SETUP = 'baseline'        # Setup 1: Current settings (already executed)
# SETUP = 'highdim'         # Setup 2: Higher dimensions for more information
# SETUP = 'deepexplore'     # Setup 3: Deeper exploration of the network
SETUP = 'ultrahighdim'    # Setup 4: Ultra high dimensions (500) for maximum separation

# ==============================================================================
# HYPERPARAMETER SETTINGS
# File paths and parameters are set based on the selected SETUP
# ==============================================================================

INPUT_EDGELIST_FILE  = r"C:\Users\Asus\Desktop\check_gene_overlap NetEM&XGDAG\Invalid data\XGDAG_PPI.txt"

# Configure output file name and parameters based on selected SETUP
if SETUP == 'baseline':
    OUTPUT_EMBEDDINGS_FILE = r"C:\Users\Asus\Desktop\check_gene_overlap NetEM&XGDAG\XGDAG_All Seed Genes\XGDAG_PPI_node2vec_embeddings_baseline.csv"
    HYPERPARAMETERS = {
        "dimensions": 128,
        "walk_length": 80,
        "num_walks": 10,
        "window": 10,
        "min_count": 3,
        "p": 1,
        "q": 1
    }
elif SETUP == 'highdim':
    OUTPUT_EMBEDDINGS_FILE = r"C:\Users\Asus\Desktop\check_gene_overlap NetEM&XGDAG\XGDAG_All Seed Genes\XGDAG_PPI_node2vec_embeddings_highdim.csv"
    HYPERPARAMETERS = {
        "dimensions": 256,      # Higher dimensions: 128 → 256
        "walk_length": 80,
        "num_walks": 10,
        "window": 10,
        "min_count": 3,
        "p": 1,
        "q": 1
    }
elif SETUP == 'deepexplore':
    OUTPUT_EMBEDDINGS_FILE = r"C:\Users\Asus\Desktop\check_gene_overlap NetEM&XGDAG\XGDAG_All Seed Genes\XGDAG_PPI_node2vec_embeddings_deepexplore.csv"
    HYPERPARAMETERS = {
        "dimensions": 128,
        "walk_length": 80,
        "num_walks": 20,        # More walks: 10 → 20
        "window": 10,
        "min_count": 3,
        "p": 0.5,               # More exploration: decreased p
        "q": 2                  # Long-distance exploration: increased q
    }
elif SETUP == 'ultrahighdim':
    OUTPUT_EMBEDDINGS_FILE = r"C:\Users\Asus\Desktop\check_gene_overlap NetEM&XGDAG\XGDAG_All Seed Genes\XGDAG_PPI_node2vec_embeddings_ultrahighdim.csv"
    HYPERPARAMETERS = {
        "dimensions": 500,      # Ultra high dimensions: 256 → 500
        "walk_length": 80,
        "num_walks": 20,        # More walks to explore the high-dimensional space
        "window": 10,
        "min_count": 3,
        "p": 1,                 # Standard BFS-DFS balance
        "q": 1
    }
else:
    raise ValueError(f"Invalid SETUP: {SETUP}. Only 'baseline', 'highdim', 'deepexplore', or 'ultrahighdim' are allowed.")

# Word2Vec Model Hyperparameters (used internally by Node2Vec)
FIT_PARAMETERS = {
    "epochs": 1,            # Number of iterations over the generated walks
    "batch_words": 4        # Number of words per batch for training
}
# ==============================================================================
# END OF SETTINGS
# ==============================================================================


def main():
    """
    Main function to load the graph, execute Node2Vec, and save embeddings.
    """
    print(f"{'='*70}")
    print(f"Running Node2Vec with Setup: {SETUP.upper()}")
    print(f"Output file: {OUTPUT_EMBEDDINGS_FILE}")
    print(f"Hyperparameters: {HYPERPARAMETERS}")
    print(f"{'='*70}\n")
    
    print("Step 1: Loading graph from file...")
    start_time = time.time()
    
    # Reading graph from edgelist file.
    # Assumes the file is tab-separated and has no comments.
    try:
        graph = nx.read_edgelist(
            INPUT_EDGELIST_FILE,
            nodetype=str,
            create_using=nx.Graph()
        )
        # Ensure the graph is undirected
        graph = graph.to_undirected()
    except FileNotFoundError:
        print(f"Error: File '{INPUT_EDGELIST_FILE}' not found. Please ensure the filename and path are correct.")
        return

    print(f"Graph loaded successfully with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
    print("-" * 30)

    print("Step 2: Initializing Node2Vec model...")
    # Instantiate Node2Vec with the graph and defined hyperparameters
    node2vec = Node2Vec(
        graph,
        dimensions=HYPERPARAMETERS["dimensions"],
        walk_length=HYPERPARAMETERS["walk_length"],
        num_walks=HYPERPARAMETERS["num_walks"],
        p=HYPERPARAMETERS["p"],
        q=HYPERPARAMETERS["q"],
        # workers=HYPERPARAMETERS["workers"],
        quiet=False # Set to False to show progress bar
    )
    print("Model initialized successfully.")
    print(f"Hyperparameters: {HYPERPARAMETERS}")
    print("-" * 30)

    print("Step 3: Training the model to generate embeddings...")
    print("This step may be time-consuming. Please wait...")
    start_time = time.time()
    
    # Training the Skip-Gram model on the generated random walks
    model = node2vec.fit(
        window=HYPERPARAMETERS["window"],
        min_count=HYPERPARAMETERS["min_count"],
        batch_words=FIT_PARAMETERS["batch_words"],
        epochs=FIT_PARAMETERS["epochs"]
    )
    
    print(f"Model training completed successfully.")
    print(f"Training time elapsed: {time.time() - start_time:.2f} seconds")
    print("-" * 30)

    print("Step 4: Extracting and saving embeddings to CSV...")
    start_time = time.time()
    
    # Extracting vectors and node names
    nodes = list(model.wv.index_to_key)
    vectors = model.wv[nodes]
    
    # Creating a pandas DataFrame
    # First column is 'gene', followed by embedding dimensions
    embedding_df = pd.DataFrame(vectors, index=nodes)
    embedding_df.reset_index(inplace=True)
    embedding_df = embedding_df.rename(columns={'index': 'gene'})
    
    # Saving DataFrame to CSV
    embedding_df.to_csv(OUTPUT_EMBEDDINGS_FILE, index=False)
    
    print(f"Embeddings saved successfully to '{OUTPUT_EMBEDDINGS_FILE}'.")
    print(f"Output file contains {len(embedding_df)} vectors with {embedding_df.shape[1] - 1} dimensions.")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
    print("-" * 30)
    print("Process completed successfully.")


if __name__ == "__main__":
    main()