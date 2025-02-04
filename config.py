# Configuration parameters for vector search algorithms
VECTOR_SEARCH_CONFIG = {
    'n_database': 100000,
    'n_queries': 600,
    'dimension': 250,
    'threshold': 5.8,  # Increased threshold to account for high dimensionality
    'seed': 1,
    'n_seeds': 100  # Number of different random seeds to use for experiments
} 