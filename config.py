# Configuration parameters for vector search algorithms
VECTOR_SEARCH_CONFIG = {   'n_database': 100000,
    'n_queries': 5000,
    'dimension': 250,
    'threshold': 5.475,  # Increased threshold to account for high dimensionality
    'seed': 1,
    'n_seeds': 5  # Number of different random seeds to use for experiments
}

# Configuration for which algorithms to run in experiments
# Set value to True to include algorithm, False to exclude
ALGORITHMS_CONFIG = {
    'Brute Force': False,
    'KD-Tree (Vanilla)': False,
    'KD-Tree (Mean Filter)': True,
    'KD-Tree (Median Filter)': False,
    'KD-Tree (Random Filter)': False,
    'KD-Tree (True Mean Filter)': False,
    'KD-Tree (Mean Variance Filter)': True  # Added new algorithm
} 


