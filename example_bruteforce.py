import numpy as np
from tqdm import tqdm
import time
from config import VECTOR_SEARCH_CONFIG

def generate_random_vectors(n_vectors: int, dimension: int, seed: int = None) -> np.ndarray:
    """Generate random vectors with components from uniform[0,1]."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(0, 1, size=(n_vectors, dimension))

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))

def brute_force_search(database: np.ndarray, queries: np.ndarray, threshold: float) -> tuple[bool, list[int] | None, list[float] | None]:
    """
    For each query vector, search through database to find any vector within threshold distance.
    
    Args:
        database: Array of shape (n_database, dimension) containing database vectors
        queries: Array of shape (n_queries, dimension) containing query vectors
        threshold: Maximum allowed Euclidean distance between any query and its match
        
    Returns:
        Tuple containing:
            - Boolean indicating if matches were found for all queries
            - If successful, list of database indices matched to each query
            - If successful, list of distances for each query-match pair
    """
    n_queries = len(queries)
    n_database = len(database)
    
    print(f"\nBrute force searching for each query...")
    print(f"Number of queries: {n_queries}")
    print(f"Database size: {n_database}")
    
    matches = []
    distances = []
    
    # For each query
    for query_idx, query in enumerate(queries):
        print(f"\nProcessing query {query_idx + 1}/{n_queries}")
        found_match = False
        
        # Try each database vector
        for db_idx in tqdm(range(n_database), desc="Searching database"):
            dist = euclidean_distance(query, database[db_idx])
            if dist <= threshold:
                matches.append(db_idx)
                distances.append(dist)
                found_match = True
                print(f"Found match at index {db_idx} with distance {dist:.4f}")
                break
        
        if not found_match:
            print(f"No match found for query {query_idx}")
            return False, None, None
    
    return True, matches, distances

if __name__ == "__main__":
    # Get parameters from config
    n_database = VECTOR_SEARCH_CONFIG['n_database']
    n_queries = VECTOR_SEARCH_CONFIG['n_queries']
    dimension = VECTOR_SEARCH_CONFIG['dimension']
    threshold = VECTOR_SEARCH_CONFIG['threshold']
    seed = VECTOR_SEARCH_CONFIG['seed']
    
    print(f"\nRunning brute force search with parameters:")
    print(f"Database size: {n_database}")
    print(f"Number of queries: {n_queries}")
    print(f"Dimension: {dimension}")
    print(f"Distance threshold: {threshold}")
    print(f"Random seed: {seed}")
    
    # Generate test data
    database = generate_random_vectors(n_database, dimension, seed)
    queries = generate_random_vectors(n_queries, dimension, seed + 1)  # Different seed
    
    # Run search and time it
    total_start = time.time()
    success, matches, distances = brute_force_search(database, queries, threshold)
    total_time = time.time() - total_start
    
    # Print results
    print(f"\nResults:")
    print(f"Solution found: {success}")
    print(f"Total time: {total_time:.2f} seconds")
    if success:
        print("\nFound matches with distances:")
        for i, (idx, dist) in enumerate(zip(matches, distances)):
            print(f"Query {i} -> Database[{idx}]: distance = {dist:.4f}")
    else:
        print("Failed to find matches for all queries") 