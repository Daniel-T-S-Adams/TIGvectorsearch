import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
import time
import sys
import os

# Add parent directory to Python path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VECTOR_SEARCH_CONFIG

def generate_random_vectors(n_vectors: int, dimension: int, seed: int = None) -> np.ndarray:
    """Generate random vectors with components from uniform[0,1]."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(0, 1, size=(n_vectors, dimension))

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))

def calculate_mean_vector(queries: np.ndarray) -> np.ndarray:
    """Compute the mean vector of all query vectors."""
    return np.mean(queries, axis=0)

def filter_relevant_vectors(database: np.ndarray, mean_query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Select the top-k closest database vectors to the mean query vector.
    
    Args:
        database: Array of database vectors
        mean_query: Mean vector of all query vectors
        k: Number of closest vectors to select
    
    Returns:
        A tuple containing:
            - Filtered database vectors
            - Their corresponding original indices
    """
    distances = np.linalg.norm(database - mean_query, axis=1)
    closest_indices = np.argsort(distances)[:k]
    return database[closest_indices], closest_indices

def kd_tree_search(kd_tree: KDTree, queries: np.ndarray) -> np.ndarray:
    """
    Search KD-tree for each query vector and return nearest neighbor indices.
    
    Args:
        kd_tree: Pre-built KDTree of the filtered database
        queries: Array of query vectors
    
    Returns:
        Array of nearest neighbor indices from the KD-tree
    """
    _, indices = kd_tree.query(queries)
    return indices

def brute_force_search(queries: np.ndarray, r_vectors: np.ndarray, r_indices: np.ndarray, worst_query_indices: np.ndarray, current_indices: np.ndarray) -> np.ndarray:
    """
    Perform brute-force search on a subset of queries.
    
    Args:
        queries: Query vectors
        r_vectors: Remaining database vectors (not in KD-tree)
        r_indices: Indices of remaining database vectors
        worst_query_indices: Queries identified as having high distance from the KD-tree search
        current_indices: Current best match indices to update
    
    Returns:
        Updated nearest neighbor indices after brute-force refinement
    """
    indices = current_indices.copy()  # Work on a copy
    for query_idx in tqdm(worst_query_indices, desc="Brute-force refinement"):
        query = queries[query_idx]
        distances = np.linalg.norm(r_vectors - query, axis=1)
        best_index = r_indices[np.argmin(distances)]
        indices[query_idx] = best_index  # Update match if better
    return indices

def hybrid_vector_search(database: np.ndarray, queries: np.ndarray, threshold: float) -> tuple[bool, list[int] | None, list[float] | None]:
    """
    Hybrid KD-tree + brute-force nearest neighbor search.

    Args:
        database: Database vectors
        queries: Query vectors
        threshold: Distance threshold for matching

    Returns:
        Tuple containing:
            - Boolean indicating if matches were found for all queries
            - If successful, list of database indices matched to each query
            - If successful, list of distances for each query-match pair
    """
    n_queries = len(queries)

    print(f"\nHybrid search started...")
    print(f"Database size: {len(database)}")
    print(f"Number of queries: {n_queries}")
    
    # Step 1: Compute filtering parameters
    max_fuel = 2_000_000_000
    base_fuel = 760_000_000
    alpha = 1700 * n_queries
    m = int((max_fuel - base_fuel) / alpha)  # Number of vectors for KD-tree

    print(f"Filtering {m} closest database vectors for KD-tree")

    # Step 2: Compute mean vector and filter database for KD-tree
    mean_query = calculate_mean_vector(queries)
    kd_vectors, kd_indices = filter_relevant_vectors(database, mean_query, m)

    # Step 3: Get remaining vectors (all vectors not in KD-tree)
    mask = np.ones(len(database), dtype=bool)
    mask[kd_indices] = False
    r_vectors = database[mask]
    r_indices = np.arange(len(database))[mask]

    # Step 4: Build KD-tree and search
    print(f"Building KD-tree with {m} vectors...")
    kd_tree = KDTree(kd_vectors)
    kd_results = kd_tree_search(kd_tree, queries)
    
    # Step 5: Get initial results from KD-tree
    final_indices = kd_indices[kd_results]
    
    print(f"Refining worst 10% of queries using brute-force...")
    brute_force_count = int(n_queries * 0.1)
    distances_to_matches = np.linalg.norm(queries - kd_vectors[kd_results], axis=1)
    worst_queries = np.argsort(-distances_to_matches)[:brute_force_count]
    
    # Step 6: Refine worst matches with brute force
    final_indices = brute_force_search(queries, r_vectors, r_indices, worst_queries, final_indices)

    # Compute final distances and check threshold
    final_distances = [euclidean_distance(queries[i], database[idx]) for i, idx in enumerate(final_indices)]
    
    # Check if all distances meet threshold
    success = all(dist <= threshold for dist in final_distances)
    if not success:
        return False, None, None
        
    return True, final_indices.tolist(), final_distances

if __name__ == "__main__":
    # Load parameters from config
    n_database = VECTOR_SEARCH_CONFIG['n_database']
    n_queries = VECTOR_SEARCH_CONFIG['n_queries']
    dimension = VECTOR_SEARCH_CONFIG['dimension']
    threshold = VECTOR_SEARCH_CONFIG['threshold']
    seed = VECTOR_SEARCH_CONFIG['seed']
    
    print(f"\nRunning hybrid search with parameters:")
    print(f"Database size: {n_database}")
    print(f"Number of queries: {n_queries}")
    print(f"Dimension: {dimension}")
    print(f"Distance threshold: {threshold}")
    print(f"Random seed: {seed}")
    
    # Generate test data
    database = generate_random_vectors(n_database, dimension, seed)
    queries = generate_random_vectors(n_queries, dimension, seed + 1)

    # Run search and time it
    total_start = time.time()
    success, matches, distances = hybrid_vector_search(database, queries, threshold)
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
