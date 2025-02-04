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

def run_kdtree(config, phase='build', data_structure=None):
    """
    Run KD-tree search algorithm with filtering optimization.
    
    Args:
        config: Configuration dictionary with parameters
        phase: Either 'build' or 'search'
        data_structure: Tuple of (KDTree, remaining vectors, remaining indices, filtered indices)
        
    Returns:
        If phase == 'build': Returns the data structure tuple
        If phase == 'search': Returns (success, matches, distances) tuple
    """
    if phase == 'build':
        # Generate or load database vectors
        n_database = config['n_database']
        dimension = config['dimension']
        seed = config['seed']
        database = generate_random_vectors(n_database, dimension, seed)
        
        # Generate query vectors to compute mean (needed for filtering)
        n_queries = config['n_queries']
        queries = generate_random_vectors(n_queries, dimension, seed + 1)
        
        # Compute filtering parameters
        max_fuel = 2_000_000_000
        base_fuel = 760_000_000
        alpha = 1700 * n_queries
        m = int((max_fuel - base_fuel) / alpha)  # Number of vectors for KD-tree
        
        print(f"Filtering {m} closest database vectors for KD-tree")
        
        # Filter vectors for KD-tree
        mean_query = calculate_mean_vector(queries)
        kd_vectors, kd_indices = filter_relevant_vectors(database, mean_query, m)
        
        # Get remaining vectors
        mask = np.ones(len(database), dtype=bool)
        mask[kd_indices] = False
        r_vectors = database[mask]
        r_indices = np.arange(len(database))[mask]
        
        # Build KD-tree
        print(f"Building KD-tree with {m} vectors...")
        kd_tree = KDTree(kd_vectors)
        
        return (kd_tree, r_vectors, r_indices, kd_indices)
        
    elif phase == 'search':
        if data_structure is None:
            raise ValueError("Must provide data structure for search phase")
            
        kd_tree, r_vectors, r_indices, kd_indices = data_structure
        
        # Generate query vectors
        n_queries = config['n_queries']
        dimension = config['dimension']
        seed = config['seed']
        threshold = config['threshold']
        queries = generate_random_vectors(n_queries, dimension, seed + 1)
        
        # Search KD-tree
        print("Searching KD-tree...")
        kd_results = kd_tree.query(queries)[1]  # Get indices only
        final_indices = kd_indices[kd_results]
        
        # Refine worst matches with brute force
        print("Refining worst 10% of queries using brute-force...")
        brute_force_count = int(n_queries * 0.1)
        distances_to_matches = np.linalg.norm(queries - kd_tree.data[kd_results], axis=1)
        worst_queries = np.argsort(-distances_to_matches)[:brute_force_count]
        
        # Brute force search on remaining vectors for worst matches
        for query_idx in tqdm(worst_queries, desc="Brute-force refinement"):
            query = queries[query_idx]
            distances = np.linalg.norm(r_vectors - query, axis=1)
            best_index = r_indices[np.argmin(distances)]
            final_indices[query_idx] = best_index
        
        # Compute final distances
        final_distances = [euclidean_distance(queries[i], 
                                           kd_tree.data[kd_results[i]] if i not in worst_queries 
                                           else r_vectors[np.where(r_indices == final_indices[i])[0][0]])
                         for i in range(n_queries)]
        
        # Check if all distances meet threshold
        success = all(dist <= threshold for dist in final_distances)
        if not success:
            return False, None, None
            
        return True, final_indices.tolist(), final_distances

if __name__ == "__main__":
    # Load parameters from config
    config = VECTOR_SEARCH_CONFIG.copy()
    
    print(f"\nRunning hybrid search with parameters:")
    print(f"Database size: {config['n_database']}")
    print(f"Number of queries: {config['n_queries']}")
    print(f"Dimension: {config['dimension']}")
    print(f"Distance threshold: {config['threshold']}")
    print(f"Random seed: {config['seed']}")
    
    # Build phase
    print("\nBuilding data structures...")
    build_start = time.time()
    data_structure = run_kdtree(config, phase='build')
    build_time = time.time() - build_start
    print(f"Build time: {build_time:.2f} seconds")
    
    # Search phase
    print("\nSearching for matches...")
    search_start = time.time()
    success, matches, distances = run_kdtree(config, phase='search', data_structure=data_structure)
    search_time = time.time() - search_start
    
    # Print results
    print(f"\nResults:")
    print(f"Solution found: {success}")
    print(f"Build time: {build_time:.2f} seconds")
    print(f"Search time: {search_time:.2f} seconds")
    print(f"Total time: {build_time + search_time:.2f} seconds")
    
    if success:
        print("\nFound matches with distances:")
        for i, (idx, dist) in enumerate(zip(matches, distances)):
            print(f"Query {i} -> Database[{idx}]: distance = {dist:.4f}")
    else:
        print("Failed to find matches for all queries")
