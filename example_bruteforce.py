import numpy as np
import time
from config import VECTOR_SEARCH_CONFIG

def generate_random_vectors(n_vectors: int, dimension: int, seed: int = None) -> np.ndarray:
    """Generate random vectors with components from uniform[0,1]."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(0, 1, size=(n_vectors, dimension))

def euclidean_distances_vectorized(queries: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Compute Euclidean distances between all queries and database vectors efficiently."""
    # Compute squared Euclidean distances using broadcasting
    # (a-b)^2 = a^2 + b^2 - 2ab
    queries_squared = np.sum(queries**2, axis=1)[:, np.newaxis]
    database_squared = np.sum(database**2, axis=1)[np.newaxis, :]
    cross_term = -2 * np.dot(queries, database.T)
    distances = np.sqrt(queries_squared + database_squared + cross_term)
    return distances

def run_bruteforce(config, phase='build', data_structure=None):
    """
    Run brute force search algorithm.
    
    Args:
        config: Configuration dictionary with parameters
        phase: Either 'build' or 'search'
        data_structure: Database vectors from build phase
        
    Returns:
        If phase == 'build': Returns the database vectors
        If phase == 'search': Returns (success, matches, distances) tuple
    """
    if phase == 'build':
        if 'database' not in config:
            raise ValueError("Database vectors must be provided in config")
        return config['database']
        
    elif phase == 'search':
        if data_structure is None:
            raise ValueError("Must provide database vectors for search phase")
        if 'queries' not in config or 'threshold' not in config:
            raise ValueError("Queries and threshold must be provided in config")
            
        database = data_structure
        queries = config['queries']
        threshold = config['threshold']
        
        # Calculate all distances at once
        all_distances = euclidean_distances_vectorized(queries, database)
        
        # Find the first match for each query that's within threshold
        matches = []
        distances = []
        
        for query_idx, query_distances in enumerate(all_distances):
            min_idx = np.argmin(query_distances)
            min_dist = query_distances[min_idx]
            
            if min_dist <= threshold:
                matches.append(int(min_idx))
                distances.append(float(min_dist))
            else:
                return False, None, None
        
        return True, matches, distances

if __name__ == "__main__":
    # Get parameters from config
    config = VECTOR_SEARCH_CONFIG.copy()
    
    print(f"\nRunning brute force search with parameters:")
    print(f"Database size: {config['n_database']}")
    print(f"Number of queries: {config['n_queries']}")
    print(f"Dimension: {config['dimension']}")
    print(f"Distance threshold: {config['threshold']}")
    print(f"Random seed: {config['seed']}")
    
    # Build phase
    print("\nBuilding (preparing database)...")
    build_start = time.time()
    database = run_bruteforce(config, phase='build')
    build_time = time.time() - build_start
    print(f"Build time: {build_time:.2f} seconds")
    
    # Search phase
    print("\nSearching for matches...")
    search_start = time.time()
    success, matches, distances = run_bruteforce(config, phase='search', data_structure=database)
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