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

def run_bruteforce(config, phase='build', data_structure=None):
    """
    Run brute force search algorithm.
    
    Args:
        config: Configuration dictionary with parameters
        phase: Either 'build' or 'search'
        data_structure: For brute force, this is just the database vectors
        
    Returns:
        If phase == 'build': Returns the database vectors
        If phase == 'search': Returns (success, matches, distances) tuple
    """
    if phase == 'build':
        # Generate or load database vectors
        n_database = config['n_database']
        dimension = config['dimension']
        seed = config['seed']
        database = generate_random_vectors(n_database, dimension, seed)
        return database
        
    elif phase == 'search':
        database = data_structure  # In brute force, data_structure is just the database
        if database is None:
            raise ValueError("Must provide database vectors for search phase")
            
        # Generate or load query vectors
        n_queries = config['n_queries']
        dimension = config['dimension']
        seed = config['seed']
        threshold = config['threshold']
        queries = generate_random_vectors(n_queries, dimension, seed + 1)
        
        n_database = len(database)
        matches = []
        distances = []
        
        # For each query
        for query_idx, query in enumerate(queries):
            found_match = False
            
            # Try each database vector
            for db_idx in tqdm(range(n_database), desc=f"Query {query_idx + 1}/{n_queries}"):
                dist = euclidean_distance(query, database[db_idx])
                if dist <= threshold:
                    matches.append(db_idx)
                    distances.append(dist)
                    found_match = True
                    break
            
            if not found_match:
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