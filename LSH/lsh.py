import numpy as np
import time
from tqdm import tqdm
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

def euclidean_lsh(v, a, b, w):
    """
    Compute a single Euclidean LSH hash value.
    """
    return int(np.floor((np.dot(a, v) + b) / w))

def hash_vectors(points, n_hashes, w, seed=None):
    """
    Compute hash vectors for points using multiple hash functions.
    """
    if seed is not None:
        np.random.seed(seed)
        
    n_points, dim = points.shape
    hash_vectors = np.zeros((n_points, n_hashes), dtype=int)
    hash_params = []
    
    for i in range(n_hashes):
        a = np.random.normal(0, 1, dim)
        b = np.random.uniform(0, w)
        hash_vectors[:, i] = [euclidean_lsh(p, a, b, w) for p in points]
        hash_params.append((a, b))
    
    return hash_vectors, hash_params

def run_lsh(config, phase='build', data_structure=None):
    """
    Run LSH algorithm.
    
    Args:
        config: Configuration dictionary with parameters
        phase: Either 'build' or 'search'
        data_structure: Tuple of (database vectors, hash vectors, hash parameters)
        
    Returns:
        If phase == 'build': Returns the data structure tuple
        If phase == 'search': Returns (success, matches, distances) tuple
    """
    # LSH parameters
    n_hashes = 20  # Number of hash functions
    w = 0.1        # Bucket width
    max_hash_dist = 1  # Maximum average hash difference allowed
    
    if phase == 'build':
        # Generate or load database vectors
        n_database = config['n_database']
        dimension = config['dimension']
        seed = config['seed']
        database = generate_random_vectors(n_database, dimension, seed)
        
        # Hash database vectors
        print(f"Computing {n_hashes} LSH hashes for database vectors...")
        db_hashes, hash_params = hash_vectors(database, n_hashes, w, seed)
        
        return (database, db_hashes, hash_params)
        
    elif phase == 'search':
        if data_structure is None:
            raise ValueError("Must provide data structure for search phase")
            
        database, db_hashes, hash_params = data_structure
        
        # Generate query vectors
        n_queries = config['n_queries']
        dimension = config['dimension']
        seed = config['seed']
        threshold = config['threshold']
        queries = generate_random_vectors(n_queries, dimension, seed + 1)
        
        # Hash query vectors using same parameters
        print(f"Computing {n_hashes} LSH hashes for query vectors...")
        query_hashes = np.zeros((len(queries), n_hashes), dtype=int)
        for i, (a, b) in enumerate(hash_params):
            query_hashes[:, i] = [euclidean_lsh(q, a, b, w) for q in queries]
        
        # Find matches for each query
        matches = []
        distances = []
        
        print("Finding matches...")
        for query_idx, (query, q_hash) in enumerate(tqdm(zip(queries, query_hashes), total=len(queries))):
            # Average hash difference per component
            hash_dists = np.abs(db_hashes - q_hash).mean(axis=1)
            candidates = np.where(hash_dists <= max_hash_dist)[0]
            
            if len(candidates) == 0:
                return False, None, None
                
            # Find closest candidate by actual distance
            candidate_dists = [euclidean_distance(query, database[idx]) for idx in candidates]
            best_idx = candidates[np.argmin(candidate_dists)]
            best_dist = min(candidate_dists)
            
            if best_dist > threshold:
                return False, None, None
                
            matches.append(best_idx)
            distances.append(best_dist)
        
        return True, matches, distances

if __name__ == "__main__":
    # Load parameters from config
    config = VECTOR_SEARCH_CONFIG.copy()
    
    print(f"\nRunning LSH search with parameters:")
    print(f"Database size: {config['n_database']}")
    print(f"Number of queries: {config['n_queries']}")
    print(f"Dimension: {config['dimension']}")
    print(f"Distance threshold: {config['threshold']}")
    print(f"Random seed: {config['seed']}")
    
    # Build phase
    print("\nBuilding data structures...")
    build_start = time.time()
    data_structure = run_lsh(config, phase='build')
    build_time = time.time() - build_start
    print(f"Build time: {build_time:.2f} seconds")
    
    # Search phase
    print("\nSearching for matches...")
    search_start = time.time()
    success, matches, distances = run_lsh(config, phase='search', data_structure=data_structure)
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