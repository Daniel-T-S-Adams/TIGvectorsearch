import numpy as np
import time
from config import VECTOR_SEARCH_CONFIG
from typing import Tuple, Optional, List

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

class BruteForce:
    def __init__(self):
        """Initialize brute force search."""
        self.database = None
        # Add timing statistics
        self.kdtree_search_time: float = 0.0  # Always 0 for brute force
        self.bruteforce_search_time: float = 0.0
        # Add match counting
        self.n_bruteforce_matches: int = 0  # Number of matches found by brute force

    def build(self, database: np.ndarray) -> None:
        """Store database for later searching."""
        self.database = database

    def search(self, queries: np.ndarray, threshold: float) -> Tuple[bool, Optional[List[int]], Optional[List[float]]]:
        """
        For each query vector, try to find a database vector within threshold distance.
        Uses brute force search (computes all pairwise distances).
        
        Args:
            queries: Array of shape (n_queries, dimension) containing query vectors
            threshold: Maximum allowed distance between any query and its match
            
        Returns:
            Tuple containing:
                - Boolean indicating if matches were found for all queries
                - If successful, list of database indices matched to each query
                - If successful, list of distances for each match
        """
        if self.database is None:
            raise ValueError("Must build before searching")
        
        matches = []
        distances = []
        
        # Reset statistics
        self.kdtree_search_time = 0.0  # Always 0 for brute force
        self.bruteforce_search_time = 0.0
        self.n_bruteforce_matches = 0
        
        # Process each query
        for query in queries:
            # Calculate distances to all database vectors
            bruteforce_start = time.time()
            dists = np.linalg.norm(self.database - query, axis=1)
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]
            self.bruteforce_search_time += time.time() - bruteforce_start
            
            if min_dist <= threshold:
                matches.append(min_idx)
                distances.append(float(min_dist))
                self.n_bruteforce_matches += 1
            else:
                return False, None, None
        
        # Print match statistics
        print("\nMatch Statistics:")
        print(f"  Total matches found: {self.n_bruteforce_matches}")
        print(f"  All matches found by brute force search")
        
        return True, matches, distances

def run_bruteforce(config, phase='build', data_structure=None):
    """
    Run brute force search algorithm.
    
    Args:
        config: Configuration dictionary with parameters
        phase: Either 'build' or 'search'
        data_structure: BruteForce instance from build phase
        
    Returns:
        If phase == 'build': Returns the BruteForce instance
        If phase == 'search': Returns (success, matches, distances) tuple
    """
    if phase == 'build':
        if 'database' not in config:
            raise ValueError("Database vectors must be provided in config")
            
        # Build brute force searcher
        brute_force = BruteForce()
        brute_force.build(config['database'])
        return brute_force
        
    elif phase == 'search':
        if data_structure is None:
            raise ValueError("Must provide BruteForce instance for search phase")
        if 'queries' not in config or 'threshold' not in config:
            raise ValueError("Queries and threshold must be provided in config")
            
        brute_force = data_structure
        return brute_force.search(config['queries'], config['threshold'])

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
    brute_force = run_bruteforce(config, phase='build')
    build_time = time.time() - build_start
    print(f"Build time: {build_time:.2f} seconds")
    
    # Search phase
    print("\nSearching for matches...")
    search_start = time.time()
    success, matches, distances = run_bruteforce(config, phase='search', data_structure=brute_force)
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