import numpy as np
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import time
import sys
import os

# Add parent directory to Python path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VECTOR_SEARCH_CONFIG

@dataclass
class KDNode:
    """Node in the KD-tree."""
    point: np.ndarray          # The vector stored at this node
    index: int                 # Index of this vector in the original database
    axis: int                  # Split axis for this node
    left: Optional['KDNode']   # Left child (values less than split)
    right: Optional['KDNode']  # Right child (values greater than split)

class KDTreeTrueMeanFilter:
    """
    KD-tree implementation with true mean filtering optimization for vector search.
    Filters database vectors based on true mean vector (all components = 0.5) before building KD-tree,
    then uses both KD-tree and brute force for efficient vector matching.
    """
    
    def __init__(self):
        """Initialize an empty KD-tree with filtering."""
        self.root: Optional[KDNode] = None
        self.r_vectors: Optional[np.ndarray] = None  # Remaining vectors not in KD-tree
        self.r_indices: Optional[np.ndarray] = None  # Indices of remaining vectors
        self.kd_indices: Optional[np.ndarray] = None  # Indices of vectors in KD-tree
        self.dimension: Optional[int] = None
        self.database: Optional[np.ndarray] = None  # Store full database for distance calculations
    
    def euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute Euclidean distance between two vectors."""
        return float(np.sqrt(np.sum((v1 - v2) ** 2)))
    
    def calculate_true_mean_vector(self, dimension: int) -> np.ndarray:
        """Return the true mean vector (all components = 0.5)."""
        return np.full(dimension, 0.5)
    
    def filter_relevant_vectors(self, database: np.ndarray, true_mean: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the top-k closest database vectors to the true mean vector.
        
        Args:
            database: Array of database vectors
            true_mean: True mean vector (all components = 0.5)
            k: Number of closest vectors to select
        
        Returns:
            A tuple containing:
                - Filtered database vectors
                - Their corresponding original indices
        """
        distances = np.linalg.norm(database - true_mean, axis=1)
        closest_indices = np.argsort(distances)[:k]
        return database[closest_indices], closest_indices
    
    def _select_axis(self, depth: int) -> int:
        """Select splitting axis based on current depth."""
        return depth % self.dimension
    
    def _build_tree(self, points: np.ndarray, indices: np.ndarray, depth: int) -> Optional[KDNode]:
        """
        Recursively build the KD-tree.
        
        Args:
            points: Array of vectors to build tree from
            indices: Original indices of these vectors in database
            depth: Current depth in tree
            
        Returns:
            Root node of the (sub)tree
        """
        if len(points) == 0:
            return None
        
        # Select axis based on depth
        axis = self._select_axis(depth)
        
        # Sort points by the selected axis
        sorted_idx = np.argsort(points[:, axis])
        points = points[sorted_idx]
        indices = indices[sorted_idx]
        
        # Select median point as pivot
        median_idx = len(points) // 2
        
        # Create node and recursively build subtrees
        node = KDNode(
            point=points[median_idx],
            index=indices[median_idx],
            axis=axis,
            left=self._build_tree(points[:median_idx], indices[:median_idx], depth + 1),
            right=self._build_tree(points[median_idx + 1:], indices[median_idx + 1:], depth + 1)
        )
        
        return node
    
    def calculate_distances_to_true_mean(self, database: np.ndarray) -> np.ndarray:
        """
        Efficiently calculate distances from all database vectors to true mean (0.5).
        Uses the formula: distance² = Σxᵢ² - Σxᵢ + 0.25d
        where d is the dimension.
        
        Args:
            database: Array of database vectors
            
        Returns:
            Array of distances to true mean
        """
        # Precompute sums and sum of squares for each vector
        sums = np.sum(database, axis=1)  # Σxᵢ for each vector
        sum_squares = np.sum(database * database, axis=1)  # Σxᵢ² for each vector
        
        # Calculate squared distances using the formula
        dimension = database.shape[1]
        squared_distances = sum_squares - sums + 0.25 * dimension
        
        # Return actual distances
        return np.sqrt(squared_distances)
    
    def build_tree(self, database: np.ndarray, n_queries: int) -> None:
        """
        Build the KD-tree from filtered database vectors.
        
        Args:
            database: Array of shape (n_database, dimension) containing vectors
            n_queries: Number of queries (used to compute filtering parameters)
        """
        self.dimension = database.shape[1]
        self.database = database
        
        # Compute filtering parameters
        max_fuel = 2_000_000_000
        base_fuel = 760_000_000
        alpha = 1700 * n_queries
        m = int((max_fuel - base_fuel) / alpha)  # Number of vectors for KD-tree
        
        print(f"Filtering {m} closest database vectors for KD-tree")
        
        # Time each step
        t0 = time.time()
        
        # Step 1: Calculate distances using optimized method
        distances = self.calculate_distances_to_true_mean(database)
        t1 = time.time()
        print(f"  Time to calculate distances: {(t1-t0)*1000:.2f}ms")
        
        # Step 2: Sort and select closest vectors
        closest_indices = np.argsort(distances)[:m]
        kd_vectors = database[closest_indices]
        self.kd_indices = closest_indices
        t2 = time.time()
        print(f"  Time to sort and select vectors: {(t2-t1)*1000:.2f}ms")
        
        # Step 3: Get remaining vectors
        mask = np.ones(len(database), dtype=bool)
        mask[self.kd_indices] = False
        self.r_vectors = database[mask]
        self.r_indices = np.arange(len(database))[mask]
        t3 = time.time()
        print(f"  Time to get remaining vectors: {(t3-t2)*1000:.2f}ms")
        
        # Step 4: Build KD-tree
        print(f"Building KD-tree with {m} vectors...")
        self.root = self._build_tree(kd_vectors, self.kd_indices, depth=0)
        t4 = time.time()
        print(f"  Time to build tree: {(t4-t3)*1000:.2f}ms")
        print(f"Total build time: {(t4-t0)*1000:.2f}ms")
    
    def _search_node(self, node: Optional[KDNode], query: np.ndarray, threshold: float,
                    best_dist: float, best_node: Optional[KDNode]) -> Tuple[float, Optional[KDNode]]:
        """
        Recursively search the tree for the closest point to query within threshold.
        
        Args:
            node: Current node in search
            query: Query vector
            threshold: Maximum allowed distance
            best_dist: Distance to best point found so far
            best_node: Best node found so far
            
        Returns:
            Tuple of (best distance found, node with best distance)
        """
        if node is None:
            return best_dist, best_node
        
        # Compute distance to current point
        dist = float(np.sqrt(np.sum((query - node.point) ** 2)))
        
        # Update best if this point is closer and within threshold
        if dist < best_dist and dist <= threshold:
            best_dist = dist
            best_node = node
            # Early return if we found a point within threshold
            return best_dist, best_node
        
        # Compute distance to splitting plane
        axis_dist = query[node.axis] - node.point[node.axis]
        
        # Recursively search subtrees
        if axis_dist <= 0:
            # Query is on left of splitting plane
            best_dist, best_node = self._search_node(node.left, query, threshold, best_dist, best_node)
            # Check right subtree if it could contain closer points
            if abs(axis_dist) <= threshold and best_dist > threshold:
                best_dist, best_node = self._search_node(node.right, query, threshold, best_dist, best_node)
        else:
            # Query is on right of splitting plane
            best_dist, best_node = self._search_node(node.right, query, threshold, best_dist, best_node)
            # Check left subtree if it could contain closer points
            if abs(axis_dist) <= threshold and best_dist > threshold:
                best_dist, best_node = self._search_node(node.left, query, threshold, best_dist, best_node)
        
        return best_dist, best_node
    
    def search(self, queries: np.ndarray, threshold: float) -> Tuple[bool, Optional[List[int]], Optional[List[float]]]:
        """
        For each query vector, try to find a database vector within threshold distance.
        Uses KD-tree for initial search and brute force for worst matches.
        
        Args:
            queries: Array of shape (n_queries, dimension) containing query vectors
            threshold: Maximum allowed distance between any query and its match
            
        Returns:
            Tuple containing:
                - Boolean indicating if matches were found for all queries
                - If successful, list of database indices matched to each query
                - If successful, list of distances for each match
        """
        if self.root is None:
            raise ValueError("Must build tree before searching")
        
        matches = []
        distances = []
        
        # Process each query
        for query in queries:
            # Search KD-tree first
            best_dist, best_node = self._search_node(
                self.root, query, threshold,
                float('inf'), None
            )
            
            # If no match found in KD-tree, try brute force on remaining vectors
            if best_node is None:
                # Calculate distances to remaining vectors
                r_distances = np.linalg.norm(self.r_vectors - query, axis=1)
                best_r_idx = np.argmin(r_distances)
                min_r_dist = r_distances[best_r_idx]
                
                if min_r_dist <= threshold:
                    matches.append(self.r_indices[best_r_idx])
                    distances.append(float(min_r_dist))
                else:
                    return False, None, None
            else:
                matches.append(best_node.index)
                distances.append(best_dist)
        
        return True, matches, distances

def run_kdtree(config, phase='build', data_structure=None):
    """
    Run KD-tree search algorithm with true mean filtering optimization.
    
    Args:
        config: Configuration dictionary with parameters
        phase: Either 'build' or 'search'
        data_structure: KDTreeTrueMeanFilter instance from build phase
        
    Returns:
        If phase == 'build': Returns the KDTreeTrueMeanFilter instance
        If phase == 'search': Returns (success, matches, distances) tuple
    """
    if phase == 'build':
        if 'database' not in config:
            raise ValueError("Database vectors must be provided in config")
            
        # Build KD-tree with filtering
        kdtree = KDTreeTrueMeanFilter()
        # Get n_queries from config or use default from VECTOR_SEARCH_CONFIG
        n_queries = config.get('n_queries', VECTOR_SEARCH_CONFIG['n_queries'])
        kdtree.build_tree(config['database'], n_queries)
        return kdtree
        
    elif phase == 'search':
        if data_structure is None:
            raise ValueError("Must provide KDTreeTrueMeanFilter instance for search phase")
        if 'queries' not in config or 'threshold' not in config:
            raise ValueError("Queries and threshold must be provided in config")
            
        kdtree = data_structure
        success, matches, distances = kdtree.search(config['queries'], config['threshold'])
        return success, matches, distances

if __name__ == "__main__":
    # Get parameters from config
    config = VECTOR_SEARCH_CONFIG.copy()
    
    print(f"\nRunning KD-tree with true mean filtering search with parameters:")
    print(f"Database size: {config['n_database']}")
    print(f"Number of queries: {config['n_queries']}")
    print(f"Dimension: {config['dimension']}")
    print(f"Distance threshold: {config['threshold']}")
    print(f"Random seed: {config['seed']}")
    
    # Build phase
    print("\nBuilding data structures...")
    build_start = time.time()
    kdtree = run_kdtree(config, phase='build')
    build_time = time.time() - build_start
    print(f"Build time: {build_time:.2f} seconds")
    
    # Search phase
    print("\nSearching for matches...")
    search_start = time.time()
    success, matches, distances = run_kdtree(config, phase='search', data_structure=kdtree)
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