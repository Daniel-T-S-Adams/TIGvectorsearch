import numpy as np
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm
import time
from config import VECTOR_SEARCH_CONFIG

@dataclass
class KDNode:
    """Node in the KD-tree."""
    point: np.ndarray          # The vector stored at this node
    index: int                 # Index of this vector in the original database
    axis: int                  # Split axis for this node
    left: Optional['KDNode']   # Left child (values less than split)
    right: Optional['KDNode']  # Right child (values greater than split)

class KDTree:
    """
    KD-tree implementation for vector search.
    Builds a balanced KD-tree from the database vectors, then uses it
    to efficiently find vectors within a threshold distance of queries.
    """
    
    def __init__(self):
        """Initialize an empty KD-tree."""
        self.root: Optional[KDNode] = None
        self.dimension: Optional[int] = None
        # Add timing statistics
        self.kdtree_search_time: float = 0.0
        self.bruteforce_search_time: float = 0.0  # Always 0 for vanilla KD-tree
        # Add match counting
        self.n_kdtree_matches: int = 0  # Number of matches found by KD-tree search
        self.n_bruteforce_matches: int = 0  # Always 0 for vanilla KD-tree
    
    def euclidean_distance_vectorized(self, query: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances between a query and multiple points efficiently."""
        # Use broadcasting to compute distances
        diff = query - points
        return np.sqrt(np.sum(diff * diff, axis=1))
    
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
    
    def build_tree(self, database: np.ndarray) -> None:
        """
        Build the KD-tree from database vectors.
        
        Args:
            database: Array of shape (n_database, dimension) containing vectors
        """
        print("\nBuilding KD-tree...")
        self.dimension = database.shape[1]
        indices = np.arange(len(database))
        self.root = self._build_tree(database, indices, depth=0)
        self.database = database  # Store for distance calculations
    
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
        
        # Reset statistics
        self.kdtree_search_time = 0.0
        self.bruteforce_search_time = 0.0  # Always 0 for vanilla KD-tree
        self.n_kdtree_matches = 0
        self.n_bruteforce_matches = 0  # Always 0 for vanilla KD-tree
        
        # Process each query
        for query in queries:
            # Search for closest point within threshold
            kdtree_start = time.time()
            best_dist, best_node = self._search_node(
                self.root, query, threshold,
                float('inf'), None
            )
            self.kdtree_search_time += time.time() - kdtree_start
            
            # If no point found within threshold, search fails
            if best_node is None:
                return False, None, None
            
            matches.append(best_node.index)
            distances.append(best_dist)
            self.n_kdtree_matches += 1
        
        # Print match statistics
        print("\nMatch Statistics:")
        print(f"  Total matches found: {self.n_kdtree_matches}")
        print(f"  All matches found by KD-tree search")
        
        return True, matches, distances

def run_kdtree(config, phase='build', data_structure=None):
    """
    Run KD-tree search algorithm.
    
    Args:
        config: Configuration dictionary with parameters
        phase: Either 'build' or 'search'
        data_structure: KDTree instance from build phase
        
    Returns:
        If phase == 'build': Returns the KDTree instance
        If phase == 'search': Returns (success, matches, distances) tuple
    """
    if phase == 'build':
        if 'database' not in config:
            raise ValueError("Database vectors must be provided in config")
            
        # Build KD-tree
        kdtree = KDTree()
        kdtree.build_tree(config['database'])
        return kdtree
        
    elif phase == 'search':
        if data_structure is None:
            raise ValueError("Must provide KDTree instance for search phase")
        if 'queries' not in config or 'threshold' not in config:
            raise ValueError("Queries and threshold must be provided in config")
            
        kdtree = data_structure
        return kdtree.search(config['queries'], config['threshold'])

if __name__ == "__main__":
    # Get parameters from config
    config = VECTOR_SEARCH_CONFIG.copy()
    
    print(f"\nRunning KD-tree search with parameters:")
    print(f"Database size: {config['n_database']}")
    print(f"Number of queries: {config['n_queries']}")
    print(f"Dimension: {config['dimension']}")
    print(f"Distance threshold: {config['threshold']}")
    print(f"Random seed: {config['seed']}")
    
    # Build phase
    print("\nBuilding KD-tree...")
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