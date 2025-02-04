import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
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

@dataclass
class KDNode:
    """Node in the KD-tree."""
    point: np.ndarray          # The vector stored at this node
    index: int                 # Index of this vector in the original database
    axis: int                  # Split axis for this node
    left: Optional['KDNode']   # Left child (values less than split)
    right: Optional['KDNode']  # Right child (values greater than split)

class KDTree:
    """Basic KD-tree implementation for vector search."""
    
    def __init__(self):
        """Initialize an empty KD-tree."""
        self.root: Optional[KDNode] = None
        self.dimension: Optional[int] = None
    
    def build_tree(self, points: np.ndarray) -> None:
        """Build the KD-tree from points."""
        self.dimension = points.shape[1]
        indices = np.arange(len(points))
        self.root = self._build_tree(points, indices, depth=0)
    
    def _build_tree(self, points: np.ndarray, indices: np.ndarray, depth: int) -> Optional[KDNode]:
        """Recursively build the KD-tree."""
        if len(points) == 0:
            return None
        
        # Select axis based on depth
        axis = depth % self.dimension
        
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
    
    def _search_node(self, node: Optional[KDNode], query: np.ndarray, threshold: float,
                    best_dist: float, best_node: Optional[KDNode]) -> Tuple[float, Optional[KDNode]]:
        """Recursively search the tree for the closest point to query within threshold."""
        if node is None:
            return best_dist, best_node
        
        # Compute distance to current point
        dist = euclidean_distance(query, node.point)
        
        # Update best if this point is closer and within threshold
        if dist < best_dist and dist <= threshold:
            best_dist = dist
            best_node = node
        
        # Early return if we found a point within threshold
        if best_dist <= threshold:
            return best_dist, best_node
        
        # Compute distance to splitting plane
        axis_dist = query[node.axis] - node.point[node.axis]
        
        # Recursively search subtrees
        if axis_dist <= 0:
            # Query is on left of splitting plane
            best_dist, best_node = self._search_node(node.left, query, threshold, best_dist, best_node)
            # Check right subtree if it could contain closer points
            if abs(axis_dist) <= threshold:
                best_dist, best_node = self._search_node(node.right, query, threshold, best_dist, best_node)
        else:
            # Query is on right of splitting plane
            best_dist, best_node = self._search_node(node.right, query, threshold, best_dist, best_node)
            # Check left subtree if it could contain closer points
            if abs(axis_dist) <= threshold:
                best_dist, best_node = self._search_node(node.left, query, threshold, best_dist, best_node)
        
        return best_dist, best_node

def run_kdtree(config, phase='build', data_structure=None):
    """
    Run KD-tree search algorithm.
    
    Args:
        config: Configuration dictionary with parameters
        phase: Either 'build' or 'search'
        data_structure: Tuple of (KDTree instance, database vectors)
        
    Returns:
        If phase == 'build': Returns the data structure tuple
        If phase == 'search': Returns (success, matches, distances) tuple
    """
    if phase == 'build':
        if 'database' not in config:
            raise ValueError("Database vectors must be provided in config")
            
        # Build KD-tree
        database = config['database']
        kdtree = KDTree()
        kdtree.build_tree(database)
        
        return (kdtree, database)
        
    elif phase == 'search':
        if data_structure is None:
            raise ValueError("Must provide data structure for search phase")
        if 'queries' not in config or 'threshold' not in config:
            raise ValueError("Queries and threshold must be provided in config")
            
        kdtree, database = data_structure
        queries = config['queries']
        threshold = config['threshold']
        
        # Search for matches
        matches = []
        distances = []
        
        print("Searching for matches...")
        for query_idx, query in enumerate(tqdm(queries)):
            best_dist, best_node = kdtree._search_node(
                kdtree.root, query, threshold,
                float('inf'), None
            )
            
            if best_node is None:
                return False, None, None
                
            matches.append(best_node.index)
            distances.append(best_dist)
        
        return True, matches, distances

if __name__ == "__main__":
    # Load parameters from config
    config = VECTOR_SEARCH_CONFIG.copy()
    
    print(f"\nRunning KD-tree search with parameters:")
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