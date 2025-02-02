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
    
    def euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute Euclidean distance between two vectors."""
        return float(np.sqrt(np.sum((v1 - v2) ** 2)))
    
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
        dist = self.euclidean_distance(query, node.point)
        
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
    
    def search(self, queries: np.ndarray, threshold: float) -> Tuple[bool, Optional[List[int]]]:
        """
        For each query vector, try to find a database vector within threshold distance.
        
        Args:
            queries: Array of shape (n_queries, dimension) containing query vectors
            threshold: Maximum allowed distance between any query and its match
            
        Returns:
            Tuple containing:
                - Boolean indicating if matches were found for all queries
                - If successful, list of database indices matched to each query
        """
        if self.root is None:
            raise ValueError("Must build tree before searching")
        
        matches = []
        
        # Process each query
        for query_idx, query in enumerate(tqdm(queries, desc="Processing queries")):
            # Search for closest point within threshold
            best_dist, best_node = self._search_node(
                self.root, query, threshold,
                float('inf'), None
            )
            
            # If no point found within threshold, search fails
            if best_node is None:
                return False, None
            
            matches.append(best_node.index)
        
        return True, matches

if __name__ == "__main__":
    # Get parameters from config
    n_database = VECTOR_SEARCH_CONFIG['n_database']
    n_queries = VECTOR_SEARCH_CONFIG['n_queries']
    dimension = VECTOR_SEARCH_CONFIG['dimension']
    threshold = VECTOR_SEARCH_CONFIG['threshold']
    seed = VECTOR_SEARCH_CONFIG['seed']
    
    print(f"\nRunning KD-tree search with parameters:")
    print(f"Database size: {n_database}")
    print(f"Number of queries: {n_queries}")
    print(f"Dimension: {dimension}")
    print(f"Distance threshold: {threshold}")
    print(f"Random seed: {seed}")
    
    # Generate test data
    np.random.seed(seed)
    database = np.random.uniform(0, 1, size=(n_database, dimension))
    np.random.seed(seed + 1)  # Different seed for queries
    queries = np.random.uniform(0, 1, size=(n_queries, dimension))
    
    # Build and search
    kdtree = KDTree()
    
    total_start = time.time()  # Start total time measurement
    
    print("\nBuilding KD-tree...")
    build_start = time.time()
    kdtree.build_tree(database)
    build_time = time.time() - build_start
    print(f"Build time: {build_time:.2f} seconds")
    
    print("\nSearching for matches...")
    search_start = time.time()
    success, matches = kdtree.search(queries, threshold)
    search_time = time.time() - search_start
    
    total_time = time.time() - total_start  # Calculate total time
    
    print("\nResults:")
    print(f"Solution found: {success}")
    print(f"Build time: {build_time:.2f} seconds")
    print(f"Search time: {search_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    if success:
        print("\nFound matches with distances:")
        for i, db_idx in enumerate(matches):
            dist = kdtree.euclidean_distance(queries[i], database[db_idx])
            print(f"Query {i} -> Database[{db_idx}]: distance = {dist:.4f}") 