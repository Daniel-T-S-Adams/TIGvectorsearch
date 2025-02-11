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

class KDTreeMeanVarianceFilter:
    """
    KD-tree implementation with mean-based filtering optimization for vector search.
    Also calculates variance along each dimension and splits along highest variance axes.
    """
    
    def __init__(self):
        """Initialize an empty KD-tree with filtering."""
        self.root: Optional[KDNode] = None
        self.r_vectors: Optional[np.ndarray] = None  # Remaining vectors not in KD-tree
        self.r_indices: Optional[np.ndarray] = None  # Indices of remaining vectors
        self.kd_indices: Optional[np.ndarray] = None  # Indices of vectors in KD-tree
        self.dimension: Optional[int] = None
        self.database: Optional[np.ndarray] = None  # Store full database for distance calculations
        self.variance: Optional[np.ndarray] = None  # Store variance along each dimension
        self.used_axes: set = set()  # Track which axes have been used for splitting
        self.sorted_variance_axes: Optional[np.ndarray] = None  # Axes sorted by variance
        # Add timing statistics
        self.kdtree_search_time: float = 0.0  # Time spent in KD-tree search
        self.bruteforce_search_time: float = 0.0  # Time spent in brute force search
        # Add match counting
        self.n_kdtree_matches: int = 0  # Number of matches found by KD-tree search
        self.n_bruteforce_matches: int = 0  # Number of matches found by brute force search
    
    def euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute Euclidean distance between two vectors."""
        return float(np.sqrt(np.sum((v1 - v2) ** 2)))
    
    def calculate_mean_and_variance(self, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both mean and variance vectors of query vectors.
        
        Args:
            queries: Array of query vectors
            
        Returns:
            Tuple containing:
                - Mean vector
                - Variance vector
        """
        mean = np.mean(queries, axis=0)
        variance = np.var(queries, axis=0)
        return mean, variance
    
    def filter_relevant_vectors(self, database: np.ndarray, mean_query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def _select_axis(self, depth: int) -> int:
        """
        Select splitting axis based on variance.
        Uses axes with highest variance first, without reusing axes.
        Falls back to cyclic selection after using all high variance axes.
        
        Args:
            depth: Current depth in tree (unused but kept for interface consistency)
            
        Returns:
            Selected axis for splitting
        """
        # If we haven't sorted axes by variance yet, do it now
        if self.sorted_variance_axes is None and self.variance is not None:
            # Get indices of axes sorted by variance (highest to lowest)
            self.sorted_variance_axes = np.argsort(-self.variance)  # Negative for descending order
        
        # Try to find unused high variance axis
        if self.sorted_variance_axes is not None:
            for axis in self.sorted_variance_axes:
                if axis < 250 and axis not in self.used_axes:  # Only consider first 250 dimensions
                    self.used_axes.add(axis)
                    return axis
        
        # If all high variance axes used or no variance info, fall back to cyclic selection
        for i in range(min(250, self.dimension)):  # Only consider first 250 dimensions
            if i not in self.used_axes:
                self.used_axes.add(i)
                return i
        
        # If all axes used, start over
        self.used_axes.clear()
        return self._select_axis(depth)
    
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
    
    def build_tree(self, database: np.ndarray, n_queries: int) -> None:
        """
        Build the KD-tree from filtered database vectors.
        
        Args:
            database: Array of shape (n_database, dimension) containing vectors
            n_queries: Number of queries (used to compute filtering parameters)
        """
        self.dimension = database.shape[1]
        self.database = database
        
        # Generate query vectors to compute mean and variance
        queries = np.random.uniform(0, 1, size=(n_queries, self.dimension))
        
        # Compute filtering parameters
        max_fuel = 2_000_000_000
        base_fuel = 760_000_000
        alpha = 1700 * n_queries
        m = int((max_fuel - base_fuel) / alpha)  # Number of vectors for KD-tree
        
        print(f"Filtering {m} closest database vectors for KD-tree")
        
        # Time each step
        t0 = time.time()
        
        # Calculate mean and variance
        mean_query, self.variance = self.calculate_mean_and_variance(queries)
        t1 = time.time()
        print(f"  Time to calculate mean and variance: {(t1-t0)*1000:.2f}ms")
        print(f"  Top 10 highest variance dimensions: {np.argsort(-self.variance)[:10]}")
        print(f"  Their variances: {self.variance[np.argsort(-self.variance)[:10]]}")
        
        # Reset axis tracking for new tree build
        self.used_axes.clear()
        self.sorted_variance_axes = None
        
        # Filter vectors for KD-tree
        kd_vectors, self.kd_indices = self.filter_relevant_vectors(database, mean_query, m)
        t2 = time.time()
        print(f"  Time to filter vectors: {(t2-t1)*1000:.2f}ms")
        
        # Get remaining vectors
        mask = np.ones(len(database), dtype=bool)
        mask[self.kd_indices] = False
        self.r_vectors = database[mask]
        self.r_indices = np.arange(len(database))[mask]
        t3 = time.time()
        print(f"  Time to get remaining vectors: {(t3-t2)*1000:.2f}ms")
        
        # Build KD-tree
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
        Search for matches to query vectors within distance threshold.
        Uses both KD-tree and brute force on remaining vectors.
        
        Args:
            queries: Array of query vectors to find matches for
            threshold: Maximum allowed distance for a match
            
        Returns:
            Tuple containing:
                - Whether matches were found for all queries
                - List of database indices for matches (or None if not all found)
                - List of distances to matches (or None if not all found)
        """
        matches: List[int] = []
        distances: List[float] = []
        
        # Reset statistics
        self.kdtree_search_time = 0.0
        self.bruteforce_search_time = 0.0
        self.n_kdtree_matches = 0
        self.n_bruteforce_matches = 0
        
        for query in queries:
            # Search KD-tree first
            best_dist = float('inf')
            best_node = None
            
            if self.root is not None:
                kdtree_start = time.time()
                best_dist, best_node = self._search_node(self.root, query, threshold, best_dist, best_node)
                self.kdtree_search_time += time.time() - kdtree_start
            
            # If no match in KD-tree, try remaining vectors
            if best_node is None and self.r_vectors is not None:
                bruteforce_start = time.time()
                r_distances = np.linalg.norm(self.r_vectors - query, axis=1)
                min_idx = np.argmin(r_distances)
                min_dist = r_distances[min_idx]
                self.bruteforce_search_time += time.time() - bruteforce_start
                
                if min_dist <= threshold:
                    matches.append(self.r_indices[min_idx])
                    distances.append(float(min_dist))
                    self.n_bruteforce_matches += 1
                    continue
            
            # Use KD-tree match if found
            if best_node is not None:
                matches.append(best_node.index)
                distances.append(best_dist)
                self.n_kdtree_matches += 1
            else:
                # No match found for this query
                return False, None, None
        
        # Print timing and match statistics
        total_time = self.kdtree_search_time + self.bruteforce_search_time
        total_matches = self.n_kdtree_matches + self.n_bruteforce_matches
        
        print("\nSearch Method Statistics:")
        print(f"  KD-tree search:     {self.kdtree_search_time:.3f}s ({(self.kdtree_search_time/total_time)*100:.1f}% of search time)")
        print(f"  Brute force search: {self.bruteforce_search_time:.3f}s ({(self.bruteforce_search_time/total_time)*100:.1f}% of search time)")
        print(f"  Total search time:  {total_time:.3f}s")
        
        print("\nMatch Statistics:")
        print(f"  Total matches found: {total_matches}")
        print(f"  KD-tree matches: {self.n_kdtree_matches} ({(self.n_kdtree_matches/total_matches)*100:.1f}%)")
        print(f"  Brute force matches: {self.n_bruteforce_matches} ({(self.n_bruteforce_matches/total_matches)*100:.1f}%)")
        
        return True, matches, distances

def run_kdtree(config: dict, phase: str = 'build', data_structure = None) -> Union[KDTreeMeanVarianceFilter, Tuple[bool, Optional[List[int]], Optional[List[float]]]]:
    """
    Run KD-tree with mean variance filtering for either build or search phase.
    
    Args:
        config: Dictionary containing either:
            - For build phase: {'database': database}
            - For search phase: {'queries': queries, 'threshold': threshold}
        phase: Either 'build' or 'search'
        data_structure: The KD-tree built in build phase (only needed for search phase)
        
    Returns:
        - For build phase: The constructed KD-tree
        - For search phase: Tuple of (success, matches, distances)
    """
    if phase == 'build':
        database = config['database']
        n_queries = VECTOR_SEARCH_CONFIG['n_queries']  # Get from global config
        
        kdtree = KDTreeMeanVarianceFilter()
        kdtree.build_tree(database, n_queries)
        return kdtree
        
    elif phase == 'search':
        if data_structure is None:
            raise ValueError("data_structure must be provided for search phase")
            
        queries = config['queries']
        threshold = config['threshold']
        
        return data_structure.search(queries, threshold)
    
    else:
        raise ValueError(f"Invalid phase: {phase}")

if __name__ == "__main__":
    # Get parameters from config
    config = VECTOR_SEARCH_CONFIG.copy()
    
    print(f"\nRunning KD-tree with mean filtering search with parameters:")
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