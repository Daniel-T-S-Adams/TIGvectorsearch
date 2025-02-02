import numpy as np
from typing import List, Optional, Tuple, Dict, Set
import heapq
from dataclasses import dataclass, field
from tqdm import tqdm
import time

@dataclass(order=True)
class DistanceEntry:
    """Helper class for priority queue operations."""
    distance: float
    index: int = field(compare=False)

class NSW:
    """
    Navigable Small World algorithm for vector search.
    Builds a graph by incrementally adding vectors and connecting each to its M nearest neighbors.
    Then uses greedy search through this graph to find vectors within a threshold distance.
    """
    
    def __init__(self, max_neighbors: int = 16):
        """
        Initialize the NSW algorithm.
        
        Args:
            max_neighbors: Maximum number of neighbors per vertex (M parameter)
        """
        self.max_neighbors = max_neighbors
        self.graph: Dict[int, Set[int]] = {}  # Adjacency list representation
        self.vectors: Optional[np.ndarray] = None
        
    def euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute Euclidean distance between two vectors."""
        return float(np.sqrt(np.sum((v1 - v2) ** 2)))
    
    def find_nearest_neighbors(self, query: np.ndarray, entry_point: int) -> List[DistanceEntry]:
        """
        Find M nearest neighbors of a query vector using greedy search through the graph.
        
        Args:
            query: Query vector
            entry_point: Index of the starting point in the graph
            
        Returns:
            List of DistanceEntry objects for the M nearest neighbors found
        """
        visited = set()
        candidates = []  # Priority queue of (distance, index)
        nearest = []    # List to store M nearest neighbors found
        
        def process_vertex(idx: int) -> None:
            if idx in visited:
                return
            visited.add(idx)
            dist = self.euclidean_distance(query, self.vectors[idx])
            entry = DistanceEntry(dist, idx)
            
            # Update nearest neighbors if needed
            if len(nearest) < self.max_neighbors:
                nearest.append(entry)
                nearest.sort(key=lambda x: x.distance)
            elif dist < nearest[-1].distance:
                nearest[-1] = entry
                nearest.sort(key=lambda x: x.distance)
            
            # Add neighbors to candidates queue
            for neighbor in self.graph[idx]:
                if neighbor not in visited:
                    heapq.heappush(candidates, DistanceEntry(
                        self.euclidean_distance(query, self.vectors[neighbor]),
                        neighbor
                    ))
        
        # Start with entry point
        process_vertex(entry_point)
        
        # Continue search until we've explored enough vertices
        while candidates and len(visited) < min(len(self.vectors), self.max_neighbors * 4):
            current = heapq.heappop(candidates)
            process_vertex(current.index)
            
            # Early stopping if we have enough neighbors and current distance is too large
            if len(nearest) == self.max_neighbors and current.distance > nearest[-1].distance * 1.5:
                break
        
        return nearest
    
    def build_graph(self, vectors: np.ndarray) -> None:
        """
        Build the NSW graph by incrementally adding vectors.
        Each vector is connected to its M nearest neighbors.
        
        Args:
            vectors: Array of shape (n_vectors, dimension) containing vectors
        """
        self.vectors = vectors
        n_vectors = len(vectors)
        
        # Start with first vector
        self.graph[0] = set()
        
        # Add remaining vectors
        for i in tqdm(range(1, n_vectors), desc="Building NSW graph"):
            # Find nearest neighbors of current vector
            entry_point = np.random.randint(i)  # Random entry point from existing graph
            neighbors = self.find_nearest_neighbors(vectors[i], entry_point)
            
            # Add edges to M nearest neighbors
            self.graph[i] = set()
            for neighbor in neighbors:
                self.graph[i].add(neighbor.index)
                self.graph[neighbor.index].add(i)  # Add bidirectional edge
    
    def search(self, queries: np.ndarray, threshold: float, n_tries: int = 3) -> Tuple[bool, Optional[List[int]]]:
        """
        For each query vector, try to find a database vector within threshold distance.
        Makes multiple attempts with different entry points for each query.
        
        Args:
            queries: Array of shape (n_queries, dimension) containing query vectors
            threshold: Maximum allowed distance between any query and its match
            n_tries: Number of attempts with different entry points per query
            
        Returns:
            Tuple containing:
                - Boolean indicating if matches were found for all queries
                - If successful, list of database indices matched to each query
        """
        if self.vectors is None:
            raise ValueError("Must build graph before searching")
        
        n_queries = len(queries)
        matches = []
        
        for query_idx, query in enumerate(tqdm(queries, desc="Processing queries")):
            found_match = False
            
            # Try multiple random entry points
            for _ in range(n_tries):
                entry_point = np.random.randint(len(self.vectors))
                neighbors = self.find_nearest_neighbors(query, entry_point)
                
                # Check if any neighbor is within threshold
                for neighbor in neighbors:
                    if neighbor.distance <= threshold:
                        matches.append(neighbor.index)
                        found_match = True
                        break
                
                if found_match:
                    break
            
            if not found_match:
                return False, None
        
        return True, matches

if __name__ == "__main__":
    # Test parameters
    n_database = 100000
    n_queries = 1
    dimension = 250
    threshold = 6.0
    seed = 42
    
    print(f"\nRunning NSW search with parameters:")
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
    nsw = NSW(max_neighbors=16)
    
    print("\nBuilding NSW graph...")
    build_start = time.time()
    nsw.build_graph(database)
    build_time = time.time() - build_start
    print(f"Build time: {build_time:.2f} seconds")
    
    print("\nSearching for matches...")
    search_start = time.time()
    success, matches = nsw.search(queries, threshold)
    search_time = time.time() - search_start
    
    print("\nResults:")
    print(f"Solution found: {success}")
    print(f"Search time: {search_time:.2f} seconds")
    if success:
        print("\nFound matches with distances:")
        for i, db_idx in enumerate(matches):
            dist = nsw.euclidean_distance(queries[i], database[db_idx])
            print(f"Query {i} -> Database[{db_idx}]: distance = {dist:.4f}") 