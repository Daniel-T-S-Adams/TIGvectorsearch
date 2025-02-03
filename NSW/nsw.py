import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import heapq
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

@dataclass(order=True)
class DistanceEntry:
    """Entry for priority queue, ordered by distance."""
    distance: float
    index: int

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
        
        Args:
            vectors: Array of vectors to build graph from
        """
        self.vectors = vectors
        n_vectors = len(vectors)
        
        # Add first vector
        self.graph[0] = set()
        
        # Add remaining vectors
        for idx in tqdm(range(1, n_vectors), desc="Building NSW graph"):
            # Find nearest neighbors among existing vertices
            entry_point = np.random.randint(idx)  # Random entry point
            neighbors = self.find_nearest_neighbors(vectors[idx], entry_point)
            
            # Add edges to graph (both directions)
            self.graph[idx] = set(n.index for n in neighbors)
            for neighbor in neighbors:
                self.graph[neighbor.index].add(idx)
                
                # Ensure max_neighbors constraint
                if len(self.graph[neighbor.index]) > self.max_neighbors:
                    # Remove furthest neighbor
                    furthest = max(
                        self.graph[neighbor.index],
                        key=lambda x: self.euclidean_distance(vectors[neighbor.index], vectors[x])
                    )
                    self.graph[neighbor.index].remove(furthest)
                    self.graph[furthest].remove(neighbor.index)

def run_nsw(config, phase='build', data_structure=None):
    """
    Run NSW algorithm.
    
    Args:
        config: Configuration dictionary with parameters
        phase: Either 'build' or 'search'
        data_structure: Tuple of (NSW instance, database vectors)
        
    Returns:
        If phase == 'build': Returns the data structure tuple
        If phase == 'search': Returns (success, matches, distances) tuple
    """
    if phase == 'build':
        # Generate or load database vectors
        n_database = config['n_database']
        dimension = config['dimension']
        seed = config['seed']
        database = generate_random_vectors(n_database, dimension, seed)
        
        # Build NSW graph
        print("Building NSW graph...")
        nsw = NSW(max_neighbors=16)
        nsw.build_graph(database)
        
        return (nsw, database)
        
    elif phase == 'search':
        if data_structure is None:
            raise ValueError("Must provide data structure for search phase")
            
        nsw, database = data_structure
        
        # Generate query vectors
        n_queries = config['n_queries']
        dimension = config['dimension']
        seed = config['seed']
        threshold = config['threshold']
        queries = generate_random_vectors(n_queries, dimension, seed + 1)
        
        # Search for matches
        matches = []
        distances = []
        n_tries = 3  # Number of attempts with different entry points per query
        
        print("Searching for matches...")
        for query_idx, query in enumerate(tqdm(queries)):
            found_match = False
            
            # Try multiple random entry points
            for _ in range(n_tries):
                entry_point = np.random.randint(len(database))
                neighbors = nsw.find_nearest_neighbors(query, entry_point)
                
                # Check if any neighbor is within threshold
                for neighbor in neighbors:
                    if neighbor.distance <= threshold:
                        matches.append(neighbor.index)
                        distances.append(neighbor.distance)
                        found_match = True
                        break
                
                if found_match:
                    break
            
            if not found_match:
                return False, None, None
        
        return True, matches, distances

if __name__ == "__main__":
    # Load parameters from config
    config = VECTOR_SEARCH_CONFIG.copy()
    
    print(f"\nRunning NSW search with parameters:")
    print(f"Database size: {config['n_database']}")
    print(f"Number of queries: {config['n_queries']}")
    print(f"Dimension: {config['dimension']}")
    print(f"Distance threshold: {config['threshold']}")
    print(f"Random seed: {config['seed']}")
    
    # Build phase
    print("\nBuilding data structures...")
    build_start = time.time()
    data_structure = run_nsw(config, phase='build')
    build_time = time.time() - build_start
    print(f"Build time: {build_time:.2f} seconds")
    
    # Search phase
    print("\nSearching for matches...")
    search_start = time.time()
    success, matches, distances = run_nsw(config, phase='search', data_structure=data_structure)
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