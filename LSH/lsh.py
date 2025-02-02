import numpy as np

def euclidean_lsh(v, a, b, w):
    """
    Compute a single Euclidean LSH hash value.
    """
    return int(np.floor((np.dot(a, v) + b) / w))

def hash_vectors(points, n_hashes, w):
    """
    Compute hash vectors for points using multiple hash functions.
    """
    n_points, dim = points.shape
    hash_vectors = np.zeros((n_points, n_hashes), dtype=int)
    hash_params = []
    
    for i in range(n_hashes):
        a = np.random.normal(0, 1, dim)
        b = np.random.uniform(0, w)
        hash_vectors[:, i] = [euclidean_lsh(p, a, b, w) for p in points]
        hash_params.append((a, b))
    
    return hash_vectors, hash_params

def find_similar_vectors(query_vectors, database_vectors, n_hashes=20, w=0.1, max_hash_dist=1):
    """
    Find database vectors similar to query vectors using LSH.
    
    Args:
        query_vectors: Query vectors to search for
        database_vectors: Database vectors to search in
        n_hashes: Number of hash functions
        w: Bucket width
        max_hash_dist: Maximum average hash difference allowed
    
    Returns:
        matches: List of arrays containing indices of similar database vectors
    """
    # Hash both query and database vectors
    query_hashes, params = hash_vectors(query_vectors, n_hashes, w)
    db_hashes = np.zeros((len(database_vectors), n_hashes), dtype=int)
    
    for i, (a, b) in enumerate(params):
        db_hashes[:, i] = [euclidean_lsh(p, a, b, w) for p in database_vectors]
    
    # Find matches for each query
    matches = []
    for q_hash in query_hashes:
        # Average hash difference per component
        hash_dists = np.abs(db_hashes - q_hash).mean(axis=1)
        matches.append(np.where(hash_dists <= max_hash_dist)[0])
    
    return matches

if __name__ == "__main__":
    # Example usage
    dim = 250
    query_vectors = np.random.uniform(0, 1, (3, dim))
    database_vectors = np.random.uniform(0, 1, (1000, dim))
    
    matches = find_similar_vectors(query_vectors, database_vectors)
    
    for i, match_indices in enumerate(matches):
        print(f"Query {i}: found {len(match_indices)} matches") 