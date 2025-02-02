import numpy as np

def sample_uniform_vectors(n_samples=400, dim=250):
    """
    Generate samples of uniform random variables in high dimension.
    
    Args:
        n_samples: Number of samples to generate (default 400)
        dim: Dimension of each random vector (default 250)
    
    Returns:
        Array of shape (n_samples, dim) containing the samples
    """
    return np.random.uniform(0, 1, (n_samples, dim))

def analyze_sample_mean(samples):
    """
    Analyze the sample mean of the given samples and compare to true mean.
    
    Args:
        samples: Array of shape (n_samples, dim) containing uniform samples
    
    Returns:
        sample_mean: The sample mean vector
        true_mean: The true mean vector (all 0.5)
        l2_distance: L2 distance between sample and true mean
        linf_distance: L∞ distance between sample and true mean
    """
    n_samples, dim = samples.shape
    
    # Calculate means
    sample_mean = np.mean(samples, axis=0)
    true_mean = np.ones(dim) * 0.5
    
    # Calculate distances
    l2_distance = np.linalg.norm(sample_mean - true_mean)
    linf_distance = np.max(np.abs(sample_mean - true_mean))
    
    return sample_mean, true_mean, l2_distance, linf_distance

if __name__ == "__main__":
    # Generate samples
    samples = sample_uniform_vectors()
    
    # Analyze the mean
    sample_mean, true_mean, l2_dist, linf_dist = analyze_sample_mean(samples)
    
    print(f"Number of samples: 400")
    print(f"Dimension: 250")
    print(f"L2 distance to true mean: {l2_dist:.6f}")
    print(f"L∞ distance to true mean: {linf_dist:.6f}")
    
    # Optional: print first few components of the means
    print("\nFirst 5 components of:")
    print(f"Sample mean: {sample_mean[:5]}")
    print(f"True mean:   {true_mean[:5]}") 