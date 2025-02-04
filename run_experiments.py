import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add algorithm directories to Python path
kd_tree_path = os.path.join(os.path.dirname(__file__), "kd_tree")
sys.path.append(kd_tree_path)
print(f"\nPython path includes: {kd_tree_path}")

# Import algorithms with debug prints
print("\nImporting algorithms...")
from config import VECTOR_SEARCH_CONFIG
print("- Imported config")
from example_bruteforce import run_bruteforce
print("- Imported brute force")
from kdtreevanilla import run_kdtree as run_kdtree_vanilla
print("- Imported vanilla KD-tree")
from kdtreemeanfilter import run_kdtree as run_kdtree_mean_filter
print("- Imported mean filtered KD-tree")
from kdtreerandomfilter import run_kdtree as run_kdtree_random_filter
print("- Imported random filtered KD-tree")
from kdtreemedianfilter import run_kdtree as run_kdtree_median_filter
print("- Imported median filtered KD-tree")
from kdtreetruemeanfilter import run_kdtree as run_kdtree_true_mean_filter
print("- Imported true mean filtered KD-tree")

def generate_random_vectors(n_vectors: int, dimension: int, seed: int = None) -> np.ndarray:
    """Generate random vectors with components from uniform[0,1]."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(0, 1, size=(n_vectors, dimension))

class AlgorithmRunner:
    def __init__(self, algorithms=None):
        self.config = VECTOR_SEARCH_CONFIG.copy()
        self.n_seeds = self.config['n_seeds']
        self.start_seed = self.config['seed']
        
        # Default algorithms with descriptive names
        self.available_algorithms = {
           # 'Brute Force': run_bruteforce,
           # 'KD-Tree (Vanilla)': run_kdtree_vanilla,
            'KD-Tree (Mean Filter)': run_kdtree_mean_filter,
            'KD-Tree (Median Filter)': run_kdtree_median_filter,
          #  'KD-Tree (Random Filter)': run_kdtree_random_filter,
            'KD-Tree (True Mean Filter)': run_kdtree_true_mean_filter
        }
        
        # Use provided algorithms if any, otherwise use all available
        self.algorithms = algorithms if algorithms is not None else self.available_algorithms.copy()
        
        # Debug print to show which algorithms are loaded
        print("\nLoaded algorithms:")
        for algo_name in self.algorithms:
            print(f"- {algo_name}")
        
        # Print seed information
        print(f"\nRunning experiments with seeds {self.start_seed} to {self.start_seed + self.n_seeds - 1}")

    def run_experiments(self):
        results = {}
        
        # For each algorithm
        print("\nStarting experiments with algorithms:")
        for algo_name, algo_func in self.algorithms.items():
            print(f"- {algo_name} (function: {algo_func.__module__}.{algo_func.__name__})")
            print(f"\nRunning {algo_name}...")
            algo_results = []
            
            # Run for each seed
            for seed_idx in range(self.n_seeds):
                current_seed = self.start_seed + seed_idx
                print(f"\nSeed {current_seed}:")
                
                try:
                    # Generate database and query vectors for this seed
                    database = generate_random_vectors(
                        self.config['n_database'],
                        self.config['dimension'],
                        seed=current_seed
                    )
                    queries = generate_random_vectors(
                        self.config['n_queries'],
                        self.config['dimension'],
                        seed=current_seed + 1  # Different seed for queries
                    )
                    
                    # Build phase
                    build_start = time.time()
                    data_structure = algo_func({'database': database}, phase='build')
                    build_time = time.time() - build_start
                    
                    # Search phase
                    search_start = time.time()
                    success, matches, distances = algo_func({
                        'queries': queries,
                        'threshold': self.config['threshold']
                    }, phase='search', data_structure=data_structure)
                    search_time = time.time() - search_start
                    
                    total_time = build_time + search_time
                    algo_results.append({
                        'build_time': build_time,
                        'search_time': search_time,
                        'total_time': total_time,
                        'success': success
                    })
                    
                    if not success:
                        print(f"  Failed to find matches for all queries")
                        print(f"  Build={build_time:.2f}s, Search={search_time:.2f}s, Total={total_time:.2f}s")
                    else:
                        print(f"  Build={build_time:.2f}s, Search={search_time:.2f}s, Total={total_time:.2f}s")
                    
                except Exception as e:
                    print(f"  Error: {str(e)}")
                    continue
            
            if algo_results:
                n_success = sum(1 for r in algo_results if r['success'])
                print(f"Successfully completed {n_success}/{len(algo_results)} runs")
                results[algo_name] = algo_results
            else:
                print(f"No runs completed")
        
        return results

    def plot_results(self, results):
        # Calculate averages
        avg_times = {}
        std_times = {}  # Also calculate standard deviations
        success_rates = {}  # Track success rates
        
        for algo_name, algo_results in results.items():
            build_times = [r['build_time'] for r in algo_results]
            search_times = [r['search_time'] for r in algo_results]
            total_times = [r['total_time'] for r in algo_results]
            
            # Calculate success rate
            n_success = sum(1 for r in algo_results if r['success'])
            success_rate = n_success / len(algo_results) if algo_results else 0
            success_rates[algo_name] = success_rate
            
            avg_times[algo_name] = {
                'build': np.mean(build_times),
                'search': np.mean(search_times),
                'total': np.mean(total_times)
            }
            std_times[algo_name] = {
                'build': np.std(build_times),
                'search': np.std(search_times),
                'total': np.std(total_times)
            }
        
        # Debug prints
        print("\nTiming Results (averaged over all runs):")
        for algo_name in avg_times:
            print(f"\n{algo_name}:")
            print(f"  Success Rate: {success_rates[algo_name]*100:.1f}%")
            print(f"  Build:  {avg_times[algo_name]['build']:.2f}s ± {std_times[algo_name]['build']:.2f}s")
            print(f"  Search: {avg_times[algo_name]['search']:.2f}s ± {std_times[algo_name]['search']:.2f}s")
            print(f"  Total:  {avg_times[algo_name]['total']:.2f}s ± {std_times[algo_name]['total']:.2f}s")
        
        # Prepare data for plotting
        algorithms = list(avg_times.keys())
        n_algorithms = len(algorithms)
        bar_width = 0.25
        index = np.arange(n_algorithms)
        
        # Create figure and axis with larger size for better readability
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        
        # Plot bars with error bars
        build_bars = ax.bar(index - bar_width, 
                           [avg_times[algo]['build'] for algo in algorithms],
                           bar_width, 
                           yerr=[std_times[algo]['build'] for algo in algorithms],
                           label='Build Time',
                           color='skyblue',
                           capsize=5)
        
        search_bars = ax.bar(index, 
                            [avg_times[algo]['search'] for algo in algorithms],
                            bar_width, 
                            yerr=[std_times[algo]['search'] for algo in algorithms],
                            label='Search Time',
                            color='lightgreen',
                            capsize=5)
        
        total_bars = ax.bar(index + bar_width, 
                           [avg_times[algo]['total'] for algo in algorithms],
                           bar_width, 
                           yerr=[std_times[algo]['total'] for algo in algorithms],
                           label='Total Time',
                           color='salmon',
                           capsize=5)
        
        # Customize plot
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Algorithm Performance (averaged over {self.n_seeds} seeds)')
        ax.set_xticks(index)
        ax.set_xticklabels(algorithms, rotation=15)
        ax.legend()
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}s',
                       ha='center', va='bottom')
        
        add_value_labels(build_bars)
        add_value_labels(search_bars)
        add_value_labels(total_bars)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig('experiment_results.png', dpi=300, bbox_inches='tight')
        
        # Display the figure
        plt.show()

def main():
    runner = AlgorithmRunner()
    results = runner.run_experiments()
    runner.plot_results(results)

if __name__ == "__main__":
    main() 