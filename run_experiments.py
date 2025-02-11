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
from config import VECTOR_SEARCH_CONFIG, ALGORITHMS_CONFIG
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
from kdtreemeanvariancefilter import run_kdtree as run_kdtree_mean_variance_filter
print("- Imported mean variance filtered KD-tree")

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
        
        # All available algorithms with descriptive names
        self.available_algorithms = {
            'Brute Force': run_bruteforce,
            'KD-Tree (Vanilla)': run_kdtree_vanilla,
            'KD-Tree (Mean Filter)': run_kdtree_mean_filter,
            'KD-Tree (Median Filter)': run_kdtree_median_filter,
            'KD-Tree (Random Filter)': run_kdtree_random_filter,
            'KD-Tree (True Mean Filter)': run_kdtree_true_mean_filter,
            'KD-Tree (Mean Variance Filter)': run_kdtree_mean_variance_filter
        }
        
        # Filter algorithms based on config
        if algorithms is not None:
            self.algorithms = algorithms
        else:
            self.algorithms = {
                name: func 
                for name, func in self.available_algorithms.items()
                if name in ALGORITHMS_CONFIG and ALGORITHMS_CONFIG[name]
            }
        
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
                    result = {
                        'build_time': build_time,
                        'search_time': search_time,
                        'total_time': total_time,
                        'success': success,
                        'data_structure': data_structure  # Store the data structure
                    }
                    
                    # Add search method timings if available
                    if hasattr(data_structure, 'kdtree_search_time'):
                        result['kdtree_search_time'] = data_structure.kdtree_search_time
                        result['bruteforce_search_time'] = data_structure.bruteforce_search_time
                    
                    algo_results.append(result)
                    
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
        # First Figure: Overall Performance
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        
        # Calculate averages for overall timings
        avg_times = {}
        std_times = {}
        success_rates = {}
        
        for algo_name, algo_results in results.items():
            build_times = [r['build_time'] for r in algo_results]
            search_times = [r['search_time'] for r in algo_results]
            total_times = [r['total_time'] for r in algo_results]
            
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
        
        # Plot overall timings
        algorithms = list(avg_times.keys())
        n_algorithms = len(algorithms)
        bar_width = 0.25
        index = np.arange(n_algorithms)
        
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
        
        # Add text labels for mean values
        for i, algo in enumerate(algorithms):
            # Build time
            build_val = avg_times[algo]['build']
            ax.text(i - bar_width, build_val, f'{build_val:.2f}s',
                   ha='center', va='bottom')
            # Search time
            search_val = avg_times[algo]['search']
            ax.text(i, search_val, f'{search_val:.2f}s',
                   ha='center', va='bottom')
            # Total time
            total_val = avg_times[algo]['total']
            ax.text(i + bar_width, total_val, f'{total_val:.2f}s',
                   ha='center', va='bottom')
        
        # Customize plot
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Overall Algorithm Performance (averaged over {self.n_seeds} seeds)')
        ax.set_xticks(index)
        ax.set_xticklabels(algorithms, rotation=15)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('overall_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Second Figure: Search Method Timings
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        
        # Calculate averages for search method timings
        search_method_times = {}
        search_method_stds = {}
        
        for algo_name, algo_results in results.items():
            # For algorithms with explicit timing breakdown
            if any('kdtree_search_time' in r for r in algo_results):
                first_search_times = [r['kdtree_search_time'] for r in algo_results]
                second_search_times = [r['bruteforce_search_time'] for r in algo_results]
            else:
                # For algorithms with only one search method
                first_search_times = [r['search_time'] for r in algo_results]
                second_search_times = [0.0] * len(algo_results)  # Zero for second search time
            
            search_method_times[algo_name] = {
                'first_search': np.mean(first_search_times),
                'second_search': np.mean(second_search_times)
            }
            search_method_stds[algo_name] = {
                'first_search': np.std(first_search_times),
                'second_search': np.std(second_search_times)
            }
        
        # Plot search method timings
        algorithms = list(search_method_times.keys())
        n_algorithms = len(algorithms)
        bar_width = 0.35
        index = np.arange(n_algorithms)
        
        first_search_bars = ax.bar(index - bar_width/2, 
                                [search_method_times[algo]['first_search'] for algo in algorithms],
                                bar_width, 
                                yerr=[search_method_stds[algo]['first_search'] for algo in algorithms],
                                label='Primary Search',
                                color='lightblue',
                                capsize=5)
        
        second_search_bars = ax.bar(index + bar_width/2, 
                                 [search_method_times[algo]['second_search'] for algo in algorithms],
                                 bar_width, 
                                 yerr=[search_method_stds[algo]['second_search'] for algo in algorithms],
                                 label='Secondary Search',
                                 color='lightpink',
                                 capsize=5)
        
        # Add text labels for mean values
        for i, algo in enumerate(algorithms):
            # First search time
            first_val = search_method_times[algo]['first_search']
            ax.text(i - bar_width/2, first_val, f'{first_val:.2f}s',
                   ha='center', va='bottom')
            # Second search time
            second_val = search_method_times[algo]['second_search']
            if second_val > 0:  # Only show non-zero values
                ax.text(i + bar_width/2, second_val, f'{second_val:.2f}s',
                       ha='center', va='bottom')
        
        # Customize plot
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Search Method Timings')
        ax.set_xticks(index)
        ax.set_xticklabels(algorithms, rotation=15)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('search_method_timings.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Third Figure: Match Statistics
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        
        # Calculate averages for match statistics
        match_stats = {}
        match_stds = {}
        
        for algo_name, algo_results in results.items():
            # Get the data structure from the last run to access match statistics
            data_structure = algo_results[-1]['data_structure']
            
            # For algorithms with both KD-tree and brute force matches
            if hasattr(data_structure, 'n_kdtree_matches') and hasattr(data_structure, 'n_bruteforce_matches'):
                kdtree_matches = [r['data_structure'].n_kdtree_matches for r in algo_results]
                bruteforce_matches = [r['data_structure'].n_bruteforce_matches for r in algo_results]
            # For algorithms with only KD-tree matches
            elif hasattr(data_structure, 'n_kdtree_matches'):
                kdtree_matches = [r['data_structure'].n_kdtree_matches for r in algo_results]
                bruteforce_matches = [0] * len(algo_results)
            # For algorithms with only brute force matches
            elif hasattr(data_structure, 'n_bruteforce_matches'):
                kdtree_matches = [0] * len(algo_results)
                bruteforce_matches = [r['data_structure'].n_bruteforce_matches for r in algo_results]
            
            match_stats[algo_name] = {
                'kdtree': np.mean(kdtree_matches),
                'bruteforce': np.mean(bruteforce_matches)
            }
            match_stds[algo_name] = {
                'kdtree': np.std(kdtree_matches),
                'bruteforce': np.std(bruteforce_matches)
            }
        
        # Plot match statistics
        algorithms = list(match_stats.keys())
        n_algorithms = len(algorithms)
        bar_width = 0.35
        index = np.arange(n_algorithms)
        
        kdtree_bars = ax.bar(index - bar_width/2, 
                           [match_stats[algo]['kdtree'] for algo in algorithms],
                           bar_width, 
                           yerr=[match_stds[algo]['kdtree'] for algo in algorithms],
                           label='KD-tree Matches',
                           color='lightgreen',
                           capsize=5)
        
        bruteforce_bars = ax.bar(index + bar_width/2, 
                               [match_stats[algo]['bruteforce'] for algo in algorithms],
                               bar_width, 
                               yerr=[match_stds[algo]['bruteforce'] for algo in algorithms],
                               label='Brute Force Matches',
                               color='lightcoral',
                               capsize=5)
        
        # Add text labels for mean values
        for i, algo in enumerate(algorithms):
            # KD-tree matches
            kdtree_val = match_stats[algo]['kdtree']
            if kdtree_val > 0:  # Only show non-zero values
                ax.text(i - bar_width/2, kdtree_val, f'{kdtree_val:.0f}',
                       ha='center', va='bottom')
            # Brute force matches
            bruteforce_val = match_stats[algo]['bruteforce']
            if bruteforce_val > 0:  # Only show non-zero values
                ax.text(i + bar_width/2, bruteforce_val, f'{bruteforce_val:.0f}',
                       ha='center', va='bottom')
        
        # Customize plot
        ax.set_ylabel('Number of Matches')
        ax.set_title('Match Statistics')
        ax.set_xticks(index)
        ax.set_xticklabels(algorithms, rotation=15)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('match_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    runner = AlgorithmRunner()
    results = runner.run_experiments()
    runner.plot_results(results)

if __name__ == "__main__":
    main() 