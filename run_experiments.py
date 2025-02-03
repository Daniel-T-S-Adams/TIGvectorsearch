import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add algorithm directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "kd-tree"))

from config import VECTOR_SEARCH_CONFIG
from example_bruteforce import run_bruteforce
from kdtreevanilla import run_kdtree

class AlgorithmRunner:
    def __init__(self, n_seeds=50, algorithms=None):
        self.n_seeds = n_seeds
        self.base_config = VECTOR_SEARCH_CONFIG.copy()
        
        # Default algorithms
        self.available_algorithms = {
            'brute_force': run_bruteforce,
            'kdtree': run_kdtree
        }
        
        self.algorithms = algorithms if algorithms is not None else self.available_algorithms

    def run_experiments(self):
        results = {}
        base_seed = self.base_config['seed']
        seeds = [base_seed + i for i in range(self.n_seeds)]
        
        for algo_name, algo_func in self.algorithms.items():
            print(f"\nRunning {algo_name}...")
            algo_results = []
            
            for seed in seeds:
                current_config = self.base_config.copy()
                current_config['seed'] = seed
                
                try:
                    # Build phase
                    build_start = time.time()
                    data_structure = algo_func(current_config, phase='build')
                    build_time = time.time() - build_start
                    
                    # Search phase
                    search_start = time.time()
                    success, matches, distances = algo_func(current_config, phase='search', data_structure=data_structure)
                    search_time = time.time() - search_start
                    
                    if not success:
                        print(f"Failed to find matches for seed {seed}")
                        continue
                        
                    total_time = build_time + search_time
                    algo_results.append({
                        'build_time': build_time,
                        'search_time': search_time,
                        'total_time': total_time
                    })
                    
                    print(f"Seed {seed}: Build={build_time:.2f}s, Search={search_time:.2f}s, Total={total_time:.2f}s")
                    
                except Exception as e:
                    print(f"Error running {algo_name} with seed {seed}: {str(e)}")
                    continue
            
            if algo_results:
                print(f"Successfully completed {len(algo_results)} runs for {algo_name}")
                results[algo_name] = algo_results
            else:
                print(f"No successful runs for {algo_name}")
        
        return results

    def plot_results(self, results):
        # Calculate averages
        avg_times = {}
        for algo_name, algo_results in results.items():
            avg_times[algo_name] = {
                'build': np.mean([r['build_time'] for r in algo_results]),
                'search': np.mean([r['search_time'] for r in algo_results]),
                'total': np.mean([r['total_time'] for r in algo_results])
            }
            
        # Debug prints
        print("\nData to be plotted:")
        for algo_name, times in avg_times.items():
            print(f"{algo_name}:")
            print(f"  Build time: {times['build']:.2f}s")
            print(f"  Search time: {times['search']:.2f}s")
            print(f"  Total time: {times['total']:.2f}s")
        
        # Prepare data
        algorithms = list(avg_times.keys())
        n_algorithms = len(algorithms)
        
        # Set up the bar positions
        bar_width = 0.25
        index = np.arange(n_algorithms)
        
        # Create figure and axis with larger size
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bars with different colors
        build_bars = ax.bar(index - bar_width, 
                           [avg_times[algo]['build'] for algo in algorithms],
                           bar_width, 
                           label='Build Time',
                           color='skyblue')
        
        search_bars = ax.bar(index, 
                            [avg_times[algo]['search'] for algo in algorithms],
                            bar_width, 
                            label='Search Time',
                            color='lightgreen')
        
        total_bars = ax.bar(index + bar_width, 
                           [avg_times[algo]['total'] for algo in algorithms],
                           bar_width, 
                           label='Total Time',
                           color='salmon')
        
        # Customize the plot
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Average Times by Algorithm')
        ax.set_xticks(index)
        ax.set_xticklabels(algorithms)
        
        # Add legend
        ax.legend()
        
        # Add value labels on top of bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}s',
                       ha='center', va='bottom')
        
        add_value_labels(build_bars)
        add_value_labels(search_bars)
        add_value_labels(total_bars)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()

def main():
    # Run all algorithms by default
    runner = AlgorithmRunner(n_seeds=VECTOR_SEARCH_CONFIG['n_seeds'])
    results = runner.run_experiments()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    for algo_name, algo_results in results.items():
        build_times = [r['build_time'] for r in algo_results]
        search_times = [r['search_time'] for r in algo_results]
        total_times = [r['total_time'] for r in algo_results]
        
        print(f"\n{algo_name}:")
        print(f"Build time  - Avg: {np.mean(build_times):.2f}s, Std: {np.std(build_times):.2f}s")
        print(f"Search time - Avg: {np.mean(search_times):.2f}s, Std: {np.std(search_times):.2f}s")
        print(f"Total time  - Avg: {np.mean(total_times):.2f}s, Std: {np.std(total_times):.2f}s")
    
    runner.plot_results(results)

if __name__ == "__main__":
    main() 