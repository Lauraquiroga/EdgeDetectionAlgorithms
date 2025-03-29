import pandas as pd
import matplotlib.pyplot as plt
import math


def plot_runtime_comparison():
    df = pd.read_csv('runtime.csv')
    print(f"Total rows in CSV: {len(df)}")
    print("\nUnique algorithms found:", df['algorithm'].unique())
    
    plt.style.use('default')
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot line for each algorithm 
    for i, algo in enumerate(['Roberts', 'Sobel', 'Canny']):
        algo_data = df[df['algorithm'] == algo]

        # Calculate pixel count for x-axis values while keeping original resolution for labels
        resolution_data = [(int(res.split('x')[0]) * int(res.split('x')[1]), res, t) 
                         for res, t in zip(algo_data['image_resolution'], algo_data['time_taken'])]
        sorted_data = sorted(resolution_data, key=lambda x: x[0])
        pixel_counts = [x[0] for x in sorted_data]
        original_resolutions = [x[1] for x in sorted_data]
        times = [x[2] for x in sorted_data]
        print(pixel_counts)
        print(f"\n{algo} data points:", len(pixel_counts))
        ax.plot(pixel_counts, times, 
               color=colors[i], marker='o', label=algo.capitalize(), 
               linewidth=2, markersize=4, alpha=0.8)
    
    # Find closest resolution for the given pixel count
    def format_resolution(x, p):
        closest_res = min(df['image_resolution'], 
                         key=lambda r: abs(int(r.split('x')[0]) * int(r.split('x')[1]) - x))
        return closest_res
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_resolution))
    
    ax.set_xlabel('Image Resolution', fontsize=12, labelpad=10)
    ax.set_ylabel('Time Taken (seconds)', fontsize=12, labelpad=10)
    ax.set_title('Edge Detection Algorithm Performance Comparison', 
                fontsize=14, pad=20)
    
    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)  
    
    ax.legend(fontsize=10, framealpha=0.9, loc='upper left')
    
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as performance_comparison.png")
    
    plt.show()

if __name__ == "__main__":
    plot_runtime_comparison()
