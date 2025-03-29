import pandas as pd
import matplotlib.pyplot as plt

def plot_runtime_comparison():
    # Read the CSV file
    df = pd.read_csv('runtime.csv')
    print(f"Total rows in CSV: {len(df)}")
    print("\nUnique algorithms found:", df['algorithm'].unique())
    
    # Set style
    plt.style.use('default')
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
    
    # Create figure and axis with larger size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot line for each algorithm with custom styling
    for i, algo in enumerate(['Roberts', 'Sobel', 'Canny']):
        algo_data = df[df['algorithm'] == algo]
        # Convert resolution strings to integers and sort
        resolutions = sorted([int(res.split('x')[0]) for res in algo_data['image_resolution']])
        times = [t for _, t in sorted(zip([int(res.split('x')[0]) for res in algo_data['image_resolution']], 
                                        algo_data['time_taken']))]
        print(f"\n{algo} data points:", len(resolutions))
        ax.plot(resolutions, times, 
               color=colors[i], marker='o', label=algo.capitalize(), 
               linewidth=2, markersize=4, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Image Resolution (pixels)', fontsize=12, labelpad=10)
    ax.set_ylabel('Time Taken (seconds)', fontsize=12, labelpad=10)
    ax.set_title('Edge Detection Algorithm Performance Comparison', 
                fontsize=14, pad=20)
    
    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)  # Put grid behind the lines
    
    # Customize legend
    ax.legend(fontsize=10, framealpha=0.9, loc='upper left')
    
    # Add light gray background to plot area
    ax.set_facecolor('#f8f9fa')
    
    # Add some padding to the layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as performance_comparison.png")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    plot_runtime_comparison()
