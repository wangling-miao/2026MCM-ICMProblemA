"""
Generate capacity distribution chart (Figure 2 in paper).
This script reads the processed features summary and creates the distribution plot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Configuration
OUTPUT_DIR = r"c:\Users\chenp\Documents\比赛\2026美赛\code\processed_data"
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "all_files_features_summary.csv")

def generate_capacity_distribution():
    """
    Generate the capacity distribution chart shown in Figure 2.
    """
    # Check if summary file exists
    if not os.path.exists(SUMMARY_FILE):
        print(f"Error: Summary file not found at {SUMMARY_FILE}")
        print("Please run process_battery_data.py first to generate the summary file.")
        return
    
    # Load features summary
    df = pd.read_csv(SUMMARY_FILE)
    
    # Extract discharge capacity values
    capacities = df['discharge_capacity'].dropna()
    
    if len(capacities) == 0:
        print("Error: No valid discharge capacity data found.")
        return
    
    print(f"Loaded {len(capacities)} samples with valid discharge capacity.")
    print(f"Capacity range: {capacities.min():.3f} Ah to {capacities.max():.3f} Ah")
    print(f"Mean capacity: {capacities.mean():.3f} Ah")
    print(f"Median capacity: {capacities.median():.3f} Ah")
    
    # Setup Plot Style (Publication Quality)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    
    # Create figure with higher DPI
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Define colors
    hist_color = '#5DADE2'  # Soft Blue
    kde_color = '#E74C3C'   # Soft Red/Coral
    edge_color = '#2E86C1'  # Darker Blue for edges
    
    # Histogram with better aesthetics
    n, bins, patches = ax.hist(
        capacities, 
        bins=35, 
        density=True, 
        alpha=0.65, 
        color=hist_color,
        edgecolor=edge_color,
        linewidth=0.8,
        label='Histogram',
        zorder=2
    )
    
    # Fit and plot kernel density estimation (KDE)
    kde = stats.gaussian_kde(capacities)
    x_range = np.linspace(capacities.min() * 0.9, capacities.max() * 1.1, 500)
    density = kde(x_range)
    
    # Plot KDE line with shadow effect
    ax.plot(
        x_range, 
        density, 
        color=kde_color, 
        linestyle='-', 
        linewidth=2.5, 
        label='Probability Density (KDE)',
        zorder=10
    )
    
    # Fill under KDE curve for extra "pop"
    ax.fill_between(x_range, density, alpha=0.1, color=kde_color, zorder=5)
    
    # Formatting Axes
    ax.set_xlabel('Discharge Capacity (Ah)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold', labelpad=10)
    
    # Title
    ax.set_title(
        f'Distribution of Battery Discharge Capacity\n(n={len(capacities)})', 
        fontsize=14, 
        fontweight='bold',
        pad=15,
        color='#333333'
    )
    
    # Custom Grid
    ax.grid(True, linestyle=':', alpha=0.6, color='gray', zorder=0)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # Add statistics annotation with styled box
    stats_text = (
        r'$\bf{Statistics}$' + '\n' +
        f'Mean:  {capacities.mean():.3f} Ah\n'
        f'Median: {capacities.median():.3f} Ah\n'
        f'Std Dev: {capacities.std():.3f} Ah\n'
        f'Range: [{capacities.min():.2f}, {capacities.max():.2f}]'
    )
    
    # Position text box
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='#CCCCCC', linewidth=1)
    ax.text(
        0.95, 0.95, 
        stats_text,
        transform=ax.transAxes,
        fontsize=10.5,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props,
        zorder=20,
        fontname='Monospace' # Monospace aligns numbers better
    )
    
    # Legend
    ax.legend(fontsize=11, loc='upper left', frameon=False)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "chart.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCapacity distribution chart saved to: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    generate_capacity_distribution()
