# Visualize dataset evolution across versions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def visualize_evolution():
    """Create comparison visualizations for dataset versions."""
    print("Creating visualizations...")
    
    # Load stats
    with open('reports/dataset_stats.json', 'r') as f:
        stats = json.load(f)
    
    # Set style
    sns.set_style('whitegrid')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dataset Evolution: Boston Housing Dataset Versioning', fontsize=16, fontweight='bold')
    
    versions = list(stats.keys())
    samples = [stats[v]['samples'] for v in versions]
    features = [stats[v]['features'] for v in versions]
    sizes = [stats[v]['size_kb'] for v in versions]
    
    # Plot 1: Sample Count
    axes[0, 0].bar(versions, samples, color=['#3498db', '#2ecc71', '#e74c3c'])
    axes[0, 0].set_title('Sample Count Across Versions', fontweight='bold')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].set_xlabel('Version')
    for i, v in enumerate(versions):
        axes[0, 0].text(i, samples[i] + 2, str(samples[i]), ha='center', fontweight='bold')
    
    # Plot 2: Feature Count
    axes[0, 1].bar(versions, features, color=['#3498db', '#2ecc71', '#e74c3c'])
    axes[0, 1].set_title('Feature Count Evolution', fontweight='bold')
    axes[0, 1].set_ylabel('Number of Features')
    axes[0, 1].set_xlabel('Version')
    for i, v in enumerate(versions):
        axes[0, 1].text(i, features[i] + 0.3, str(features[i]), ha='center', fontweight='bold')
    
    # Plot 3: File Size
    axes[1, 0].plot(versions, sizes, marker='o', linewidth=2, markersize=10, color='#9b59b6')
    axes[1, 0].fill_between(range(len(versions)), sizes, alpha=0.3, color='#9b59b6')
    axes[1, 0].set_title('File Size Changes', fontweight='bold')
    axes[1, 0].set_ylabel('File Size (KB)')
    axes[1, 0].set_xlabel('Version')
    axes[1, 0].grid(True, alpha=0.3)
    for i, v in enumerate(versions):
        axes[1, 0].text(i, sizes[i] + 0.5, f'{sizes[i]:.1f} KB', ha='center', fontweight='bold')
    
    # Plot 4: Target Distribution for v3.0 (histogram for regression)
    df_v3 = pd.read_csv('data/features/housing_data_featured.csv')
    target_col = 'MEDV' if 'MEDV' in df_v3.columns else df_v3.columns[-1]
    
    axes[1, 1].hist(df_v3[target_col], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title(f'Target Distribution ({target_col}) - v3.0', fontweight='bold')
    axes[1, 1].set_xlabel(f'{target_col}')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'reports/version_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")
    
    plt.close()
    
    # Create second figure: Target distribution across versions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Target Distribution Across Versions', fontsize=14, fontweight='bold')
    
    for idx, version in enumerate(versions):
        path = {
            'v1.0': 'data/raw/housing_data.csv',
            'v2.0': 'data/processed/housing_data_clean.csv',
            'v3.0': 'data/features/housing_data_featured.csv'
        }[version]
        
        df = pd.read_csv(path)
        target_col = 'MEDV' if 'MEDV' in df.columns else df.columns[-1]
        
        axes[idx].hist(df[target_col], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{version}', fontweight='bold')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_xlabel(f'{target_col}')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save second plot
    output_path2 = 'reports/target_distribution.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Target distribution plot saved to {output_path2}")
    
    plt.close()

if __name__ == '__main__':
    visualize_evolution()
