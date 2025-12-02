# Analyze and compare dataset versions
import pandas as pd
import json
import os

def analyze_versions():
    """Generate statistical analysis for all dataset versions."""
    print("Analyzing dataset versions...")
    
    versions = {
        'v1.0': 'data/raw/housing_data.csv',
        'v2.0': 'data/processed/housing_data_clean.csv',
        'v3.0': 'data/features/housing_data_featured.csv'
    }
    
    stats = {}
    
    for version, path in versions.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            file_size = os.path.getsize(path) / 1024  # KB
            
            stats[version] = {
                'samples': len(df),
                'features': len(df.columns),
                'size_kb': round(file_size, 2),
                'numeric_features': len(df.select_dtypes(include=['number']).columns),
                'target_distribution': df['target'].value_counts().to_dict() if 'target' in df.columns else {}
            }
            
            print(f"\n{version}:")
            print(f"  Samples: {stats[version]['samples']}")
            print(f"  Features: {stats[version]['features']}")
            print(f"  Size: {stats[version]['size_kb']} KB")
    
    # Save to JSON
    os.makedirs('reports', exist_ok=True)
    output_path = 'reports/dataset_stats.json'
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Statistics saved to {output_path}")
    
    return stats

if __name__ == '__main__':
    analyze_versions()
