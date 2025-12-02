# Step 1: Download raw Boston Housing dataset
import pandas as pd
import os

def download_housing_data():
    """Download and save the raw Boston Housing dataset."""
    print("Downloading Boston Housing dataset...")
    
    # Load Boston Housing from online source (sklearn deprecated it)
    try:
        df = pd.read_csv(
            'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
        )
        print("✓ Loaded Boston Housing dataset")
    except Exception as e:
        print(f"Error loading from online source: {e}")
        print("Creating sample dataset...")
        # Fallback: create a small sample dataset
        import numpy as np
        np.random.seed(42)
        df = pd.DataFrame({
            'CRIM': np.random.uniform(0.006, 89, 506),
            'ZN': np.random.uniform(0, 100, 506),
            'INDUS': np.random.uniform(0.46, 27.74, 506),
            'CHAS': np.random.choice([0, 1], 506),
            'NOX': np.random.uniform(0.385, 0.871, 506),
            'RM': np.random.uniform(3.561, 8.780, 506),
            'AGE': np.random.uniform(2.9, 100, 506),
            'DIS': np.random.uniform(1.1296, 12.1265, 506),
            'RAD': np.random.randint(1, 25, 506),
            'TAX': np.random.randint(187, 712, 506),
            'PTRATIO': np.random.uniform(12.6, 22, 506),
            'B': np.random.uniform(0.32, 396.9, 506),
            'LSTAT': np.random.uniform(1.73, 37.97, 506),
            'MEDV': np.random.uniform(5, 50, 506)
        })
    
    # Ensure output directory exists
    os.makedirs('data/raw', exist_ok=True)
    
    # Save to CSV
    output_path = 'data/raw/housing_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✓ Raw dataset saved to {output_path}")
    print(f"  - Samples: {len(df)}")
    print(f"  - Features: {len(df.columns)}")
    print(f"  - Size: {os.path.getsize(output_path) / 1024:.2f} KB")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df

if __name__ == '__main__':
    download_housing_data()
