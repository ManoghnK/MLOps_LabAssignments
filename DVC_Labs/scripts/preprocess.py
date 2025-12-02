# Step 2: Clean and normalize the dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data():
    """
    Clean and normalize the Housing dataset.
    - Remove duplicates
    - Detect and remove outliers using IQR method
    - Normalize features using StandardScaler
    """
    print("Loading raw data...")
    df = pd.read_csv('data/raw/housing_data.csv')
    initial_samples = len(df)
    
    print(f"Initial samples: {initial_samples}")
    
    # Step 1: Remove duplicates
    df = df.drop_duplicates()
    after_dedup = len(df)
    print(f"✓ Removed {initial_samples - after_dedup} duplicates")
    
    # Step 2: Remove outliers using IQR method
    # Identify target column
    target_col = 'MEDV' if 'MEDV' in df.columns else None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier condition
    outlier_condition = (
        (df[numeric_cols] < (Q1 - 1.5 * IQR)) |
        (df[numeric_cols] > (Q3 + 1.5 * IQR))
    ).any(axis=1)
    
    df_clean = df[~outlier_condition].copy()
    outliers_removed = after_dedup - len(df_clean)
    print(f"✓ Removed {outliers_removed} outliers using IQR method")
    
    # Step 3: Normalize features (not target)
    scaler = StandardScaler()
    df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
    print(f"✓ Normalized features using StandardScaler")
    
    # Save cleaned data
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/housing_data_clean.csv'
    df_clean.to_csv(output_path, index=False)
    
    print(f"\n✓ Cleaned dataset saved to {output_path}")
    print(f"  - Samples: {len(df_clean)} ({(len(df_clean)/initial_samples)*100:.1f}% of original)")
    print(f"  - Features: {len(df_clean.columns)}")
    print(f"  - Size: {os.path.getsize(output_path) / 1024:.2f} KB")
    
    return df_clean

if __name__ == '__main__':
    preprocess_data()
