# Step 3: Create feature-engineered dataset
import pandas as pd
import numpy as np
import os

def feature_engineering():
    """
    Create derived features for the Housing dataset.
    New features for Boston Housing:
    - RM_LSTAT_interaction: rooms * lower status population
    - DIS_NOX_ratio: distance to employment / NOX pollution
    - TAX_PTRATIO_ratio: tax rate / pupil-teacher ratio
    - feature_squared: polynomial features
    - feature_interactions: cross-feature interactions
    """
    print("Loading cleaned data...")
    df = pd.read_csv('data/processed/housing_data_clean.csv')
    
    print(f"Creating derived features...")
    
    # Identify columns (check both uppercase and lowercase)
    target_col = None
    if 'MEDV' in df.columns:
        target_col = 'MEDV'
    elif 'medv' in df.columns:
        target_col = 'medv'
    
    feature_cols = [col for col in df.columns if col != target_col]
    
    features_added = 0
    
    # Boston Housing specific features (handle both uppercase and lowercase column names)
    if 'rm' in df.columns and 'lstat' in df.columns:
        df['rm_lstat_interaction'] = df['rm'] * df['lstat']
        features_added += 1
        print("  ✓ Created rm_lstat_interaction")
    elif 'RM' in df.columns and 'LSTAT' in df.columns:
        df['RM_LSTAT_interaction'] = df['RM'] * df['LSTAT']
        features_added += 1
        print("  ✓ Created RM_LSTAT_interaction")
    
    if 'dis' in df.columns and 'nox' in df.columns:
        df['dis_nox_ratio'] = df['dis'] / (df['nox'] + 0.0001)
        features_added += 1
        print("  ✓ Created dis_nox_ratio")
    elif 'DIS' in df.columns and 'NOX' in df.columns:
        df['DIS_NOX_ratio'] = df['DIS'] / (df['NOX'] + 0.0001)
        features_added += 1
        print("  ✓ Created DIS_NOX_ratio")
    
    if 'tax' in df.columns and 'ptratio' in df.columns:
        df['tax_ptratio_ratio'] = df['tax'] / (df['ptratio'] + 0.0001)
        features_added += 1
        print("  ✓ Created tax_ptratio_ratio")
    elif 'TAX' in df.columns and 'PTRATIO' in df.columns:
        df['TAX_PTRATIO_ratio'] = df['TAX'] / (df['PTRATIO'] + 0.0001)
        features_added += 1
        print("  ✓ Created TAX_PTRATIO_ratio")
    
    if 'age' in df.columns and 'dis' in df.columns:
        df['age_dis_interaction'] = df['age'] * df['dis']
        features_added += 1
        print("  ✓ Created age_dis_interaction")
    elif 'AGE' in df.columns and 'DIS' in df.columns:
        df['AGE_DIS_interaction'] = df['AGE'] * df['DIS']
        features_added += 1
        print("  ✓ Created AGE_DIS_interaction")
    
    # Universal polynomial features (works for any dataset)
    numeric_features = [col for col in feature_cols if df[col].dtype in ['float64', 'int64']]
    if len(numeric_features) >= 2:
        df[f'{numeric_features[0]}_squared'] = df[numeric_features[0]] ** 2
        df[f'{numeric_features[1]}_squared'] = df[numeric_features[1]] ** 2
        df[f'{numeric_features[0]}_{numeric_features[1]}_interaction'] = (
            df[numeric_features[0]] * df[numeric_features[1]]
        )
        features_added += 3
        print(f"  ✓ Created polynomial features")
    
    print(f"\n✓ Created {features_added} new features")
    
    # Save feature-engineered data
    os.makedirs('data/features', exist_ok=True)
    output_path = 'data/features/housing_data_featured.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Feature-engineered dataset saved to {output_path}")
    print(f"  - Samples: {len(df)}")
    print(f"  - Features: {len(df.columns)} (added {features_added} new features)")
    print(f"  - Size: {os.path.getsize(output_path) / 1024:.2f} KB")
    
    # Print target statistics
    if target_col and target_col in df.columns:
        print(f"\n  Target ({target_col}) statistics:")
        print(f"    - Mean: {df[target_col].mean():.2f}")
        print(f"    - Median: {df[target_col].median():.2f}")
        print(f"    - Std: {df[target_col].std():.2f}")
        print(f"    - Min: {df[target_col].min():.2f}")
        print(f"    - Max: {df[target_col].max():.2f}")
    
    return df

if __name__ == '__main__':
    feature_engineering()
