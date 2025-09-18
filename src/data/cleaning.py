# File: src/data/cleaning.py
import pandas as pd
import numpy as np
from pathlib import Path

def standardize_departments(df):
    """Standardize department names"""
    print("Before standardization:", df['Department'].unique())
    
    dept_mapping = {
        'Hr': 'HR', 
        'hr': 'HR',
        'HRR': 'HR',
        'I.T': 'IT', 
        'It': 'IT',
        'Finanace': 'Finance',
        'Slaes': 'Sales',
        'Markting': 'Marketing'
    }
    df['Department'] = df['Department'].replace(dept_mapping)
    df['Department'] = df['Department'].fillna('Unknown')

    print("After standardization:", df['Department'].unique())
    return df

def fix_salary_data(df):

    """Fix salary data quality issues"""
    print(f"Salary range before: ${df['Salary'].min()} - ${df['Salary'].max()}")
    
    # 1. Remove negative salaries (data error)
    original_count = len(df)
    df = df[df['Salary'] >= 0]
    removed_count = original_count - len(df)
    print(f"Removed {removed_count} records with negative salary")
    
    # 2. Remove extreme outliers (e.g., below $10,000 or above $500,000)
    df = df[(df['Salary'] >= 10000) & (df['Salary'] <= 500000)]
    
    print(f"Salary range after: ${df['Salary'].min()} - ${df['Salary'].max()}")
    return df

def clean_hr_data(input_path: Path, output_path: Path):
    """
    Clean the HR dataset by handling missing values, duplicates, and data inconsistencies.
    
    Args:
        input_path (Path): Path to the raw CSV file
        output_path (Path): Path to save the cleaned CSV file
    """
    print("Loading data for cleaning...")
    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # 1. Handle duplicates
    duplicates = df_clean.duplicated().sum()
    print(f"Found {duplicates} duplicate rows")
    df_clean = df_clean.drop_duplicates()
    
    # 2. Apply data fixes FIRST
    df_clean = standardize_departments(df_clean)
    df_clean = fix_salary_data(df_clean)
    
    # 3. Handle missing values
    print("Checking for missing values...")
    missing_values = df_clean.isnull().sum()
    missing_percent = (df_clean.isnull().sum() / len(df_clean)) * 100
    
    missing_df = pd.DataFrame({
        'missing_count': missing_values,
        'missing_percentage': missing_percent
    }).sort_values('missing_count', ascending=False)
    
    # Display columns with missing values
    missing_cols = missing_df[missing_df['missing_count'] > 0]
    print(f"Columns with missing values: {len(missing_cols)}")
    
    # Handle specific columns with missing values
    # (Adjust based on your EDA findings)
    if 'ColumnWithMissingData' in df_clean.columns:
        # Example: Fill numeric columns with median
        if df_clean['ColumnWithMissingData'].dtype in ['int64', 'float64']:
            median_val = df_clean['ColumnWithMissingData'].median()
            df_clean['ColumnWithMissingData'].fillna(median_val, inplace=True)
            print(f"Filled ColumnWithMissingData with median: {median_val}")
        # Example: Fill categorical columns with mode
        else:
            mode_val = df_clean['ColumnWithMissingData'].mode()[0]
            df_clean['ColumnWithMissingData'].fillna(mode_val, inplace=True)
            print(f"Filled ColumnWithMissingData with mode: {mode_val}")
    
    # 4. Handle data inconsistencies (example)
    # Standardize text columns
    text_columns = df_clean.select_dtypes(include=['object']).columns
    for col in text_columns:
        df_clean[col] = df_clean[col].str.strip().str.title()
    
    # 5. Fix data types if needed
    # Example: Convert specific columns to category type
    categorical_cols = ['Department', 'Gender', 'JobRole', 'Education', 'Attrition']
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('category')
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save cleaned data
    df_clean.to_csv(output_path, index=False)
    print(f"Cleaned data saved!!")
    print(f"Final shape: {df_clean.shape}")
    
    return df_clean

if __name__ == "__main__":
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    input_csv = project_root / 'data' / 'processed' / 'hr_dataset.csv'
    output_csv = project_root / 'data' / 'processed' / 'hr_dataset_cleaned.csv'
    
    # Run cleaning process
    clean_hr_data(input_csv, output_csv)