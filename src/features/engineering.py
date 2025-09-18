# File: src/features/engineering.py
import pandas as pd
import numpy as np
from pathlib import Path

def engineer_features(input_path: Path, output_path: Path):
    """
    Create new features for the HR dataset to improve predictive power.
    
    Args:
        input_path (Path): Path to the cleaned CSV file
        output_path (Path): Path to save the feature-engineered CSV file
    """
    print("Loading data for feature engineering...")
    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")
    
    # Create a copy to avoid modifying the original
    df_eng = df.copy()
    
    # 1. Create interaction features
    # Salary to bonus ratio
    if all(col in df_eng.columns for col in ['Salary', 'Bonus']):
        df_eng['SalaryBonusRatio'] = df_eng['Salary'] / (df_eng['Bonus'] + 1)
    
    # 2. Create polynomial features
    if 'Age' in df_eng.columns:
        df_eng['AgeSquared'] = df_eng['Age'] ** 2

    if 'Salary' in df_eng.columns:
        # Remove or fix invalid values first
        df_eng.loc[df_eng['Salary'] <= -1, 'Salary'] = np.nan
        
        mean_salary = df_eng['Salary'].mean()
        df_eng['Salary'] = df_eng['Salary'].fillna(mean_salary)
        
        df_eng['Salary'] = np.log1p(df_eng['Salary'])
    
    # 3. Create categorical aggregations
    # Average salary by department
    if 'Department' in df_eng.columns and 'Salary' in df_eng.columns:
        dept_salary_means = df_eng.groupby('Department')['Salary'].mean().to_dict()
        df_eng['DeptSalaryRatio'] = df_eng['Salary'] / df_eng['Department'].map(dept_salary_means)

    # Average salary by job role
    if 'JobRole' in df_eng.columns and 'Salary' in df_eng.columns:
        role_salary_means = df_eng.groupby('JobRole')['Salary'].mean().to_dict()
        df_eng['RoleSalaryRatio'] = df_eng['Salary'] / df_eng['JobRole'].map(role_salary_means)
    
    # 4. Create tenure-based features
    if 'Attrition' in df_eng.columns:
        df_eng['AttritionFlag'] = df_eng['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # 5. Create flag features
    if all(col in df_eng.columns for col in ['Salary', 'Bonus']):
        df_eng['TotalCompensationCalculated'] = df_eng['Salary'] + df_eng['Bonus']
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save feature-engineered data
    df_eng.to_csv(output_path, index=False)
    print(f"Feature-engineered data saved!!")
    print(f"Final shape: {df_eng.shape}")
    
    return df_eng

if __name__ == "__main__":
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    input_csv = project_root / 'data' / 'processed' / 'hr_dataset_cleaned.csv'
    output_csv = project_root / 'data' / 'processed' / 'hr_dataset_engineered.csv'
    
    # Run feature engineering process
    engineer_features(input_csv, output_csv)