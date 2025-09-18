import pandas as pd
from pathlib import Path

def load_and_process_hr_data(raw_data_path: Path, processed_data_path: Path):
    """
    Loads raw HR data from an Excel file, converts it to CSV,
    and saves it to the processed data directory.

    Args:
        raw_data_path (Path): The path to the raw Excel data file.
        processed_data_path (Path): The path to save the processed CSV file.
    """
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {raw_data_path}")

    print(f"Loading raw data from: {raw_data_path}")
    try:
        raw_df = pd.read_excel(raw_data_path)
        print("Raw data loaded successfully.")
    except Exception as e:
        raise ValueError(f"Error loading Excel file: {e}")

    # Ensure the parent directory for processed data exists
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving processed data to: {processed_data_path}")
    raw_df.to_csv(processed_data_path, index=False)
    print("Processed data saved successfully.")

if __name__ == "__main__":
    # Define base project directory (assuming script is in src/data/)
    project_root = Path(__file__).parent.parent.parent

    # Define paths relative to the project root
    raw_excel_file = project_root / 'data' / 'raw' / 'hr_dataset.xlsx'
    processed_csv_file = project_root / 'data' / 'processed' / 'hr_dataset.csv'

    try:
        load_and_process_hr_data(raw_excel_file, processed_csv_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error during data loading and processing: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")