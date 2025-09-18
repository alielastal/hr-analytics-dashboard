# File: run_pipeline.py
import subprocess
import sys
from pathlib import Path

def run_phase(phase_name, command):
    """Run a phase and check for errors"""
    print(f"\n{'='*50}")
    print(f"STARTING PHASE: {phase_name}")
    print(f"{'='*50}")
    print(command)
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {phase_name} completed successfully!")
        print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR in {phase_name}:")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Run the entire pipeline sequentially"""
    project_root = Path(__file__).parent
    
    # Phase 1: Data Loading (if needed)
    print("Phase 1: Data loading already completed")
    
    # Phase 2: Data Cleaning
    success = run_phase(
        "Data Cleaning", 
        f"python {project_root / 'src' / 'data' / 'cleaning.py'}"
    )
    if not success:
        sys.exit(1)
    
    # Phase 2: Feature Engineering  
    success = run_phase(
        "Feature Engineering",
        f"python {project_root / 'src' / 'features' / 'engineering.py'}"
    )
    if not success:
        sys.exit(1)
    
    # Phase 3: EDA & Attrition Analysis
    print(f"\n{'='*50}")
    print("PHASE 3: Please run notebooks/03_attrition_analysis.ipynb")
    print("Manually execute all cells in the Jupyter notebook")
    print(f"{'='*50}")
    
    print("\n🎯 AFTER RUNNING PHASE 3, CHECK FOR:")
    print("✅ Department names are consistent (no duplicates)")
    print("✅ Salaries are in realistic ranges ($30,000-$200,000+)") 
    print("✅ Attrition patterns make logical sense")
    
    print(f"\n{'='*50}")
    print("PIPELINE EXECUTION COMPLETED!")
    print("Review Phase 3 results before proceeding to Phase 4 (Modeling)")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()