# File: run_dashboard.py
import subprocess
import sys
from pathlib import Path

def run_dashboard():
    """Run the Streamlit dashboard"""
    dashboard_path = Path(__file__).parent / 'src' / 'app' / 'main.py'
    
    print("🚀 Starting HR Analytics Dashboard...")
    print("📊 Dashboard will open in your browser shortly")
    print("⏹️  Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard server stopped")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

if __name__ == "__main__":
    run_dashboard()