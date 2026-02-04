#!/usr/bin/env python
"""
Run the Streamlit application.

Usage:
    python run_app.py

Or directly:
    streamlit run app/Home.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Get the app path
    app_path = Path(__file__).parent / "app" / "Home.py"

    if not app_path.exists():
        print(f"Error: App not found at {app_path}")
        sys.exit(1)

    print("🚀 Starting Multi-Agent XAI Text Classifier...")
    print(f"📁 App path: {app_path}")
    print("\n" + "=" * 50)
    print("The app will open in your browser automatically.")
    print("Press Ctrl+C to stop the server.")
    print("=" * 50 + "\n")

    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
    ])


if __name__ == "__main__":
    main()
