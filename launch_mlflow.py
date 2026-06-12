"""
launch_mlflow.py
Usage:
    python launch_mlflow.py
Then open: http://127.0.0.1:5000
"""

import os
import sys
from pathlib import Path

# Critical for Windows multiprocessing — must be set before any subprocess
if __name__ == "__main__":
    # Windows multiprocessing guard
    from multiprocessing import freeze_support
    freeze_support()

    PROJECT_ROOT = Path(__file__).parent
    DB_PATH      = str(PROJECT_ROOT / "mlruns" / "mlflow.db")
    TRACKING_URI = f"sqlite:///{DB_PATH}"

    # Must be set before MLflow imports
    os.environ["MLFLOW_TRACKING_URI"]        = TRACKING_URI
    os.environ["MLFLOW_BACKEND_STORE_URI"]   = TRACKING_URI

    print("=" * 60)
    print("  MLflow 3.x UI — Windows single-process mode")
    print(f"  Tracking URI : {TRACKING_URI}")
    print(f"  URL          : http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    from mlflow.server import _run_server

    _run_server(
        file_store_path=TRACKING_URI,
        registry_store_uri=TRACKING_URI,
        default_artifact_root=str(PROJECT_ROOT / "mlruns"),
        serve_artifacts=False,
        artifacts_only=False,
        artifacts_destination=None,
        host="127.0.0.1",
        port=5000,
        workers=1,
        # Force uvicorn to use single-process — no multiprocessing supervisor
        uvicorn_opts="--no-access-log",
    )
