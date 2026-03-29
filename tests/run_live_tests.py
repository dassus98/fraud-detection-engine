"""
Orchestrator: starts the API server as a subprocess, waits until healthy,
runs the live test suite, then shuts down cleanly.

Usage:  python tests/run_live_tests.py
"""
import subprocess
import sys
import time
import os
import requests


PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"
STARTUP_TIMEOUT = 30   # seconds


def _wait_for_server(timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)
    return False


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    python = sys.executable

    print("Starting API server...")
    server = subprocess.Popen(
        [python, "-m", "uvicorn", "src.api.main:app",
         "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=project_root,
    )

    try:
        if not _wait_for_server(STARTUP_TIMEOUT):
            print(f"Server did not become healthy within {STARTUP_TIMEOUT}s.")
            server.terminate()
            sys.exit(1)

        elapsed = 0
        deadline = time.time() + STARTUP_TIMEOUT
        while time.time() < deadline:
            try:
                requests.get(f"{BASE}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)

        print(f"Server is up. Running tests...\n")

        # Import and run tests directly in the same process.
        # Avoids spawning another Python that might not find the server.
        sys.path.insert(0, project_root)
        from tests.test_api_live import run_single_tests, run_batch_tests
        run_single_tests()
        run_batch_tests()

    finally:
        server.terminate()
        server.wait()
        print("Server stopped.")


if __name__ == "__main__":
    main()
