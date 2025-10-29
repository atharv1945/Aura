import subprocess
import sys
import os
from dotenv import load_dotenv
import time # For timing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

dotenv_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded environment variables from {dotenv_path}")
else:
    print(f"Warning: .env file not found at {dotenv_path}. GOOGLE_API_KEY should be set externally.")

SCRIPTS_TO_RUN = [
    os.path.join("scripts", "generate_data.py"),
    os.path.join("scripts", "preprocessing.py"),
    os.path.join("scripts", "train_transformer.py"),
    os.path.join("scripts", "train_gnn.py"),
    os.path.join("scripts", "aura_risk_score.py"),
    os.path.join("scripts", "explainability_engine.py"),
]

def run_script(script_path: str) -> bool:
    """Executes a python script using the same interpreter and checks for errors."""
    full_script_path = os.path.join(PROJECT_ROOT, script_path)
    if not os.path.exists(full_script_path):
        print(f"\n[ERROR] Script not found: {full_script_path}", file=sys.stderr)
        return False

    print(f"\n{'='*25} RUNNING SCRIPT: {script_path} {'='*25}")
    start_time = time.time()

    if "explainability_engine.py" in script_path and not os.getenv("GOOGLE_API_KEY"):
        print("\n[!] WARNING: GOOGLE_API_KEY environment variable is not set.")
        print("    The explainability engine requires an API key to function.")
        print("    Please create a .env file in the project root or set the environment variable.")
        print("    Skipping this step.")
        return True 

    process = subprocess.run([sys.executable, full_script_path],
                             check=False, cwd=PROJECT_ROOT) 

    end_time = time.time()
    duration = end_time - start_time

    if process.returncode != 0:
        print(f"\n--- SCRIPT FAILED: {script_path} (Exit Code: {process.returncode}) ---", file=sys.stderr)
        print(f"--- Duration: {duration:.2f} seconds ---", file=sys.stderr)
        return False
    else:
        print(f"\n--- SCRIPT SUCCEEDED: {script_path} ---")
        print(f"--- Duration: {duration:.2f} seconds ---")
        return True

def main():
    """Orchestrates the execution of the entire AURA pipeline."""
    print(f">>> STARTING AURA END-TO-END PIPELINE (Project Root: {PROJECT_ROOT}) <<<")
    overall_start_time = time.time()

    for script in SCRIPTS_TO_RUN:
        if not run_script(script):
            print(f"\n[ERROR] PIPELINE HALTED due to an error in {script}.")
            overall_end_time = time.time()
            print(f">>> Total Pipeline Duration (Before Halt): {overall_end_time - overall_start_time:.2f} seconds <<<")
            sys.exit(1) 

    overall_end_time = time.time()
    print("\n>>> AURA END-TO-END PIPELINE COMPLETED SUCCESSFULLY <<<")
    print(f">>> Total Pipeline Duration: {overall_end_time - overall_start_time:.2f} seconds <<<")

if __name__ == "__main__":
    main()

