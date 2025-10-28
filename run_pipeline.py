import subprocess
import sys
import os
from dotenv import load_dotenv

# --- FIX: Load .env file at the beginning to check for the API key ---
load_dotenv()

# Define the order of scripts to be executed.
# Paths are relative to the project root directory.
SCRIPTS_TO_RUN = [
    # --- FIX: Call the correct data generation script ---
    "scripts/generate_data.py", 
    "scripts/preprocessing.py",
    "scripts/train_transformer.py",
    "scripts/train_gnn.py",
    "scripts/aura_risk_score.py",
    "scripts/explainability_engine.py",
]

def run_script(script_path: str) -> bool:
    """Executes a python script using the same interpreter and checks for errors."""
    print(f"\n{'='*25} RUNNING SCRIPT: {script_path} {'='*25}")

    # Check for the explainability engine and remind the user about the API key.
    if "explainability_engine.py" in script_path and not os.getenv("GOOGLE_API_KEY"):
        print("\n[!] WARNING: GOOGLE_API_KEY environment variable is not set.")
        print("    The explainability engine will fail. Please run 'python scripts/test_api_key.py' to verify.")
        print("    Skipping this step.")
        return True # Return True to allow the pipeline to finish, but skip this script

    # Use subprocess.run to execute the script.
    # We use sys.executable to ensure it uses the same python interpreter (e.g., from your venv)
    process = subprocess.run([sys.executable, script_path], check=False)

    if process.returncode != 0:
        print(f"\n--- SCRIPT FAILED: {script_path} with exit code {process.returncode} ---", file=sys.stderr)
        return False
    
    print(f"\n--- SCRIPT SUCCEEDED: {script_path} ---")
    return True

def main():
    """Orchestrates the execution of the entire AURA pipeline."""
    print(">>> STARTING AURA END-TO-END PIPELINE <<<")

    for script in SCRIPTS_TO_RUN:
        if not run_script(script):
            print(f"\n[ERROR] PIPELINE HALTED due to an error in {script}.")
            sys.exit(1) # Exit with an error code
    
    print("\n>>> AURA END-TO-END PIPELINE COMPLETED SUCCESSFULLY <<<")

if __name__ == "__main__":
    main()
