import os
import pandas as pd
from simulate_censoring_pipeline import process_and_save_censored_datasets

# Create results/ folder if it doesn't exist
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load datasets (assumes CSV files are in the same directory as this script)
datasets = {
    "pbc": (pd.read_csv("./data/pbc_finale.csv"), os.path.join(RESULTS_DIR, "pbc")),
    "metabric": (pd.read_csv("./data/metabric_finale.csv"), os.path.join(RESULTS_DIR, "metabric"))
    }

# Create subdirectories for each dataset
for _, (_, path) in datasets.items():
    os.makedirs(path, exist_ok=True)

# Run the simulation on all datasets
if __name__ == "__main__":
    process_and_save_censored_datasets(datasets, desired_censor_rates=[0.1, 0.3, 0.5, 0.7, 0.9], random_state=42)
