# Censoring Simulation Pipeline

This repository contains the code used to generate semi-synthetic censoring in real-world survival datasets. The simulation reproduces different censoring rates while preserving realistic time distributions using an iterative probability correction method.

## Files

- `simulate_censoring_pipeline.py`: Core module containing all functions for simulating censoring.
- `main.py`: Entry point that loads datasets, applies the simulation, and saves results.
- `requirements.txt`: List of required Python packages.
- `data/`: Folder containing preprocessed survival datasets in CSV format.
- `results/`: Output folder where simulated datasets will be stored (automatically created).

## Setup Instructions

1. (Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the simulation pipeline:

```bash
python main.py
```

This will:
- Load the preprocessed CSV datasets from the `data/` folder
- Simulate censoring at different rates (e.g. 10%, 30%, 50%, 70%, 90%)
- Save the generated datasets in corresponding subfolders under `results/`


- All datasets in `data/` are already preprocessed and ready to use.
- You can modify the list of censoring rates and number of replications directly in `main.py`.
- Each generated file includes a `true_time` column preserving the original event time.