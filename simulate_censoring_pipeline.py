import os
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

def extract_censor_dist(df_event: pd.DataFrame, df_all: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    Estimate the censoring distribution using a Cox model.
    Inputs:
        - df_event: DataFrame with event==1 rows
        - df_all: Full DataFrame with all patients
    Returns:
        - uniq_times: array of unique time values
        - censor_cdf: matrix of CDF values for each patient
    """
    df_all_copy = df_all.copy()
    df_all_copy['event'] = 1 - df_all_copy['event']
    
    cph = CoxPHFitter(penalizer=0.0001)
    cph.fit(df_all_copy, duration_col='time', event_col='event')
    
    censor_curves = cph.predict_survival_function(df_event)
    uniq_times = np.sort(df_all['time'].unique())
    censor_cdf = 1 - censor_curves.values.T
    
    return uniq_times, censor_cdf

def calculate_normalized_cdf_sampling(
    censor_cdf: np.ndarray,
    uniq_times: np.ndarray,
    df_event: pd.DataFrame,
    n_draws: int = 1000,
    n_iterations: int = 20,
    epsilon: float = 1e-10
) -> (list, list):
    """
    Compute normalized CDF probabilities with iterative correction.
    Inputs:
        - censor_cdf: censoring CDF matrix
        - uniq_times: time grid
        - df_event: uncensored patients
    Returns:
        - cdf_T: raw probabilities
        - normalized_cdf_T: corrected normalized probabilities
    """
    cdf_T = []
    for idx, row in df_event.iterrows():
        event_time = row['time']
        time_index = np.searchsorted(uniq_times, event_time)
        if time_index >= len(uniq_times):
            time_index = len(uniq_times) - 1
        probability = censor_cdf[idx, time_index]
        cdf_T.append(probability)

    cdf_T_with_epsilon = [p if p != 0 else epsilon for p in cdf_T]
    probas = np.array(cdf_T_with_epsilon, dtype=float)
    probas_norm = probas / probas.sum()

    n_probas = len(probas_norm)
    deviation = []

    Z = pd.DataFrame(np.zeros((n_probas, n_draws)), columns=[f'gen{i}' for i in range(n_draws)])
    for k in range(n_draws):
        selection = np.random.choice(range(n_probas), size=int(round(np.mean(probas)*n_probas, ndigits=0)), replace=False, p=probas_norm)
        Z.loc[selection, f'gen{k}'] = 1

    Znew = Z.copy()
    probas_norm_new = probas_norm
    for _ in range(n_iterations):
        q = Znew.mean(axis=1) - probas
        deviation.append(np.abs(q).sum())
        probas_norm_new = [j * (1 - q[i]) for i, j in enumerate(probas_norm_new)]
        probas_norm_new = probas_norm_new / np.sum(probas_norm_new)
        Znew = pd.DataFrame(np.zeros((n_probas, n_draws)), columns=[f'gen{i}' for i in range(n_draws)])
        for k in range(n_draws):
            selection = np.random.choice(range(n_probas), size=int(round(np.mean(probas)*n_probas, ndigits=0)), replace=False, p=probas_norm_new)
            Znew.loc[selection, f'gen{k}'] = 1

    return cdf_T_with_epsilon, probas_norm_new

def find_time_for_cdf(cdf_value, uniq_times, censor_cdf_row):
    """
    Interpolates the time corresponding to a given CDF value.
    """
    index = np.searchsorted(censor_cdf_row, cdf_value, side='left')
    if index < len(censor_cdf_row) and censor_cdf_row[index] == cdf_value:
        return uniq_times[index]
    if index == 0:
        return uniq_times[0]
    if index >= len(censor_cdf_row):
        return uniq_times[-1]

    cdf_left = censor_cdf_row[index - 1]
    cdf_right = censor_cdf_row[index]
    time_left = uniq_times[index - 1]
    time_right = uniq_times[index]

    interpolated_time = time_left + (cdf_value - cdf_left) * (time_right - time_left) / (cdf_right - cdf_left)
    return interpolated_time

def generate_synthetic_censoring_times(df_all, df_event, cdf_T, normalized_cdf_T, uniq_times, censor_cdf, desired_censor_rate, random_state=None):
    """
    Generate synthetic censoring times to achieve a target censoring rate.
    """
    if random_state is not None:
        np.random.seed(random_state)

    num_to_censor = int(len(df_event) * desired_censor_rate)
    uncensored_indices = np.random.choice(df_event.index, size=num_to_censor, replace=False, p=normalized_cdf_T)

    df_event = df_event.copy()
    df_event['true_time'] = df_event['time'].copy()

    for i in uncensored_indices:
        cdf_Ti = cdf_T[i]
        u = np.random.uniform(0, cdf_Ti)
        censor_time = find_time_for_cdf(u, uniq_times, censor_cdf[i])
        df_event.at[i, 'event'] = 0
        df_event.at[i, 'time'] = censor_time

    df_event = df_event[df_event['time'] != 0].reset_index(drop=True)
    return df_event

def apply_censoring_simulation(df_all, dataset_name='', desired_censor_rates=None, random_state=None, base_path='', n_replications=5):
    """
    Apply censoring simulation across different target rates and save the results.
    """
    if desired_censor_rates is None:
        desired_censor_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if 'time' not in df_all.columns or 'event' not in df_all.columns:
        raise ValueError("DataFrame must contain 'time' and 'event' columns.")

    for rate in desired_censor_rates:
        for repl in range(1, n_replications + 1):
            df_all_copy = df_all.copy()
            df_event = df_all_copy[df_all_copy.event == 1].copy().reset_index(drop=True)

            uniq_times, censor_cdf = extract_censor_dist(df_event, df_all_copy)
            cdf_T, normalized_cdf_T = calculate_normalized_cdf_sampling(censor_cdf, uniq_times, df_event)
            df_combined = generate_synthetic_censoring_times(df_all_copy, df_event, cdf_T, normalized_cdf_T, uniq_times, censor_cdf, rate, random_state=random_state + repl)

            censor_rate = np.round(((df_combined['event'] == 0).mean()) * 100).astype(int)
            filename = f"{dataset_name}_{censor_rate}_repl_{repl}.csv"
            file_path = os.path.join(base_path, filename)
            df_combined.to_csv(file_path, index=False)

            print(f"{dataset_name} with censoring rate {censor_rate}% (replication {repl}) saved to {file_path}")

def process_and_save_censored_datasets(datasets, desired_censor_rates=None, random_state=42):
    """
    Process all datasets and apply synthetic censoring simulation.
    """
    for dataset_name, (data, save_path) in datasets.items():
        print(f"Processing {dataset_name} dataset...")
        apply_censoring_simulation(df_all=data, dataset_name=dataset_name, 
                                   desired_censor_rates=desired_censor_rates, 
                                   random_state=random_state, base_path=save_path)
