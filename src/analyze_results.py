#!/usr/bin/env python3

import os
import pandas as pd
import glob
import yaml
import json

# --- Configuration ---
# This script assumes it is located in the root of your 'cbf_safety_filter' package.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to the directory where all your scenario result folders are created.
RESULTS_BASE_DIR = os.path.join(PACKAGE_ROOT, "plots/sim")

# Path to the directory where the scenario YAML files were generated.
CONFIG_BASE_DIR = os.path.join(os.path.dirname((os.path.dirname(PACKAGE_ROOT))), "share", "cbf_safety_filter", "config", "generated_scenarios")

# The name of the final summary file.
SUMMARY_FILENAME = "batch_summary.csv"

def analyze_all_scenarios():
    """
    Scans all subdirectories in the results folder, reads each run_data.csv,
    correlates it with its scenario file, and compiles a summary of the outcomes.
    """
    if not os.path.isdir(RESULTS_BASE_DIR):
        print(f"Error: Results directory not found at '{RESULTS_BASE_DIR}'")
        print("Please ensure you have run the simulations first.")
        return

    # Find all run_data.csv files recursively within the results directory
    csv_files = glob.glob(os.path.join(RESULTS_BASE_DIR, "**/run_data.csv"), recursive=True)

    if not csv_files:
        print(f"No 'run_data.csv' files found in '{RESULTS_BASE_DIR}'.")
        return

    print(f"Found {len(csv_files)} result files. Analyzing...")

    all_summaries = []

    for csv_file_path in csv_files:
        try:
            # Extract the scenario name from the directory path
            scenario_name = os.path.basename(os.path.dirname(csv_file_path))
            
            # --- Load corresponding scenario data ---
            scenario_yaml_path = os.path.join(CONFIG_BASE_DIR, f"{scenario_name}.yaml")
            gamma = None
            beta = None
            obstacles_json = None

            if os.path.exists(scenario_yaml_path):
                with open(scenario_yaml_path, 'r') as f:
                    scenario_config = yaml.safe_load(f)
                    hocbf_params = scenario_config.get('hocbf_controller', {}).get('ros__parameters', {})
                    gamma = hocbf_params.get('gamma_js')
                    beta = hocbf_params.get('beta_js')
                    
                    # Serialize obstacle info into a JSON string for easy storage in one column
                    obstacles = scenario_config.get('obstacles', [])
                    obstacles_json = json.dumps(obstacles)
            else:
                print(f"Warning: Corresponding scenario file not found for {scenario_name}. Skipping extra data.")

            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(csv_file_path)
            
            if df.empty:
                print(f"Warning: {scenario_name} has an empty data file. Skipping.")
                summary = {
                    'scenario': scenario_name,
                    'status': 'EMPTY_DATA',
                    'duration_s': 0,
                    'min_h_val': None,
                    'min_psi_val': None,
                    'qp_infeasible_count': 0,
                    'gamma': gamma,
                    'beta': beta,
                    'obstacles_json': obstacles_json
                }
                all_summaries.append(summary)
                continue

            # Extract summary metrics from the DataFrame
            final_status = df['final_run_status'].iloc[0] # Status is the same for all rows
            total_duration = df['time'].iloc[-1]
            min_h = df['min_h'].min()
            min_psi = df['min_psi'].min()
            qp_infeasible_count = df['qp_infeasible'].sum()

            summary = {
                'scenario': scenario_name,
                'status': final_status,
                'duration_s': round(total_duration, 2),
                'min_h_val': round(min_h, 4),
                'min_psi_val': round(min_psi, 4),
                'qp_infeasible_count': int(qp_infeasible_count),
                'gamma': gamma,
                'beta': beta,
                'obstacles_json': obstacles_json
            }
            all_summaries.append(summary)

        except Exception as e:
            print(f"Error processing file {csv_file_path}: {e}")

    # Convert the list of summaries into a final DataFrame
    summary_df = pd.DataFrame(all_summaries)
    
    # Sort the results for better readability
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by='scenario')

    # Save the final summary to a CSV file in the base results directory
    summary_file_path = os.path.join(RESULTS_BASE_DIR, SUMMARY_FILENAME)
    summary_df.to_csv(summary_file_path, index=False)

    print("\n" + "="*50)
    print("Batch Analysis Complete")
    print("="*50)
    print(f"Summary report saved to: {summary_file_path}\n")

    # Print a high-level overview to the console
    if not summary_df.empty:
        status_counts = summary_df['status'].value_counts()
        print("Run Outcomes:")
        print(status_counts.to_string())
        
        failures = summary_df[summary_df['status'] != 'SUCCESS']
        if not failures.empty:
            print(f"\nTotal Failures (Timeouts/Errors): {len(failures)}")

        infeasible_runs = summary_df[summary_df['qp_infeasible_count'] > 0]
        if not infeasible_runs.empty:
            print(f"Scenarios with at least one QP infeasibility: {len(infeasible_runs)}")
            
    print("\nAnalysis finished.")


if __name__ == '__main__':
    analyze_all_scenarios()
