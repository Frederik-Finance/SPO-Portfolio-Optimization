"""
compute_and_plot_shap_evolution.py

This script loads DRL agent model snapshots, computes SHAP values for each,
aggregates them, and plots the evolution of feature importance over epochs.
"""
import torch
import numpy as np
import pandas as pd
import shap
import os
import matplotlib.pyplot as plt
import csv

# Assuming src components are accessible via PYTHONPATH or relative paths
from drl_spo_project.src.data_loader import load_all_data
from drl_spo_project.src.drl_environment import PortfolioEnv
from drl_spo_project.src.drl_agent import PPOAgent
from drl_spo_project.src.spo_layer import DifferentiableMVO
from drl_spo_project.src.spo_loss import SPOPlusLoss
from drl_spo_project.src.xai_utils import explain_with_shap

# --- Configuration ---
MANIFEST_CSV_PATH = 'drl_spo_project/training_logs/shap_models_manifest.csv'
DATA_PATH_PREFIX = 'data/' # Relative to project root
OUTPUT_DIR = 'drl_spo_project/visualizations/shap_over_time/' # For plots
DEVICE = torch.device('cpu')

NUM_BACKGROUND_SAMPLES = 50  # Fewer samples for speed during evolution analysis
NUM_EXPLAIN_SAMPLES = 10     # Fewer samples for speed
TARGET_OUTPUT_FOR_EVOLUTION = 0 # Index of the actor output to analyze (e.g., first asset)
TOP_N_FEATURES_TO_PLOT = 10

def compute_shap_for_model(actor_model, background_tensor, explain_tensor, policy_feature_names, target_output_index):
    """Computes SHAP values for a given model, data, and target output."""
    actor_model.eval()
    shap_values_list = explain_with_shap(
        actor_model,
        background_tensor,
        explain_tensor,
        policy_feature_names
    )

    if not isinstance(shap_values_list, list) or not shap_values_list:
        print("  Error: explain_with_shap did not return a valid list of SHAP values.")
        # Handle potential 3D array if action_dim > 1 but returned as single array
        if isinstance(shap_values_list, np.ndarray) and shap_values_list.ndim == 3:
            if target_output_index >= shap_values_list.shape[2]:
                print(f"  Target output index {target_output_index} out of bounds for 3D SHAP array.")
                return None
            shap_values_for_target = shap_values_list[:, :, target_output_index]
        # Handle 2D array if action_dim == 1
        elif isinstance(shap_values_list, np.ndarray) and shap_values_list.ndim == 2 and target_output_index == 0:
            shap_values_for_target = shap_values_list
        else:
            return None
    elif target_output_index >= len(shap_values_list):
        print(f"  Error: Target output index {target_output_index} is out of bounds for SHAP values list (len: {len(shap_values_list)}).")
        return None
    else:
        shap_values_for_target = shap_values_list[target_output_index]

    if shap_values_for_target is None or not isinstance(shap_values_for_target, np.ndarray) or shap_values_for_target.ndim != 2:
        print(f"  Error: SHAP values for target output {target_output_index} are not a 2D numpy array. Shape: {shap_values_for_target.shape if hasattr(shap_values_for_target, 'shape') else 'N/A'}")
        return None

    # Calculate mean absolute SHAP values for each feature for this model
    mean_abs_shap = np.mean(np.abs(shap_values_for_target), axis=0)
    return mean_abs_shap

def main():
    print("Starting SHAP evolution analysis...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # --- 1. Load Data and Initialize Environment (once) ---
    print("Loading data and initializing environment...")
    try:
        all_data = load_all_data(data_path_prefix=DATA_PATH_PREFIX)
        env = PortfolioEnv(processed_data=all_data)
    except Exception as e:
        print(f"Failed to load data or initialize environment: {e}")
        import traceback; traceback.print_exc()
        return

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy_feature_names = env.policy_feature_column_names

    if not policy_feature_names:
        print("Error: policy_feature_column_names is empty. Cannot proceed.")
        return
    print(f"Env initialized. Policy state dim: {state_dim}, Action dim: {action_dim}, Policy features: {len(policy_feature_names)}")

    # --- 2. Prepare Fixed Background and Explanation Datasets (once) ---
    print(f"Preparing fixed background ({NUM_BACKGROUND_SAMPLES}) and explanation ({NUM_EXPLAIN_SAMPLES}) datasets...")
    # Use full_pivot_features_for_env for sampling, then select policy_feature_column_names
    # Sample from the end of the dataset if it represents more recent/relevant data
    dataset_len = len(env.full_pivot_features_for_env)

    bg_sample_size = min(NUM_BACKGROUND_SAMPLES, dataset_len)
    ex_sample_size = min(NUM_EXPLAIN_SAMPLES, dataset_len)

    background_data_full_df = env.full_pivot_features_for_env.tail(max(bg_sample_size, ex_sample_size) + 5).sample(n=bg_sample_size, random_state=42, replace=False if bg_sample_size <= dataset_len else True)
    background_data_policy_df = background_data_full_df[policy_feature_names]
    background_tensor = torch.tensor(background_data_policy_df.values, dtype=torch.float32).to(DEVICE)

    explain_data_full_df = env.full_pivot_features_for_env.tail(max(bg_sample_size, ex_sample_size) + 5).sample(n=ex_sample_size, random_state=43, replace=False if ex_sample_size <= dataset_len else True)
    explain_data_policy_df = explain_data_full_df[policy_feature_names]
    explain_tensor = torch.tensor(explain_data_policy_df.values, dtype=torch.float32).to(DEVICE)
    print("Datasets prepared.")

    # --- 3. Load Model Manifest ---
    if not os.path.exists(MANIFEST_CSV_PATH):
        print(f"Error: Model manifest file not found at {MANIFEST_CSV_PATH}")
        return

    model_snapshots = []
    with open(MANIFEST_CSV_PATH, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                model_snapshots.append({'epoch': int(row['epoch']), 'path': row['model_snapshot_path']})
            except ValueError:
                print(f"Skipping row with invalid epoch: {row}")
                continue

    if not model_snapshots:
        print("No valid model snapshots found in manifest.")
        return
    # Sort by epoch
    model_snapshots.sort(key=lambda x: x['epoch'])
    print(f"Found {len(model_snapshots)} model snapshots in manifest.")

    # --- 4. Initialize Agent Structure (dummy MVO/SPO for actor loading) ---
    # This agent instance is primarily for its structure; policy state_dict will be loaded per snapshot.
    print("Initializing base DRL agent structure...")
    try:
        # These components might not be strictly necessary if only loading actor_mean's state_dict
        # and not running the full agent logic, but PPOAgent constructor requires them.
        mvo_solver = DifferentiableMVO(num_assets=action_dim, max_weight_per_asset=0.35).to(DEVICE)
        spo_loss_module = SPOPlusLoss(num_assets=action_dim, mvo_max_weight_per_asset=0.35).to(DEVICE)

        agent = PPOAgent(state_dim, action_dim,
                         mvo_solver_instance=mvo_solver,
                         spo_loss_instance=spo_loss_module)
    except Exception as e:
        print(f"Error initializing base PPOAgent structure: {e}")
        return

    # --- 5. Iterate Through Snapshots, Compute SHAP, and Aggregate ---
    all_epochs_shap_means = []
    processed_epochs = []

    for snapshot in model_snapshots:
        epoch = snapshot['epoch']
        model_path = snapshot['path']
        print(f"\nProcessing Epoch {epoch}, Model: {model_path}...")

        if not os.path.exists(model_path):
            print(f"  Warning: Model file not found at {model_path}. Skipping epoch {epoch}.")
            continue

        try:
            # Load the actor_mean's state_dict into the policy_old.actor_mean part of the agent
            # The manifest saves agent.policy_old.actor_mean.state_dict()
            agent.policy_old.actor_mean.load_state_dict(torch.load(model_path, map_location=DEVICE))
            actor_model_to_explain = agent.policy_old.actor_mean

            mean_abs_shap_for_epoch = compute_shap_for_model(
                actor_model_to_explain,
                background_tensor,
                explain_tensor,
                policy_feature_names,
                TARGET_OUTPUT_FOR_EVOLUTION
            )

            if mean_abs_shap_for_epoch is not None:
                all_epochs_shap_means.append(mean_abs_shap_for_epoch)
                processed_epochs.append(epoch)
                print(f"  Successfully computed SHAP for epoch {epoch}.")
            else:
                print(f"  Failed to compute SHAP for epoch {epoch}.")

        except Exception as e:
            print(f"  Error processing model for epoch {epoch}: {e}")
            import traceback; traceback.print_exc()
            continue

    if not all_epochs_shap_means:
        print("No SHAP values were successfully computed for any epoch. Exiting.")
        return

    # --- 6. Aggregate and Plot Evolution ---
    print("\nAggregating SHAP values and plotting evolution...")
    shap_evolution_df = pd.DataFrame(all_epochs_shap_means, index=processed_epochs, columns=policy_feature_names)

    # Select top N features based on mean absolute SHAP over all processed epochs
    overall_mean_abs_shap = shap_evolution_df.mean(axis=0).sort_values(ascending=False)
    top_n_to_plot = min(TOP_N_FEATURES_TO_PLOT, len(overall_mean_abs_shap))

    if top_n_to_plot == 0:
        print("No features to plot after aggregation. Exiting.")
        return

    top_features = overall_mean_abs_shap.head(top_n_to_plot).index.tolist()
    df_to_plot = shap_evolution_df[top_features]

    plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style
    fig, ax = plt.subplots(figsize=(15, 8))

    palette = plt.cm.get_cmap('tab10', top_n_to_plot) # Get a colormap

    for i, feature in enumerate(df_to_plot.columns):
        ax.plot(df_to_plot.index, df_to_plot[feature], label=feature, color=palette(i), marker='o', markersize=4, linestyle='-')

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Mean Absolute SHAP Value (for Output {TARGET_OUTPUT_FOR_EVOLUTION})")
    ax.set_title(f"Evolution of Top {top_n_to_plot} Feature Importances (Output {TARGET_OUTPUT_FOR_EVOLUTION})")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    plot_filename = f"mean_abs_shap_evolution_output_{TARGET_OUTPUT_FOR_EVOLUTION}.png"
    plot_path = os.path.join(OUTPUT_DIR, plot_filename)
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved SHAP evolution plot to {plot_path}")

    # Save the aggregated data to CSV for further analysis if needed
    csv_filename = f"mean_abs_shap_evolution_data_output_{TARGET_OUTPUT_FOR_EVOLUTION}.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    shap_evolution_df.to_csv(csv_path)
    print(f"Saved SHAP evolution data to {csv_path}")

    print("SHAP evolution analysis finished.")

if __name__ == '__main__':
    # This script assumes it's run from the project root or that
    # PYTHONPATH is set up for src imports.
    # Example: PYTHONPATH=. python drl_spo_project/visualizations/compute_and_plot_shap_evolution.py
    main()
