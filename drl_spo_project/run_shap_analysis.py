import torch
import numpy as np
import pandas as pd
import shap
import os
import matplotlib.pyplot as plt

from src.data_loader import load_all_data
from src.drl_environment import PortfolioEnv
from src.drl_agent import PPOAgent
from src.spo_layer import DifferentiableMVO
from src.spo_loss import SPOPlusLoss
from src.xai_utils import explain_with_shap

def run_shap():
    # --- Configuration ---
    MODEL_PATH = './models/ppo_portfolio_final.pth'
    NUM_BACKGROUND_SAMPLES = 100
    NUM_EXPLAIN_SAMPLES = 20
    # TARGET_SHAP_OUTPUT_INDEX = 0 # Changed to list or 'all'
    TARGET_SHAP_OUTPUT_INDICES = 'all' # Can be 'all' or a list like [0, 2]
    DEVICE = torch.device('cpu')
    DATA_PATH_PREFIX = 'data/'
    VIZ_DIR = './visualizations/shap/'

    print(f"Using device: {DEVICE}")
    if not os.path.exists(VIZ_DIR):
        os.makedirs(VIZ_DIR)
        print(f"Created directory: {VIZ_DIR}")

    # --- Load Data ---
    print("Loading data...")
    try:
        all_data = load_all_data(data_path_prefix=DATA_PATH_PREFIX)
        if all_data['chosen_etf_prices'].empty or all_data['combined_features'].empty:
             print("Critical dataframes (chosen_etf_prices or combined_features) are empty. Exiting.")
             return
    except Exception as e:
        print(f"Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return
    print("Data loaded successfully.")

    # --- Initialize Environment ---
    print("Initializing DRL environment...")
    try:
        env = PortfolioEnv(processed_data=all_data)
    except ValueError as e:
        print(f"Error initializing environment: {e}")
        return

    state_dim = env.observation_space.shape[0] # This is based on policy_feature_column_names due to Env changes
    action_dim = env.action_space.shape[0] # This is n_etfs

    # Use policy-specific feature names for SHAP
    policy_feature_names = env.policy_feature_column_names
    if not policy_feature_names:
        print("Error: policy_feature_column_names is empty in the environment. Cannot proceed with SHAP.")
        return
    print(f"Environment initialized. Policy State dim: {state_dim}, Action dim (n_etfs): {action_dim}")
    print(f"Number of policy features for SHAP: {len(policy_feature_names)}")

    # --- True Forward Return Feature Indices are no longer needed for PPOAgent ---
    # true_return_indices logic removed. PPOAgent constructor was changed.

    # --- Initialize MVO Solver, SPO+ Loss (dummy for agent init if only actor is needed) ---
    print("Initializing dummy MVO solver and SPO+ loss module for agent structure (if PPOAgent requires them)...")
    try:
        mvo_solver = DifferentiableMVO(num_assets=action_dim, max_weight_per_asset=0.35).to(DEVICE)
        spo_loss_module = SPOPlusLoss(num_assets=action_dim, mvo_max_weight_per_asset=0.35).to(DEVICE)
    except Exception as e:
        print(f"Error initializing MVO/SPO_Loss components: {e}")
        return

    # --- Initialize Agent and Load Model ---
    print("Initializing DRL agent...")
    try:
        # PPOAgent constructor no longer takes true_forward_return_feature_indices
        agent = PPOAgent(state_dim, action_dim,
                         # true_forward_return_feature_indices=None, # Argument removed
                         mvo_solver_instance=mvo_solver,
                         spo_loss_instance=spo_loss_module,
                         lr_actor=0.0003, lr_critic=0.001, gamma=0.99,
                         K_epochs=10, eps_clip=0.2, action_std_init=0.6,
                         spo_plus_loss_coeff=1.0)
    except Exception as e:
        print(f"Error initializing PPOAgent: {e}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Exiting.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    agent.policy_old.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    actor_model_to_explain = agent.policy_old.actor_mean.to(DEVICE)
    actor_model_to_explain.eval()
    print("Model loaded and actor_mean set to eval mode.")

    # --- Prepare Data for SHAP ---
    # Data for SHAP should only contain policy-observed features.
    # env.full_pivot_features_for_env contains all features including forward returns.
    # We sample from the full data, then select policy_feature_column_names.
    print("Preparing data for SHAP analysis...")
    if len(env.full_pivot_features_for_env) <= max(NUM_BACKGROUND_SAMPLES, NUM_EXPLAIN_SAMPLES) :
        print(f"Warning: Dataset size ({len(env.full_pivot_features_for_env)}) is small. Background and explanation samples might overlap or be the entire dataset.")

    if len(env.full_pivot_features_for_env) > NUM_BACKGROUND_SAMPLES:
        background_data_full_df = env.full_pivot_features_for_env.sample(n=NUM_BACKGROUND_SAMPLES, random_state=42)
    else:
        background_data_full_df = env.full_pivot_features_for_env.copy()
    # Select only policy features for SHAP background
    background_data_policy_df = background_data_full_df[policy_feature_names]
    background_observations_tensor = torch.tensor(background_data_policy_df.values, dtype=torch.float32).to(DEVICE)

    if len(env.full_pivot_features_for_env) > NUM_EXPLAIN_SAMPLES:
        explain_data_full_df = env.full_pivot_features_for_env.sample(n=NUM_EXPLAIN_SAMPLES, random_state=43)
    else:
        explain_data_full_df = env.full_pivot_features_for_env.copy()
    # Select only policy features for SHAP explanation data
    explain_data_policy_df = explain_data_full_df[policy_feature_names]
    observations_to_explain_tensor = torch.tensor(explain_data_policy_df.values, dtype=torch.float32).to(DEVICE)

    print(f"Background samples (policy features): {background_observations_tensor.shape[0]}, Explain samples (policy features): {observations_to_explain_tensor.shape[0]}")

    # --- Generate SHAP Values ---
    print("Generating SHAP values...")
    shap_values_list_all_outputs = explain_with_shap(
        actor_model_to_explain,
        background_observations_tensor,
        observations_to_explain_tensor,
        policy_feature_names # Pass policy-specific feature names
    )

    if not isinstance(shap_values_list_all_outputs, list) or not all(isinstance(arr, np.ndarray) for arr in shap_values_list_all_outputs):
        print("Error: explain_with_shap did not return the expected list of numpy arrays.")
        if isinstance(shap_values_list_all_outputs, np.ndarray) and shap_values_list_all_outputs.ndim == 3:
            print("Output was a single 3D numpy array. Converting to list of 2D arrays.")
            # Assuming shape is (num_samples, num_features, num_outputs) -> convert to list of (num_samples, num_features)
            shap_values_list_all_outputs = [shap_values_list_all_outputs[:, :, i] for i in range(shap_values_list_all_outputs.shape[2])]
        elif isinstance(shap_values_list_all_outputs, np.ndarray) and shap_values_list_all_outputs.ndim == 2 and action_dim == 1:
             print("Output was a single 2D numpy array and action_dim is 1. Wrapping in a list.")
             shap_values_list_all_outputs = [shap_values_list_all_outputs]
        else:
            print("Cannot proceed with SHAP value processing due to unexpected format.")
            return

    if not shap_values_list_all_outputs: # Empty list
        print("Error: explain_with_shap returned an empty list of SHAP values.")
        return

    # Determine which output indices to process
    output_indices_to_plot = []
    if TARGET_SHAP_OUTPUT_INDICES == 'all':
        output_indices_to_plot = list(range(len(shap_values_list_all_outputs)))
    elif isinstance(TARGET_SHAP_OUTPUT_INDICES, list):
        output_indices_to_plot = [i for i in TARGET_SHAP_OUTPUT_INDICES if i < len(shap_values_list_all_outputs)]
        if len(output_indices_to_plot) != len(TARGET_SHAP_OUTPUT_INDICES):
            print(f"Warning: Some target indices were out of bounds. Valid indices: {output_indices_to_plot}")
    else:
        print(f"Error: TARGET_SHAP_OUTPUT_INDICES format is invalid: {TARGET_SHAP_OUTPUT_INDICES}")
        return

    if not output_indices_to_plot:
        print("No valid output indices to plot SHAP values for.")
        return

    # --- Visualize SHAP Values for each target output index ---
    for current_output_index in output_indices_to_plot:
        if current_output_index >= len(shap_values_list_all_outputs):
            print(f"Skipping output index {current_output_index} as it's out of bounds for SHAP values list (len: {len(shap_values_list_all_outputs)}).")
            continue

        shap_values_for_target_output = shap_values_list_all_outputs[current_output_index]
        # shap_values_for_target_output should have shape (NUM_EXPLAIN_SAMPLES, num_policy_features)

        # Data for plotting should be the policy-specific features
        df_for_plot = explain_data_policy_df

        print(f"Visualizing SHAP values for output index {current_output_index}...")
        print(f"SHAP values shape: {shap_values_for_target_output.shape}, Data for plot shape: {df_for_plot.shape}")


        # Summary Plot (Bar)
        plt.figure()
        shap.summary_plot(shap_values_for_target_output, df_for_plot, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance (Bar) for Output Index {current_output_index}')
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, f'summary_bar_output_{current_output_index}.png'))
        plt.close()
        print(f"Saved summary bar plot for output index {current_output_index}.")

        # Summary Plot (Beeswarm)
        plt.figure()
        shap.summary_plot(shap_values_for_target_output, df_for_plot, plot_type="dot", show=False) # 'dot' is beeswarm
        plt.title(f'SHAP Feature Importance (Beeswarm) for Output Index {current_output_index}')
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, f'summary_beeswarm_output_{current_output_index}.png'))
        plt.close()
        print(f"Saved summary beeswarm plot for output index {current_output_index}.")

    print(f"SHAP visualizations saved to {VIZ_DIR} directory.")

if __name__ == '__main__':
    # Ensure necessary packages are installed
    try:
        import shap
        import matplotlib
    except ImportError as e:
        print(f"Missing one or more required packages (shap, matplotlib): {e}")
        print("Please install them using pip install shap matplotlib")
    else:
        run_shap()
