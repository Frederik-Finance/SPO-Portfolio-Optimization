import torch
import torch.nn as nn
import shap
import pandas as pd
import argparse
import os
import csv
import numpy as np

class ActorCritic(nn.Module):
    """
    """
    def __init__(self, state_dim: int, action_dim: int):
        """
        """
        super(ActorCritic, self).__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        """
        return self.actor_mean(state)

def analyze_single_model_shap(model_snapshot_path: str, epoch_number: int,
                              output_csv_path: str,
                              background_data_tensor: torch.Tensor,
                              explain_data_tensor: torch.Tensor,
                              target_shap_output_index: int,
                              state_dim: int, action_dim: int,
                              feature_names: list,
                              device: torch.device):
    """
    """
    print(f"Processing model for epoch {epoch_number} from {model_snapshot_path}...")

    try:
        model = ActorCritic(state_dim, action_dim).to(device)
        actor_mean_state_dict = torch.load(model_snapshot_path, map_location=device)
        model.actor_mean.load_state_dict(actor_mean_state_dict)
        model.eval()
        print(f"  Model for epoch {epoch_number} loaded successfully to {device}.")
    except Exception as e:
        print(f"  Error loading model for epoch {epoch_number} from {model_snapshot_path}: {e}")
        return

    def model_output_wrapper(x: torch.Tensor) -> torch.Tensor:
        return model(x)[:, target_shap_output_index]

    try:
        explainer = shap.DeepExplainer(model_output_wrapper, background_data_tensor)
        shap_values_np = explainer.shap_values(explain_data_tensor)
        print(f"  SHAP values computed for epoch {epoch_number}. Shape: {shap_values_np.shape}")
    except Exception as e:
        print(f"  Error during SHAP computation for epoch {epoch_number}: {e}")
        return

    if shap_values_np.ndim == 1:
        mean_abs_shap_values_np = np.abs(shap_values_np)
    else:
        mean_abs_shap_values_np = np.abs(shap_values_np).mean(axis=0)

    num_shap_vals = len(mean_abs_shap_values_np)
    num_feat_names = len(feature_names)
    effective_len = min(num_shap_vals, num_feat_names)

    mean_abs_shap_values_for_csv = mean_abs_shap_values_np[:effective_len]

    if num_shap_vals != num_feat_names:
        print(f"  Warning for epoch {epoch_number}: Mismatch between SHAP values ({num_shap_vals}) "
              f"and feature names ({num_feat_names}). Using {effective_len} values for CSV.")

    try:
        with open(output_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch_number] + mean_abs_shap_values_for_csv.tolist())
        print(f"  Appended SHAP data for epoch {epoch_number} to {output_csv_path}")
    except Exception as e:
        print(f"  Error appending SHAP data for epoch {epoch_number} to {output_csv_path}: {e}")


def main_shap_analysis_from_manifest(
    manifest_path: str, output_csv_path: str,
    background_data_path: str, explain_data_path: str,
    target_shap_output_index: int, state_dim: int, action_dim: int,
    feature_names_path: str):
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for SHAP analysis.")

    try:
        with open(feature_names_path, 'r') as f:
            feature_names_line = f.read().strip()
        if not feature_names_line:
            raise ValueError("Feature names file is empty.")
        feature_names = [name.strip() for name in feature_names_line.split(',')]
        print(f"Loaded {len(feature_names)} feature names from {feature_names_path}.")
        if len(feature_names) != state_dim:
            print(f"Warning: Number of feature names ({len(feature_names)}) from {feature_names_path} "
                  f"does not match provided state_dim ({state_dim}). Ensure consistency.")
    except Exception as e:
        print(f"Critical Error: Failed to load feature names from {feature_names_path}: {e}. Cannot proceed.")
        return

    try:
        background_data_tensor = torch.load(background_data_path).to(device)
        explain_data_tensor = torch.load(explain_data_path).to(device)
        print(f"Background data loaded: {background_data_tensor.shape}, Explain data loaded: {explain_data_tensor.shape}")
        if background_data_tensor.shape[1] != state_dim or explain_data_tensor.shape[1] != state_dim:
            print(f"Critical Warning: Data dimension mismatch with state_dim. "
                  f"Background features: {background_data_tensor.shape[1]}, "
                  f"Explain features: {explain_data_tensor.shape[1]}, Expected state_dim: {state_dim}. "
                  "Ensure data matches model input dimension.")
            return
    except Exception as e:
        print(f"Critical Error: Failed to load background/explanation data: {e}. Cannot proceed.")
        return

    try:
        manifest_df = pd.read_csv(manifest_path)
        if not all(col in manifest_df.columns for col in ['epoch', 'model_snapshot_path']):
            raise ValueError("Manifest CSV must contain 'epoch' and 'model_snapshot_path' columns.")
        print(f"Loaded SHAP model manifest from {manifest_path} with {len(manifest_df)} entries.")
    except Exception as e:
        print(f"Critical Error: Failed to load or parse manifest file {manifest_path}: {e}. Cannot proceed.")
        return

    header = ['epoch'] + feature_names
    if not os.path.exists(output_csv_path):
        try:
            output_dir = os.path.dirname(output_csv_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
            print(f"Initialized output CSV {output_csv_path} with header.")
        except Exception as e:
            print(f"Critical Error: Could not write header to {output_csv_path}: {e}. Cannot proceed.")
            return
    else:
        print(f"Output CSV {output_csv_path} already exists. Will append new data.")

    for index, row in manifest_df.iterrows():
        epoch = int(row['epoch'])
        model_path = row['model_snapshot_path']

        analyze_single_model_shap(
            model_snapshot_path=model_path,
            epoch_number=epoch,
            output_csv_path=output_csv_path,
            background_data_tensor=background_data_tensor,
            explain_data_tensor=explain_data_tensor,
            target_shap_output_index=target_shap_output_index,
            state_dim=state_dim,
            action_dim=action_dim,
            feature_names=feature_names,
            device=device
        )

    print("Completed SHAP analysis for all models in the manifest.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Perform SHAP analysis on DRL agent model snapshots listed in a manifest file."
    )
    parser.add_argument(
        "--manifest_path", type=str, required=True,
        help="Path to the manifest CSV file. Expected columns: 'epoch', 'model_snapshot_path'."
    )
    parser.add_argument(
        "--output_csv_path", type=str, required=True,
        help="Path to the output CSV file where SHAP importance evolution will be logged (e.g., shap_evolution.csv)."
    )
    parser.add_argument(
        "--background_data_path", type=str, required=True,
        help="Path to the background data tensor (.pt file) for SHAP explainer initialization. "
             "Shape: (n_samples, state_dim)."
    )
    parser.add_argument(
        "--explain_data_path", type=str, required=True,
        help="Path to the data tensor (.pt file) whose instances will be explained by SHAP. "
             "Shape: (n_instances_to_explain, state_dim)."
    )
    parser.add_argument(
        "--feature_names_path", type=str, required=True,
        help="Path to a text file containing a comma-separated list of feature names, "
             "matching the order and number of state features."
    )
    parser.add_argument(
        "--target_shap_output_index", type=int, required=True,
        help="Index of the actor model's output neuron to explain (e.g., for a specific asset's predicted return)."
    )
    parser.add_argument(
        "--state_dim", type=int, required=True,
        help="Dimensionality of the state space (number of input features to the model)."
    )
    parser.add_argument(
        "--action_dim", type=int, required=True,
        help="Dimensionality of the action space (number of output neurons from actor_mean)."
    )

    args = parser.parse_args()

    main_shap_analysis_from_manifest(
        manifest_path=args.manifest_path,
        output_csv_path=args.output_csv_path,
        background_data_path=args.background_data_path,
        explain_data_path=args.explain_data_path,
        target_shap_output_index=args.target_shap_output_index,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        feature_names_path=args.feature_names_path
    )
