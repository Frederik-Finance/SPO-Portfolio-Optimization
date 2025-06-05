"""
plot_training_dynamics.py

This script generates various visualizations related to the DRL agent's
training process and internal mechanics. It reads data from CSV files that
are assumed to be logged during training or evaluation runs.

The visualizations include:
1.  Portfolio weight dynamics over timesteps for a specific evaluation episode.
2.  Evolution of the agent's exploration noise (action_log_std or actual std dev) over epochs.
3.  A heatmap of correlations between input features, predicted returns, and actual rewards.
4.  Distributions of key PPO algorithm metrics (e.g., advantages, policy ratios).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np # For exp in noise plot

# Define fixed input and output directories as per user's request
TRAINING_LOGS_DIR = "drl_spo_project/training_logs/"
TRAINING_DYNAMICS_PLOTS_DIR = "drl_spo_project/visualizations/training_dynamics/"

def plot_portfolio_weight_dynamics(csv_path: str, output_path_dir: str):
    """
    Loads portfolio weight evolution data from a CSV and generates a stacked
    area chart showing how ETF weights change over timesteps within the
    last evaluation episode of the last epoch.

    Args:
        csv_path (str): Path to the input CSV file.
                        Expected columns: 'epoch', 'eval_episode_num', 'timestep',
                        and columns ending with '_weight' (e.g., 'ETF_A_weight').
        output_path_dir (str): Directory to save the generated plot.
    """
    print(f"Loading portfolio weight data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Portfolio weights CSV file not found at {csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Portfolio weights CSV file at {csv_path} is empty.")
        return

    if df.empty:
        print("Portfolio weights DataFrame is empty. No plot will be generated.")
        return

    required_cols = ['epoch', 'eval_episode_num', 'timestep']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain columns: {', '.join(required_cols)}. Found: {df.columns.tolist()}")
        return

    weight_cols = [col for col in df.columns if col.endswith('_weight')]
    if not weight_cols:
        print("Error: No weight columns (e.g., 'ETF_A_weight') found in CSV.")
        return

    # Select the last epoch and last episode for plotting
    specific_epoch = df['epoch'].max()
    epoch_df = df[df['epoch'] == specific_epoch]
    specific_episode = epoch_df['eval_episode_num'].max()

    plot_df = epoch_df[epoch_df['eval_episode_num'] == specific_episode]

    if plot_df.empty:
        print(f"No data found for Epoch {specific_epoch}, Episode {specific_episode}. Skipping plot.")
        return

    plot_df = plot_df.set_index('timestep')

    plt.figure(figsize=(12, 7))
    plt.stackplot(plot_df.index,
                  [plot_df[col] for col in weight_cols],
                  labels=weight_cols,
                  alpha=0.8)

    plt.xlabel("Timestep")
    plt.ylabel("Portfolio Weight")
    plt.title(f"Portfolio Weight Dynamics (Epoch {specific_epoch}, Episode {specific_episode})")
    plt.legend(loc='upper right')
    plt.ylim(0, max(1.0, plot_df[weight_cols].sum(axis=1).max() * 1.05))
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    output_filename = f"portfolio_weights_dynamics_e{specific_epoch}_ep{specific_episode}.png"
    final_output_path = os.path.join(output_path_dir, output_filename)
    plt.savefig(final_output_path)
    plt.close()
    print(f"Saved portfolio weight dynamics plot to {final_output_path}")


def plot_exploration_noise_evolution(csv_path: str, output_path_dir: str):
    """
    Loads exploration noise data (action_log_std) from a CSV and generates a
    line chart showing its evolution over epochs, plotting the actual standard deviation.

    Args:
        csv_path (str): Path to the input CSV file.
                        Expected columns: 'epoch', and columns starting with
                        'log_std_' (e.g., 'log_std_etf_A').
        output_path_dir (str): Directory to save the generated plot.
    """
    print(f"Loading exploration noise data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Exploration noise CSV file not found at {csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Exploration noise CSV file at {csv_path} is empty.")
        return

    if df.empty:
        print("Exploration noise DataFrame is empty. No plot will be generated.")
        return

    if 'epoch' not in df.columns:
        print("Error: 'epoch' column not found in CSV for noise data.")
        return

    df = df.set_index('epoch')
    log_std_cols = [col for col in df.columns if col.startswith('log_std_')]

    if not log_std_cols:
        print("Error: No log_std columns (e.g., 'log_std_etf_A') found in CSV.")
        return

    plt.figure(figsize=(12, 7))
    for col in log_std_cols:
        data_to_plot = np.exp(df[col]) # Always plot actual standard deviation
        label = col.replace('log_std_', 'std_')
        plt.plot(df.index, data_to_plot, label=label, marker='o', linestyle='-')

    plt.xlabel("Epoch")
    plt.ylabel("Action Standard Deviation")
    plt.title("Evolution of Agent's Exploration Noise (Standard Deviation)")
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    final_output_path = os.path.join(output_path_dir, "exploration_noise_std_dev_evolution.png")
    plt.savefig(final_output_path)
    plt.close()
    print(f"Saved exploration noise evolution plot to {final_output_path}")

def plot_feature_correlation_heatmap(csv_path: str, output_path_dir: str):
    """
    Loads feature analysis data from a CSV and generates a heatmap of the
    Pearson correlation matrix between the features, predicted returns, and actual reward.

    Args:
        csv_path (str): Path to the input CSV file.
                        Expected columns: feature columns (e.g., 'feature_1'),
                        predicted return columns (e.g., 'predicted_return_etf_A'),
                        and 'actual_reward'.
        output_path_dir (str): Directory to save the generated plot.
    """
    print(f"Loading feature analysis data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Feature analysis CSV file not found at {csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Feature analysis CSV file at {csv_path} is empty.")
        return

    if df.empty or df.shape[0] < 2:
        print("Feature analysis DataFrame is empty or has insufficient data for correlation. No plot.")
        return

    df_corr = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Feature, Prediction, and Reward Correlation Heatmap")
    plt.tight_layout()

    final_output_path = os.path.join(output_path_dir, "feature_correlation_heatmap.png")
    plt.savefig(final_output_path)
    plt.close()
    print(f"Saved feature correlation heatmap to {final_output_path}")

def plot_ppo_metrics_distributions(csv_path: str, output_path_dir: str):
    """
    Loads PPO algorithm metrics data from a CSV and generates histograms
    to visualize their distributions.

    Args:
        csv_path (str): Path to the input CSV file.
                        Expected columns: 'epoch', 'advantage_value',
                        'policy_ratio_value', 'clipped_surrogate_objective_value'.
        output_path_dir (str): Directory to save the generated plot.
    """
    print(f"Loading PPO metrics distribution data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: PPO metrics CSV file not found at {csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: PPO metrics CSV file at {csv_path} is empty.")
        return

    if df.empty:
        print("PPO metrics DataFrame is empty. No plots will be generated.")
        return

    metric_cols = ['advantage_value', 'policy_ratio_value', 'clipped_surrogate_objective_value']

    available_metric_cols = [col for col in metric_cols if col in df.columns]
    if len(available_metric_cols) != len(metric_cols):
        missing_cols = set(metric_cols) - set(available_metric_cols)
        print(f"Warning: Missing PPO metric columns in CSV: {missing_cols}. Plotting available ones.")

    if not available_metric_cols:
        print("No PPO metric columns found to plot.")
        return

    num_metrics = len(available_metric_cols)

    if num_metrics <= 3:
        n_rows, n_cols = 1, num_metrics
    else:
        n_rows, n_cols = 2, int(np.ceil(num_metrics / 2.0))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    for i, metric in enumerate(available_metric_cols):
        ax = axes[i]
        sns.histplot(df[metric].dropna(), kde=True, ax=ax, bins=30)
        ax.set_title(f"Distribution of {metric.replace('_', ' ').title()}")
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle='--', alpha=0.7)

    for j in range(num_metrics, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    final_output_path = os.path.join(output_path_dir, "ppo_metrics_distributions.png")
    plt.savefig(final_output_path)
    plt.close()
    print(f"Saved PPO metrics distributions plot to {final_output_path}")


if __name__ == "__main__":
    # Ensure output directory exists
    if not os.path.exists(TRAINING_DYNAMICS_PLOTS_DIR):
        os.makedirs(TRAINING_DYNAMICS_PLOTS_DIR)
        print(f"Created output directory: {TRAINING_DYNAMICS_PLOTS_DIR}")

    # Construct full paths to CSV files
    weights_csv_path = os.path.join(TRAINING_LOGS_DIR, "portfolio_weights_evolution.csv")
    noise_csv_path = os.path.join(TRAINING_LOGS_DIR, "exploration_noise_evolution.csv")
    feature_analysis_csv_path = os.path.join(TRAINING_LOGS_DIR, "feature_analysis_data.csv")
    ppo_metrics_csv_path = os.path.join(TRAINING_LOGS_DIR, "ppo_metrics_distribution.csv")

    # Call all plotting functions
    plot_portfolio_weight_dynamics(weights_csv_path, TRAINING_DYNAMICS_PLOTS_DIR)
    plot_exploration_noise_evolution(noise_csv_path, TRAINING_DYNAMICS_PLOTS_DIR) # plot_actual_std is now hardcoded to True inside the function
    plot_feature_correlation_heatmap(feature_analysis_csv_path, TRAINING_DYNAMICS_PLOTS_DIR)
    plot_ppo_metrics_distributions(ppo_metrics_csv_path, TRAINING_DYNAMICS_PLOTS_DIR)

    print("Training dynamics plotting script finished.")
