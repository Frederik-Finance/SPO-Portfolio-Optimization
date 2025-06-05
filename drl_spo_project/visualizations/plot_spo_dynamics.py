"""
plot_spo_dynamics.py

This script generates visualizations related to the SPO (Smart Predict-then-Optimize)
process within the DRL agent's learning. It uses data logged to CSV files,
which would typically capture aspects of the SPO+ Loss calculation and the
resulting portfolio weights.

The visualizations include:
1.  Comparison of portfolio weights derived from predicted returns (w*(ĉ)) versus
    those from true returns (w*(c)) for specific scenarios.
2.  Evolution of the individual components of the SPO+ Loss function over
    training epochs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define fixed input and output directories as per user's request
TRAINING_LOGS_DIR = "drl_spo_project/training_logs/"
SPO_DYNAMICS_PLOTS_DIR = "drl_spo_project/visualizations/spo_dynamics/"

def plot_w_star_comparison(csv_path: str, output_path_dir: str):
    """
    Loads data comparing portfolio weights w*(ĉ) (from predicted returns) and
    w*(c) (from true returns) and generates grouped bar charts for each
    logged scenario.

    Args:
        csv_path (str): Path to the input CSV file.
                        Expected columns: 'scenario_id', 'etf_name',
                        'w_star_c_hat_weight' (for w*(ĉ)),
                        'w_star_c_weight' (for w*(c)).
        output_path_dir (str): Directory to save the generated plots.
    """
    print(f"Loading w_star comparison data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: w_star comparison CSV file not found at {csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: w_star comparison CSV file at {csv_path} is empty.")
        return

    if df.empty:
        print("w_star comparison DataFrame is empty. No plots will be generated.")
        return

    # Validate required columns
    required_cols = ['scenario_id', 'etf_name', 'w_star_c_hat_weight', 'w_star_c_weight']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain columns: {', '.join(required_cols)}. Found: {df.columns.tolist()}")
        return

    # Iterate through each unique scenario_id and create a separate plot
    for scenario_id in df['scenario_id'].unique():
        scenario_df = df[df['scenario_id'] == scenario_id]

        if scenario_df.empty:
            print(f"No data for scenario_id {scenario_id}. Skipping this scenario.")
            continue

        # Melt the DataFrame to have 'weight_type' (w*(ĉ) or w*(c)) and 'weight' columns
        # This format is suitable for seaborn's grouped barplot.
        melted_df = scenario_df.melt(id_vars=['scenario_id', 'etf_name'],
                                     value_vars=['w_star_c_hat_weight', 'w_star_c_weight'],
                                     var_name='weight_type', value_name='weight')

        # Map original column names to more readable legend labels
        legend_labels = {
            'w_star_c_hat_weight': 'w*(ĉ) (Predicted)',
            'w_star_c_weight': 'w*(c) (True)'
        }
        melted_df['weight_type'] = melted_df['weight_type'].map(legend_labels)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='etf_name', y='weight', hue='weight_type', data=melted_df, palette="viridis")

        plt.xlabel("ETF Name")
        plt.ylabel("Portfolio Weight")
        plt.title(f"Comparison of Portfolio Weights: w*(ĉ) vs. w*(c) for Scenario {scenario_id}")
        plt.legend(title="Weight Type")
        # Adjust y-axis to start at 0 and accommodate the max weight, ensuring some padding
        plt.ylim(0, max(1.0, melted_df['weight'].max() * 1.1))
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the plot for the current scenario
        output_filename = f"w_star_comparison_scenario_{scenario_id}.png"
        final_output_path = os.path.join(output_path_dir, output_filename)
        plt.savefig(final_output_path)
        plt.close() # Close figure to free memory
        print(f"Saved w_star comparison plot for scenario {scenario_id} to {final_output_path}")

def plot_spo_loss_components_evolution(csv_path: str, output_path_dir: str):
    """
    Loads data on the evolution of SPO+ loss components over training epochs
    and generates a line chart.

    Args:
        csv_path (str): Path to the input CSV file.
                        Expected columns: 'epoch',
                        'spo_max_term_val_mean',
                        'spo_term_2_r_hat_w_star_c_mean',
                        'spo_term_r_true_w_star_c_mean',
                        'total_spo_plus_loss'.
        output_path_dir (str): Directory to save the generated plot.
    """
    print(f"Loading SPO+ loss components data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: SPO+ loss components CSV file not found at {csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: SPO+ loss components CSV file at {csv_path} is empty.")
        return

    if df.empty:
        print("SPO+ loss components DataFrame is empty. No plot will be generated.")
        return

    # Validate required columns - corrected to match CSV headers from main.py
    required_cols = ['epoch', 'spo_max_term_val_mean', 'spo_term_2_r_hat_w_star_c_mean',
                     'spo_term_r_true_w_star_c_mean', 'total_spo_plus_loss']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain columns: {', '.join(required_cols)}. Found columns: {df.columns.tolist()}")
        return

    # Use 'update_batch_id' as the x-axis for more granular tracking, as 'epoch' might have multiple updates
    # If 'update_batch_id' is not unique per epoch, we might need to create a combined index or group by epoch.
    # For now, assuming 'update_batch_id' is a continuous sequence of updates.
    # If 'epoch' is preferred, ensure it's unique or handle duplicates (e.g., take mean per epoch).
    # Given the CSV snippet, 'update_batch_id' seems to be the more granular x-axis.
    df['x_axis_id'] = df['update_batch_id'] # Use update_batch_id for x-axis

    # Define columns for the main components and the total loss
    main_components = ['spo_max_term_val_mean', 'spo_term_2_r_hat_w_star_c_mean', 'spo_term_r_true_w_star_c_mean']
    total_loss_column = 'total_spo_plus_loss'

    # Create a figure with two subplots, sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Evolution of SPO+ Loss Components Over Training", fontsize=16)

    # Plot main components on the first subplot (ax1)
    for col in main_components:
        if col in df.columns:
            ax1.plot(df['x_axis_id'], df[col], label=col, marker='o', linestyle='-', markersize=4)
        else:
            print(f"Warning: Column '{col}' not found in SPO loss data. Skipping this line on ax1.")

    ax1.set_ylabel("Loss Component Value")
    ax1.set_title("Individual SPO+ Loss Components")
    ax1.legend(loc='best')
    ax1.grid(True, which="both", ls="--", alpha=0.7)

    # Plot total SPO+ loss on the second subplot (ax2)
    if total_loss_column in df.columns:
        ax2.plot(df['x_axis_id'], df[total_loss_column], label=total_loss_column, color='red', marker='x', linestyle='-', markersize=4)
    else:
        print(f"Warning: Column '{total_loss_column}' not found in SPO loss data. Skipping this line on ax2.")

    ax2.set_xlabel("Update Batch ID")
    ax2.set_ylabel("Total SPO+ Loss")
    ax2.set_title("Total SPO+ Loss")
    ax2.legend(loc='best')
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Use scientific notation for small numbers

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap

    # Save the plot
    final_output_path = os.path.join(output_path_dir, "spo_loss_components_evolution.png")
    plt.savefig(final_output_path)
    plt.close() # Close figure
    print(f"Saved SPO+ loss components evolution plot to {final_output_path}")

if __name__ == "__main__":
    # Ensure output directory exists
    if not os.path.exists(SPO_DYNAMICS_PLOTS_DIR):
        os.makedirs(SPO_DYNAMICS_PLOTS_DIR)
        print(f"Created output directory: {SPO_DYNAMICS_PLOTS_DIR}")

    # Construct full paths to CSV files
    w_star_csv_path = os.path.join(TRAINING_LOGS_DIR, "w_star_comparison_data.csv")
    spo_loss_csv_path = os.path.join(TRAINING_LOGS_DIR, "spo_loss_components_evolution.csv")

    # Call plotting functions
    plot_w_star_comparison(w_star_csv_path, SPO_DYNAMICS_PLOTS_DIR)
    plot_spo_loss_components_evolution(spo_loss_csv_path, SPO_DYNAMICS_PLOTS_DIR)

    print("SPO dynamics plotting script finished.")
