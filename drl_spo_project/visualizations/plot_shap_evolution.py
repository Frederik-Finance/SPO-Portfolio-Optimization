"""
plot_shap_evolution.py

This script visualizes the evolution of SHAP (SHapley Additive exPlanations)
feature importances over training epochs. It reads data logged by
`periodic_shap_analyzer.py` (typically stored in `feature_analysis_data.csv`).

The script generates a suite of plots to understand feature importance dynamics:
1.  Overall Top N Features Bar Chart: Static view of average absolute importance.
2.  Individual Lines Chart: Trend of SHAP values for top N features.
3.  Stacked Area Chart: Relative absolute importance of top N features (plus "Others").
4.  Faceted Line Plots: SHAP evolution for each top N feature in its own subplot.
5.  Heatmap: SHAP values of top N features across epochs.
6.  Rank Evolution Plot: Changes in feature rank (based on absolute SHAP) over epochs.

These visualizations help in understanding how the DRL agent's reliance on
different input features changes as it learns.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np # For ceil in faceted plot and arange in rank plot

# Define fixed input and output directories and top_n_features
TRAINING_LOGS_DIR = "drl_spo_project/training_logs/"
SHAP_EVOLUTION_CSV = os.path.join(TRAINING_LOGS_DIR, "feature_analysis_data.csv")
SHAP_PLOTS_DIR = "drl_spo_project/visualizations/shap_over_time/"
TOP_N_FEATURES_CONFIG = 10 # Configured number of top features to focus on

# --- Helper function for consistent styling and saving ---
def _save_plot(fig, path, tight_layout_rect=None):
    """Saves the plot with tight layout and closes the figure."""
    if tight_layout_rect:
        fig.tight_layout(rect=tight_layout_rect)
    else:
        fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved plot to {path}")

# --- Plotting Function 1: Overall Top N Features (Bar Chart) ---
def plot_overall_top_n_barchart(mean_abs_shap_overall_top_n: pd.Series, output_dir: str, actual_top_n: int):
    """Plots a bar chart of overall mean absolute SHAP values for top N features."""
    if mean_abs_shap_overall_top_n.empty:
        print("No data for overall top N barchart. Skipping.")
        return

    fig = plt.figure(figsize=(10, max(6, actual_top_n * 0.6)))
    colors = sns.color_palette("viridis_r", n_colors=len(mean_abs_shap_overall_top_n)) # _r for reversed viridis
    mean_abs_shap_overall_top_n.sort_values(ascending=True).plot(kind='barh', color=colors, ax=fig.gca())
    plt.xlabel("Overall Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title(f"Top {actual_top_n} Features by Overall Mean Absolute SHAP Value")
    _save_plot(fig, os.path.join(output_dir, f"overall_top_{actual_top_n}_features_barchart.png"))

# --- Plotting Function 2: Line Chart of Top N Feature SHAP Values ---
def plot_individual_lines_top_n(df_plot: pd.DataFrame, top_feature_names: list, output_dir: str, actual_top_n: int, y_label: str):
    """Plots line chart for SHAP evolution of top N features on a single plot."""
    if df_plot.empty or not top_feature_names:
        print("No data or no top features for individual lines plot. Skipping.")
        return

    fig = plt.figure(figsize=(13, 7))
    palette = sns.color_palette("tab10", n_colors=max(10, len(top_feature_names)))

    for i, feature in enumerate(top_feature_names):
        if feature in df_plot.columns:
            plt.plot(df_plot.index, df_plot[feature], label=feature, marker='o', markersize=4, linestyle='-', color=palette[i % len(palette)])

    plt.xlabel("Epoch")
    plt.ylabel(y_label) # Use the provided y_label
    plt.title(f"Evolution of Top {actual_top_n} SHAP Feature Importances")
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize='small')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    _save_plot(fig, os.path.join(output_dir, f"top_{actual_top_n}_shap_evolution_lines.png"), tight_layout_rect=[0, 0, 0.82, 1])

# --- Plotting Function 3: Stacked Area Chart of Relative Feature Importance ---
def plot_stacked_area_relative_importance(df_abs_for_stacking: pd.DataFrame, feature_columns: list, top_feature_names: list,
                                          output_dir: str, actual_top_n_for_plot: int):
    """Plots stacked area chart of relative SHAP importance using absolute values."""
    if df_abs_for_stacking.empty or not feature_columns:
        print("No data or no features for stacked area plot. Skipping.")
        return

    # Ensure df_abs_for_stacking indeed contains absolute values for feature columns
    # This should be handled by the caller, but a check or re-abs can be added if necessary.
    # df_abs_for_stacking[feature_columns] = df_abs_for_stacking[feature_columns].abs()

    total_shap_per_epoch = df_abs_for_stacking[feature_columns].sum(axis=1)
    relative_df = df_abs_for_stacking[feature_columns].copy()
    for epoch_idx, total_shap_at_epoch in total_shap_per_epoch.items():
        if total_shap_at_epoch == 0:
            relative_df.loc[epoch_idx] = 0
        else:
            relative_df.loc[epoch_idx] = relative_df.loc[epoch_idx] / total_shap_at_epoch
    
    plot_df_relative = pd.DataFrame(index=relative_df.index)
    plot_columns_ordered = []

    # Use the provided top_feature_names (selected based on overall abs mean)
    if top_feature_names:
        plot_df_relative = relative_df[top_feature_names].copy()
        plot_columns_ordered.extend(top_feature_names)

    other_feature_names = [col for col in feature_columns if col not in top_feature_names]
    if other_feature_names:
        plot_df_relative['Others'] = relative_df[other_feature_names].sum(axis=1)
        plot_columns_ordered.append('Others')
    
    if plot_df_relative.empty or plot_df_relative.shape[1] == 0 : # Check if any columns to plot
        print("No data to plot for stacked area chart after processing. Skipping.")
        return

    fig = plt.figure(figsize=(13, 7))
    num_plot_items = len(plot_columns_ordered)
    
    base_palette = sns.color_palette("tab20", n_colors=max(20, num_plot_items)) # tab20 for more colors
    colors = base_palette[:num_plot_items]

    if 'Others' in plot_columns_ordered and num_plot_items > 0:
        num_main_features = num_plot_items -1
        colors = sns.color_palette("tab20", n_colors=max(20,num_main_features))[:num_main_features] + ['#bdbdbd'] # Grey for Others

    plt.stackplot(plot_df_relative.index,
                  [plot_df_relative[col] for col in plot_columns_ordered],
                  labels=plot_columns_ordered,
                  alpha=0.85,
                  colors=colors if len(colors) == num_plot_items else None)

    plt.xlabel("Epoch")
    plt.ylabel("Relative Mean Absolute SHAP Value")
    title_suffix = f"(Top {actual_top_n_for_plot} + Others)" if 'Others' in plot_columns_ordered and actual_top_n_for_plot > 0 else "(All Features)"
    if not top_feature_names and 'Others' in plot_columns_ordered and len(plot_columns_ordered) == 1:
        title_suffix = "(All Features as 'Others')"
    plt.title(f"Relative SHAP Feature Importance {title_suffix}")
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize='small')
    plt.ylim(0, 1)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    _save_plot(fig, os.path.join(output_dir, f"relative_shap_evolution_stacked_area.png"), tight_layout_rect=[0, 0, 0.82, 1])

# --- Plotting Function 4: Faceted Line Plots ---
def plot_faceted_lines_top_n(df_plot: pd.DataFrame, top_feature_names: list, output_dir: str, y_label_stem: str):
    """Plots SHAP evolution for each top N feature in a separate facet."""
    if df_plot.empty or not top_feature_names:
        print("No data or no top features for faceted plot. Skipping.")
        return

    actual_top_n = len(top_feature_names)
    # Dynamically determine number of columns, aiming for a reasonable aspect ratio
    if actual_top_n <= 4:
        ncols = actual_top_n
    elif actual_top_n <= 9:
        ncols = 3
    elif actual_top_n <= 16:
        ncols = 4
    else: # For more than 16, cap at 5 cols to prevent plots from becoming too wide
        ncols = 5
        
    nrows = (actual_top_n + ncols - 1) // ncols # Ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(min(5 * ncols, 20), 3.5 * nrows), sharex=True, squeeze=False) # Adjusted height, capped width
    axes_flat = axes.flatten()
    # Using a qualitative palette like 'tab10' or 'Set1' can be good for distinguishing lines if colors were per facet.
    # Since each line is alone in its facet, a consistent color or a simple sequence is fine.
    # Using a single color for all lines in facets, or a distinct color per feature if you prefer.
    # Let's use distinct colors from a good palette for each feature, even in facets for consistency if they appear elsewhere.
    palette = sns.color_palette("tab10", n_colors=max(10, actual_top_n))


    for i, feature in enumerate(top_feature_names):
        if feature in df_plot.columns:
            ax = axes_flat[i]
            ax.plot(df_plot.index, df_plot[feature], marker='o', markersize=3, linestyle='-', color=palette[i % len(palette)])
            ax.set_title(feature, fontsize=10, loc='left') # Title on left
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8) # Add horizontal line at y=0
            
            if i % ncols == 0: # Only add y-label to the first column of subplots
                ax.set_ylabel(y_label_stem, fontsize=9)
            
            # Add x-label only to the bottom-most row of subplots
            # Calculate if current subplot is in the last row
            is_last_row = (i // ncols == nrows - 1)
            # Or if it's in the second to last row but subsequent plots are hidden
            is_effectively_last_row = (i // ncols == nrows - 2 and (i + ncols) >= actual_top_n)

            if is_last_row or is_effectively_last_row:
                 ax.set_xlabel("Epoch", fontsize=9)
            
            ax.tick_params(axis='both', which='major', labelsize=8)


    for j in range(actual_top_n, nrows * ncols): # Hide unused subplots
        fig.delaxes(axes_flat[j])

    fig.suptitle(f"Faceted SHAP Evolution for Top {actual_top_n} Features", fontsize=14, y=0.99 if nrows > 1 else 1.03)
    # Ensure tight_layout is called before savefig by _save_plot
    _save_plot(fig, os.path.join(output_dir, f"faceted_top_{actual_top_n}_shap_evolution.png"))

# Called from the main suite function:
# plot_faceted_lines_top_n(df, top_feature_names, output_dir, y_label_stem=shap_value_y_label)
# --- Plotting Function 5: Heatmap of SHAP Evolution ---
def plot_heatmap_top_n(df_plot: pd.DataFrame, top_feature_names: list, output_dir: str, cbar_label: str):
    """Plots a heatmap of SHAP values for top N features over epochs."""
    if df_plot.empty or not top_feature_names:
        print("No data or no top features for heatmap. Skipping.")
        return
    
    heatmap_data = df_plot[top_feature_names].copy()
    num_epochs = len(df_plot.index)
    num_features = len(top_feature_names)
    
    fig_height = max(5, num_features * 0.4)
    fig_width = max(8, num_epochs * 0.25)
    fig_width = min(fig_width, 20)
    fig_height = min(fig_height, 18)

    fig = plt.figure(figsize=(fig_width, fig_height))
    annotate_heatmap = num_epochs <= 25 and num_features <= 15
    
    # Determine a symmetric color scale if data has positive and negative values
    vmin, vmax = None, None
    if not heatmap_data.empty:
        min_val = heatmap_data.min().min()
        max_val = heatmap_data.max().max()
        if min_val < 0 and max_val > 0:
            abs_max = max(abs(min_val), abs(max_val))
            vmin, vmax = -abs_max, abs_max
            cmap = "coolwarm" # Good diverging palette
        else:
            cmap = "viridis" # Good sequential palette
    else:
        cmap = "viridis"


    sns.heatmap(heatmap_data.T, annot=annotate_heatmap, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax,
                linewidths=.5, cbar_kws={'label': cbar_label}, ax=fig.gca())
    plt.yticks(rotation=0, fontsize=9)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Feature", fontsize=10)
    plt.title(f"Heatmap of Top {len(top_feature_names)} SHAP Feature Importances", fontsize=12)
    _save_plot(fig, os.path.join(output_dir, f"heatmap_top_{len(top_feature_names)}_shap_evolution.png"))

# --- Plotting Function 6: Rank Evolution Plot (Based on Absolute SHAP) ---
def plot_rank_evolution_top_n(df_input_for_ranking: pd.DataFrame, feature_columns: list, top_features_to_plot: list, output_dir: str, actual_top_n: int):
    """Plots the evolution of feature ranks (based on absolute SHAP) over time."""
    if df_input_for_ranking.empty or not feature_columns or not top_features_to_plot:
        print("No data or relevant features for rank evolution plot. Skipping.")
        return

    # Rank based on absolute SHAP values, descending (lower rank is more important)
    ranks_df = df_input_for_ranking[feature_columns].abs().rank(axis=1, method='min', ascending=False)
    ranks_to_plot_df = ranks_df[top_features_to_plot]

    fig = plt.figure(figsize=(13, 7))
    palette = sns.color_palette("tab10", n_colors=max(10, len(top_features_to_plot)))

    for i, feature in enumerate(top_features_to_plot):
        if feature in ranks_to_plot_df.columns:
            plt.plot(ranks_to_plot_df.index, ranks_to_plot_df[feature], label=feature, marker='.', markersize=6, linestyle='-', color=palette[i % len(palette)])

    plt.xlabel("Epoch")
    plt.ylabel("Feature Rank (by Abs. SHAP; Lower is More Important)")
    plt.title(f"Rank Evolution of Top {actual_top_n} Features (by Overall Abs. SHAP)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize='small')
    
    all_ranks_in_plot = ranks_to_plot_df.stack().dropna()
    if not all_ranks_in_plot.empty:
        min_rank = 1
        max_rank = all_ranks_in_plot.max() # Max rank value
        # Adjust y-ticks to be integers and not too dense
        step = max(1, int(np.ceil((max_rank - min_rank + 1) / 10.0)))
        plt.yticks(np.arange(min_rank, int(max_rank) + 1, step=step))
    
    plt.gca().invert_yaxis() 
    plt.grid(True, which="both", ls="--", alpha=0.5)
    _save_plot(fig, os.path.join(output_dir, f"rank_evolution_top_{actual_top_n}_features.png"), tight_layout_rect=[0, 0, 0.82, 1])

# --- Main Orchestration Function ---
def plot_shap_evolution_suite(csv_path: str, output_dir: str, top_n_config: int):
    """
    Loads SHAP evolution data and generates a comprehensive suite of visualizations.
    """
    print(f"Loading SHAP data from: {csv_path}")
    try:
        df_orig = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file at {csv_path} is empty.")
        return

    if df_orig.empty:
        print("DataFrame is empty. No plots will be generated.")
        return
    if 'epoch' not in df_orig.columns:
        print("Error: 'epoch' column not found in CSV. This column is required.")
        return

    df = df_orig.set_index('epoch')
    feature_columns = [col for col in df.columns if col != 'epoch']

    if not feature_columns:
        print("No feature columns found (excluding 'epoch'). No plots will be generated.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    sns.set_theme(style="whitegrid")

    # Determine top N features based on the mean of their *absolute* values
    mean_abs_shap_overall = df[feature_columns].abs().mean().sort_values(ascending=False)
    
    top_feature_names = []
    num_actually_plotted_top_features = 0

    if top_n_config > 0:
        num_actually_plotted_top_features = min(top_n_config, len(mean_abs_shap_overall))
        if num_actually_plotted_top_features > 0:
            top_feature_names = mean_abs_shap_overall.head(num_actually_plotted_top_features).index.tolist()
    
    if top_n_config > 0 and not top_feature_names:
        print(f"Configured to show top {top_n_config} features, but no features identified as 'top'.")
    else:
        print(f"Focusing on {num_actually_plotted_top_features} top features: {top_feature_names if top_feature_names else 'None (or all, depending on plot)'}")

    # Determine appropriate Y-axis label for plots showing raw SHAP values
    # Default to "Mean SHAP Value" as it's general. If your data source guarantees "mean absolute", you can change this.
    # Or, as your original script described, values *are* "mean absolute SHAP values".
    # If so, the input data from your image showing negative values would be inconsistent with that description.
    # For flexibility, we use "SHAP Value" or "Mean SHAP Value".
    # Let's assume the "mean" is calculated *before* this script.
    shap_value_y_label = "SHAP Value per Epoch"


    # --- Generate Plots ---

    # Plot 1: Overall Top N Features Bar Chart (Static, based on mean absolute SHAP)
    if num_actually_plotted_top_features > 0:
        plot_overall_top_n_barchart(mean_abs_shap_overall.head(num_actually_plotted_top_features),
                                    output_dir,
                                    num_actually_plotted_top_features)
    elif top_n_config > 0:
        print("Skipping overall top N barchart as no top features were selected/available.")

    if len(df.index) <= 1:
        print("Not enough epochs (<=1) to plot evolution. Only static plots (if any) were generated.")
        return

    # Plot 2: Individual Lines Chart (plots values from df directly)
    if top_feature_names:
        plot_individual_lines_top_n(df, top_feature_names, output_dir, num_actually_plotted_top_features, shap_value_y_label)
    elif top_n_config > 0:
        print("Skipping individual lines plot as no top features were identified.")

    # Plot 3: Stacked Area Chart (uses absolute values for relative importance)
    # Create a df with absolute values for stacking (if not already all positive)
    df_abs_for_stacking = df.copy()
    # No need to re-abs if all are positive, but this is safer for the logic:
    df_abs_for_stacking[feature_columns] = df_abs_for_stacking[feature_columns].abs() 
    plot_stacked_area_relative_importance(df_abs_for_stacking, feature_columns, top_feature_names, output_dir, num_actually_plotted_top_features)

    # Plot 4: Faceted Line Plots (plots values from df directly)
    if top_feature_names:
        plot_faceted_lines_top_n(df, top_feature_names, output_dir, y_label_stem=shap_value_y_label)
    elif top_n_config > 0:
        print("Skipping faceted lines plot as no top features were identified.")

    # Plot 5: Heatmap (plots values from df directly, cbar reflects this)
    if top_feature_names:
        plot_heatmap_top_n(df, top_feature_names, output_dir, cbar_label=shap_value_y_label)
    elif top_n_config > 0:
        print("Skipping heatmap plot as no top features were identified.")

    # Plot 6: Rank Evolution Plot (based on ranks of absolute SHAP values from df)
    if top_feature_names:
        plot_rank_evolution_top_n(df, feature_columns, top_feature_names, output_dir, num_actually_plotted_top_features)
    elif top_n_config > 0:
        print("Skipping rank evolution plot as no top features were identified.")


if __name__ == "__main__":
    if not os.path.exists(SHAP_PLOTS_DIR):
        os.makedirs(SHAP_PLOTS_DIR)
        print(f"Created output directory: {SHAP_PLOTS_DIR}")

    plot_shap_evolution_suite(SHAP_EVOLUTION_CSV, SHAP_PLOTS_DIR, TOP_N_FEATURES_CONFIG)
    print("SHAP evolution plotting suite finished.")