import torch
import numpy as np
import pandas as pd 
import os
import csv
import shutil

from src.data_loader import load_all_data
from src.drl_environment import PortfolioEnv
from src.drl_agent import PPOAgent
from src.spo_layer import DifferentiableMVO
from src.spo_loss import SPOPlusLoss

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yfinance as yf

csv_files_headers = {}
csv_loggers_paths = {}
log_output_dir = 'drl_spo_project/training_logs/'

def initialize_csv_loggers(env_instance, feature_names_list):
    global csv_files_headers, csv_loggers_paths, log_output_dir

    etf_names = env_instance.chosen_etf_prices.columns.tolist()

    csv_files_headers = {
        'shap_models_manifest': ['epoch', 'model_snapshot_path'],
        'portfolio_weights_evolution': ['epoch', 'eval_episode_num', 'timestep'] + [f'{etf}_weight' for etf in etf_names],
        'exploration_noise_evolution': ['epoch'] + [f'log_std_{etf}' for etf in etf_names],
        'feature_analysis_data': ['epoch'] + feature_names_list + \
                                 [f'pred_return_{etf}' for etf in etf_names],
        'ppo_metrics_distribution': ['epoch', 'update_batch_id', 'sample_in_batch_id',
                                     'advantage_value', 'policy_ratio_value', 'clipped_surrogate_objective_value'],
        'spo_loss_components_evolution': ['epoch', 'update_batch_id', 'spo_max_term_val_mean',
                                          'spo_term_2_r_hat_w_star_c_mean', 'spo_term_r_true_w_star_c_mean',
                                          'total_spo_plus_loss']
    }
    csv_loggers_paths = {name: os.path.join(log_output_dir, f"{name}.csv") for name in csv_files_headers}

    for name, header_list in csv_files_headers.items():
        if header_list:
            try:
                with open(csv_loggers_paths[name], 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header_list)
                print(f"Initialized CSV logger: {csv_loggers_paths[name]}")
            except Exception as e:
                print(f"Error initializing CSV logger for {name}: {e}")


def append_to_csv(log_name, data_row):
    global csv_loggers_paths
    if log_name not in csv_loggers_paths:
        print(f"Warning: Log name '{log_name}' not found in csv_loggers_paths.")
        return
    try:
        with open(csv_loggers_paths[log_name], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data_row)
    except Exception as e:
        print(f"Error appending to CSV {log_name}: {e}")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def calculate_performance_metrics(portfolio_values, risk_free_rate_annual=0.0):
    if isinstance(portfolio_values, np.ndarray):
        portfolio_values = pd.Series(portfolio_values)

    returns = portfolio_values.pct_change().dropna()

    if returns.empty:
        print("Warning: Returns series is empty in calculate_performance_metrics. Returning zeroed metrics.")
        return {
            "Total Return": 0, "Annualized Return": 0, "Annualized Volatility": 0,
            "Sharpe Ratio": 0, "Max Drawdown": 0, "Sortino Ratio": 0
        }

    N = 252

    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    num_years = len(portfolio_values) / N
    annualized_return = ( (1 + total_return) ** (1 / num_years) ) - 1 if num_years > 0 else 0


    annualized_volatility = returns.std() * np.sqrt(N)

    risk_free_rate_periodic = risk_free_rate_annual / N
    excess_returns = returns - risk_free_rate_periodic
    sharpe_ratio = (excess_returns.mean() * np.sqrt(N)) / returns.std() if returns.std() != 0 else 0

    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(N)
    sortino_ratio = (annualized_return - risk_free_rate_annual) / downside_deviation if downside_deviation != 0 else 0
    
    metrics = {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Sortino Ratio": sortino_ratio,
    }
    return metrics

def plot_performance(portfolio_values, title="Portfolio Performance", metrics=None, dates=None):
    if isinstance(portfolio_values, pd.Series) and isinstance(portfolio_values.index, pd.DatetimeIndex):
        pv_series = portfolio_values
    elif dates is not None and len(dates) == len(portfolio_values):
        pv_series = pd.Series(portfolio_values, index=pd.to_datetime(dates))
    else:
        print("Warning: Could not create DateTimeIndex. Using integer index for plotting.")
        pv_series = pd.Series(portfolio_values)

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    fig.suptitle(title, fontsize=16)

    returns = pv_series.pct_change().dropna()

    if returns.empty:
        print("Warning: Cannot generate detailed plots as returns series is empty.")
        if not pv_series.empty:
            axes[0].plot(pv_series.index, pv_series.values, label="Portfolio Value", color="blue")
            axes[0].set_ylabel("Portfolio Value")
            axes[0].set_title("Equity Curve")
            axes[0].grid(True)
            axes[0].legend()
        for i in range(1, 4):
            axes[i].set_visible(False)
        plt.tight_layout(rect=[0.1, 0, 1, 0.96])
        plot_filename = os.path.join(log_output_dir, "performance_summary_plot.png")
        plt.savefig(plot_filename)
        print(f"Performance plot (equity curve only) saved to {plot_filename}")
        plt.close(fig)
        return

    # Plot 0: Equity Curve
    axes[0].plot(pv_series.index, pv_series.values, label="Portfolio Value", color="blue")
    axes[0].set_ylabel("Portfolio Value")
    axes[0].set_title("Equity Curve")
    axes[0].grid(True)
    axes[0].legend()

    # Plot 1: Periodic Returns
    axes[1].bar(returns.index, returns.values, label="Periodic Returns", color="green", width=0.8)
    axes[1].set_ylabel("Returns")
    axes[1].set_title("Periodic Returns")
    axes[1].grid(True)
    axes[1].legend()

    # Plot 2: Drawdown Chart
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    axes[2].plot(drawdown.index, drawdown.values, label="Drawdown", color="red")
    axes[2].fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
    axes[2].set_ylabel("Drawdown")
    axes[2].set_title("Portfolio Drawdown")
    axes[2].grid(True)
    axes[2].legend()

    # Plot 3: Rolling Sharpe Ratio
    N_annualization = 252
    rolling_window = 63
    if len(returns) >= rolling_window:
        rolling_mean_returns = returns.rolling(window=rolling_window).mean()
        rolling_std_returns = returns.rolling(window=rolling_window).std()
        rolling_sharpe = (rolling_mean_returns / rolling_std_returns) * np.sqrt(N_annualization)
        rolling_sharpe.replace([np.inf, -np.inf], np.nan, inplace=True)

        axes[3].plot(rolling_sharpe.index, rolling_sharpe.values, label=f"{rolling_window}-Day Rolling Sharpe Ratio", color="purple")
        axes[3].set_ylabel("Rolling Sharpe Ratio")
        axes[3].set_title(f"{rolling_window}-Day Rolling Sharpe Ratio (Annualized)")
        axes[3].grid(True)
        axes[3].legend()
    else:
        axes[3].text(0.5, 0.5, f"Not enough data for {rolling_window}-day rolling Sharpe", horizontalalignment='center', verticalalignment='center', transform=axes[3].transAxes)
        axes[3].set_title(f"{rolling_window}-Day Rolling Sharpe Ratio (Annualized)")

    if isinstance(pv_series.index, pd.DatetimeIndex):
        for ax_idx in range(4):
             if axes[ax_idx].get_visible():
                axes[ax_idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

    if metrics:
        stats_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
        fig.text(0.02, 0.5, stats_text, transform=fig.transFigure,
                 verticalalignment='center', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.tight_layout(rect=[0.1, 0, 1, 0.96])
    plot_filename = os.path.join(log_output_dir, "performance_summary_plot.png")
    plt.savefig(plot_filename)
    print(f"Performance plot saved to {plot_filename}")
    plt.close(fig)



def main():
    global log_output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    TRAIN_TEST_CUTOFF_DATE = '2023-01-01'

    # --- Parameters ---
    data_path_prefix = 'data/'
    state_dim = None
    action_dim = None
    lr_actor = 0.0003
    lr_critic = 0.001
    gamma = 0.99
    K_epochs = 80
    eps_clip = 0.2
    action_std_init = 0.6
    
    spo_plus_loss_coeff = 1.0
    mvo_max_weight_per_asset = 0.25

    max_episodes = 80
    max_timesteps_per_episode = 40
    update_timestep_threshold = 40
    
    save_model_freq = 4
    eval_freq = 5

    # --- NEW: Covariance Lookback Parameters ---
    cov_calc_lookback_window = 252 * 1  # 1 year lookback
    min_cov_calculation_samples = 60

    # --- Directory Setup ---
    model_save_path_base = './models/'
    if not os.path.exists(model_save_path_base):
        os.makedirs(model_save_path_base)

    if os.path.exists(log_output_dir):
        shutil.rmtree(log_output_dir)
        print(f"Cleared old log directory: {log_output_dir}")
    os.makedirs(log_output_dir, exist_ok=True)

    shap_models_dir = os.path.join(log_output_dir, 'shap_model_snapshots/')
    os.makedirs(shap_models_dir, exist_ok=True)
    print(f"Logging directory setup at: {log_output_dir}")
    print(f"SHAP model snapshots will be saved in: {shap_models_dir}")

    # --- Training Phase ---
    print("--- Training Phase ---")
    print(f"Loading training data up to {TRAIN_TEST_CUTOFF_DATE}...")
    try:
        train_all_data = load_all_data(data_path_prefix=data_path_prefix, end_date=TRAIN_TEST_CUTOFF_DATE)
        if train_all_data['chosen_etf_prices'].empty or train_all_data['combined_features'].empty:
            print("Critical training dataframes are empty. Exiting.")
            return
        # chosen_etf_prices_df = train_all_data['chosen_etf_prices'] # This line might be unused if env handles prices internally
    except Exception as e:
        print(f"Failed to load training data: {e}")
        import traceback; traceback.print_exc()
        return
    print("Training data loaded successfully.")

    # --- Environment Initialization for Training ---
    print("Initializing DRL environment for training...")
    try:
        train_env = PortfolioEnv(processed_data=train_all_data)
        policy_feature_names = train_env.policy_feature_column_names # Keep this for loggers
        initialize_csv_loggers(train_env, policy_feature_names) # Use train_env
    except Exception as e:
        print(f"Error initializing training environment or CSV loggers: {e}")
        import traceback; traceback.print_exc()
        return

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    print(f"Training environment initialized. State dim: {state_dim}, Action dim (n_etfs): {action_dim}")

    # ----- The initial static covariance matrix calculation has been removed -----

    # --- Agent Initialization ---
    print("Initializing MVO solver, SPO+ loss module, and DRL agent...")
    mvo_solver = DifferentiableMVO(num_assets=action_dim, max_weight_per_asset=mvo_max_weight_per_asset).to(device)
    spo_loss_module = SPOPlusLoss(num_assets=action_dim, mvo_max_weight_per_asset=mvo_max_weight_per_asset).to(device)
    agent = PPOAgent(state_dim, action_dim,
                       mvo_solver_instance=mvo_solver,
                       spo_loss_instance=spo_loss_module,
                       lr_actor=lr_actor, lr_critic=lr_critic,
                       gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip,
                       action_std_init=action_std_init,
                       spo_plus_loss_coeff=spo_plus_loss_coeff)
    agent.policy.to(device)
    agent.policy_old.to(device)
    print("Agent initialized.")

    # --- Training Loop ---
    print("Starting training loop...")
    timestep_count = 0
    episode_rewards_history = []
    portfolio_value_history = []
    current_agent_update_count = 0

    for episode in range(1, max_episodes + 1):
        state = train_env.reset()
        
        episode_reward = 0
        for t in range(1, max_timesteps_per_episode + 1):
            timestep_count += 1
            
            # --- Get Rolling Covariance at each timestep ---
            current_cov_np = train_env.get_rolling_covariance(cov_calc_lookback_window, min_cov_calculation_samples)
            current_covariance_matrix_tensor = torch.tensor(current_cov_np, dtype=torch.float32).to(device)
            current_covariance_matrix_tensor += 1e-6 * torch.eye(train_env.n_etfs, device=device)
            
            # --- Agent takes an action ---
            action_portfolio_weights, predicted_returns_for_step = agent.select_action(state, current_covariance_matrix_tensor)
            next_state, reward, done, info = train_env.step(action_portfolio_weights)
            
            true_fwd_returns = info.get('true_forward_returns', np.zeros(action_dim, dtype=np.float32))
            if true_fwd_returns is None or not isinstance(true_fwd_returns, np.ndarray) or true_fwd_returns.shape[0] != action_dim:
                if not (true_fwd_returns is None and action_dim == 0):
                    print(f"Warning: true_forward_returns from info dict is invalid or shape mismatch. Using zeros. Received: {true_fwd_returns}")
                true_fwd_returns = np.zeros(action_dim, dtype=np.float32)

            agent.store_reward_terminal(reward, done, true_fwd_returns)
            state = next_state
            episode_reward += reward
            
            # --- Update Agent ---
            if timestep_count % update_timestep_threshold == 0 and len(agent.buffer['states']) > 0:
                update_metrics = agent.update(current_covariance_matrix_tensor) # Use the latest covariance matrix for the update
                if update_metrics:
                    current_agent_update_count += 1
                    spo_c = update_metrics['spo_components']
                    append_to_csv('spo_loss_components_evolution', [
                        episode, current_agent_update_count,
                        spo_c['spo_max_term_val_mean'], spo_c['spo_term_2_r_hat_w_star_c_mean'],
                        spo_c['spo_term_r_true_w_star_c_mean'], update_metrics['total_spo_loss']
                    ])
                    adv_batch = update_metrics['advantages_batch']
                    ratios_batch = update_metrics['ratios_batch']
                    obj_batch = update_metrics['policy_objective_batch']
                    for i in range(len(adv_batch)):
                        append_to_csv('ppo_metrics_distribution', [
                            episode, current_agent_update_count, i,
                            adv_batch[i], ratios_batch[i], obj_batch[i]
                        ])
            
            if done:
                break
        
        log_std_values = agent.policy.action_log_std.detach().cpu().numpy().flatten()
        append_to_csv('exploration_noise_evolution', [episode] + log_std_values.tolist())

        episode_rewards_history.append(episode_reward)
        portfolio_value_history.append(train_env.current_portfolio_value)
        print(f"Episode: {episode}, Timesteps: {t}, Reward: {episode_reward:.2f}, Portfolio Value: {train_env.current_portfolio_value:.2f}")

        # --- Periodic Evaluation during Training ---
        if episode % eval_freq == 0:
            print(f"--- Starting Evaluation for Training Epoch {episode} ---")
            eval_state = train_env.reset_for_eval()

            for eval_ep_num in range(1, 3):
                for t_eval in range(1, max_timesteps_per_episode + 1):
                    # --- Get Rolling Covariance for Evaluation Step ---
                    eval_cov_np = train_env.get_rolling_covariance(cov_calc_lookback_window, min_cov_calculation_samples)
                    eval_covariance_matrix_tensor = torch.tensor(eval_cov_np, dtype=torch.float32).to(device)
                    eval_covariance_matrix_tensor += 1e-6 * torch.eye(train_env.n_etfs, device=device)

                    eval_portfolio_weights, eval_predicted_returns = agent.select_action(eval_state, eval_covariance_matrix_tensor, is_eval=True)
                    append_to_csv('portfolio_weights_evolution',
                                  [episode, eval_ep_num, t_eval] + eval_portfolio_weights.tolist())
                    
                    next_eval_state, eval_reward, eval_done, eval_info = train_env.step(eval_portfolio_weights)

                    append_to_csv('feature_analysis_data',
                                  [episode] + eval_state.tolist() + \
                                  eval_predicted_returns.tolist())
                    eval_state = next_eval_state
                    if eval_done:
                        break
                print(f"Evaluation Episode {eval_ep_num} for Training Epoch {episode} ended. Final Value: {train_env.current_portfolio_value:.2f}")
            print(f"--- Finished Evaluation for Training Epoch {episode} ---")

        # --- Save Model Snapshot ---
        if episode % save_model_freq == 0:
            model_filename = os.path.join(model_save_path_base, f"ppo_portfolio_ep{episode}.pth")
            agent.save_model(model_filename)
            print(f"Saved main model at episode {episode} to {model_filename}")

            actual_shap_snapshot_path = os.path.join(shap_models_dir, f"actor_model_ep{episode}.pth")
            try:
                torch.save(agent.policy_old.actor_mean_layers.state_dict(), actual_shap_snapshot_path)
                append_to_csv('shap_models_manifest', [episode, actual_shap_snapshot_path])
                print(f"Saved SHAP model snapshot to {actual_shap_snapshot_path}")
            except Exception as e:
                print(f"Error saving SHAP model snapshot: {e}")

    print("Training finished.")
    final_model_path = os.path.join(model_save_path_base, "ppo_portfolio_final.pth")
    print(f"Saving final model to {final_model_path}")
    agent.save_model(final_model_path)

    if hasattr(train_env, 'close') and callable(train_env.close):
        train_env.close()

    print(f"Training visualizations can be generated from console output or by extending plotting scripts to use CSVs.")

    # --- Backtesting Phase ---
    print("\n--- Backtesting Phase ---")
    print(f"Loading backtesting data from {TRAIN_TEST_CUTOFF_DATE} onwards...")
    try:
        backtest_all_data = load_all_data(data_path_prefix=data_path_prefix, start_date=TRAIN_TEST_CUTOFF_DATE)
        if backtest_all_data['chosen_etf_prices'].empty or backtest_all_data['combined_features'].empty:
            print("Critical backtesting dataframes are empty. Exiting.")
            return
    except Exception as e:
        print(f"Failed to load backtesting data: {e}")
        import traceback; traceback.print_exc()
        return
    print("Backtesting data loaded successfully.")

    print("Initializing DRL environment for backtesting...")
    try:
        backtest_env = PortfolioEnv(processed_data=backtest_all_data)
    except Exception as e:
        print(f"Error initializing backtesting environment: {e}")
        import traceback; traceback.print_exc()
        return
    print(f"Backtesting environment initialized. State dim: {backtest_env.observation_space.shape[0]}, Action dim (n_etfs): {backtest_env.action_space.shape[0]}")

    print(f"Loading trained model from {final_model_path} for backtesting...")
    # Re-initialize agent or ensure it's clean if action_dim could change.
    # For now, assume the same agent structure with loaded weights.
    # state_dim and action_dim should be from the training phase for model compatibility.
    backtest_agent = PPOAgent(state_dim, action_dim, # Use state_dim, action_dim from training
                           mvo_solver_instance=mvo_solver, # Can reuse or re-init
                           spo_loss_instance=spo_loss_module, # Can reuse or re-init
                           lr_actor=lr_actor, lr_critic=lr_critic, # These LR are not used for eval
                           gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip,
                           action_std_init=action_std_init, # Not critical for eval
                           spo_plus_loss_coeff=spo_plus_loss_coeff)
    try:
        backtest_agent.load_model(final_model_path)
        backtest_agent.policy.eval()        # Set policy to evaluation mode
        backtest_agent.policy_old.eval()    # Set policy_old to evaluation mode
        print("Trained model loaded successfully.")
    except Exception as e:
        print(f"Error loading model for backtesting: {e}. Ensure the model path is correct and compatible.")
        return

    print("Fetching benchmark data for backtest period...")
    benchmark_series = None # Initialize to None
    if not backtest_env.dates.empty:
        benchmark_ticker = "SPY" # Example benchmark
        try:
            # Ensure dates are in a format yfinance accepts (e.g., YYYY-MM-DD strings)
            start_date_str = pd.to_datetime(backtest_env.dates[0]).strftime('%Y-%m-%d')
            end_date_str = pd.to_datetime(backtest_env.dates[-1]).strftime('%Y-%m-%d')

            spy_data = yf.download(benchmark_ticker, start=start_date_str, end=end_date_str, progress=False, auto_adjust=True) # Use auto_adjust for simplicity
            if not spy_data.empty and 'Close' in spy_data:
                # Ensure benchmark_series has a DatetimeIndex
                benchmark_series_temp = spy_data['Close'].pct_change().dropna()
                benchmark_series_temp.index = pd.to_datetime(benchmark_series_temp.index)

                benchmark_series = benchmark_series_temp
                print(f"Benchmark data ({benchmark_ticker}) loaded successfully for {len(benchmark_series)} periods.")
            else:
                print(f"Warning: Could not download or process benchmark data for {benchmark_ticker}. Proceeding without benchmark.")
        except Exception as e:
            print(f"Error downloading benchmark data for {benchmark_ticker}: {e}. Proceeding without benchmark.")
    else:
        print("Warning: Backtest dates are empty. Cannot fetch benchmark data.")

    print("Starting backtest loop...")
    backtest_portfolio_values = []
    backtest_dates = []

    state = backtest_env.reset_for_backtest()
    backtest_portfolio_values.append(backtest_env.current_portfolio_value)
    backtest_dates.append(pd.to_datetime(backtest_env.current_date))

    while True:
        # Get Rolling Covariance for Backtest Step
        current_cov_np_backtest = backtest_env.get_rolling_covariance(cov_calc_lookback_window, min_cov_calculation_samples)
        current_covariance_matrix_tensor_backtest = torch.tensor(current_cov_np_backtest, dtype=torch.float32).to(device)
        current_covariance_matrix_tensor_backtest += 1e-6 * torch.eye(backtest_env.n_etfs, device=device) # Use backtest_env.n_etfs

        # Agent selects action deterministically
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            # Using policy_old as it's typically the stable model used for updates/final eval
            if hasattr(backtest_agent.policy_old, '_get_positive_action_mean'):
                positive_mean_action_tensor = backtest_agent.policy_old._get_positive_action_mean(state_tensor)
            elif hasattr(backtest_agent.policy_old, 'actor_mean_layers'):
                 positive_mean_action_tensor = backtest_agent.policy_old.actor_mean_layers(state_tensor)
                 positive_mean_action_tensor = torch.sigmoid(positive_mean_action_tensor)
            else:
                action_mean, _ = backtest_agent.policy_old.actor_mean_layers(state_tensor), None
                positive_mean_action_tensor = torch.sigmoid(action_mean)

        if not hasattr(backtest_agent, 'mvo_solver') or backtest_agent.mvo_solver is None:
             backtest_agent.mvo_solver = DifferentiableMVO(num_assets=backtest_env.n_etfs, max_weight_per_asset=mvo_max_weight_per_asset).to(device)

        backtest_action_weights_tensor = backtest_agent.mvo_solver(positive_mean_action_tensor, current_covariance_matrix_tensor_backtest)
        backtest_action_weights_np = backtest_action_weights_tensor.cpu().numpy().flatten()
        
        next_state, reward, done, info = backtest_env.step(backtest_action_weights_np)

        backtest_portfolio_values.append(backtest_env.current_portfolio_value)
        current_date_info = info.get('current_date', backtest_env.current_date)
        backtest_dates.append(pd.to_datetime(current_date_info))
        
        state = next_state
        if done:
            print("Backtest loop finished.")
            break

    if hasattr(backtest_env, 'close') and callable(backtest_env.close):
        backtest_env.close()

    # --- Backtest Performance Analysis ---
    print("\n--- Backtest Performance Analysis ---")
    if not backtest_dates or len(backtest_dates) != len(backtest_portfolio_values):
        print("Warning: Mismatch between backtest dates and portfolio values. Plotting with default index.")
        pv_series_backtest = pd.Series(backtest_portfolio_values)
    else:
        pv_series_backtest = pd.Series(backtest_portfolio_values, index=pd.DatetimeIndex(backtest_dates))
        if pv_series_backtest.index.has_duplicates:
             print("Warning: Duplicate dates found in backtest. Aggregating portfolio values (last).")
             pv_series_backtest = pv_series_backtest.groupby(pv_series_backtest.index).last()

    backtest_performance_metrics = calculate_performance_metrics(pv_series_backtest)
    print("\n--- Backtest Performance Metrics ---")
    if backtest_performance_metrics:
        for k, v in backtest_performance_metrics.items():
            if isinstance(v, (int, float)):
                print(f"{k}: {v:.4f}")
    else:
        print("No metrics calculated, possibly due to empty returns.")

    plot_performance(pv_series_backtest, title="Backtest Performance (Out-of-Sample)",
                     metrics=backtest_performance_metrics,
                     dates=pv_series_backtest.index.tolist())

    print("\n--- Generating Pyfolio Report ---")
    if not isinstance(pv_series_backtest.index, pd.DatetimeIndex):
        pv_series_backtest.index = pd.to_datetime(pv_series_backtest.index)

    if pv_series_backtest.index.tz is not None:
        pv_series_backtest.index = pv_series_backtest.index.tz_localize(None)

    backtest_returns_for_qs = pv_series_backtest.pct_change().dropna()

    skip_pyfolio_report = False
    if backtest_returns_for_qs.empty:
        print("Warning: Initial backtest returns are empty. Skipping Pyfolio report.")
        skip_pyfolio_report = True
    else:
        if benchmark_series is not None and not benchmark_series.empty:
            if not isinstance(benchmark_series.index, pd.DatetimeIndex): # Should be, but double check
                benchmark_series.index = pd.to_datetime(benchmark_series.index)
            if benchmark_series.index.tz is not None: # Should be, but double check
                benchmark_series.index = benchmark_series.index.tz_localize(None)

            common_index = backtest_returns_for_qs.index.intersection(benchmark_series.index)

            if common_index.empty:
                print("Warning: No common dates between backtest returns and benchmark returns. Proceeding without benchmark for Pyfolio.")
                benchmark_series = None
            else:
                backtest_returns_for_qs = backtest_returns_for_qs.loc[common_index]
                benchmark_series = benchmark_series.loc[common_index]

                backtest_returns_for_qs.dropna(inplace=True)
                if benchmark_series is not None:
                    benchmark_series.dropna(inplace=True)

                if benchmark_series is not None and not benchmark_series.empty and not backtest_returns_for_qs.empty:
                    final_common_index = backtest_returns_for_qs.index.intersection(benchmark_series.index)

                    if final_common_index.empty:
                        print("Warning: Returns or benchmark became empty after NaN removal post-alignment. Proceeding without benchmark or skipping report.")
                        benchmark_series = None
                        if backtest_returns_for_qs.loc[final_common_index].empty:
                             backtest_returns_for_qs = pd.Series([], dtype='float64')
                    else:
                        backtest_returns_for_qs = backtest_returns_for_qs.loc[final_common_index]
                        benchmark_series = benchmark_series.loc[final_common_index]
                elif backtest_returns_for_qs.empty:
                     print("Warning: Backtest returns became empty after NaN removal post-alignment.")
                elif benchmark_series is not None and benchmark_series.empty:
                     print("Warning: Benchmark series became empty after NaN removal. Proceeding without benchmark.")
                     benchmark_series = None

        if backtest_returns_for_qs.empty:
            if not skip_pyfolio_report:
                 print("Skipping Pyfolio report as backtest returns are empty after all processing.")
            skip_pyfolio_report = True

    if not skip_pyfolio_report:
        pyfolio_report_output_path = os.path.join(log_output_dir, 'Pyfolio_Tear_Sheet.pdf')
        try:
            print(f"Generating Pyfolio tear sheet. Returns length: {len(backtest_returns_for_qs)}")
            if benchmark_series is not None and not benchmark_series.empty:
                print(f"Benchmark length: {len(benchmark_series)}")

            returns_for_pyfolio = backtest_returns_for_qs.copy()
            if not isinstance(returns_for_pyfolio.index, pd.DatetimeIndex):
                raise ValueError("Returns index must be DatetimeIndex for Pyfolio.")
            returns_for_pyfolio.index.name = 'date'

            benchmark_rets_for_pyfolio = None
            if benchmark_series is not None and not benchmark_series.empty:
                benchmark_rets_for_pyfolio = benchmark_series.copy()
                if not isinstance(benchmark_rets_for_pyfolio.index, pd.DatetimeIndex):
                     raise ValueError("Benchmark returns index must be DatetimeIndex for Pyfolio.")
                benchmark_rets_for_pyfolio.index.name = 'date'

            original_backend = matplotlib.get_backend()
            matplotlib.use('Agg') 

            # with PdfPages(pyfolio_report_output_path) as pdf:
            #     plt.close('all')
            #     pf.create_full_tear_sheet(
            #         returns_for_pyfolio,
            #         benchmark_rets=benchmark_rets_for_pyfolio,
            #         round_trips=False 
            #     )
            #     for i in plt.get_fignums():
            #         pdf.savefig(plt.figure(i))
            #     plt.close('all')

            matplotlib.use(original_backend)
            print(f"Pyfolio tear sheet saved to {pyfolio_report_output_path}")

        except Exception as e:
            print(f"Error generating Pyfolio report: {e}")
            import traceback
            traceback.print_exc()
            if matplotlib.get_backend().lower() != original_backend.lower(): # Check if backend changed before restoring
                matplotlib.use(original_backend)
    else:
        print("Pyfolio report generation skipped due to empty or problematic returns data.")

    # --- Training Curves Plot (remains) ---
    fig_train, ax_train = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_train[0].plot(episode_rewards_history, label="Episode Reward")
    ax_train[0].set_title("Training Episode Rewards")
    ax_train[0].set_ylabel("Reward")
    ax_train[0].legend()
    ax_train[1].plot(portfolio_value_history, label="End of Episode Portfolio Value")
    ax_train[1].set_title("Training Portfolio Value at Episode End")
    ax_train[1].set_ylabel("Portfolio Value")
    ax_train[1].set_xlabel("Episode")
    ax_train[1].legend()
    plt.tight_layout()
    training_curve_path = os.path.join(log_output_dir, "training_curves.png")
    plt.savefig(training_curve_path)
    print(f"Training curves saved to {training_curve_path}")
    plt.close(fig_train)

    print(f"Training visualizations and performance metrics generated.")

if __name__ == '__main__':
    main()


