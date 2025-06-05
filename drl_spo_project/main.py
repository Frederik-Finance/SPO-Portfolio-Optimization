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
        return {
            "Total Return": 0, "Annualized Return": 0, "Annualized Volatility": 0,
            "Sharpe Ratio": 0, "Max Drawdown": 0, "Sortino Ratio": 0,
            "Cumulative Returns Plot": None, "Daily Returns Plot": None
        }

    N = 252

    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    num_years = len(portfolio_values) / N
    annualized_return = ( (1 + total_return) ** (1 / num_years) ) - 1 if num_years > 0 else 0


    annualized_volatility = returns.std() * np.sqrt(N)

    risk_free_rate_periodic = (1 + risk_free_rate_annual)**(1/N) - 1
    excess_returns = returns - risk_free_rate_periodic
    sharpe_ratio = (excess_returns.mean() * N) / (returns.std() * np.sqrt(N)) if (returns.std() * np.sqrt(N)) != 0 else 0


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
    if isinstance(portfolio_values, np.ndarray):
        if dates is None or len(dates) != len(portfolio_values):
            print("Warning: Dates not provided or length mismatch for numpy array. Using integer index for plotting.")
            index_for_plot = np.arange(len(portfolio_values))
        else:
            index_for_plot = dates
        pv_series = pd.Series(portfolio_values, index=index_for_plot)
    else:
        pv_series = portfolio_values

    returns = pv_series.pct_change().dropna()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=16)

    axes[0].plot(pv_series.index, pv_series.values, label="Portfolio Value", color="blue")
    axes[0].set_ylabel("Portfolio Value")
    axes[0].set_title("Equity Curve")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].bar(returns.index, returns.values, label="Periodic Returns", color="green", width=0.8)
    axes[1].set_ylabel("Returns")
    axes[1].set_title("Periodic Returns")
    axes[1].grid(True)
    axes[1].legend()

    if isinstance(pv_series.index, pd.DatetimeIndex):
        fig.autofmt_xdate()
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))


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

    max_episodes = 100
    max_timesteps_per_episode = 50
    update_timestep_threshold = 40
    
    save_model_freq = 4
    eval_freq = 5    

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


    cov_calc_lookback_window = 252 * 1
    min_cov_calculation_samples = 60

    print("Loading data...")
    try:
        all_data = load_all_data(data_path_prefix=data_path_prefix)
        if all_data['chosen_etf_prices'].empty or all_data['combined_features'].empty:
             print("Critical dataframes (chosen_etf_prices or combined_features) are empty. Exiting.")
             return 
        chosen_etf_prices_df = all_data['chosen_etf_prices']
    except Exception as e:
        print(f"Failed to load data: {e}")
        import traceback; traceback.print_exc()
        return
    print("Data loaded successfully.")

    print("Initializing DRL environment...")
    try:
        env = PortfolioEnv(processed_data=all_data)
        policy_feature_names = env.policy_feature_column_names
        initialize_csv_loggers(env, policy_feature_names)
    except Exception as e:
        print(f"Error initializing environment or CSV loggers: {e}")
        import traceback; traceback.print_exc()
        return
        
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Environment initialized. State dim: {state_dim}, Action dim (n_etfs): {action_dim}")

    print("Calculating initial static covariance matrix...")
    initial_returns_for_cov = chosen_etf_prices_df.pct_change().dropna()
    if len(initial_returns_for_cov) < min_cov_calculation_samples: 
        initial_covariance_matrix_np = np.eye(env.n_etfs) * 0.01
    else:
        initial_covariance_matrix_np = initial_returns_for_cov.cov().values
    initial_covariance_matrix_tensor = torch.tensor(initial_covariance_matrix_np, dtype=torch.float32).to(device)
    initial_covariance_matrix_tensor += 1e-6 * torch.eye(env.n_etfs, device=device)
    current_covariance_matrix_tensor = initial_covariance_matrix_tensor.clone()
    print("Initial static covariance matrix calculated.")

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

    print("Starting training loop...")
    timestep_count = 0
    episode_rewards_history = []
    portfolio_value_history = []
    current_agent_update_count = 0

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        
        episode_reward = 0
        for t in range(1, max_timesteps_per_episode + 1):
            timestep_count += 1
            
            action_portfolio_weights, predicted_returns_for_step = agent.select_action(state, current_covariance_matrix_tensor)
            next_state, reward, done, info = env.step(action_portfolio_weights)
            
            true_fwd_returns = info.get('true_forward_returns', np.zeros(action_dim, dtype=np.float32))
            if true_fwd_returns is None or not isinstance(true_fwd_returns, np.ndarray) or true_fwd_returns.shape[0] != action_dim:
                if not (true_fwd_returns is None and action_dim == 0):
                    print(f"Warning: true_forward_returns from info dict is invalid or shape mismatch. Using zeros. Received: {true_fwd_returns}")
                true_fwd_returns = np.zeros(action_dim, dtype=np.float32)

            agent.store_reward_terminal(reward, done, true_fwd_returns)
            state = next_state
            episode_reward += reward
            
            if timestep_count % update_timestep_threshold == 0 and len(agent.buffer['states']) > 0 :
                update_metrics = agent.update(current_covariance_matrix_tensor)
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
        portfolio_value_history.append(env.current_portfolio_value)
        print(f"Episode: {episode}, Timesteps: {t}, Reward: {episode_reward:.2f}, Portfolio Value: {env.current_portfolio_value:.2f}")

        if episode % eval_freq == 0:
            print(f"--- Starting Evaluation for Training Epoch {episode} ---")
            eval_state = env.reset_for_eval() if hasattr(env, 'reset_for_eval') else env.reset()

            for eval_ep_num in range(1, 3):
                for t_eval in range(1, max_timesteps_per_episode + 1):
                    eval_portfolio_weights, eval_predicted_returns = agent.select_action(eval_state, current_covariance_matrix_tensor, is_eval=True)
                    append_to_csv('portfolio_weights_evolution',
                                 [episode, eval_ep_num, t_eval] + eval_portfolio_weights.tolist())
                    next_eval_state, eval_reward, eval_done, eval_info = env.step(eval_portfolio_weights)

                    append_to_csv('feature_analysis_data',
                                 [episode] + eval_state.tolist() + \
                                 eval_predicted_returns.tolist())
                    eval_state = next_eval_state
                    if eval_done:
                        break
                print(f"Evaluation Episode {eval_ep_num} for Training Epoch {episode} ended. Final Value: {env.current_portfolio_value:.2f}")
            print(f"--- Finished Evaluation for Training Epoch {episode} ---")

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

    if hasattr(env, 'close') and callable(env.close):
        env.close()

    print(f"Training visualizations (basic mean losses) can be generated from console output or by extending plotting scripts to use CSVs.")


    print("\n--- Running Final Evaluation and Performance Analysis ---")

    final_eval_env = env

    final_eval_portfolio_values = []
    final_eval_dates = []

    state = final_eval_env.reset()
    initial_value = final_eval_env.current_portfolio_value
    final_eval_portfolio_values.append(initial_value)
    if hasattr(final_eval_env, 'current_date'):
        final_eval_dates.append(pd.to_datetime(final_eval_env.current_date))


    final_policy_to_eval = agent.policy_old

    for t_eval in range(1, max_timesteps_per_episode * 2):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            positive_mean_action_tensor = final_policy_to_eval._get_positive_action_mean(state_tensor)

        final_eval_portfolio_weights = agent.mvo_solver(positive_mean_action_tensor, current_covariance_matrix_tensor)
        final_eval_portfolio_weights_np = final_eval_portfolio_weights.cpu().numpy().flatten()

        next_state, reward, done, info = final_eval_env.step(final_eval_portfolio_weights_np)
        
        final_eval_portfolio_values.append(final_eval_env.current_portfolio_value)
        if hasattr(final_eval_env, 'current_date'):
            final_eval_dates.append(pd.to_datetime(final_eval_env.current_date))

        state = next_state
        if done:
            break

    if final_eval_dates:
        pv_series_final = pd.Series(final_eval_portfolio_values, index=pd.DatetimeIndex(final_eval_dates))
    else:
        pv_series_final = pd.Series(final_eval_portfolio_values)

    performance_metrics = calculate_performance_metrics(pv_series_final)
    print("\n--- Final Performance Metrics ---")
    for k, v in performance_metrics.items():
        print(f"{k}: {v:.4f}")

    plot_performance(pv_series_final, title="Final Agent Performance", metrics=performance_metrics, dates=final_eval_dates if final_eval_dates else None)

    fig_train, ax_train = plt.subplots(2,1, figsize=(10,8), sharex=True)
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
