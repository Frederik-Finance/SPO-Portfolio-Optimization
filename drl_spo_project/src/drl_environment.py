import random
import gym
from gym import spaces
import numpy as np
import pandas as pd
import os




MIN_VALID_STEPS = 20

class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, processed_data, initial_investment=100000, transaction_cost_pct=0.001):
        super(PortfolioEnv, self).__init__()

        self.initial_investment = initial_investment
        self.transaction_cost_pct = transaction_cost_pct

        self.chosen_etf_prices_original = processed_data['chosen_etf_prices'].copy()
        self.feature_df_original = processed_data['combined_features'].copy()

        if self.chosen_etf_prices_original.empty:
            raise ValueError("Critical error: 'chosen_etf_prices' is empty in processed_data.")
        if self.feature_df_original.empty:
            raise ValueError("Critical error: 'combined_features' is empty in processed_data.")

        try:
            pivot_features = self.feature_df_original.unstack(level='Ticker')
            pivot_features.columns = ['_'.join(str(idx) for idx in col).strip() for col in pivot_features.columns.values]
            self.pivot_features_for_env = pivot_features
        except Exception as e:
            raise ValueError(f"Error pivoting/unstacking combined_features_df: {e}. Check its structure.")

        common_dates = self.chosen_etf_prices_original.index.intersection(self.pivot_features_for_env.index)
        
        if len(common_dates) < MIN_VALID_STEPS:
            raise ValueError(f"Not enough common dates ({len(common_dates)}) after aligning prices and features. Need at least {MIN_VALID_STEPS}.")

        self.chosen_etf_prices = self.chosen_etf_prices_original.loc[common_dates].copy()
        self.pivot_features_for_env = self.pivot_features_for_env.loc[common_dates].copy()
        
        self.pivot_features_for_env.dropna(axis=1, how='all', inplace=True)
        self.pivot_features_for_env.ffill(inplace=True)
        self.pivot_features_for_env.bfill(inplace=True)
        if self.pivot_features_for_env.isnull().values.any():
            print("Warning: NaNs found in pivot_features_for_env after ffill/bfill. This might indicate issues.")

        self.dates = common_dates.unique().sort_values()
        
        if len(self.dates) < MIN_VALID_STEPS:
            raise ValueError(f"Not enough valid dates ({len(self.dates)}) after final processing. Need at least {MIN_VALID_STEPS}.")

        malformed_cols = [c for c in self.pivot_features_for_env.columns if 'Unnamed:' in c]
        if malformed_cols:
            print(f"Dropping malformed columns: {malformed_cols}")
            self.pivot_features_for_env = self.pivot_features_for_env.drop(columns=malformed_cols)

        self.forward_return_column_names = [col for col in self.pivot_features_for_env.columns if col.startswith('M1_fwd_rtn_')]

        self.full_pivot_features_for_env = self.pivot_features_for_env.copy()

        all_feature_names_from_full_pivot = self.full_pivot_features_for_env.columns.tolist()

        excluded_patterns = ['timestep', 'actual returns', 'actual_returns', 'actual_reward', 'eval_episode_num']

        refined_policy_feature_column_names = []
        for col_name in all_feature_names_from_full_pivot:
            is_excluded = False
            if col_name in self.forward_return_column_names:
                is_excluded = True
            else:
                for pattern in excluded_patterns:
                    if col_name == pattern or col_name.startswith(pattern + '_'):
                        is_excluded = True
                        break
            if not is_excluded:
                refined_policy_feature_column_names.append(col_name)

        self.policy_feature_column_names = refined_policy_feature_column_names

        if not self.policy_feature_column_names:
            raise ValueError("No policy features found after excluding forward returns and other specified patterns. Check feature naming or exclusion logic.")

        self.n_features = len(self.policy_feature_column_names)
        self.n_etfs = self.chosen_etf_prices.shape[1] 
        
        if self.n_etfs == 0:
            raise ValueError("Number of ETFs is zero. Check 'chosen_etf_prices' columns.")
        if self.n_features == 0:
            raise ValueError("Number of features for the environment is zero. Check 'pivot_features_for_env'.")
        
        self.current_timestep = 0


        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_etfs,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32)

        self.current_step = 0
        self.current_portfolio_value = self.initial_investment
        self.current_weights = np.zeros(self.n_etfs + 1) 
        self.current_weights[self.n_etfs] = 1.0
    
    def _get_observation(self):
        current_date = self.dates[self.current_step]
        all_features_for_date = self.full_pivot_features_for_env.loc[current_date]
        obs = all_features_for_date[self.policy_feature_column_names].values.astype(np.float32)
        return obs

    def reset(self):
        max_start_step = len(self.dates) - MIN_VALID_STEPS
        if max_start_step < 0:
            max_start_step = 0
        self.current_step = random.randint(0, max_start_step)
        self.current_portfolio_value = self.initial_investment
        self.current_weights = np.zeros(self.n_etfs + 1)
        self.current_weights[self.n_etfs] = 1.0
        return self._get_observation()

    def step(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array(action).flatten()

        action = np.clip(action, 0, 1)
        if np.sum(action) > 1:
            action = action / np.sum(action)

        etf_weights = action
        cash_weight = 1.0 - np.sum(etf_weights)
        new_weights = np.append(etf_weights, cash_weight)

        trades = new_weights[:-1] - self.current_weights[:-1]
        transaction_amount = np.sum(np.abs(trades)) * self.current_portfolio_value
        transaction_costs = transaction_amount * self.transaction_cost_pct
        
        portfolio_value_before_market_move = self.current_portfolio_value - transaction_costs

        current_date = self.dates[self.current_step]
        current_prices = self.chosen_etf_prices.loc[current_date].values
        
        if np.any(current_prices <= 1e-6):
            print(f"Warning: Zero or near-zero price found for an ETF at date {current_date}. Prices: {current_prices}")

        done = False
        next_date_idx = self.current_step + 1
        if next_date_idx >= len(self.dates):
            done = True
            next_prices = current_prices
        else:
            next_market_date = self.dates[next_date_idx]
            next_prices = self.chosen_etf_prices.loc[next_market_date].values

        new_portfolio_value_after_market_move = 0
        for i in range(self.n_etfs):
            if current_prices[i] > 1e-6:
                new_portfolio_value_after_market_move += \
                    (etf_weights[i] * portfolio_value_before_market_move / current_prices[i]) * next_prices[i]

        new_portfolio_value_after_market_move += cash_weight * portfolio_value_before_market_move
        
        reward = new_portfolio_value_after_market_move - self.current_portfolio_value
        
        self.current_portfolio_value = new_portfolio_value_after_market_move
        self.current_weights = new_weights
        self.current_step += 1

        if self.current_step >= len(self.dates) -1 :
            done = True

        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        
        action_date = self.dates[self.current_step - 1]
        true_fwd_returns_for_step = self.full_pivot_features_for_env.loc[action_date, self.forward_return_column_names].values.astype(np.float32)

        ordered_true_fwd_returns = np.zeros(self.n_etfs, dtype=np.float32)
        etf_tickers_ordered = self.chosen_etf_prices.columns.tolist()

        for i, ticker_col_name in enumerate(etf_tickers_ordered):
            target_fwd_col = None
            for fwd_col in self.forward_return_column_names:
                if fwd_col.endswith(f'_{ticker_col_name}'):
                    target_fwd_col = fwd_col
                    break

            if target_fwd_col:
                original_fwd_col_idx = self.forward_return_column_names.index(target_fwd_col)
                ordered_true_fwd_returns[i] = true_fwd_returns_for_step[original_fwd_col_idx]
            else:
                print(f"Warning: Could not find matching forward return column for ETF ticker {ticker_col_name}")
                ordered_true_fwd_returns[i] = np.nan


        info_dict = {'true_forward_returns': ordered_true_fwd_returns}

        return obs, reward, done, info_dict

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step} / {len(self.dates)}')
        print(f'Date: {self.dates[self.current_step-1] if self.current_step > 0 else "Initial"}')
        print(f'Portfolio Value: {self.current_portfolio_value:.2f}')
        formatted_weights = [f"{w:.4f}" for w in self.current_weights]
        etf_names = self.chosen_etf_prices.columns.tolist()
        weights_info = ", ".join([f"{etf_names[i]}: {formatted_weights[i]}" for i in range(self.n_etfs)]) + f", Cash: {formatted_weights[self.n_etfs]}"
        print(f'Weights: {weights_info}')

    # In drl_environment.py, inside the PortfolioEnv class
    def get_rolling_covariance(self, lookback_window=252, min_samples=60):
        """
        Calculates the covariance matrix based on a rolling window of past returns.
        """
        # Determine the current date of the environment
        current_date = self.dates[self.current_step]
        
        # Define the start date for the lookback period
        start_date = current_date - pd.DateOffset(days=lookback_window)
        
        # Slice the original prices DataFrame to get the historical window
        historical_prices = self.chosen_etf_prices_original.loc[start_date:current_date]
        
        # Calculate periodic returns for the window
        returns = historical_prices.pct_change().dropna()
        
        # Ensure there are enough samples to calculate a stable covariance
        if len(returns) < min_samples:
            # Fallback to an identity matrix if data is insufficient
            return np.eye(self.n_etfs) * 0.01
        else:
            return returns.cov().values


if __name__ == '__main__':
    try:
        from data_loader import load_all_data, ETF_SYMBOLS
    except ImportError:
        print("Error: data_loader.py not found. Ensure it's in the 'src' directory or PYTHONPATH is set.")
        exit()

    RUN_WITH_REAL_DATA = False
    NUM_EXAMPLE_STEPS = 5

    if RUN_WITH_REAL_DATA:
        print("--- Attempting to initialize PortfolioEnv with REAL data ---")
        try:
            current_script_path = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_script_path)
            data_path = os.path.join(project_root, 'data')
            
            print(f"Loading real data using data_path_prefix: '{data_path}'")
            processed_data_real = load_all_data(data_path_prefix=data_path)
            
            if processed_data_real['chosen_etf_prices'].empty or processed_data_real['combined_features'].empty:
                raise ValueError("Real data loading resulted in empty DataFrames.")

            print("Real data loaded. Initializing PortfolioEnv...")
            env_real = PortfolioEnv(processed_data_real)
            obs_real = env_real.reset()
            print("\n--- Real Environment Test ---")
            print(f"Observation space shape: {env_real.observation_space.shape}")
            print(f"Action space shape: {env_real.action_space.shape}")
            print(f"Initial observation (first 10 features): {obs_real[:10]}")
            print(f"Number of ETFs: {env_real.n_etfs}, Names: {env_real.chosen_etf_prices.columns.tolist()}")

            for i in range(min(NUM_EXAMPLE_STEPS, len(env_real.dates) -1 )):
                action_real = env_real.action_space.sample()
                obs_real, reward_real, done_real, info_real = env_real.step(action_real)
                print(f"\nStep {i+1} (Real Data):")
                env_real.render()
                print(f"Reward: {reward_real:.2f}")
                if done_real:
                    print("Episode finished.")
                    break
            if not done_real:
                print("\nFinished example steps with real data.")

        except FileNotFoundError as e_real:
            print(f"Data file not found for real data run: {e_real}.")
            print("Switching to dummy data example.")
            RUN_WITH_REAL_DATA = False
        except ValueError as e_val:
            print(f"ValueError during real data setup: {e_val}.")
            print("Switching to dummy data example.")
            RUN_WITH_REAL_DATA = False
        except Exception as e_exc:
            print(f"An unexpected error occurred during real data run: {e_exc}")
            import traceback
            traceback.print_exc()
            print("Switching to dummy data example.")
            RUN_WITH_REAL_DATA = False
    
    if not RUN_WITH_REAL_DATA:
        print("\n--- Setting up PortfolioEnv with DUMMY Data ---")
        num_dummy_steps = 100
        default_dummy_tickers = ['SPY', 'QQQ', 'GLD', 'BND', 'XLE']
        dummy_chosen_tickers = ETF_SYMBOLS[:5] if 'ETF_SYMBOLS' in globals() and len(ETF_SYMBOLS) >=5 else default_dummy_tickers
        num_dummy_etfs = len(dummy_chosen_tickers)
        num_dummy_base_features = 10

        dummy_dates = pd.date_range(start='2020-01-01', periods=num_dummy_steps, freq='B')
        dummy_prices_data = np.random.rand(num_dummy_steps, num_dummy_etfs) * 100 + 50
        dummy_chosen_etf_prices = pd.DataFrame(dummy_prices_data, columns=dummy_chosen_tickers, index=dummy_dates)

        feature_names = [f'feature_{j}' for j in range(num_dummy_base_features)]
        econ_feature_names = [f'econ_feat_{k}' for k in range(3)]
        
        df_list_for_combined = []
        for ticker in dummy_chosen_tickers:
            ticker_data = np.random.rand(num_dummy_steps, num_dummy_base_features)
            df_ticker = pd.DataFrame(ticker_data, columns=feature_names, index=dummy_dates)
            df_ticker['Ticker'] = ticker
            df_ticker.reset_index(inplace=True)
            df_ticker.rename(columns={'index': 'Date'}, inplace=True)
            df_list_for_combined.append(df_ticker)
        
        dummy_combined_features_long = pd.concat(df_list_for_combined)
        
        dummy_econ_data = np.random.rand(num_dummy_steps, len(econ_feature_names))
        dummy_econ_df = pd.DataFrame(dummy_econ_data, columns=econ_feature_names, index=dummy_dates)
        dummy_econ_df.reset_index(inplace=True)
        dummy_econ_df.rename(columns={'index': 'Date'}, inplace=True)

        dummy_combined_features_long = pd.merge(dummy_combined_features_long, dummy_econ_df, on='Date', how='left')
        dummy_combined_features_long.set_index(['Date', 'Ticker'], inplace=True)
        
        dummy_processed_data = {
            'chosen_etf_prices': dummy_chosen_etf_prices,
            'combined_features': dummy_combined_features_long
        }
        
        try:
            print("Initializing PortfolioEnv with dummy data...")
            env_dummy = PortfolioEnv(dummy_processed_data)
            obs_dummy = env_dummy.reset()
            print("\n--- Dummy Environment Test ---")
            print(f"Observation space shape: {env_dummy.observation_space.shape}")
            print(f"Action space shape: {env_dummy.action_space.shape}")
            print(f"Initial observation (first 10 features): {obs_dummy[:10]}")
            print(f"Number of ETFs: {env_dummy.n_etfs}, Names: {env_dummy.chosen_etf_prices.columns.tolist()}")

            for i in range(min(NUM_EXAMPLE_STEPS, len(env_dummy.dates)-1)):
                action_dummy = env_dummy.action_space.sample() 
                obs_dummy, reward_dummy, done_dummy, info_dummy = env_dummy.step(action_dummy)
                print(f"\nStep {i+1} (Dummy Data):")
                env_dummy.render()
                print(f"Reward: {reward_dummy:.2f}")
                if done_dummy:
                    print("Episode finished.")
                    break
            if not done_dummy:
                print("\nFinished example steps with dummy data.")
        except Exception as e_dummy:
            print(f"Error initializing or running dummy environment: {e_dummy}")
            import traceback
            traceback.print_exc()
