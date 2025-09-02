# DRL-SPO: A Decision-Focused Deep Reinforcement Learning Framework for Portfolio Optimization

## Introduction

This project implements a sophisticated, end-to-end portfolio optimization framework using Deep Reinforcement Learning (DRL). The core innovation lies in its use of a Decision-Focused Learning objective based on the SPO+ (Smart Predict-then-Optimize) framework.

Instead of training a model to simply minimize prediction error (e.g., predict future returns as accurately as possible), this project trains a DRL agent to produce predictions that are maximally useful for a downstream portfolio optimization task. The agent, powered by the Proximal Policy Optimization (PPO) algorithm, learns to generate return estimates that, when fed into a Mean-Variance Optimization (MVO) solver, yield the highest possible portfolio performance.

This approach directly bridges the gap between financial prediction and portfolio construction, optimizing the entire decision-making pipeline from raw data to final asset allocation.

## Key Features

- **Decision-Focused Reinforcement Learning:** Utilizes a custom `SPOPlusLoss` function that enables the PPO agent to learn from the "regret" of its decisions, directly optimizing for the quality of the final portfolio allocation rather than just prediction accuracy.
- **Differentiable MVO Layer:** Integrates a convex Mean-Variance Optimization problem directly into the neural network using `cvxpylayers`. This allows gradients to flow back through the optimizer, enabling true end-to-end training.
- **End-to-End Automated Pipeline:** Provides a complete workflow from data ingestion and feature engineering to agent training, out-of-sample backtesting, and performance visualization.
- **Rich, Multi-Source Feature Set:** Leverages a wide array of data sources to inform the agent's decisions, including:
  - Market Data (prices, volume)
  - Macroeconomic Indicators (GDP, CPI, etc.)
  - Interest Rates & Yield Spreads
  - Market Sentiment (Put/Call Ratios, Volatility)
- **Automated Asset Universe Selection:** Employs the `AffinityPropagation` clustering algorithm to intelligently select a diversified subset of ETFs, reducing dimensionality while preserving market representation.
- **Robust Backtesting Engine:** Features a rigorous out-of-sample backtesting module that compares the DRL agent's performance against standard benchmarks (e.g., equal-weighted buy-and-hold) while accounting for realistic transaction costs and rebalancing fees.
- **Explainable AI (XAI) for Finance:** Includes capabilities for SHAP (SHapley Additive exPlanations) analysis to interpret the model's behavior, identifying which features are most influential in its decision-making process over time.
- **Comprehensive Logging and Visualization:** Generates detailed logs for all training metrics (losses, weights, rewards) and a suite of plots for analyzing training dynamics, backtest performance, and portfolio evolution.

## System Architecture

The system is composed of four main components that interact in a loop during the training process. The project is organized into a modular structure, with each Python script in the `src/` directory responsible for a specific component of the pipeline.

```
+--------------------------+      +--------------------------+
|      PortfolioEnv        |      |        PPOAgent          |
| (drl_environment.py)     |      |      (drl_agent.py)      |
+--------------------------+      +--------------------------+
| - Provides current state |----->| - Actor predicts returns |
|   (market features)      |      |   (r_hat)                |
| - Executes trades        |      | - Critic estimates value |
| - Calculates reward      |      +-----------+--------------+
+-----------+--------------+                  |
            |                                 | (r_hat)
            | (reward, next_state)            |
            |                                 v
+-----------+--------------+      +--------------------------+
|      SPOPlusLoss         |      |    DifferentiableMVO     |
|      (spo_loss.py)       |<-----|     (spo_layer.py)       |
+--------------------------+      +--------------------------+
| - Calculates SPO+ loss   |      | - Solves for optimal     |
|   (decision-focused)     |----->|   portfolio weights (w)  |
| - Informs agent update   |      |   using r_hat & covariance|
+--------------------------+      +--------------------------+
```

### `main.py`

- **Role:** The main entry point and orchestrator of the project.
- **Functionality:** Sets hyperparameters, initializes the data loader, environment, and agent, manages the main training loop, and runs the final out-of-sample backtest.

### `src/data_loader.py`

- **Role:** Data ingestion and feature engineering.
- **Functionality:** Downloads market data, loads supplemental data, performs automated asset selection, engineers a rich feature set, and prepares the ground truth returns for the SPO+ loss.

### `src/drl_environment.py`

- **Role:** The portfolio simulation environment.
- **Functionality:** Implements a custom OpenAI Gym environment, manages the portfolio state, calculates rewards, simulates transaction costs, and provides the agent with the state and true forward returns.

### `src/drl_agent.py`

- **Role:** The core DRL agent logic.
- **Functionality:** Implements the `PPOAgent` with an `ActorCritic` architecture. The Actor predicts future returns (`r_hat`), which are then passed to the `DifferentiableMVO` layer. It manages the experience replay buffer and the combined PPO and SPO+ loss update step.

### `src/spo_layer.py`

- **Role:** The differentiable optimization layer.
- **Functionality:** Defines the `DifferentiableMVO` `nn.Module` using `cvxpylayers` to embed a robust Mean-Variance Optimization problem into the PyTorch computation graph.

### `src/spo_loss.py`

- **Role:** The decision-focused loss function.
- **Functionality:** Implements the `SPOPlusLoss` as a `nn.Module`, calculating the "regret" by comparing the agent's decision with the "oracle" decision.

### `run_shap_analysis.py`

- **Role:** Explainable AI (XAI) analysis.
- **Functionality:** Loads a trained model and uses the SHAP library to compute Shapley values for each feature, providing insights into the model's drivers.

## The DRL-SPO+ Learning Algorithm

This project's intelligence is centered around a hybrid learning algorithm that combines Deep Reinforcement Learning (DRL) with a decision-focused learning objective known as SPO+ (Smart Predict-then-Optimize). This allows the agent to learn not just to predict, but to predict in a way that leads to high-quality portfolio decisions.

The process can be broken down into three core components: the Prediction Model, the Differentiable Optimizer, and the Decision-Focused Loss Function.

### 1. The Prediction Model (PPO Actor)

The agent uses an Actor-Critic architecture, trained with Proximal Policy Optimization (PPO).

- **The Actor's Job:** The Actor network takes the current market state (the rich feature set) and outputs a vector of predicted future returns (`r_hat`) for each ETF.
- **The Critic's Job:** The Critic network functions as it normally does in PPO, estimating the value of the current state to help stabilize training.

### 2. The Differentiable Optimizer (DifferentiableMVO Layer)

This is the bridge between prediction and action. The predicted returns (`r_hat`) from the Actor are fed into a differentiable Mean-Variance Optimization (MVO) layer.

- **Function:** This layer solves a classic portfolio optimization problem: `maximize (r_hat' * w) - k * (w' * Σ * w)`, where `w` are the portfolio weights, `Σ` is the asset covariance matrix, and `k` is a risk-aversion parameter.
- **Differentiability:** By using `cvxpylayers`, this optimization problem becomes a differentiable node in the computation graph, allowing gradients to flow back through the MVO solver.
- **Output:** The output of this layer is the final portfolio weights (`w_hat`), which is the action the agent executes in the environment.

### 3. The Decision-Focused Loss (SPOPlusLoss)

This is the intellectual core of the algorithm. During the training update step, the agent calculates a special loss based on the quality of its decision, known as "regret."

1.  **Calculate the "Oracle" Decision:** First, the agent calculates the theoretically best possible portfolio weights, `w*`, by feeding the true future returns (`r_true`) into the MVO solver. This `w*` represents the perfect-foresight decision.
2.  **Calculate Regret:** The `SPOPlusLoss` function then computes the difference in portfolio performance between the agent's decision (`w_hat`) and the oracle's decision (`w*`), when evaluated against the true returns `r_true`.
3.  **Generate a Gradient:** The SPO+ formulation provides a loss whose gradient effectively tells the Actor network how to change its `r_hat` predictions to make the resulting `w_hat` closer to the optimal `w*`.

### End-to-End Information Flow

The entire process ties together like this: `State -> [Actor] -> r_hat -> [MVO Solver] -> w_hat (Action) -> [Environment]`

During training, the `SPOPlusLoss` looks at the `r_hat` that was predicted and the `r_true` that actually occurred. It calculates the regret and sends a powerful gradient signal back to the Actor. `Loss(r_hat, r_true) -> Gradient -> [Actor Update]`

This end-to-end, decision-focused training loop is what makes the agent so effective. It learns to produce predictions that are not just statistically accurate, but are tailored to be maximally effective for the specific MVO problem it is trying to solve.

## Data Sources and Features

The agent's decisions are informed by a rich and diverse dataset aggregated from multiple sources.

### Data Sources

- **Yahoo Finance:** The primary source for historical daily price and volume data for a wide universe of ETFs, accessed via the `yfinance` library.
- **Local Excel Files:** The project relies on three key Excel files located in the `data/` directory:
  - `ETF_Summary.xlsx`: Contains ETF-specific sentiment and options data (put/call ratios, short interest, implied volatility).
  - `US_Economic_Data.xlsx`: Provides time series for major US macroeconomic indicators (GDP, CPI), aligned by release date to prevent lookahead bias.
  - `US Rates.xlsx`: Contains historical data for US Treasury interest rates.

### Feature Engineering

A comprehensive feature engineering pipeline creates the state vector for the DRL agent, providing a holistic view of the market environment:

- **Momentum & Technical Features:** Historical returns over multiple periods (1-12 months) and trading volume.
- **Market Sentiment Features:** Put/Call Ratio, Short Interest, Implied Volatility, and Market Capitalization.
- **Macroeconomic Indicators:** GDP, CPI, Industrial Production, Non-Farm Payrolls, and Consumer Sentiment, all time-aligned to their public release dates.
- **Interest Rate Features:** Key US Treasury yields (10Y, 2Y, etc.), the 10Y-2Y yield spread, and their rates of change.

## Project Structure

```
drl_spo_project/
├── data/                 # Raw data files (CSVs, Excel)
├── models/               # Saved model weights
├── src/                  # Source code
│   ├── __init__.py
│   ├── data_loader.py    # Loads and preprocesses data
│   ├── drl_agent.py      # PPO agent implementation
│   ├── drl_environment.py# Gym environment for portfolio management
│   ├── spo_layer.py      # Differentiable MVO layer
│   └── spo_loss.py       # SPO+ loss function
├── training_logs/        # Output logs, charts, and data from training/backtesting
├── visualizations/       # Scripts for creating visualizations
├── main.py               # Main script to run training and backtesting
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup and Usage

Follow these steps to set up the environment and run the project.

### 1. Setup Instructions

**Clone the Repository:**

```bash
git clone <your-repository-url>
cd drl_spo_project
```

**Create and Activate a Virtual Environment (Recommended):**

```bash
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

**Install Required Dependencies:**

```bash
pip install -r requirements.txt
```

**Note:** `cvxpylayers` can sometimes have complex dependencies. If you encounter installation errors, please consult the official `cvxpylayers` documentation.

**Verify Data Files:** Ensure the necessary Excel files are located in the `data/` directory:

- `ETF_Summary.xlsx`
- `US_Economic_Data.xlsx`
- `US Rates.xlsx`

### 2. How to Run

**Execute the Main Script:**

```bash
python main.py
```

**Expected Workflow:** The script will load data, initialize the environment and agent, run the training loop, save model checkpoints and logs, and automatically launch an out-of-sample backtest upon completion, generating final performance charts.

### 3. Configuration

Key parameters for the simulation can be adjusted directly in the `main()` function of `main.py`:

- **Data Split:** `TRAIN_TEST_CUTOFF_DATE` sets the boundary between training and backtesting data.
- **Training Loop:** `max_episodes` and `max_timesteps_per_episode` control the training duration.
- **PPO & SPO+ Hyperparameters:** `lr_actor`, `lr_critic`, `gamma`, `spo_plus_loss_coeff`, and `kappa` (risk aversion) can be tuned.
- **Backtesting Simulation:** `REBALANCE_FREQUENCY_DAYS` and `REBALANCING_FEE_PCT` control the backtest realism.
