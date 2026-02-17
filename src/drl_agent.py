import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
import os

try:
    from .spo_layer import DifferentiableMVO
    from .spo_loss import SPOPlusLoss
except ImportError:
    from spo_layer import DifferentiableMVO
    from spo_loss import SPOPlusLoss

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.6):
        super(ActorCritic, self).__init__()

        self.actor_mean_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim)
        )
        self.softplus = nn.Softplus()
        self.epsilon = 1e-6

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * np.log(action_std_init))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def _get_positive_action_mean(self, state):
        raw_action_mean = self.actor_mean_layers(state)
        # Use Tanh to bound returns between -1 and 1.
        # Scale to 0.001 (10bps).
        # Target is 0.0005 (5bps).
        # This relationship (Max Pred ~ 2x Target) ensures the constraint is feasible but BINDING.
        # If predictions were huge (e.g. 0.05), constraint would be slack -> Equal Weights.
        scaled_action_mean = torch.tanh(raw_action_mean) * 0.001 
        return scaled_action_mean

    def act(self, state):
        """
        """
        positive_action_mean = self._get_positive_action_mean(state)

        action_std = torch.exp(self.action_log_std).expand_as(positive_action_mean)
        # print(f"DEBUG: Mean {positive_action_mean.mean():.6f} | Std {action_std.mean():.6f}")
        cov_mat = torch.diag_embed(action_std * action_std)
        
        try:
            dist = MultivariateNormal(positive_action_mean, covariance_matrix=cov_mat)
        except ValueError as e:
            jitter = 1e-5 * torch.eye(cov_mat.size(-1), device=cov_mat.device, dtype=cov_mat.dtype)
            dist = MultivariateNormal(positive_action_mean, covariance_matrix=cov_mat + jitter)


        sampled_action = dist.sample()
        action_logprob = dist.log_prob(sampled_action)
        state_value = self.critic(state)

        return sampled_action.detach(), action_logprob.detach(), state_value.detach(), positive_action_mean.detach()

    def evaluate(self, state, sampled_action_from_buffer):
        """
        """
        positive_action_mean = self._get_positive_action_mean(state)

        action_std = torch.exp(self.action_log_std).expand_as(positive_action_mean)
        cov_mat = torch.diag_embed(action_std * action_std)
        
        try:
            dist = MultivariateNormal(positive_action_mean, covariance_matrix=cov_mat)
        except ValueError as e:
            jitter = 1e-5 * torch.eye(cov_mat.size(-1), device=cov_mat.device, dtype=cov_mat.dtype)
            dist = MultivariateNormal(positive_action_mean, covariance_matrix=cov_mat + jitter)


        action_logprobs = dist.log_prob(sampled_action_from_buffer)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

    def forward(self):
        raise NotImplementedError

import torch
import torch.optim as optim
import torch.nn as nn

class PPOAgent:
    def __init__(self, state_dim, action_dim,
                 mvo_solver_instance: DifferentiableMVO,
                 spo_loss_instance: SPOPlusLoss,
                 lr_actor=0.0003, lr_critic=0.001, gamma=0.99,
                 K_epochs=80, eps_clip=0.2, action_std_init=0.0005,
                 spo_plus_loss_coeff=1.0, kappa=0.0): 

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.action_dim = action_dim
        self.spo_plus_loss_coeff = spo_plus_loss_coeff
        self.kappa = kappa 

        self.mvo_solver = mvo_solver_instance
        self.spo_loss_fn = spo_loss_instance

        if self.mvo_solver is None:
            raise ValueError("mvo_solver_instance must be provided.")
        if self.spo_loss_fn is None:
            raise ValueError("spo_loss_instance must be provided.")

        self.policy = ActorCritic(state_dim, action_dim, action_std_init=action_std_init)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor_mean_layers.parameters(), 'lr': lr_actor},
            {'params': self.policy.action_log_std, 'lr': lr_actor * 0.1},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init=action_std_init)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.buffer = {
            "states": [], "actions": [], "logprobs": [], "rewards": [],
            "is_terminals": [], "state_values": [], "true_forward_returns": [],
            "prediction_actions_for_mvo": [], # Renamed from mean_actions
            "target_returns": [] # Store target return used
        }

    def select_action(self, state, current_covariance_matrix: torch.Tensor, is_eval=False, target_return=0.0005, kappa=None):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                sampled_action_tensor, action_logprob, state_val, positive_mean_action_tensor = self.policy_old.act(state_tensor)
                
                device = positive_mean_action_tensor.device
                cv_matrix_tensor_for_mvo = current_covariance_matrix.to(device)

                if cv_matrix_tensor_for_mvo.ndim == 2 and positive_mean_action_tensor.ndim == 2:
                    cv_matrix_tensor_for_mvo = cv_matrix_tensor_for_mvo.unsqueeze(0)
                elif cv_matrix_tensor_for_mvo.shape[0] == 1 and positive_mean_action_tensor.shape[0] > 1 :
                    cv_matrix_tensor_for_mvo = cv_matrix_tensor_for_mvo.repeat(positive_mean_action_tensor.shape[0],1,1)

                # IMPORTANT FIX:
                # During TRAINING (is_eval=False), we must use the SAMPLED action (noisy prediction)
                # to generate the portfolio weights. This ensures the reward signal corresponds
                # to the action that was actually taken (exploration).
                # During EVAL, we use the MEAN action (clean prediction) for best performance.
                
                if is_eval:
                    action_for_mvo = positive_mean_action_tensor
                else:
                    # Sampled action might have negative values if using Normal dist without softplus?
                    # The ActorCritic uses softplus on mean, but adds noise. 
                    # We should probably clamp or ensure positivity if MVO expects valid returns.
                    # MVO handles negative returns (just might not satisfy target).
                    # Actually policy.act() generally returns 'sampled_action' from MultivariateNormal(mean, cov).
                    # 'positive_mean_action' is softplus(mean).
                    # The sampled action is centered around positive_mean. It could be negative.
                    # Let's ensure it's comparable to returns by softplus or abs? 
                    # Or just feed raw. The MVO is the 'decoder'.
                    # Let's use the sampled_action directly.
                    action_for_mvo = sampled_action_tensor

                portfolio_weights_tensor = self.mvo_solver(
                    action_for_mvo, 
                    cv_matrix_tensor_for_mvo, 
                    target_return=target_return # Constraints
                ) 
                
                # --- Debugging: Check for Equal Weights / Fallback ---
                w_np = portfolio_weights_tensor.cpu().numpy().flatten()
                pred_np = action_for_mvo.cpu().numpy().flatten()
                
                is_equal_weight = np.allclose(w_np, 1.0/len(w_np), atol=1e-3)
                if is_equal_weight:
                    print(f"[DEBUG] Equal Weights Detected! Pred Range: [{pred_np.min():.5f}, {pred_np.max():.5f}] | Target: {target_return}")
                # -----------------------------------------------------

            if not is_eval:
                self.buffer['states'].append(state_tensor.cpu())
                self.buffer['actions'].append(sampled_action_tensor.cpu())
                self.buffer['logprobs'].append(action_logprob.cpu())
                self.buffer['state_values'].append(state_val.cpu())
                self.buffer['prediction_actions_for_mvo'].append(action_for_mvo.cpu())
                self.buffer['target_returns'].append(torch.tensor(target_return, dtype=torch.float32).cpu())

            return portfolio_weights_tensor.cpu().numpy().flatten(), sampled_action_tensor.cpu().numpy().flatten()

    def store_reward_terminal(self, reward, is_terminal, true_forward_returns_for_step):
        self.buffer['rewards'].append(torch.tensor(reward, dtype=torch.float32))
        self.buffer['is_terminals'].append(torch.tensor(is_terminal, dtype=torch.bool))
        
        target_device = self.buffer['actions'][0].device if self.buffer['actions'] and len(self.buffer['actions']) > 0 and self.buffer['actions'][0] is not None else torch.device('cpu')
        self.buffer['true_forward_returns'].append(torch.tensor(true_forward_returns_for_step, dtype=torch.float32).unsqueeze(0).to(target_device))

    def clear_buffer(self):
        for key in self.buffer.keys():
            self.buffer[key] = []

    def update(self, current_covariance_matrix: torch.Tensor, kappa=None): 
        if not self.buffer['states'] or len(self.buffer['states']) < 1:
            return None

        returns = []
        discounted_reward = 0
        for i in reversed(range(len(self.buffer['rewards']))):
            reward = self.buffer['rewards'][i]
            is_terminal = self.buffer['is_terminals'][i]
            if is_terminal:
                discounted_reward = reward
            else:
                discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
            
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        old_states = torch.cat(self.buffer['states']).detach()
        old_sampled_actions = torch.cat(self.buffer['actions']).detach()
        old_logprobs = torch.cat(self.buffer['logprobs']).detach().squeeze()
        old_state_values = torch.cat(self.buffer['state_values']).detach().squeeze()
        old_true_forward_returns = torch.cat(self.buffer['true_forward_returns']).detach()
        old_prediction_actions = torch.cat(self.buffer['prediction_actions_for_mvo']).detach()
        old_target_returns = torch.tensor(self.buffer['target_returns'], dtype=torch.float32).detach()

        # Advantage Calculation
        advantages = returns - old_state_values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Covariance Matrix Handling
        num_samples_in_batch = old_prediction_actions.shape[0]
        if current_covariance_matrix.ndim == 2:
            cv_matrix_tensor_for_loss = current_covariance_matrix.unsqueeze(0).repeat(num_samples_in_batch, 1, 1)
        elif current_covariance_matrix.shape[0] == 1 and num_samples_in_batch > 1:
            cv_matrix_tensor_for_loss = current_covariance_matrix.repeat(num_samples_in_batch, 1, 1)
        elif current_covariance_matrix.shape[0] == num_samples_in_batch:
            cv_matrix_tensor_for_loss = current_covariance_matrix
        else:
             # Just repeat the last if we can't match? 
             # Simpler: Assume covariance matrix passed is for the *current* step, but we are updating on stored batch.
             # Actually, storing covariance matrices in buffer is memory intensive.
             # But here we are passing 'current' which might be wrong if covariances change per step.
             # Ideally, we should store covariances or recompute them.
             # Given code structure, let's assume slowly changing covariance or simple repeat for now.
             cv_matrix_tensor_for_loss = current_covariance_matrix.repeat(num_samples_in_batch, 1, 1) if current_covariance_matrix.shape[0] != num_samples_in_batch else current_covariance_matrix

        cv_matrix_tensor_for_loss = cv_matrix_tensor_for_loss.to(old_prediction_actions.device)
        old_target_returns = old_target_returns.to(old_prediction_actions.device)

        # SPO+ Loss Calculation
        # Input: The actions we *used* for MVO (prediction_actions)
        actual_spo_plus_loss, spo_components = self.spo_loss_fn(
            old_prediction_actions, 
            old_true_forward_returns,
            cv_matrix_tensor_for_loss,
            old_target_returns # Pass batch of targets
        )

        total_policy_loss_agg = 0
        total_value_loss_agg = 0
        final_advantages_batch, final_ratios_batch, final_policy_objective_batch = None, None, None

        for k_epoch_iter in range(self.K_epochs):
            logprobs, state_values_eval, dist_entropy = self.policy.evaluate(old_states, old_sampled_actions)
            state_values_eval = torch.squeeze(state_values_eval)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss_per_sample = -torch.min(surr1, surr2)
            value_loss = self.MseLoss(state_values_eval, returns)

            # NOTE: SPO+ is computed ONCE per batch (on the collected trajectory), 
            # NOT recomputed per PPO epoch (since it depends on "oracle" which is fixed for the data).
            # Also, recomputing it would require differentiating through MVO each time which is slow.
            # However, technically we want to update the policy to minimize SPO+ loss on *new* actions?
            # No, standard DRL-SPO usually treats SPO+ as an auxiliary loss on the *trajectory* features.
            # The 'old_prediction_actions' are fixed (from rollout). 
            # If we want to optimize the policy to produce BETTER predictions, we need gradients w.r.t policy parameters.
            # But 'old_prediction_actions' are detached from graph! 
            # Wait. If 'old_prediction_actions' comes from buffer as .detach(), 
            # then 'actual_spo_plus_loss' will have NO gradient w.r.t. current policy!
            
            # IMPT: We need to re-evaluate the ACTOR on the states to get *current* predictions for SPO+ loss gradient.
            # Like we do for 'logprobs' in PPO.
            
            # Re-evaluate current policy to get new predictions for SPO+
            # We need the 'mean' action or 'sampled'?
            # Usually we want the 'mean' (deterministic prediction) to match the oracle.
            current_mean_actions_for_loss = self.policy._get_positive_action_mean(old_states) 
            # Or sampled? The paper definition is on the parameters \hat{\mu}. 
            # The actor outputs distribution parameters. The mean is the best estimate of \hat{\mu}.
            
            current_spo_loss, _ = self.spo_loss_fn(
                current_mean_actions_for_loss,
                old_true_forward_returns,
                cv_matrix_tensor_for_loss,
                old_target_returns
            )

            loss = policy_loss_per_sample.mean() + \
                   0.5 * value_loss + \
                   self.spo_plus_loss_coeff * current_spo_loss - \
                   0.01 * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss_agg += policy_loss_per_sample.mean().item()
            total_value_loss_agg += value_loss.item()
            
            if k_epoch_iter == self.K_epochs - 1:
                final_advantages_batch = advantages.detach().cpu().numpy()
                final_ratios_batch = ratios.detach().cpu().numpy()
                final_policy_objective_batch = policy_loss_per_sample.detach().cpu().numpy()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.clear_buffer()

        avg_policy_loss = total_policy_loss_agg / self.K_epochs
        avg_value_loss = total_value_loss_agg / self.K_epochs

        update_metrics = {
            'total_spo_loss': actual_spo_plus_loss.item() if actual_spo_plus_loss is not None else 0.0,
            'avg_policy_loss': avg_policy_loss,
            'avg_value_loss': avg_value_loss,
            'spo_components': spo_components,
            'advantages_batch': final_advantages_batch,
            'ratios_batch': final_ratios_batch,
            'policy_objective_batch': final_policy_objective_batch
        }
        return update_metrics

    def save_model(self, filepath):
        torch.save(self.policy_old.state_dict(), filepath)

    def load_model(self, filepath):
        self.policy.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
        self.policy_old.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
if __name__ == '__main__':
    print("Performing a basic instantiation test of PPOAgent...")
    try:
        dummy_state_dim = 10
        dummy_n_etfs = 2

        dummy_mvo = DifferentiableMVO(num_assets=dummy_n_etfs)
        dummy_spo_loss = SPOPlusLoss(num_assets=dummy_n_etfs)

        agent = PPOAgent(state_dim=dummy_state_dim,
                         action_dim=dummy_n_etfs,
                         mvo_solver_instance=dummy_mvo,
                         spo_loss_instance=dummy_spo_loss,
                         spo_plus_loss_coeff=0.01) # Adjusted coefficient for testing
        print("PPOAgent successfully instantiated with dummy MVO and SPO Loss module.")
    except Exception as e:
        print(f"Error during PPOAgent instantiation in drl_agent.py __main__: {e}")
        import traceback
        traceback.print_exc()
