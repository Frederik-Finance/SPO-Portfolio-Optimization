import torch
import torch.nn as nn
import numpy as np

try:
    from .spo_layer import DifferentiableMVO
except ImportError:
    from spo_layer import DifferentiableMVO


class SPOPlusLoss(nn.Module):
    def __init__(self, num_assets, mvo_max_weight_per_asset=1.0, kappa=0.0, scaling_factor=1000.0):
        """
        SPO+ Loss for Constraint-Side Uncertainty.
        
        Ref: "Local Consistency of SPO+ Loss"
        Loss L = 1 - mu_stress^T y*(mu)  (Normalized form in paper)
        
        In our case:
        Target Constraint: mu^T w >= r_tgt
        Regret = Maximizing Constraint Satisfaction? No, minimizing violation.
        
        Paper Formulation:
        L_{SPO+}(\hat{\mu}, \mu) = \mu_s^T y^*(\mu_s) - \mu_s^T y^*(\mu)
        
        where y*(v) solves: min 1/2 ||y||^2 s.t. v^T y >= r_tgt
        
        For feasible v, v^T y*(v) = r_tgt (binding constraint).
        So L = r_tgt - \mu_s^T y^*(\mu)
        
        Args:
            scaling_factor: Multiplier to scale the loss to be comparable with PPO loss.
                            Since returns are ~1e-3, loss is ~1e-4. PPO is ~1. 
                            Factor of 1000-10000 is appropriate.
        """
        super(SPOPlusLoss, self).__init__()
        self.num_assets = num_assets
        self.scaling_factor = scaling_factor
        
        # Oracle Solver (uses True parameters)
        # We assume the oracle also solves the MinVariance problem with True returns.
        self.oracle_solver = DifferentiableMVO(
            num_assets=num_assets, 
            max_weight_per_asset=mvo_max_weight_per_asset
        )

    def forward(self, predicted_returns_hat, true_returns, covariance_matrix, target_return, kappa=None):
        """
        Compute SPO+ Loss.
        
        mu_s = 2 * mu_hat - mu
        w*(mu) = Optimal weights using TRUE returns (Oracle)
        
        Loss = r_tgt - mu_s^T w*(mu)
        
        Note: If mu_s^T w*(mu) > r_tgt, loss is negative?
        SPO+ loss is usually non-negative. 
        In the paper, L = 1 - mu_s^T y*. 
        The paper assumes min-norm, where dual variables are positive.
        Essentially we want to maximize mu_s^T w*(mu) to satisfy the constraint for mu_s.
        If we minimize (r_tgt - val), we push val up.
        """
        # Dimensions
        if predicted_returns_hat.ndim == 1:
            predicted_returns_hat = predicted_returns_hat.unsqueeze(0)
        if true_returns.ndim == 1:
            true_returns = true_returns.unsqueeze(0)
        
        batch_size = predicted_returns_hat.shape[0]
        device = predicted_returns_hat.device
        
        # Scalar handling for target_return
        if not isinstance(target_return, torch.Tensor):
            target_return = torch.full((batch_size,), float(target_return), device=device)
        elif target_return.ndim == 0:
            target_return = target_return.repeat(batch_size)
            
        # 1. Oracle Solution w*(mu)
        # Minimize Variance s.t. mu^T w >= r_tgt
        # We generally assume true_returns allows feasibility.
        with torch.no_grad(): 
            w_star_mu = self.oracle_solver(true_returns, covariance_matrix, target_return).detach()

        # 2. Stress Parameter
        # mu_s = 2 * mu_hat - mu
        mu_s = 2 * predicted_returns_hat - true_returns
        
        # 3. SPO+ Loss Calculation
        # L = Constraints(mu_s on w*(mu))
        # We want mu_s^T w*(mu) >= r_tgt.
        # Loss = ReLU(r_tgt - mu_s^T w*(mu)) ?
        # The paper form '1 - ...' implies linear penalty, allowing negative values if ... > 1.
        # However, for minimization tasks, 'Regret' is usually positive.
        # In the paper, L >= Regret >= 0.
        # Let's use the direct linear form from paper: L = r_tgt - mu_s^T w*(mu)
        # Wait, if we minimize this, we push mu_s^T w* to +infinity.
        # The paper's problem is P(v): min ||y||^2 s.t. v^T y = 1.
        # The minimizer y*(v) has length proportional to 1/||v||.
        # If we just maximize mu_s^T w*, and w* is fixed, mu_hat grows indefinitely.
        # BUT w* is fixed. mu_hat is network output.
        # Network output usually bounded or regularized.
        # Also, mu_hat determines the action in the PPO step (w_hat).
        # This Loss is an AUXILIARY loss to shape the features.
        
        # Let's stick to the paper's literal definition:
        # L = 1 - \mu_s^T y^*(\mu)
        # Here: L = target_return - \mu_s^T w^*(\mu)
        
        # Calculate Term: mu_s^T w*(mu)
        term_mu_s_w_star = torch.sum(mu_s * w_star_mu, dim=1)
        
        # Loss
        raw_loss = target_return - term_mu_s_w_star
        
        # Scaling
        scaled_loss = raw_loss * self.scaling_factor
        
        final_loss = scaled_loss.mean()

        spo_component_means = {
           'raw_spo_loss_mean': raw_loss.mean().item(),
           'term_mu_s_w_star_mean': term_mu_s_w_star.mean().item(),
           'target_return_mean': target_return.mean().item()
        }

        return final_loss, spo_component_means

if __name__ == '__main__':
    print("Testing SPO+ Loss (Constraint-Side)...")
    loss_mod = SPOPlusLoss(num_assets=3)
    
    # Fake data
    p_ret = torch.tensor([[0.02, 0.02, 0.02]], requires_grad=True)
    t_ret = torch.tensor([[0.01, 0.03, 0.01]])
    cov = torch.eye(3).unsqueeze(0)
    tgt = 0.015
    
    l, comps = loss_mod(p_ret, t_ret, cov, tgt)
    print("Loss:", l.item())
    print("Comps:", comps)
    
    l.backward()
    print("Grad:", p_ret.grad)
