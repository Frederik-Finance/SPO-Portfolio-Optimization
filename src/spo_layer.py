import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

class DifferentiableMVO(nn.Module):
    def __init__(self, num_assets, max_weight_per_asset=1.0, debug=False):
        """
        Differentiable Layer solving the Minimum Variance Portfolio problem
        subject to a Soft Return Constraint:
            min (1/2) * w^T * Sigma * w + Penalty * ReLU(Target - mu^T w)^2
            s.t. sum(w) = 1
                 w >= 0
        
        This formulation DOES NOT constrain mu^T w >= Target strictly.
        Instead, it penalizes violation. This guarantees feasibility (the solver never crashes).
        The penalty strength acts as the "strictness" of the constraint.
        """
        super(DifferentiableMVO, self).__init__()
        self.num_assets = num_assets
        self.max_weight_per_asset = max_weight_per_asset
        self.debug = debug

        # --- Define Optimization Problem (Parametric) ---
        w = cp.Variable(num_assets)
        
        # Parameters
        mu_hat_param = cp.Parameter(num_assets, name="mu_hat")
        L_transpose_param = cp.Parameter((num_assets, num_assets), name="L_transpose")
        target_return_param = cp.Parameter(name="target_return", value=0.0005)
        
        # Penalty parameter for soft constraint
        # A large value forces the solver to prioritize the constraint
        penalty_strength = 10000.0 

        # Objective components
        # 1. Risk: (1/2) || L^T w ||^2
        risk = 0.5 * cp.sum_squares(L_transpose_param @ w)
        
        # 2. Return Violation: pos(Target - mu^T w)
        # We square it to make it quadratic (differentiable and compatible with QP solvers)
        # Using pos() is equivalent to ReLU
        expected_return = mu_hat_param @ w
        violation = cp.pos(target_return_param - expected_return)
        penalty = penalty_strength * cp.square(violation)
        
        # Combined Objective
        objective = cp.Minimize(risk + penalty)

        # Constraints (Simplex only)
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]
        
        if max_weight_per_asset < 1.0:
            constraints.append(w <= max_weight_per_asset)

        # Create CvxpyLayer
        problem = cp.Problem(objective, constraints)
        self.cvxpylayer = CvxpyLayer(problem, parameters=[L_transpose_param, mu_hat_param, target_return_param], variables=[w])
        
        if self.debug:
               print(f"Initialized DifferentiableMVO (Soft Constrained) with {num_assets} assets.")


    def forward(self, predicted_returns, covariance_matrix, target_return=0.0005, kappa=None):
        """
        Forward pass.
        """
        # Ensure batch dim
        if predicted_returns.ndim == 1:
            predicted_returns = predicted_returns.unsqueeze(0)
        if covariance_matrix.ndim == 2:
            covariance_matrix = covariance_matrix.unsqueeze(0).repeat(predicted_returns.shape[0], 1, 1)

        batch_size = predicted_returns.shape[0]
        device = predicted_returns.device
        dtype = predicted_returns.dtype
        
        # Prepare target_return as a tensor
        if isinstance(target_return, float):
             target_return_t = torch.full((batch_size,), target_return, device=device, dtype=dtype)
        elif isinstance(target_return, torch.Tensor):
             if target_return.ndim == 0:
                  target_return_t = target_return.repeat(batch_size)
             else:
                  target_return_t = target_return
        else:
             target_return_t = torch.full((batch_size,), 0.0005, device=device, dtype=dtype)

        final_weights_list = []

        for i in range(batch_size):
            mu_i = predicted_returns[i]
            Sigma_i = covariance_matrix[i]
            r_tgt_i = target_return_t[i]
            
            # --- Cholesky Decomposition ---
            try:
                L_i = torch.linalg.cholesky(Sigma_i)
                L_i_transpose = L_i.T.contiguous()
                
                # --- Solve ---
                # Soft Constraint formulation is always feasible.
                # using SCS (default) or ECOS. 
                # Since we have sum_squares terms, SCS is standard.
                w_sol_tuple = self.cvxpylayer(
                    L_i_transpose, 
                    mu_i, 
                    r_tgt_i, 
                    solver_args={'solve_method': 'SCS', 'eps': 1e-4, 'max_iters': 2500, 'acceleration_lookback': 0}
                )
                w_sol = w_sol_tuple[0]
                
                if torch.isnan(w_sol).any():
                    if self.debug:
                         print(f"MVO NaN detected sample {i}")
                    w_sol = torch.ones_like(mu_i) / self.num_assets
                    
                final_weights_list.append(w_sol)

            except Exception as e:
                if self.debug:
                    print(f"MVO Error sample {i}: {e}")
                
                 # Fallback: Equal Weights
                w_eq = torch.ones_like(mu_i) / self.num_assets
                final_weights_list.append(w_eq)

        return torch.stack(final_weights_list)
