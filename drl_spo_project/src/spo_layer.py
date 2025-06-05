import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import cvxpy.error

class DifferentiableMVO(nn.Module):
    def __init__(self, num_assets, max_weight_per_asset=1.0):
        """
        """
        super(DifferentiableMVO, self).__init__()
        self.num_assets = num_assets
        self.max_weight_per_asset = max_weight_per_asset

        y_cvx = cp.Variable(num_assets, name="y_decision_var")
        mu_cvx_param = cp.Parameter(num_assets, name="expected_returns_param")
        L_transpose_param = cp.Parameter((num_assets, num_assets), name="cov_L_transpose_param")

        objective = cp.Minimize(0.5 * cp.sum_squares(L_transpose_param @ y_cvx))
        constraints = [(mu_cvx_param @ y_cvx) == 1, y_cvx >= 0]

        if self.max_weight_per_asset < 1.0:
            constraints.append(y_cvx <= self.max_weight_per_asset * cp.sum(y_cvx))

        problem = cp.Problem(objective, constraints)
        self.cvxpylayer = CvxpyLayer(problem, parameters=[L_transpose_param, mu_cvx_param], variables=[y_cvx])

        print(f"DifferentiableMVO layer initialized for Max Sharpe (num_assets={num_assets}).")
        print(f"Objective: Minimize 0.5 * sum_squares(L_transpose @ y), s.t. mu^T y = 1, y >= 0")
        if self.max_weight_per_asset < 1.0:
            print(f"Constraint: y_i <= {self.max_weight_per_asset} * sum(y_j) also included.")
        print("Output 'y' will be normalized to get weights 'w = y / sum(y)'.")


    def forward(self, predicted_returns, covariance_matrix):
        """
        """
        if predicted_returns.ndim == 1:
            predicted_returns = predicted_returns.unsqueeze(0)
        if covariance_matrix.ndim == 2:
            covariance_matrix = covariance_matrix.unsqueeze(0).repeat(predicted_returns.shape[0], 1, 1)

        batch_size = predicted_returns.shape[0]
        num_assets = self.num_assets
        final_weights_list = []
        
        DEBUG_MVO_LAYER = False

        for i in range(batch_size):
            mu_i = predicted_returns[i]
            Sigma_i = covariance_matrix[i]
            y_solution_flat = torch.ones_like(mu_i)

            if not torch.any(mu_i > 1e-6):
                if DEBUG_MVO_LAYER:
                    print(f"MVO WARNING Sample {i}: All mu_i non-positive or too small. mu_i: {mu_i.detach().cpu().numpy()}. Fallback to equal y.")
            else:
                try:
                    L_i = torch.linalg.cholesky(Sigma_i)
                    L_i_transpose = L_i.T.contiguous()

                    solver_verbose = False
                    y_solution_tuple = self.cvxpylayer(
                        L_i_transpose, mu_i,
                        solver_args={'verbose': solver_verbose,
                                     'solve_method': 'ECOS',
                                     'max_iters': 10000,
                                     'feastol': 1e-8,
                                     'abstol': 1e-8,
                                     'reltol': 1e-8}
                    )
                    y_candidate = y_solution_tuple[0]

                    if torch.isnan(y_candidate).any() or torch.isinf(y_candidate).any():
                        if DEBUG_MVO_LAYER:
                            print(f"MVO WARNING Sample {i}: Solver returned NaN/Inf. mu: {mu_i.detach().cpu().numpy()}. y_sol: {y_candidate.detach().cpu().numpy()}. Fallback to equal y.")
                    elif (y_candidate < -1e-5).any():
                        if DEBUG_MVO_LAYER:
                            print(f"MVO WARNING Sample {i}: Solver returned negative y despite y>=0 constraint. mu: {mu_i.detach().cpu().numpy()}. y: {y_candidate.detach().cpu().numpy()}. Fallback to equal y.")
                    else:
                        y_solution_flat = y_candidate

                except cp.error.SolverError as se:
                    if DEBUG_MVO_LAYER:
                        print(f"MVO ERROR Sample {i}: CVXPY SolverError. mu: {mu_i.detach().cpu().numpy()}. Error: {se}. Fallback to equal y.")
                except RuntimeError as re:
                    if DEBUG_MVO_LAYER:
                        print(f"MVO ERROR Sample {i}: RuntimeError (likely Cholesky decomposition failed). mu: {mu_i.detach().cpu().numpy()}. Error: {re}. Check Sigma_i. Fallback to equal y.")
                except Exception as e:
                    if DEBUG_MVO_LAYER:
                        print(f"MVO ERROR Sample {i}: Generic Error in cvxpylayer. mu: {mu_i.detach().cpu().numpy()}. Error: {e}, Type: {type(e)}. Fallback to equal y.")

            sum_y = torch.sum(y_solution_flat)

            if sum_y.abs() > 1e-6:
                weights_w = y_solution_flat / sum_y
                weights_w = torch.relu(weights_w)
                weights_w = weights_w / torch.sum(weights_w)
            else:
                if DEBUG_MVO_LAYER:
                    print(f"MVO WARNING Sample {i}: Sum of y is near zero: {sum_y.item()}. mu: {mu_i.detach().cpu().numpy()}. Fallback to equal w.")
                weights_w = torch.ones_like(mu_i) / num_assets

            if self.max_weight_per_asset < 1.0:
                weights_w_clipped = torch.clamp(weights_w, 0, self.max_weight_per_asset)
                current_sum_clipped = torch.sum(weights_w_clipped)
                
                if current_sum_clipped > 1e-6:
                    weights_w = weights_w_clipped / current_sum_clipped
                else:
                    if DEBUG_MVO_LAYER and not torch.allclose(weights_w, weights_w_clipped):
                         print(f"MVO WARNING Sample {i}: Post-clipping for max_weight led to zero sum. Reverting to equal w.")
                    weights_w = torch.ones_like(mu_i) / num_assets
            
            final_weights_list.append(weights_w.unsqueeze(0))

        final_weights_batch = torch.cat(final_weights_list, dim=0)
        return final_weights_batch

if __name__ == '__main__':
    print("Differentiable MVO Layer (spo_layer.py) - Implemented with cvxpylayers")
    num_assets_example = 3
    batch_s = 5

    mvo_layer = DifferentiableMVO(num_assets=num_assets_example, max_weight_per_asset=0.6)

    all_returns = []
    all_covs = []

    dummy_predicted_returns_1 = torch.tensor([[0.1, 0.2, 0.05]], dtype=torch.float32)
    rand_matrix_1 = torch.rand(num_assets_example, num_assets_example)
    dummy_cov_matrix_1 = torch.matmul(rand_matrix_1, rand_matrix_1.transpose(0,1)) / num_assets_example
    dummy_cov_matrix_1 = dummy_cov_matrix_1 + 1e-4 * torch.eye(num_assets_example)
    all_returns.append(dummy_predicted_returns_1)
    all_covs.append(dummy_cov_matrix_1.unsqueeze(0))

    dummy_predicted_returns_2 = torch.tensor([[-0.01, -0.02, -0.005]], dtype=torch.float32)
    rand_matrix_2 = torch.rand(num_assets_example, num_assets_example) + 0.1
    dummy_cov_matrix_2 = torch.matmul(rand_matrix_2, rand_matrix_2.transpose(0,1)) / num_assets_example
    dummy_cov_matrix_2 = dummy_cov_matrix_2 + 1e-4 * torch.eye(num_assets_example)
    all_returns.append(dummy_predicted_returns_2)
    all_covs.append(dummy_cov_matrix_2.unsqueeze(0))

    dummy_predicted_returns_3 = torch.tensor([[10.0, 0.001, 0.001]], dtype=torch.float32)
    sigma_vals_3 = torch.tensor([0.5, 0.01, 0.01], dtype=torch.float32)
    dummy_cov_matrix_3 = torch.diag(sigma_vals_3) + 1e-4 * torch.eye(num_assets_example)
    all_returns.append(dummy_predicted_returns_3)
    all_covs.append(dummy_cov_matrix_3.unsqueeze(0))

    dummy_predicted_returns_4 = torch.tensor([[1e-7, 2e-7, 0.5e-7]], dtype=torch.float32)
    rand_matrix_4 = torch.rand(num_assets_example, num_assets_example) + 0.2
    dummy_cov_matrix_4 = torch.matmul(rand_matrix_4, rand_matrix_4.transpose(0,1)) / num_assets_example
    dummy_cov_matrix_4 = dummy_cov_matrix_4 + 1e-4 * torch.eye(num_assets_example)
    all_returns.append(dummy_predicted_returns_4)
    all_covs.append(dummy_cov_matrix_4.unsqueeze(0))
    
    dummy_predicted_returns_5 = torch.rand(1, num_assets_example, requires_grad=True) * 0.1 + 0.01
    rand_matrix_5 = torch.rand(num_assets_example, num_assets_example) + 0.3
    dummy_cov_matrix_5 = torch.matmul(rand_matrix_5, rand_matrix_5.transpose(0,1)) / num_assets_example
    dummy_cov_matrix_5 = dummy_cov_matrix_5 + 1e-4 * torch.eye(num_assets_example)
    all_returns.append(dummy_predicted_returns_5)
    all_covs.append(dummy_cov_matrix_5.unsqueeze(0))


    batched_predicted_returns = torch.cat(all_returns, dim=0).requires_grad_(True)
    batched_covariance_matrices = torch.cat(all_covs, dim=0)


    print(f"\nBatched predicted returns (mu) shape: {batched_predicted_returns.shape}")
    print(f"Batched covariance matrices (Sigma) shape: {batched_covariance_matrices.shape}")

    optimal_weights = None
    print("\n--- Running MVO Layer Forward Pass ---")
    try:
        optimal_weights = mvo_layer(batched_predicted_returns, batched_covariance_matrices)
    except Exception as e:
        print(f"!!! UNHANDLED EXCEPTION DURING MVO FORWARD PASS: {e}")
        import traceback
        traceback.print_exc()
    print("--- MVO Layer Forward Pass Complete ---")


    if optimal_weights is not None:
        print(f"\nOptimal weights (w = y/sum(y)):\n{optimal_weights.detach().numpy()}")
        print(f"Sum of weights per sample: {torch.sum(optimal_weights, dim=1).detach().numpy()}")
        print(f"Max weight per sample: {torch.max(optimal_weights, dim=1)[0].detach().numpy()}")
        print(f"Min weight per sample: {torch.min(optimal_weights, dim=1)[0].detach().numpy()}")
        print(f"Shape of optimal weights: {optimal_weights.shape}")

        if not batched_predicted_returns.requires_grad:
            batched_predicted_returns.requires_grad_(True)

        print("\n--- Running Gradient Test ---")
        optimal_weights_for_grad = mvo_layer(batched_predicted_returns, batched_covariance_matrices)

        target_dummy_weights = torch.ones_like(optimal_weights_for_grad) / num_assets_example
        loss = torch.sum((optimal_weights_for_grad - target_dummy_weights)**2)

        print(f"Dummy loss for gradient check: {loss.item()}")

        if batched_predicted_returns.grad is not None:
            batched_predicted_returns.grad.zero_()
        
        try:
            loss.backward()
            print(f"Gradients for predicted_returns (mu.grad):\n{batched_predicted_returns.grad}")
            if batched_predicted_returns.grad is not None:
                grad_sum_abs = torch.abs(batched_predicted_returns.grad).sum()
                print(f"Sum of absolute gradients: {grad_sum_abs.item()}")
                print(f"Are gradients non-zero? {'Yes' if grad_sum_abs > 1e-9 else 'No (or very small)'}")
            else:
                print("No gradients computed for predicted_returns.")
        except Exception as e:
            print(f"!!! ERROR DURING BACKWARD PASS: {e}")
            import traceback
            traceback.print_exc()
        print("--- Gradient Test Complete ---")
    else:
        print("Optimal weights calculation failed or was skipped.")
