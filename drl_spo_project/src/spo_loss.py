import torch
import torch.nn as nn
import numpy as np
try:
    from .spo_layer import DifferentiableMVO
except ImportError:
    from spo_layer import DifferentiableMVO


class SPOPlusLoss(nn.Module):
    def __init__(self, num_assets, mvo_weight_bounds=(0,1), mvo_risk_aversion=None, mvo_target_return=None, mvo_max_weight_per_asset=1.0):
        """
        """
        super(SPOPlusLoss, self).__init__()
        self.num_assets = num_assets
        
        self.mvo_solver = DifferentiableMVO(
            num_assets=num_assets, 
            max_weight_per_asset=mvo_max_weight_per_asset
        )

    def forward(self, predicted_returns_c_hat, true_returns_c, covariance_matrix):
        """
        """
        if predicted_returns_c_hat.ndim == 1:
            predicted_returns_c_hat = predicted_returns_c_hat.unsqueeze(0)
        if true_returns_c.ndim == 1:
            true_returns_c = true_returns_c.unsqueeze(0)
        
        batch_size_pred = predicted_returns_c_hat.shape[0]
        if covariance_matrix.ndim == 2: 
            covariance_matrix = covariance_matrix.unsqueeze(0).repeat(batch_size_pred, 1, 1)
        elif covariance_matrix.shape[0] == 1 and batch_size_pred > 1:
            covariance_matrix = covariance_matrix.repeat(batch_size_pred, 1, 1)
        elif covariance_matrix.shape[0] != batch_size_pred:
            raise ValueError(f"Covariance matrix batch size {covariance_matrix.shape[0]} "
                             f"does not match predicted returns batch size {batch_size_pred} and cannot be broadcast.")

        with torch.no_grad(): 
            w_star_c = self.mvo_solver(true_returns_c, covariance_matrix).detach()

        r_hat = predicted_returns_c_hat
        r_true = true_returns_c

        effective_mu_for_max_term = -r_true + 2 * r_hat
        w_for_max_term = self.mvo_solver(effective_mu_for_max_term, covariance_matrix)
        max_term_val = torch.sum(effective_mu_for_max_term * w_for_max_term, dim=1)

        term_2_r_hat_w_star_c = -2 * torch.sum(r_hat * w_star_c, dim=1)
        
        term_r_true_w_star_c_contribution = torch.sum(r_true * w_star_c, dim=1)
        
        spo_plus_loss_per_item = max_term_val + term_2_r_hat_w_star_c + term_r_true_w_star_c_contribution
        
        final_loss = spo_plus_loss_per_item.mean()

        spo_component_means = {
           'spo_max_term_val_mean': max_term_val.mean().item(),
           'spo_term_2_r_hat_w_star_c_mean': term_2_r_hat_w_star_c.mean().item(),
           'spo_term_r_true_w_star_c_mean': term_r_true_w_star_c_contribution.mean().item()
        }

        return final_loss, spo_component_means

if __name__ == '__main__':
    print("SPO+ Loss (spo_loss.py) - Testing with Functional MVO Layer and Component Return")
    
    num_assets_example = 3 
    batch_s = 2

    mvo_max_weight = 0.6
    try:
        spo_loss_fn = SPOPlusLoss(num_assets=num_assets_example, mvo_max_weight_per_asset=mvo_max_weight)
        print(f"SPOPlusLoss instantiated successfully with MVO max_weight_per_asset={mvo_max_weight}.")
    except Exception as e:
        print(f"Error instantiating SPOPlusLoss: {e}")
        import traceback
        traceback.print_exc()
        exit()

    dummy_predicted_returns = torch.rand(batch_s, num_assets_example, requires_grad=True) + 0.01 
    dummy_true_returns = torch.rand(batch_s, num_assets_example) + 0.005

    dummy_cov_list = []
    for _ in range(batch_s):
        rand_m = torch.rand(num_assets_example, num_assets_example)
        cov_m = torch.matmul(rand_m, rand_m.transpose(0,1)) / num_assets_example 
        cov_m = cov_m + 1e-3 * torch.eye(num_assets_example)
        dummy_cov_list.append(cov_m.unsqueeze(0))
    dummy_covariance_matrices_batched = torch.cat(dummy_cov_list, dim=0)

    print(f"\nDummy predicted_returns_c_hat:\n{dummy_predicted_returns}")
    print(f"Dummy true_returns_c:\n{dummy_true_returns}")

    spo_loss_value, components = None, None

    print("\n--- Testing SPOPlusLoss Forward Pass ---")
    try:
        spo_loss_value, components = spo_loss_fn(dummy_predicted_returns, dummy_true_returns, dummy_covariance_matrices_batched)
        print(f"\nCalculated SPO+ Loss value: {spo_loss_value.item()}")
        print(f"Returned SPO+ Loss Components (means): {components}")

        assert spo_loss_value.ndim == 0, "SPO+ Loss should be a scalar tensor."
        assert isinstance(components, dict), "Components should be a dictionary."
        assert 'spo_max_term_val_mean' in components, "A key component is missing in the returned dictionary."

        print("\n--- Testing SPOPlusLoss Backward Pass ---")
        if dummy_predicted_returns.grad is not None:
            dummy_predicted_returns.grad.zero_()
        spo_loss_value.backward()
        print("Backward pass successful.")

        if dummy_predicted_returns.grad is not None:
            print(f"Gradients for predicted_returns_c_hat (first sample):\n{dummy_predicted_returns.grad[0]}")
            if torch.all(torch.abs(dummy_predicted_returns.grad) < 1e-10):
                print("Warning: Gradients are all zero or very close to zero. This might indicate an issue.")
            else:
                print("Gradients have flowed and are non-zero (checked first sample).")
        else:
            print("Error: No gradients computed for predicted_returns_c_hat after backward pass.")

    except Exception as e:
        print(f"!!! Error during SPOPlusLoss forward or backward pass: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n--- SPO+ Loss Test Finished ---")
