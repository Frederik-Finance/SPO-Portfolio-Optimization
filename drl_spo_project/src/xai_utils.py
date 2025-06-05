import numpy as np
import pandas as pd
import shap

def define_market_regimes(benchmark_price_series, window=60, quantiles=[0.33, 0.66], regime_names=['Low', 'Medium', 'High']):
    """
    """
    if not isinstance(benchmark_price_series, pd.Series):
        raise ValueError("benchmark_price_series must be a pandas Series.")
    if len(regime_names) != len(quantiles) + 1:
        raise ValueError("Length of regime_names must be len(quantiles) + 1.")

    rolling_vol = benchmark_price_series.pct_change().rolling(window=window).std().dropna()
    
    q_low = rolling_vol.quantile(quantiles[0])
    q_high = rolling_vol.quantile(quantiles[1])
    
    regime_labels = pd.Series(index=rolling_vol.index, dtype=str)
    regime_labels.loc[rolling_vol <= q_low] = regime_names[0]
    regime_labels.loc[(rolling_vol > q_low) & (rolling_vol <= q_high)] = regime_names[1]
    regime_labels.loc[rolling_vol > q_high] = regime_names[2]
    
    return regime_labels.reindex(benchmark_price_series.index).ffill().ffill()

def calculate_feature_importance(model, data_X, data_y, feature_names, method='permutation'):
    """
    """
    print(f"Placeholder: Calculating dummy feature importance (method '{method}' not yet implemented).")
    importances = np.random.rand(len(feature_names))
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
    return importance_df

def explain_with_shap(drl_actor_model, background_observations_tensor, observations_to_explain_tensor, feature_names, target_output_indices=None):
    """
    """
    print("Generating SHAP explanations for DRL actor model...")
    drl_actor_model.eval()

    try:
        explainer = shap.DeepExplainer(drl_actor_model, background_observations_tensor)
        
        shap_values = explainer.shap_values(observations_to_explain_tensor, check_additivity=False)

        print(f"SHAP values generated. Type: {type(shap_values)}")
        if isinstance(shap_values, list):
            print(f"  Number of model outputs explained: {len(shap_values)}")
            if shap_values:
                 print(f"  Shape of SHAP values for the first output: {shap_values[0].shape if hasattr(shap_values[0], 'shape') else 'N/A'}")
        elif hasattr(shap_values, 'shape'):
            print(f"  Shape of SHAP values (single output or aggregated): {shap_values.shape}")
        
        return shap_values

    except Exception as e:
        print(f"Error during SHAP explanation: {e}")
        print("Ensure SHAP library is correctly installed and compatible with the PyTorch model structure.")
        print("Falling back to dummy SHAP values as a placeholder.")
        num_explain_samples = observations_to_explain_tensor.shape[0]
        num_features = background_observations_tensor.shape[1]
        
        num_outputs = 1
        try:
            with torch.no_grad():
                 model_output_sample = drl_actor_model(observations_to_explain_tensor[:1])
                 if model_output_sample.ndim > 1:
                     num_outputs = model_output_sample.shape[1]
        except Exception as model_err:
            print(f"  Could not determine model output dimension for dummy SHAP: {model_err}. Assuming single output.")

        dummy_shap_values_list = [
            np.random.rand(num_explain_samples, num_features) for _ in range(num_outputs)
        ]
        if num_outputs == 1:
            return dummy_shap_values_list[0]
        return dummy_shap_values_list


def explain_with_lime(model_predict_function, data_X_instance, feature_names, num_features_to_explain=5):
    """
    """
    print("Placeholder: Generating dummy LIME explanation.")
    dummy_lime_explanation = sorted([(feature_names[i], np.random.rand()) for i in range(min(num_features_to_explain, len(feature_names)))], key=lambda x: x[1], reverse=True)
    return dummy_lime_explanation


if __name__ == '__main__':
    print("XAI Utilities (xai_utils.py) - Updated with define_market_regimes and SHAP integration.")

    print("\n--- Testing define_market_regimes ---")
    dates_regime = pd.to_datetime([f'2023-01-{d:02d}' for d in range(1, 31)]) \
                 + pd.to_timedelta(np.random.randint(1, 100, size=30), unit='D')
    dates_regime = dates_regime.sort_values()
    dummy_prices = pd.Series(np.random.rand(len(dates_regime)) * 10 + 100, index=dates_regime)
    
    test_window = min(10, len(dummy_prices) - 2)
    if test_window > 1 :
        regimes = define_market_regimes(dummy_prices, window=test_window)
        print("Market Regimes defined (first 5):")
        print(regimes.head())
        print("Regime counts:")
        print(regimes.value_counts())
    else:
        print("Skipping define_market_regimes test due to insufficient dummy data points for the window.")


    print("\n--- Testing explain_with_shap (conceptual) ---")
    import torch
    import torch.nn as nn

    class DummyActorModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(DummyActorModel, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)
        def forward(self, x):
            return self.linear(x)

    dummy_input_dim = 10
    dummy_output_dim = 3
    dummy_actor = DummyActorModel(dummy_input_dim, dummy_output_dim)

    dummy_bg_obs = torch.rand(20, dummy_input_dim)
    dummy_explain_obs = torch.rand(3, dummy_input_dim)
    dummy_feature_names = [f'feature_{i}' for i in range(dummy_input_dim)]

    shap_output = explain_with_shap(dummy_actor, dummy_bg_obs, dummy_explain_obs, dummy_feature_names)

    if isinstance(shap_output, list):
        print(f"SHAP output is a list with {len(shap_output)} item(s) (typically one per model output).")
        if shap_output:
            print(f"  Shape of SHAP values for the first output: {shap_output[0].shape}")
    elif hasattr(shap_output, 'shape'):
        print(f"SHAP output is an array with shape: {shap_output.shape}")


    print("\n--- Testing other XAI placeholders ---")
    dummy_model_for_fi = lambda x: np.sum(x, axis=1)
    fi_output = calculate_feature_importance(dummy_model_for_fi, dummy_bg_obs.numpy(), np.random.rand(20), dummy_feature_names)
    print(f"Feature importance output (head):\n{fi_output.head(2)}")

    lime_output = explain_with_lime(lambda x: np.sum(x, axis=1, keepdims=True), dummy_explain_obs[0].numpy(), dummy_feature_names)
    print(f"LIME output (example):\n{lime_output[:2]}")
