import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_metrics(log_dir='./training_logs'):
    """
    Plots the training metrics from the training_metrics.csv file.
    """
    metrics_file = os.path.join(log_dir, 'training_metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return

    df = pd.read_csv(metrics_file)

    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('Training Metrics Evolution', fontsize=16)

    # Plot 1: PPO Losses
    axes[0].plot(df.index, df['policy_loss'], label='Policy Loss')
    axes[0].plot(df.index, df['value_loss'], label='Value Loss')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('PPO Losses')
    axes[0].grid(True)
    axes[0].legend()

    # Plot 2: SPO+ Loss
    axes[1].plot(df.index, df['total_spo_plus_loss'], label='Total SPO+ Loss', color='red')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('SPO+ Loss')
    axes[1].grid(True)
    axes[1].legend()

    # Plot 3: SPO+ Loss Components
    axes[2].stackplot(df.index,
                      df['spo_max_term_val_mean'],
                      df['spo_term_2_r_hat_w_star_c_mean'],
                      df['spo_term_r_true_w_star_c_mean'],
                      labels=['Max Term', '2*r_hat*w_star_c', 'r_true*w_star_c'],
                      alpha=0.7)
    axes[2].set_ylabel('Component Value')
    axes[2].set_title('SPO+ Loss Components')
    axes[2].grid(True)
    axes[2].legend(loc='upper left')

    plt.xlabel('Update Step')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = os.path.join(log_dir, "training_metrics_summary.png")
    plt.savefig(plot_filename)
    print(f"Training metrics plot saved to {plot_filename}")
    plt.close(fig)

if __name__ == '__main__':
    plot_training_metrics()
