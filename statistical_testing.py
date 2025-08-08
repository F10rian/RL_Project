
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os
import numpy as np
from scipy.stats import mannwhitneyu

def extract_scalar_from_event(file_path, tag):
    ea = event_accumulator.EventAccumulator(file_path)
    ea.Reload()
    scalars = ea.Scalars(tag)
    return [s.value for s in scalars]


def build_auc_distribution_from_logs(
    logs_dir: str,
    scalar_tag: str,
    episode_tag: str = None  # Optional: x-axis values, e.g., episode number
):
    """
    Computes AUC of scalar_tag over episodes for each run in logs_dir.

    Parameters:
    - logs_dir: str, directory with subfolders (one per run/seed)
    - scalar_tag: str, name of the reward metric in TensorBoard (e.g., 'eval/episode_reward')
    - episode_tag: str (optional), tag for the x-axis values (e.g., episode number). If None, use step field.

    Returns:
    - List[float]: AUC values for each run
    """
    aucs = []
    max_steps = 0

    for root, _, files in os.walk(logs_dir):
        event_files = [f for f in files if f.startswith("events.out.tfevents")]
        if not event_files:
            continue

        event_file = os.path.join(root, event_files[0])

        try:
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()

            scalar_events = ea.Scalars(scalar_tag)
            if not scalar_events:
                continue

            y = [e.value for e in scalar_events]

            x = [e.step for e in scalar_events]

            max_steps = max(x)

            if len(x) != len(y):
                continue

            auc = np.trapezoid(y, x)
            aucs.append(auc)

        except Exception as e:
            print(f"Failed to process {event_file}: {e}")

    return aucs


def compute_mann_whitney_u_test(path_1, path_2):
    """
    Computes the Mann-Whitney U test between two sets of AUC values.

    Parameters:
    - path_1: str, path to the first set of logs
    - path_2: str, path to the second set of logs

    Returns:
    - float: U statistic
    - float: p-value
    """
    aucs_1 = build_auc_distribution_from_logs(path_1, 'rollout/ep_rew_mean')
    aucs_2 = build_auc_distribution_from_logs(path_2, 'rollout/ep_rew_mean')

    if not aucs_1 or not aucs_2:
        raise ValueError("One or both paths do not contain valid AUC data.")

    u_statistic, p_value = mannwhitneyu(aucs_1, aucs_2, alternative='two-sided')

    print(f"Mann–Whitney U-statistic: {u_statistic:.3f}")
    print(f"P-value: {p_value:.4f}")

    plt.boxplot([aucs_1, aucs_2], labels=["Pretrain", "Transfer"])
    plt.ylabel("AUC of Episode Reward")
    plt.title("AUC Distribution per Run")
    plt.grid(True)
    plt.show()
    
    return u_statistic, p_value


compute_mann_whitney_u_test("./log_baseline_7x7", "./transfer_5x5_to_7x7")