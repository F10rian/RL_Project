from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu
from evaluation.evaluation import build_auc_distribution, get_max_step_from_first_event_file, load_model_logs


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
    metric = "rollout/ep_rew_mean"

    max_steps = get_max_step_from_first_event_file(path_1, path_2, metric)

    x_trunc_list_1, y_trunc_list_1 = load_model_logs(path_1, metric, max_steps=max_steps, use_running_max=False)
    x_trunc_list_2, y_trunc_list_2 = load_model_logs(path_2, metric, max_steps=max_steps, use_running_max=False)

    aucs_1 = build_auc_distribution(x_trunc_list=x_trunc_list_1, y_trunc_list=y_trunc_list_1)
    aucs_2 = build_auc_distribution(x_trunc_list=x_trunc_list_2, y_trunc_list=y_trunc_list_2)

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



if __name__ == "__main__":
    compute_mann_whitney_u_test("./log_baseline_7x7", "./transfer_5x5_to_7x7")

