def print_results(model_names, average_max_reward, mean_auc_reward, exceeding_baseline_x_vals, baseline_x, baseline_y, baseline_avg_max_reward, baseline_mean_auc_reward):

    # Print header
    print("\nModel Comparison Table")
    print("| Model                    | Final Performance (avg max reward) | Sample Efficiency (mean AUC reward) | Exceeding Baseline (x where reward > baseline y) |")
    print("|--------------------------|------------------------------------|-------------------------------------|--------------------------------------------------|")

    # Print baseline row
    print(f"| {model_names[0]}           | {baseline_avg_max_reward:.2f}                          | {baseline_mean_auc_reward:.2f}                          | Threshold: {baseline_y:.2f} @ step {baseline_x:.0f}           |")

    # Print each transfer model row (in case of multiple in future)
    for i in range(len(model_names) - 1):
        exceed_str = f"{exceeding_baseline_x_vals[i]}"
        print(f"| {model_names[i+1]:<25} | {average_max_reward[i]:.2f}                          | {mean_auc_reward[i]:.2f}                          | {exceed_str} |")
