import glob
import os
import numpy as np

from tensorboard.backend.event_processing import event_accumulator


def get_max_step_from_first_event_file(paths, metric) -> int:
    x_max_list = []
    for path in paths:
        event_file = glob.glob(os.path.join(f"{path}/DQN_1", "events.out.tfevents.*"))[0]

        # Check the min max_steps from files
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        scalar_events_1 = ea.Scalars(metric)
        x = max([e.step for e in scalar_events_1])
        x_max_list.append(x)

    max_steps = min(x_max_list)

    return max_steps


def extract_scalar_from_event(event_file, scalar_tag):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    scalar_events = ea.Scalars(scalar_tag)
    if not scalar_events:
        return None, None

    x = [e.step for e in scalar_events]
    y = [e.value for e in scalar_events]
    return x, y


def load_model_logs(
    logs_dir: str,
    scalar_tag: str,
    max_steps: int,
    use_running_max: bool = False,
):
    x_trunc_list = []
    y_trunc_list = []

    for root, _, files in os.walk(logs_dir):
        event_files = [f for f in files if f.startswith("events.out.tfevents")]
        if not event_files:
            continue

        event_file = os.path.join(root, event_files[0])

        try:
            x, y = extract_scalar_from_event(event_file, scalar_tag)

            if x is None or y is None:
                continue

            # Truncate to max_steps
            x_trunc = []
            y_trunc = []
            for xi, yi in zip(x, y):
                if xi <= max_steps:
                    x_trunc.append(xi)
                    y_trunc.append(yi)

            if len(x_trunc) < 2:
                continue

            if use_running_max:
                y_trunc = np.maximum.accumulate(y_trunc)
            
            x_trunc_list.append(x_trunc)
            y_trunc_list.append(y_trunc)

        except Exception as e:
            print(f"Failed to process {event_file}: {e}")
    return x_trunc_list, y_trunc_list


def build_auc_distribution(
    x_values,
    y_values
):
    """
    Computes truncated AUC over step range [0, max_steps] for each run.
    """
    aucs = []
    for x_trunc, y_trunc in zip(x_values, y_values):
        auc = np.trapezoid(y_trunc, x_trunc)
        aucs.append(auc)
    return aucs


def calculate_sample_efficiency(x_values, y_values):
    aucs = build_auc_distribution(x_values=x_values, y_values=y_values)
    return np.mean(aucs)


def calculate_final_performance(y_values):
    max_rewards = []
    for y in y_values:
        max_rewards.append(np.max(y))
    return np.mean(max_rewards)


def calculate_convergence_speed(baseline_x_values, baseline_y_values, transfer_x_values, transfer_y_values):
    mean_baseline_y_values = np.mean(np.array(baseline_y_values), axis=0)
    mean_baseline_x_values = np.mean(np.array(baseline_x_values), axis=0)
    
    baseline_max_idx = np.argmax(mean_baseline_y_values)
    baseline_x = mean_baseline_x_values[baseline_max_idx]
    baseline_y = mean_baseline_y_values[baseline_max_idx]

    exceeding_baseline_vals = []

    for x_values, y_values in zip(transfer_x_values, transfer_y_values):
        mean_x_values = np.mean(x_values)
        mean_y_values = np.mean(y_values)


        idx = np.argmax(mean_y_values > baseline_y)

        if idx == 0:
            exceeding_baseline_vals.append(-1)
        else:
            exceeding_baseline_vals.append(mean_x_values[idx])

    return baseline_x, baseline_y, exceeding_baseline_vals


def main():
    model_names = ["Baseline 7x7", "Transfer (5x5 → 7x7)"]
    model_paths = [
        "./transfer_5x5_to_7x7"
    ]
    baseline_model_path = "./log_baseline_7x7"

    max_steps = get_max_step_from_first_event_file(model_paths + [baseline_model_path], "rollout/ep_rew_mean")

    baseline_x_values, baseline_y_values = load_model_logs(baseline_model_path, "rollout/ep_rew_mean", max_steps=max_steps, use_running_max=False)

    baseline_avg_max_reward = calculate_final_performance(y_values=baseline_y_values)
    baseline_mean_auc_reward = calculate_sample_efficiency(x_values=baseline_x_values, y_values=baseline_y_values)
    avg_max_rewards = []
    mean_auc_rewards = []

    for model_path in model_paths:
        x_values, y_values = load_model_logs(model_path, "rollout/ep_rew_mean", max_steps=max_steps, use_running_max=False)

        avg_max_rewards.append(calculate_final_performance(y_values=y_values))
        mean_auc_rewards.append(calculate_sample_efficiency(x_values=x_values, y_values=y_values))

    baseline_x, baseline_y, exceeding_baseline_x_vals = calculate_convergence_speed(
        baseline_x_values=baseline_x_values,
        baseline_y_values=baseline_y_values,
        transfer_x_values=x_values,
        transfer_y_values=y_values
    )
    print_results(model_names, avg_max_rewards, mean_auc_rewards, exceeding_baseline_x_vals, baseline_x, baseline_y, baseline_avg_max_reward, baseline_mean_auc_reward)
    



def print_results(model_names, average_max_reward, mean_auc_reward, exceeding_baseline_x_vals, baseline_x, baseline_y, baseline_avg_max_reward, baseline_mean_auc_reward):

    # Print header
    print("\n📊 Model Comparison Table")
    print("| Model                    | Final Performance (avg max reward) | Sample Efficiency (mean AUC reward) | Exceeding Baseline (x where reward > baseline y) |")
    print("|--------------------------|------------------------------------|-------------------------------------|--------------------------------------------------|")

    # Print baseline row
    print(f"| Baseline (7x7)           | {baseline_avg_max_reward:.2f}                          | {baseline_mean_auc_reward:.2f}                          | Threshold: {baseline_y:.2f} @ step {baseline_x:.0f}           |")

    # Print each transfer model row (in case of multiple in future)
    for model_name, avg_max, auc, exceed_vals in zip(
        model_names,
        average_max_reward,
        mean_auc_reward,
        exceeding_baseline_x_vals
    ):
        exceed_str = ", ".join([f"{x:.0f}" if x != -1 else "N/A" for x in exceed_vals])
        print(f"| {model_name:<25} | {avg_max:.2f}                          | {auc:.2f}                          | {exceed_str} |")






if __name__ == "__main__":
    main()

