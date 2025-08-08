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
    baseline_max_idx = np.argmax(baseline_y_values)
    baseline_x = baseline_x_values[baseline_max_idx]
    baseline_y = baseline_y_values[baseline_max_idx]

    exceeding_baseline_vals = []

    for x_val, y_val in zip(transfer_x_values, transfer_y_values):
        idx = np.argmax(y_val > baseline_y)

        if idx == 0:
            exceeding_baseline_vals.append(-1)
        else:
            exceeding_baseline_vals.append(x_val[idx])

    return baseline_x, exceeding_baseline_vals


def main():
    model_paths = [
        "./log_baseline_7x7",
        "./transfer_5x5_to_7x7"
    ]

    max_steps = get_max_step_from_first_event_file(model_paths, "rollout/ep_rew_mean")

    for model_path in model_paths:
        x_values, y_values = load_model_logs(model_path, "rollout/ep_rew_mean", max_steps=max_steps, use_running_max=False)

        average_max_reward = calculate_final_performance(y_values=y_values)









if __name__ == "__main__":
    main()

