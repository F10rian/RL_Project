import os
import numpy as np

import numpy as np
from scipy.integrate import quad
from print_stats import print_results
from evaluation_helper import (
    get_x_for_given_y,
    make_cubic_mean_function,
    extract_scalar_from_event, 
    get_max_step_from_first_event_file
)
from constants import ROOT_PATH


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


def calculate_sample_efficiency(mean_func):
    a, b = mean_func.x[0], mean_func.x[-1]
    auc, _ = quad(mean_func, a, b)
    return auc


def calculate_final_performance(mean_func):
    return mean_func.y.max()


def calculate_convergence_speed(baseline_func, transfer_funcs):
    x_grid = np.linspace(baseline_func.x.min(), baseline_func.x.max(), 1000)
    y_grid = baseline_func(x_grid)

    idx_max = np.argmax(y_grid)
    bl_x_at_max = int(x_grid[idx_max])
    bl_y_at_max = y_grid[idx_max]

    exceeding_baseline_vals = []

    for transfer_func in transfer_funcs:
        x_at_max = get_x_for_given_y(transfer_func, bl_y_at_max)
        if x_at_max is None:
            exceeding_baseline_vals.append(-1)
        else:
            exceeding_baseline_vals.append(int(x_at_max))

    return bl_x_at_max, bl_y_at_max, exceeding_baseline_vals


def main():

    baseline_model_path = f"{ROOT_PATH}/log_baseline_7x7"
    model_names = [
        "Baseline 7x7", 
        "Transfer (5x5 → 7x7)"]
    
    model_paths = [
        f"{ROOT_PATH}/log_transfer_5x5_to_7x7"
    ]
    max_steps = get_max_step_from_first_event_file(model_paths + [baseline_model_path], "rollout/ep_rew_mean")

    # Calculate performance of the baseline model
    baseline_x_values, baseline_y_values = load_model_logs(baseline_model_path, "rollout/ep_rew_mean", max_steps=max_steps, use_running_max=False)
    baseline_mean_function = make_cubic_mean_function(x=baseline_x_values, y=baseline_y_values)

    baseline_avg_max_reward = calculate_final_performance(mean_func=baseline_mean_function)
    baseline_mean_auc_reward = calculate_sample_efficiency(mean_func=baseline_mean_function)

    # Calculate performance of the transfer model
    avg_max_rewards = []
    mean_auc_rewards = []
    model_mean_functions = []

    for model_path in model_paths:
        x_values, y_values = load_model_logs(model_path, "rollout/ep_rew_mean", max_steps=max_steps, use_running_max=False)

        model_mean_function = make_cubic_mean_function(x=x_values, y=y_values)
        model_mean_functions.append(model_mean_function)

        avg_max_rewards.append(calculate_final_performance(mean_func=model_mean_function))
        mean_auc_rewards.append(calculate_sample_efficiency(mean_func=model_mean_function))

    # Calculate convergence speed
    baseline_x, baseline_y, exceeding_baseline_x_vals = calculate_convergence_speed(
        baseline_func=baseline_mean_function,
        transfer_funcs=model_mean_functions
    )
    print_results(
        model_names=model_names,
        average_max_reward=avg_max_rewards,
        mean_auc_reward=mean_auc_rewards,
        exceeding_baseline_x_vals=exceeding_baseline_x_vals,    
        baseline_x=baseline_x,
        baseline_y=baseline_y,
        baseline_avg_max_reward=baseline_avg_max_reward,
        baseline_mean_auc_reward=baseline_mean_auc_reward
    )
   

if __name__ == "__main__":
    main()

