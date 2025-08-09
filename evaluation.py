import glob
import os
import numpy as np

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import brentq

from tensorboard.backend.event_processing import event_accumulator

from plot_stats import print_results


def make_cubic_function(x, y, curve_kind="cubic", steps=200):
    functions = []
    for xi, yi in zip(x, y):
        func = interp1d(xi, yi, kind=curve_kind)
        functions.append(func)

    min_x = max(xi[0] for xi in x)
    max_x = min(xi[-1] for xi in x)
    x_common = np.linspace(min_x, max_x, steps)

    # calculate mean values
    values = np.vstack([func(x_common) for func in functions])
    mean_values = np.mean(values, axis=0)

    # calculate mean function
    mean_function = interp1d(x_common, mean_values, kind=curve_kind)
    return mean_function


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


def calculate_sample_efficiency(mean_func):
    a, b = mean_func.x[0], mean_func.x[-1]
    auc, _ = quad(mean_func, a, b)
    return auc


def calculate_final_performance(mean_func):
    return mean_func.y.max()


def get_x_for_given_y(f, y_target):
    x_data = np.asarray(f.x)
    y_data = np.asarray(f.y)

    # Case 1: never reaches target
    if np.all(y_data < y_target):
        return None

    # Case 2: already above/equal at start
    if y_data[0] >= y_target:
        return float(x_data[0])

    # Case 3: find first crossing interval
    for i in range(1, len(y_data)):
        if y_data[i] >= y_target:
            x_low, x_high = x_data[i-1], x_data[i]
            g = lambda x: f(x) - y_target
            return float(brentq(g, x_low, x_high))

    return None



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
    model_names = ["Baseline 7x7", "Transfer (5x5 → 7x7)"]
    model_paths = [
        "./transfer_5x5_to_7x7"
    ]
    baseline_model_path = "./log_baseline_7x7"

    max_steps = get_max_step_from_first_event_file(model_paths + [baseline_model_path], "rollout/ep_rew_mean")

    baseline_x_values, baseline_y_values = load_model_logs(baseline_model_path, "rollout/ep_rew_mean", max_steps=max_steps, use_running_max=False)

    baseline_mean_function = make_cubic_function(x=baseline_x_values, y=baseline_y_values)

    baseline_avg_max_reward = calculate_final_performance(mean_func=baseline_mean_function)
    baseline_mean_auc_reward = calculate_sample_efficiency(mean_func=baseline_mean_function)
    avg_max_rewards = []
    mean_auc_rewards = []

    model_mean_functions = []

    for model_path in model_paths:
        x_values, y_values = load_model_logs(model_path, "rollout/ep_rew_mean", max_steps=max_steps, use_running_max=False)

        model_mean_function = make_cubic_function(x=x_values, y=y_values)
        model_mean_functions.append(model_mean_function)

        avg_max_rewards.append(calculate_final_performance(mean_func=model_mean_function))
        mean_auc_rewards.append(calculate_sample_efficiency(mean_func=model_mean_function))

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


