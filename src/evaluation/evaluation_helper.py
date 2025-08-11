import os
import glob
import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import brentq
from tensorboard.backend.event_processing import event_accumulator


def make_cubic_mean_function(x, y, curve_kind="cubic", steps=200):
    min_x = max(xi[0] for xi in x)
    max_x = min(xi[-1] for xi in x)
    x_common = np.linspace(min_x, max_x, steps)

    values = make_cubic_function(x, y, curve_kind, steps)

    # calculate mean function
    mean_values = np.mean(values, axis=0)
    mean_function = interp1d(x_common, mean_values, kind=curve_kind)
    return mean_function


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
    return values


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