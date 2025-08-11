import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tbparse import SummaryReader
from evaluation_helper import make_cubic_function


def add_reward_plot(log_folders, label, color=None, alpha=0.3):
    """
    Add reward data from multiple log folders to the current plot.
    
    Args:
        log_folders: List of paths to tensorboard log folders
        label: Label for this dataset in the legend
        color: Color for the mean line and band (auto-selected if None)
        alpha: Transparency of the min-max band
    """
    existing_folders = [folder for folder in log_folders if os.path.exists(folder)]
    print(f"Found {len(existing_folders)} log folders for '{label}': {existing_folders}")

    if not existing_folders:
        print(f"No log folders found for '{label}'! Please check the paths.")
        return

    # Collect all reward data
    all_rewards = []
    all_steps = []

    for folder in existing_folders:
        try:
            reader = SummaryReader(folder)
            df = reader.scalars
            reward_df = df[df["tag"] == "rollout/ep_rew_mean"]
            
            if not reward_df.empty:
                all_rewards.append(reward_df["value"].values)
                all_steps.append(reward_df["step"].values)
                print(f"Loaded data from {folder}: {len(reward_df)} points")
            else:
                print(f"No reward data found in {folder}")
        except Exception as e:
            print(f"Error reading {folder}: {e}")

    values = make_cubic_function(all_steps, all_rewards, curve_kind="cubic", steps=200)
    print(values.shape)

    common_steps = np.linspace(1, 100001, 200)  # 1000 points to match values.shape

    # Convert to numpy array for easier calculation
    reward_matrix = np.array(values)

    # Calculate statistics
    mean_rewards = np.mean(reward_matrix, axis=0)
    min_rewards = np.min(reward_matrix, axis=0)
    max_rewards = np.max(reward_matrix, axis=0)

    # Plot with matching colors for band and mean line
    plt.fill_between(common_steps, min_rewards, max_rewards, alpha=alpha, 
                     color=color)  # Remove label to exclude from legend
    plt.plot(common_steps, mean_rewards, linewidth=2, color=color,
             label=f'{label} (n={len(all_rewards)})')

def setup_plot():
    """Setup the plot with proper formatting."""
    plt.figure(figsize=(12, 8))
    # Increase font sizes globally
    plt.rcParams.update({'font.size': 20})  # Base font size
    plt.rcParams.update({'axes.titlesize': 28})  # Title font size
    plt.rcParams.update({'axes.labelsize': 24})  # Axis label font size
    plt.rcParams.update({'xtick.labelsize': 20})  # X-axis tick font size
    plt.rcParams.update({'ytick.labelsize': 20})  # Y-axis tick font size
    plt.rcParams.update({'legend.fontsize': 20})  # Legend font size
    plt.rcParams['figure.dpi'] = 300

def finalize_plot(title="Reward Progress Comparison"):
    """Finalize the plot with labels, legend, and formatting."""
    # Set axis limits to ensure 0,0 is at the intersection of axes
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.xlabel("Timesteps", fontsize=24)
    plt.ylabel("Mean Episode Reward", fontsize=24)
    plt.title(title, fontsize=28)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)

# Example usage:
if __name__ == "__main__":
    # Setup the plot
    setup_plot()

    parser = argparse.ArgumentParser(description="Pfad-Name-Paare einlesen")

    parser.add_argument(
        'entries',
        nargs='+',
        help='Enter paths to the log folders followed by their labels, e.g. path1 label1 path2 label2 ...'
    )
    args = parser.parse_args()
    entries = args.entries

    # Check if even number of entries
    if len(entries) % 2 != 0:
        print("Every path must have exactly one name associated with it.")
        sys.exit(1)

    # Create pairs
    colors =[ 'green', 'blue', 'orange', 'gray', 'purple', 'red', 'cyan', 'magenta', 'yellow', 'black']
    # cut colors to the number of entries
    if len(entries) // 2 < len(colors):
        colors = colors[:len(entries) // 2]
    elif len(entries) // 2 > len(colors):
        colors = colors * (len(entries) // 2 // len(colors) + 1)
    # Create pairs of (path, name, color)
    path_name_pairs = list(zip(entries[::2], entries[1::2], colors))

    for path, name, color in path_name_pairs:
        # get all the subfolders except zip folders from the path
        log_folders = [os.path.join(path, subfolder) for subfolder in os.listdir(path) if os.path.isdir(os.path.join(path, subfolder)) and not subfolder.endswith('.zip')]
        add_reward_plot(log_folders, name, color=color)
    
    # Finalize and show the plot
    finalize_plot("Transfer Learning: 5x5 to 7x7 Crossing")
    plt.savefig("images/transfer_learning_5x5_to_7x7_mean.svg", format="svg", bbox_inches="tight")
    plt.show()