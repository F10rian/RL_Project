from tbparse import SummaryReader
import matplotlib.pyplot as plt
import numpy as np
import os

def add_reward_plot(log_folders, label, color=None, alpha=0.3):
    """
    Add reward data from multiple log folders to the current plot.
    
    Args:
        log_folders: List of paths to tensorboard log folders
        label: Label for this dataset in the legend
        color: Color for the mean line and band (auto-selected if None)
        alpha: Transparency of the min-max band
    """
    # Check which folders exist
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

    if not all_rewards:
        print(f"No reward data found in any folder for '{label}'!")
        return

    # Find common step range
    min_steps = min(len(steps) for steps in all_steps)
    print(f"Using first {min_steps} steps for analysis of '{label}'")

    # Align all data to same step count and calculate running max
    aligned_running_max = []
    common_steps = all_steps[0][:min_steps]

    for i, rewards in enumerate(all_rewards):
        # Calculate running maximum for this run
        running_max = np.maximum.accumulate(rewards[:min_steps])
        aligned_running_max.append(running_max)

    # Convert to numpy array for easier calculation
    running_max_matrix = np.array(aligned_running_max)

    # Calculate statistics on running maxima
    mean_running_max = np.mean(running_max_matrix, axis=0)
    min_running_max = np.min(running_max_matrix, axis=0)
    max_running_max = np.max(running_max_matrix, axis=0)

    # Plot with matching colors for band and mean line
    plt.fill_between(common_steps, min_running_max, max_running_max, alpha=alpha, 
                     color=color)  # Remove label to exclude from legend
    plt.plot(common_steps, mean_running_max, linewidth=2, color=color,
             label=f'{label} Best Performance (n={len(aligned_running_max)})')

def setup_plot():
    """Setup the plot with proper formatting."""
    plt.figure(figsize=(12, 8))

def finalize_plot(title="Best Performance Comparison"):
    """Finalize the plot with labels, legend, and formatting."""
    # Set axis limits to ensure 0,0 is at the intersection of axes
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Best Episode Reward (Running Max)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

# Example usage:
if __name__ == "__main__":
    # Setup the plot
    setup_plot()
    
    # Add first dataset
    log_folders_1 = ["log_baseline/DQN_1", 
                     "log_baseline/DQN_2",
                     "log_baseline/DQN_3",
                     "log_baseline/DQN_4",
                     "log_baseline/DQN_5",
                     "log_baseline/DQN_6",
                     "log_baseline/DQN_7",
                     "log_baseline/DQN_8",
                     "log_baseline/DQN_9",
                     "log_baseline/DQN_10",
                     "log_baseline/DQN_11",
                     "log_baseline/DQN_12",
                     "log_baseline/DQN_13",
                     "log_baseline/DQN_14",
                     "log_baseline/DQN_15",
                     "log_baseline/DQN_16",
                     "log_baseline/DQN_17",
                     "log_baseline/DQN_18",
                     "log_baseline/DQN_19",
                     "log_baseline/DQN_20",]
    
    add_reward_plot(log_folders_1, "5x5 baseline", color='green')

    log_folders_2 = ["log_baseline_7x7/DQN_1", 
                    "log_baseline_7x7/DQN_2",  
                    "log_baseline_7x7/DQN_3",
                    "log_baseline_7x7/DQN_4",
                    "log_baseline_7x7/DQN_5",
                    "log_baseline_7x7/DQN_6",
                    "log_baseline_7x7/DQN_7",
                    "log_baseline_7x7/DQN_8",
                    "log_baseline_7x7/DQN_9",
                    "log_baseline_7x7/DQN_10",
                    "log_baseline_7x7/DQN_11",
                    "log_baseline_7x7/DQN_12",
                    "log_baseline_7x7/DQN_13",
                    "log_baseline_7x7/DQN_14",
                    "log_baseline_7x7/DQN_15",
                    "log_baseline_7x7/DQN_16",
                    "log_baseline_7x7/DQN_17",
                    "log_baseline_7x7/DQN_18",
                    "log_baseline_7x7/DQN_19",
                    "log_baseline_7x7/DQN_20",                   
                     ]
    
    add_reward_plot(log_folders_2, "7x7 baseline", color='blue')

    log_folders_3 = ["transfer_5x5_to_7x7/DQN_1", 
                    "transfer_5x5_to_7x7/DQN_2",
                    "transfer_5x5_to_7x7/DQN_3",
                    "transfer_5x5_to_7x7/DQN_4",
                    "transfer_5x5_to_7x7/DQN_5",
                    "transfer_5x5_to_7x7/DQN_6",
                    "transfer_5x5_to_7x7/DQN_7",
                    "transfer_5x5_to_7x7/DQN_8",
                    "transfer_5x5_to_7x7/DQN_9",
                    "transfer_5x5_to_7x7/DQN_10",
                    "transfer_5x5_to_7x7/DQN_11",
                    "transfer_5x5_to_7x7/DQN_12",
                    "transfer_5x5_to_7x7/DQN_13",
                    "transfer_5x5_to_7x7/DQN_14",
                    "transfer_5x5_to_7x7/DQN_15",
                    "transfer_5x5_to_7x7/DQN_16",
                    "transfer_5x5_to_7x7/DQN_17",
                    "transfer_5x5_to_7x7/DQN_18",
                    "transfer_5x5_to_7x7/DQN_19",
                    "transfer_5x5_to_7x7/DQN_20",                   
                                     
                     ]
    
    add_reward_plot(log_folders_3, "transfer", color='gray')

    
    
    # You can add more datasets like this:
    # log_folders_2 = ["other_experiment/run1", "other_experiment/run2"]
    # add_reward_plot(log_folders_2, "Other Experiment", color='blue')
    
    # Finalize and show the plot
    finalize_plot("Transfer Learning: 5x5 to 7x7 Crossing")
    plt.show()