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

    # Align all data to same step count
    aligned_rewards = []
    common_steps = all_steps[0][:min_steps]

    for i, rewards in enumerate(all_rewards):
        aligned_rewards.append(rewards[:min_steps])

    # Convert to numpy array for easier calculation
    reward_matrix = np.array(aligned_rewards)

    # Calculate statistics
    mean_rewards = np.mean(reward_matrix, axis=0)
    min_rewards = np.min(reward_matrix, axis=0)
    max_rewards = np.max(reward_matrix, axis=0)

    # Plot with matching colors for band and mean line
    plt.fill_between(common_steps, min_rewards, max_rewards, alpha=alpha, 
                     color=color)  # Remove label to exclude from legend
    plt.plot(common_steps, mean_rewards, linewidth=2, color=color,
             label=f'{label} Mean (n={len(aligned_rewards)})')

def setup_plot():
    """Setup the plot with proper formatting."""
    plt.figure(figsize=(12, 8))

def finalize_plot(title="Reward Progress Comparison"):
    """Finalize the plot with labels, legend, and formatting."""
    # Set axis limits to ensure 0,0 is at the intersection of axes
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Mean Episode Reward", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

# Example usage:
if __name__ == "__main__":
    # Setup the plot
    setup_plot()
    
    # Add first dataset
    log_folders_1 = ["dqn_crossing_tensorboard/5x5_to_7x7_after_40k_steps", 
                     "dqn_crossing_tensorboard/5x5_to_7x7_after_80k_steps", 
                     "dqn_crossing_tensorboard/5x5_to_7x7_after_120k_steps",
                     "dqn_crossing_tensorboard/5x5_to_7x7_after_160k_steps",
                     "dqn_crossing_tensorboard/5x5_to_7x7_after_200k_steps"]
    
    add_reward_plot(log_folders_1, "5x5 to 7x7 Transfer", color='blue')
    
    # You can add more datasets like this:
    # log_folders_2 = ["other_experiment/run1", "other_experiment/run2"]
    # add_reward_plot(log_folders_2, "Other Experiment", color='blue')
    
    # Finalize and show the plot
    finalize_plot("Transfer Learning: 5x5 to 7x7 Crossing")
    plt.show()