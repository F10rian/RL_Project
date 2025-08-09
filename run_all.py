import subprocess
import sys

# List of scripts to run in order
commands = [
    [
        "test2.py",
        "--mode", "finetune_sweep",
        "--env", "MiniGrid-Crossing-7x7-v0",
        "--tensorboard_log", "log_transfer_5x5_to_7x7",
        "--batch_size", "512",
        "--buffer_size", "1_000",
        "--lr", "5e-4",
        "--exp_init_eps", "1.0",
        "--exp_fraction", "0.8",
        "--model_path", "log_baseline_5x5/MiniGrid-Crossing-5x5-v0",
        "--steps", "10_000",
        "--verbose", "1"
    ]
]


"""[
        "python", "test2.py",
        "--mode", "train",
        "--env", "MiniGrid-Crossing-5x5-v0",
        "--tensorboard_log", "log_baseline_5x5",
        "--num_models", "20",
        "--batch_size", "512",
        "--buffer_size", "100_000",
        "--lr", "5e-4",
        "--exp_init_eps", "1.0",
        "--exp_fraction", "0.8",
        "--verbose", "0"
    ],
    [
        "python", "test2.py",
        "--mode", "train",
        "--env", "MiniGrid-Crossing-7x7-v0",
        "--tensorboard_log", "log_baseline_7x7",
        "--num_models", "20",
        "--batch_size", "512",
        "--buffer_size", "100_000",
        "--lr", "5e-4",
        "--exp_init_eps", "1.0",
        "--exp_fraction", "0.8",
        "--verbose", "0"
    ],
    [
        "python", "test2.py",
        "--mode", "finetune_sweep",
        "--env", "MiniGrid-Crossing-7x7-v0",
        "--tensorboard_log", "log_transfer_5x5_to_7x7",
        "--num_models", "20",
        "--batch_size", "512",
        "--buffer_size", "100_000",
        "--lr", "1e-4",
        "--exp_init_eps", "0.5",
        "--exp_fraction", "0.8",
        "--verbose", "0"
    ],
    [
        "python", "test2.py",
        "--mode", "train",
        "--env", "MiniGrid-Crossing-9x9-v0",
        "--tensorboard_log", "log_baseline_9x9",
        "--num_models", "20",
        "--batch_size", "512",
        "--buffer_size", "100_000",
        "--lr", "5e-4",
        "--exp_init_eps", "1.0",
        "--exp_fraction", "0.8",
        "--verbose", "0"
    ],
    ]"""

python_exe = sys.executable
for script in commands:
    print(f"Running {script}...")
    script_with_venv = [python_exe] + script
    subprocess.run(script_with_venv, check=True)
    print(f"{script} finished.\n")

print("All scripts completed.")