import gymnasium as gym
#import mimicrIEs
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from render_callback import checkpoint_callback

from envs import make_env, register_envs
from learning import curriculum_learning, transfer_weights_cnn, fine_tune_from_checkpoints, fine_tune_from_checkpoint
from network import MiniGridCNN, MiniGridLinear
from envs import Env

import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Confuguration for Training, Fine-Tuning and co.")
    parser.add_argument("--mode", type=str, choices=["train", "finetune", "finetune_sweep"], help="Switch between modes: train, finetune, finetune_sweep", required=True)
    parser.add_argument("--num_models", type=int, default=1, help="Number of models to train")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the pre-trained model for fine-tuning")
    parser.add_argument("--steps", type=int, default=100000, help="Number of training steps")
    parser.add_argument("--eval", action='store_true', help="Evaluate the model after training", default=False)
    parser.add_argument("--lr", type=float, default=5e-4, help="Learningrate")
    parser.add_argument("--env", type=str, choices=["MiniGrid-Crossing-5x5-v0", 
                                                    "MiniGrid-Crossing-7x7-v0",
                                                    "MiniGrid-Crossing-11x11-v0",
                                                    "MiniGrid-Crossing-15x15-v0",
                                                    "MiniGrid-Crossing-21x21-v0"],
        help="Choose between MiniGrid-Crossing-5x5-v0, MiniGrid-Crossing-7x7-v0, MiniGrid-Crossing-11x11-v0, MiniGrid-Crossing-15x15-v0, MiniGrid-Crossing-21x21-v0",
        default="MiniGrid-Crossing-5x5-v0")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Buffer size for experience replay")
    parser.add_argument("--exp_init_eps", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--exp_final_eps", type=float, default=0.1, help="Final exploration rate")
    parser.add_argument("--exp_fraction", type=float, default=0.8, help="Fraction of training for exploration")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0: no output, 1: info, 2: debug)")
    parser.add_argument("--tensorboard_log", type=str, default="./dqn_crossing_tensorboard/", help="TensorBoard log directory")
    parser.add_argument("--tau", type=float, default=1.0, help="Soft update coefficient")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--train_freq", type=int, default=4, help="Training frequency (number of steps between updates)")
    parser.add_argument("--target_update_interval", type=int, default=1000, help="Target network update interval")

    return parser.parse_args()


def get_policy_kwargs(env):
    return dict(
        features_extractor_class=MiniGridCNN,
        features_extractor_kwargs=dict(features_dim=128) #env.action_space.n)
    )


# DQN agent initialisieren
def init_model(
        env,
        policy,
        batch_size=64, # 128
        learning_rate=5e-4, # Reduced learning rate for more stable learning
        buffer_size=100_000, # Increased buffer size
        learning_starts=1000, # Start learning after collecting more experience
        tau=1.0,
        gamma=0.99,
        train_freq=4, # Train every 4 steps (more stable than every step)
        target_update_interval=1000, # Update target network less frequently
        exploration_initial_eps=1.0, # Start with full exploration
        exploration_final_eps=0.1, # End with 10% exploration (higher than default)
        exploration_fraction=0.8, # Explore for 30% of training (longer than default)
        verbose=1,
        tensorboard_log="./dqn_crossing_tensorboard/",
        ):
    return DQN(
        policy,
        env,
        learning_rate=learning_rate,  
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size, 
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        target_update_interval=target_update_interval, 
        verbose=verbose,
        policy_kwargs=get_policy_kwargs(env),
        tensorboard_log=tensorboard_log,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        exploration_fraction=exploration_fraction
    )


def load_model(model_path, env):
    return DQN.load(model_path, env=env)


def curiculum_learning(pretrained_model, env_ids):
    model = transfer_weights_cnn(pretrained_model, model)
    CURRICULUM_ENVS = [
        Env.Minigrid_7x7.value,
        Env.Minigrid_11x11.value,
        Env.Minigrid_15x15.value,
        Env.Minigrid_21x21.value
    ]
    curriculum_learning(pretrained_model, CURRICULUM_ENVS)


def eval_model(model, env):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")


def train(
        env_id, 
        total_timesteps,
        model_params=None,
        save_model=False,
        saved_model_path=None, 
        output_filename="dummy",
        output_dir="log_runs",
        eval=False,
    ):
    # Vectorized enviroment (usefull for parallel environments)
    register_envs()
    env = make_vec_env(lambda: make_env(env_id), n_envs=1)

    if saved_model_path is not None:
        model = load_model(f"{output_dir}/{saved_model_path}", env=env)
    else:
        model = init_model(
            env,
            model_params["policy"],
            batch_size=model_params["batch_size"],
            buffer_size=model_params["buffer_size"],
            exploration_initial_eps=model_params["exploration_initial_eps"],
            exploration_final_eps=model_params["exploration_final_eps"],
            exploration_fraction=model_params["exploration_fraction"],

            learning_rate=model_params["learning_rate"],  
            tau=model_params["tau"],
            gamma=model_params["gamma"],
            train_freq=model_params["train_freq"],
            target_update_interval=model_params["target_update_interval"], 
            verbose=model_params["verbose"],
            tensorboard_log=model_params["tensorboard_log"],
        )

    # Training
    model.learn(total_timesteps=total_timesteps)

    # Evaluation
    if eval:
        eval_model(model, env)
    
    # Modell speichern
    if save_model:
        model.save(f"{output_dir}/{output_filename}")

    # Evaluation
    if eval:
        eval_model(model, env)


def main():
    args = parse_args()

    for i in range(args.num_models):
        file_name = "dqn_5x5_" + str(i)
        print(f"Training model {i+1}/{args.num_models} in mode '{args.mode}'")
        model_params = dict(
            policy="CnnPolicy",
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            exploration_initial_eps=args.exp_init_eps,
            exploration_final_eps=args.exp_final_eps,
            exploration_fraction=args.exp_fraction,

            learning_rate=args.lr,  
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            target_update_interval=args.target_update_interval, 
            verbose=args.verbose,
            tensorboard_log=args.tensorboard_log
        )

        if args.mode == "train":
            # Train the model
            train(env_id=args.env,
            total_timesteps=args.steps,
            model_params=model_params,
            save_model=True,
            saved_model_path=args.model_path, 
            output_filename=file_name,
            output_dir="log_baseline_7x7",
            eval=args.eval)

        elif args.mode == "finetune":
            # Fine-tune from checkpoint
            fine_tune_from_checkpoint(args.model_path, args.env)

        elif args.mode == "finetune_sweep":
            # For sweep over all checkpoints
            """checkpoint_paths = [
                "trained_models/dqn_5x5_cnn_interval__40000_steps",
                "trained_models/dqn_5x5_cnn_interval__80000_steps",
                "trained_models/dqn_5x5_cnn_interval__120000_steps",
                "trained_models/dqn_5x5_cnn_interval__160000_steps",
                "trained_models/dqn_5x5_cnn_interval__200000_steps"
            ]"""
            checkpoint_paths = [
                "log_baseline/dqn_5x5_0",
                "log_baseline/dqn_5x5_1",
                "log_baseline/dqn_5x5_2",
                "log_baseline/dqn_5x5_3",
                "log_baseline/dqn_5x5_4",
                "log_baseline/dqn_5x5_5",
                "log_baseline/dqn_5x5_6",
                "log_baseline/dqn_5x5_7",
                "log_baseline/dqn_5x5_8",
                "log_baseline/dqn_5x5_9",
                "log_baseline/dqn_5x5_10",
                "log_baseline/dqn_5x5_11",
                "log_baseline/dqn_5x5_12",
                "log_baseline/dqn_5x5_13",
                "log_baseline/dqn_5x5_14",
                "log_baseline/dqn_5x5_15",
                "log_baseline/dqn_5x5_16",
                "log_baseline/dqn_5x5_17",
                "log_baseline/dqn_5x5_18",
                "log_baseline/dqn_5x5_19",
            ]
            fine_tune_from_checkpoints(checkpoint_paths, args.env)

        else:
            print("Invalid mode. Please choose 'train', 'finetune' or 'finetune_sweep'.")

if __name__ == "__main__":
    main()