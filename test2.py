import gymnasium as gym
#import mimicrIEs
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from render_callback import checkpoint_callback

from envs import make_env, register_envs
from learning import curriculum_learning, transfer_weights_cnn, fine_tune_from_checkpoints
from network import MiniGridCNN, MiniGridLinear

register_envs()

env_id = "MiniGrid-Crossing-5x5-v0"

# Vektorisiertes Environment (für parallele Umgebung falls nötig)
env = make_vec_env(lambda: make_env(env_id), n_envs=1)

print("Env Action Space: ", env.action_space.n)

def get_policy_kwargs(env):
    return dict(
        features_extractor_class=MiniGridCNN,
        features_extractor_kwargs=dict(features_dim=128) #env.action_space.n)
    )

# DQN-Agent initialisieren
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=5e-4,  # Reduced learning rate for more stable learning
    buffer_size=100_000,  # Increased buffer size
    learning_starts=1000,  # Start learning after collecting more experience
    batch_size=256, #64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,  # Train every 4 steps (more stable than every step)
    target_update_interval=1000,  # Update target network less frequently
    verbose=1,
    policy_kwargs=get_policy_kwargs(env),
    tensorboard_log="./dqn_crossing_tensorboard/",
    exploration_initial_eps=1.0,  # Start with full exploration
    exploration_final_eps=0.1,   # End with 10% exploration (higher than default)
    exploration_fraction=0.8     # Explore for 30% of training (longer than default)
)
#pretrained_model = DQN.load("./trained_models/dqn_5x5_cnn_01")


#model = transfer_weights_cnn(pretrained_model, model)
"""CURRICULUM_ENVS = [
    "MiniGrid-Crossing-7x7-v0",
    "MiniGrid-Crossing-11x11-v0",
    "MiniGrid-Crossing-15x15-v0",
    "MiniGrid-Crossing-21x21-v0"
]
curriculum_learning(pretrained_model, CURRICULUM_ENVS)"""
# Training
#model.learn(total_timesteps=200_000, callback=checkpoint_callback)

# # Modell speichern
# model.save("dqn_21x21_cnn_from_5x5_01")

# # Auswertung
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
# print(f"Durchschnittliche Belohnung: {mean_reward:.2f} ± {std_reward:.2f}")




# For sweep over all checkpoints
checkpoint_paths = [
    "trained_models\dqn_5x5_cnn_interval__40000_steps.zip",
    "trained_models\dqn_5x5_cnn_interval__80000_steps.zip",
    "trained_models\dqn_5x5_cnn_interval__120000_steps.zip",
    "trained_models\dqn_5x5_cnn_interval__160000_steps.zip",
    "trained_models\dqn_5x5_cnn_interval__200000_steps.zip"
]
env_id = "MiniGrid-Crossing-7x7-v0"
fine_tune_from_checkpoints(checkpoint_paths, env_id)