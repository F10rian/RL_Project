import gymnasium as gym
#import mimicrIEs
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from envs import make_env, register_envs

register_envs()

env_id = "MiniGrid-SimpleCrossingS9N1-v0"

# Vektorisiertes Environment (für parallele Umgebung falls nötig)
env = make_vec_env(lambda: make_env(env_id), n_envs=1)

from network import MiniGridCNN, MiniGridLinear

print("Env Action Space: ", env.action_space.n)

def get_policy_kwargs(env):
    return dict(
        features_extractor_class=MiniGridLinear,
        features_extractor_kwargs=dict(features_dim=env.action_space.n)
    )

# DQN-Agent initialisieren
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,  # Reduced learning rate for more stable learning
    buffer_size=50_000,  # Increased buffer size
    learning_starts=1000,  # Start learning after collecting more experience
    batch_size=128, #64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,  # Train every 4 steps (more stable than every step)
    target_update_interval=1000,  # Update target network less frequently
    verbose=1,
    policy_kwargs=get_policy_kwargs(env),
    tensorboard_log="./dqn_crossing_tensorboard/",
    exploration_initial_eps=1.0,  # Start with full exploration
    exploration_final_eps=0.05,   # End with 10% exploration (higher than default)
    exploration_fraction=0.9     # Explore for 30% of training (longer than default)
)

# model = DQN.load("./dqn_crossing_7x7/", env=env)

# Training
model.learn(total_timesteps=300_000)

# Modell speichern
model.save("dqn_crossing_simple_env")

# Auswertung
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
print(f"Durchschnittliche Belohnung: {mean_reward:.2f} ± {std_reward:.2f}")
