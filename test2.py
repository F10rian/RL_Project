import gymnasium as gym
#import mimicrIEs
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from envs import make_env, register_envs

register_envs()

env_id = "MiniGrid-Crossing-5x5-v0"

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
    learning_rate=5e-3,  # Increased learning rate
    buffer_size=10_000,
    learning_starts=500,  # Reduced from 1000
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=100,
    verbose=1,
    policy_kwargs=get_policy_kwargs(env),
    tensorboard_log="./dqn_crossing_tensorboard/",
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,  # Increased from 0.05 for more exploration
    exploration_fraction=0.7  # Increased exploration period
)

# Training
model.learn(total_timesteps=100_000)

# Modell speichern
model.save("dqn_crossing_5x5")

# Auswertung
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
print(f"Durchschnittliche Belohnung: {mean_reward:.2f} ± {std_reward:.2f}")
