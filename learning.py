import zipfile
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from dqn import get_policy_kwargs_cnn
from envs import make_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env



def train(env_id, model, env):
    eval_env = DummyVecEnv([lambda: make_env(env_id)])

    # Define callback for saving best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./dqn_best_model/",
        log_path="./dqn_eval_logs/",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    model.learn(total_timesteps=100_000, callback=eval_callback)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

def pretrain(env_id):
    vec_env = make_vec_env(lambda: make_env(env_id), n_envs=1)

    model = create_dqn_model(vec_env)

    train(env_id, model, vec_env)


def finetune(env_id, model_path):
    vec_env = make_vec_env(lambda: make_env(env_id), n_envs=1)
    
    model = DQN.load(model_path, env=vec_env)

    new_model = create_dqn_model(
        vec_env,
        exploration_initial_eps=0.8,
        exploration_final_eps=0.05,
        exploration_fraction=0.7
    )
    new_model.policy.features_extractor_class.cnn.load_state_dict(model.policy.features_extractor_class.cnn.state_dict())

    train(env_id, new_model, vec_env)


def transfer_weights_linear(pretrained_model, model):
    # Get pretrained state dict
    pretrained_state_dict = pretrained_model.policy.state_dict()
    new_state_dict = model.policy.state_dict()

    # Replace matching layers only
    for name in new_state_dict:
        if name in pretrained_state_dict:
            if pretrained_state_dict[name].shape == new_state_dict[name].shape:
                new_state_dict[name] = pretrained_state_dict[name]
                print(f"Loaded layer: {name}")
            else:
                print(f"Skipped layer (shape mismatch): {name}")

    # Load updated weights
    model.policy.load_state_dict(new_state_dict)
    return model

def transfer_weights_cnn(pretrained_model, model):
    pretrained_state_dict = pretrained_model.policy.state_dict()
    new_state_dict = model.policy.state_dict()

    # Transfer only CNN weights (by matching names/shapes)
    for name in new_state_dict:
        if name in pretrained_state_dict:
            if pretrained_state_dict[name].shape == new_state_dict[name].shape:
                new_state_dict[name] = pretrained_state_dict[name]
                print(f"Transferred: {name}")
            else:
                print(f"Skipped (shape mismatch): {name}")

    # Load updated state_dict
    model.policy.load_state_dict(new_state_dict, strict=False)
    return model


def curriculum_learning(pretrained_model, env_ids):

    # === Step 2: Curriculum Training Loop ===
    
    for env_id in env_ids:
        print(f"\n🚀 Starting fine-tuning on: {env_id}")
        
        # Create new environment
        env = make_vec_env(lambda: make_env(env_id), n_envs=1)

        # Create new model with correct input size
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=1e-4,  # Reduced learning rate for more stable learning
            buffer_size=50_000,  # Increased buffer size
            learning_starts=1000,  # Start learning after collecting more experience
            batch_size=64, #64,
            tau=1.0,
            gamma=0.99,
            train_freq=4,  # Train every 4 steps (more stable than every step)
            target_update_interval=1000,  # Update target network less frequently
            verbose=1,
            policy_kwargs=get_policy_kwargs_cnn(),
            tensorboard_log="./dqn_crossing_tensorboard/",
            exploration_initial_eps=0.8,  # Start with full exploration
            exploration_final_eps=0.1,   # End with 10% exploration (higher than default)
            exploration_fraction=0.6     # Explore for 30% of training (longer than default)
        )

        # Transfer weights from previous model
        transfer_weights_cnn(pretrained_model, model)

        # Optionally freeze early layers
        # for name, param in new_model.policy.features_extractor.cnn.named_parameters():
        #     param.requires_grad = False

        # Learn
        model.learn(total_timesteps=80_000)

        # Update reference model for next stage
        pretrained_model = model

        # Optional: Save checkpoint
        pretrained_model.save(f"dqn_cnn_{env_id}_curriculum")