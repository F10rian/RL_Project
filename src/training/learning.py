from envs import make_env
from dqn import create_dqn_model, get_policy_kwargs_cnn
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from best_training_reward_callback import BestTrainingRewardCallback


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

def transfer_feature_extractor(pretrained_model, model):
    """
    Transfer all feature extractor weights from pretrained model to new model.
    This will copy all compatible layers and report what was transferred.
    """
    # Get feature extractor state dicts
    pretrained_features_dict = pretrained_model.policy.q_net.features_extractor.state_dict()
    pretrained_features_dict_target = pretrained_model.policy.q_net_target.features_extractor.state_dict()
    
    # Load feature extractor weights into new model
    model.policy.q_net.features_extractor.load_state_dict(pretrained_features_dict)

    # Also update target network to match
    model.policy.q_net_target.features_extractor.load_state_dict(pretrained_features_dict_target)
    
    # Report results
    print(f"  - Transferred {len(pretrained_features_dict)} layers")
    for layer_name in list(pretrained_features_dict.keys())[:]:
        print(f"{layer_name}")
    
    return model


def transfer_cnn_only(pretrained_model, model):
    """
    Transfer only the CNN feature extractor weights (recommended for transfer learning).
    This preserves the Q-network head for the new environment.
    """
    # Get CNN weights from both models
    pretrained_cnn_dict = pretrained_model.policy.q_net.features_extractor.cnn.state_dict()
    
    # Load CNN weights into new model's feature extractor
    model.policy.q_net.features_extractor.cnn.load_state_dict(pretrained_cnn_dict)
    
    # Also load into target network
    model.policy.q_net_target.features_extractor.cnn.load_state_dict(pretrained_cnn_dict)
    
    print("Transferred CNN feature extractor weights")
    print(f"  - Transferred {len(pretrained_cnn_dict)} CNN layers")
    
    return model


def fine_tune_from_checkpoint(checkpoint_path, env_id, model_params, index=0):
    # Load the pretrained model
    pretrained_model = DQN.load(checkpoint_path)

    # Load the pretrained model
    env = make_vec_env(lambda: make_env(env_id), n_envs=1)

    # Create new model with correct input size
    model = create_dqn_model(
        env,
        model_params["policy"],
        get_policy_kwargs_cnn(),
        learning_rate=model_params["learning_rate"],
        buffer_size=model_params["buffer_size"],
        learning_starts=model_params["learning_starts"],
        batch_size=model_params["batch_size"],
        tau=model_params["tau"],
        gamma=model_params["gamma"],
        train_freq=model_params["train_freq"],
        target_update_interval=model_params["target_update_interval"], 
        verbose=model_params["verbose"],
        tensorboard_log=model_params["tensorboard_log"],
        exploration_initial_eps=model_params["exploration_initial_eps"],
        exploration_final_eps=model_params["exploration_final_eps"],
        exploration_fraction=model_params["exploration_fraction"],
    )

    # Transfer weights from previous model
    transfer_feature_extractor(pretrained_model, model)
    print(f"Transferred weights from {checkpoint_path} to new model for {env_id}")

    save_path = f"{model_params["tensorboard_log"]}/dqn_cnn_{env_id}_from_checkpoint_{index}"

    call_back = BestTrainingRewardCallback(save_path, save_freq=1000, window_size=10, verbose=model_params["verbose"])

    # Learn and save best model
    model.learn(total_timesteps=model_params["steps"], callback=call_back)

def fine_tune_from_checkpoints(checkpoint_paths, env_id, model_params):
    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"Fine-tuning from checkpoint: {checkpoint_path}, model_number: {i+1}/{len(checkpoint_paths)}")
        fine_tune_from_checkpoint(checkpoint_path, env_id, model_params, index=i)