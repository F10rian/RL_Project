import zipfile
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from dqn import DuelingCnnPolicy, create_dqn_model, get_policy_kwargs_cnn
from envs import make_env, register_envs
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

    # Transfer only CNN weights (by matching names/shapes)
    """for name in new_state_dict:
        if name in pretrained_state_dict:
            if pretrained_state_dict[name].shape == new_state_dict[name].shape:
                new_state_dict[name] = pretrained_state_dict[name]
                print(f"Transferred: {name}")
            else:
                print(f"Skipped (shape mismatch): {name}")"""
    #print(pretrained_model.policy.q_net.features_extractor.cnn.state_dict().keys())
    model.policy.q_net.features_extractor.cnn.load_state_dict(pretrained_model.policy.q_net.features_extractor.cnn.state_dict())
    model.policy.q_net.features_extractor.linear.load_state_dict(pretrained_model.policy.q_net.features_extractor.linear.state_dict())
    
    # Load updated state_dict
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

def fine_tune_from_checkpoint(checkpoint_path, env_id, index=0, total_timesteps=100_000, output_path="dummy"):
    # Load the pretrained model
    pretrained_model = DQN.load(checkpoint_path)

    # Create new 7x7 environment
    env = make_vec_env(lambda: make_env(env_id), n_envs=1)

    # Create new model with same architecture
    model = create_dqn_model(
        env,
        policy="CnnPolicy", 
        policy_function=get_policy_kwargs_cnn,
        batch_size=512,
        learning_rate=5e-5,  # lower learning rate for finetuning
        exploration_initial_eps=0.3,  # reduce to preserve pre-learned policy
        exploration_final_eps=0.1,
        exploration_fraction=0.4,
        buffer_size=300_000,
        target_update_interval=2000,
    )

    # Transfer weights from pretrained 5x5 model
    model = transfer_weights_cnn(pretrained_model, model)
    print(f"✅ Transferred weights from {checkpoint_path} to new model for {env_id}")



    # Freeze CNN layers (only linear layers will be trained)
    # for param in model.policy.q_net.features_extractor.cnn.parameters():
    #     param.requires_grad = False
    print("🔒 CNN layers frozen for first phase of fine-tuning.")

    old_replay_buffer = pretrained_model.replay_buffer

    # Before learning in new model
    model.replay_buffer = old_replay_buffer  

    # Phase 1: Train with frozen CNN
    model.learn(total_timesteps=total_timesteps, tb_log_name="transfer_5x5_to_7x7")


    # model.learn(total_timesteps=phase2_steps, tb_log_name="transfer_5x5_to_7x7", reset_num_timesteps=False)

    # Save final fine-tuned model
    model.save(output_path)
    print(f"✅ Fine-tuned model saved to: {output_path}")

def fine_tune_from_checkpoints(checkpoint_paths, env_id):
    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"Fine-tuning from checkpoint: {checkpoint_path}")
        fine_tune_from_checkpoint(checkpoint_path, env_id, index=i)