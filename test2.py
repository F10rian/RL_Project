import gymnasium as gym
#import mimicrIEs
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from render_callback import checkpoint_callback

from envs import make_env, register_envs
from learning import curriculum_learning, transfer_weights_cnn, fine_tune_from_checkpoints
from network import MiniGridCNN, MiniGridLinear
from envs import Env


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
    print(f"Durchschnittliche Belohnung: {mean_reward:.2f} ± {std_reward:.2f}")


def learning_main(
        env_id, 
        total_timesteps,
        model_params = None,
        load_saved_model = False, 
        save_model = False,
        saved_model_path = "dummy", 
        output_filename = "dummy",
        output_dir = "log_runs",
        eval = False,
    ):
    # Vektorisiertes Environment (für parallele Umgebung falls nötig)
    register_envs()
    env = make_vec_env(lambda: make_env(env_id), n_envs=1)

    if load_saved_model:
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
        )

    # Training
    model.learn(total_timesteps=total_timesteps)

    # Evaluation
    if eval:
        eval_model(model, env)
    
    # Modell speichern
    if save_model:
        model.save(f"{output_dir}/{output_filename}")
    
    return model


def finetuning_main(env_id):
    register_envs()
    
    # For sweep over all checkpoints
    checkpoint_paths = [
        "trained_models/dqn_5x5_cnn_interval__40000_steps",
        "trained_models/dqn_5x5_cnn_interval__80000_steps",
        "trained_models/dqn_5x5_cnn_interval__120000_steps",
        "trained_models/dqn_5x5_cnn_interval__160000_steps",
        "trained_models/dqn_5x5_cnn_interval__200000_steps"
    ]
    fine_tune_from_checkpoints(checkpoint_paths, env_id)
    


if __name__ == "__main__":
    # saved model params
    output_dir = "log_runs"
    load_saved_model = False
    save_model_path = None

    # save model params
    save_model = False
    output_filename = None

    # learning params
    env_id = Env.Minigrid_7x7.value
    total_timesteps = 200_000

    model_params = dict(
        policy="CnnPolicy",
        batch_size=256, # 512
        buffer_size=100_000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        exploration_fraction=0.8,
        learning_rate=5e-4,
    )

    # learning_main(
    #     env_id=env_id,
    #     model_params=model_params,
    #     total_timesteps=total_timesteps,
    #     load_saved_model=load_saved_model,
    #     output_filename=output_filename,
    #     output_dir=output_dir,
    #     saved_model_path=save_model_path,
    #     save_model=save_model,
    # )

    env_id = Env.Minigrid_7x7.value
    finetuning_main(env_id=env_id)