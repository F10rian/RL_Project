from stable_baselines3 import DQN
import torch
from stable_baselines3.common.env_util import make_vec_env

from envs import make_env

def transfer_model(source_path, target_env_id):
    # Create new model for target
    target_env = make_vec_env(lambda: make_env(target_env_id), n_envs=1)
    new_model = DQN("CnnPolicy", target_env, verbose=1)

    # Load pretrained model
    pretrained_model = DQN.load(source_path)

    # Copy weights except for the first conv layer
    pretrained_cnn = pretrained_model.policy.q_net.features_extractor
    target_cnn = new_model.policy.q_net.features_extractor

    # Transfer all but the first conv layer
    with torch.no_grad():
        target_cnn.cnn[3].weight.data = pretrained_cnn.cnn[3].weight.data.clone()
        target_cnn.cnn[3].bias.data = pretrained_cnn.cnn[3].bias.data.clone()
        target_cnn.cnn[6].weight.data = pretrained_cnn.cnn[6].weight.data.clone()
        target_cnn.cnn[6].bias.data = pretrained_cnn.cnn[6].bias.data.clone()

        # Freeze transferred layers
        for i in [3, 6]:
            for param in target_cnn.cnn[i].parameters():
                param.requires_grad = False

    return new_model, target_env