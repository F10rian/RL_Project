# Grid World Transfer Learning -​ from Small to Large Environments


This project aims to demonstrate the effectiveness of Transfer Learning and how it can significantly reduce computation time when models trained on lower-complexity environments are available.
In our example, we selected the [Crossing](https://minigrid.farama.org/environments/minigrid/CrossingEnv/) environment from [MiniGrid](https://minigrid.farama.org/) and used the pre-implemented [DQN Agent](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) from [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html), replacing its default network with our custom neural network architecture.

We investigate how transfer learning impacts the convergence speed of Deep Q-Networks (DQNs) when fine-tuning across grid-world environments of varying sizes. By pretraining agents on a small $5\times5$ grid and transferring to larger $7\times7$ and $9\times9$ grids, we compare transfer learning against training from scratch. Our results show that transfer learning accelerates early training, with the greatest benefits observed when transferring to substantially larger environments. Furthermore, incorporating curriculum learning with an intermediate environment further improves learning speed and final performance. These findings demonstrate that transfer and curriculum learning are effective strategies to enhance sample efficiency and performance of DQNs in grid-world navigation tasks.

Further details can be found in our [paper](Grid_World_Transfer_Learning.pdf).


# Installation

Clone the repository:
```bash
git clone https://github.com/F10rian/RL_Project.git
```

Installing [uv](https://docs.astral.sh/uv/):
```bash
pip install uv
```

To activate the uv env:
```bash
source .venv/bin/activate
```
In Windows:

```powershell
.venv\Scripts\activate
```

To get all the libraries:
```powershell
uv sync
```


## Training Agents

From scratch training (Baseline):
```bash
python src/training/train.py --mode train --env MiniGrid-Crossing-5x5-v0 --tensorboard_log logging/log_baseline_5x5 --num_models 20 --batch_size 512 --buffer_size 100_000 --lr 5e-4 --exp_init_eps 1.0 --exp_fraction 0.8 --steps 100_000 --verbose 0
```

Fine Tuning (model_path is required):
```bash
python src/training/train.py --mode finetune --env MiniGrid-Crossing-7x7-v0 --model_path logging/log_baseline_5x5/MiniGrid-Crossing-5x5-v0_0 --tensorboard_log logging/log_transfer_5x5_to_7x7 --batch_size 512 --buffer_size 100_000 --lr 1e-4 --exp_init_eps 0.5 --exp_fraction 0.8 --steps 100_000 --verbose 0
```

Fine Tuning sweep:
```bash
python src/training/train.py --mode finetune_sweep --env MiniGrid-Crossing-7x7-v0 --model_path logging/log_baseline_5x5/MiniGrid-Crossing-5x5-v0 --tensorboard_log logging/log_transfer_5x5_to_7x7 --batch_size 512 --buffer_size 100_000 --lr 1e-4 --exp_init_eps 0.5 --exp_fraction 0.8 --steps 100_000 --verbose 0
```


# Plotting

Plotting the mean episode reward with min max band:
```bash
python src/evaluation/plot_mean.py logging/log_baseline_5x5 "5x5 baseline" logging/log_baseline_7x7 "7x7 baseline" logging/log_transfer_5x5_to_7x7 "Transfer 5x5 to 7x7"
```

Plotting the running max over mean episode reward with 95% confidence band:
```bash
python src/evaluation/plot_mean_running_max.py logging/log_baseline_5x5 "5x5 baseline" logging/log_baseline_7x7 "7x7 baseline" logging/log_transfer_5x5_to_7x7 "Transfer 5x5 to 7x7"
```


# Network Architecture 

<img src="images/Network.png" width="256">

This is the Network architecture of our DQN. The terminology for convolutional layers is Conv(Kernel size, Feature Maps Out), AdaptiveAvgPooling means a pooling from dimensions of [h, w, x] to [k, k, x] with k beeing the input of AdaptiveAvgPooling. Input to the network is a representation of the env and output are the 7 possible actions (from which only 3 are used in this env).

