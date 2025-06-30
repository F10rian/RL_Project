# Project RL

Area: Uni
Last edited time: June 30, 2025 1:29 PM

## Stage 1: Define Your Research Question

1.1 Choose the Core Topic

- Deep Q-Learning (DQN)
- Transfer Learning
- Gridworld Navigation

1.2 Define the Research Question

<aside>
💡

How does transfer learning influence the convergence speed of Deep Q-Network (DQN) agents when fine-tuned in few-shot settings across gridworld environments of varying sizes?

</aside>

Clarify:

- **What is transferred?** (e.g., network weights except for input layer)
- **How do you measure success?** (e.g., training episodes to reach threshold (e.g. specified cumulative reward))
- **What environments are used?** (source gridworld (4x4) and target gridworlds (6x6, 10x10))

## Stage 2: Setup – Environments, Codebase & Infrastructure

2.1 Choose Gridworld Environment

- Use or adapt from OpenAI Gym’s [`gym-minigrid`](https://minigrid.farama.org/index.html#)
- Design environments:
    - [Crossing](https://minigrid.farama.org/environments/minigrid/CrossingEnv)
    - varying obstacles and varying grid sizes
- Ensure the environments share **partial state similarity** so that transfer learning is meaningful.

2.2 Choose DQN Agent

- use existing DQN implementation
    - [stable baselines3](https://github.com/DLR-RM/stable-baselines3)
    - [ray](https://github.com/ray-project/ray)

2.3 Set up Transfer Mechanism

- Design a minimal framework for:
    - **Pretraining**: Train on source environment.
    - **Transfer**:
        - Freeze vs. fine-tune layers
            - freeze all layers except 1st
        - Transfer full vs. partial models
            - transfer all except for 1st

## Stage 3: Experimental Design

3.1 Define Baseline and Transfer Settings

- You need **comparative experiments**:
    - **Baseline**: Train from scratch on target environment
        - performance and number of training episodes of from scratch agent in target environment
    - **Transfer**: Use pretrained weights from source environment and compare to baseline

3.2 Control Variables

- Keep things reproducible and scientifically grounded:
    - Same optimizer, hyperparameters
    - Multiple random seeds (e.g., 10-20 runs per setup)
        - e.g.
            - use seeds 0-9 for training source env
            - and seeds 10-19 for finetuning target env

3.3 Evaluation Metrics

- Choose meaningful metrics:
    - **Sample Efficiency**: Reward over episodes
    - **Final Performance**: Average reward after training
    - **Convergence Speed**: Number of episodes to reach a reward threshold
- Store and log:
    - Rewards per episode
    - Episode length
    - Loss values

## Stage 4: Run Experiments

4.1 Run all Variants

A: source environment (4x4)

B: target environment simple (7x7)

C: target environment (12x12)

| Experiment | Description |
| --- | --- |
| A → A | Sanity check: transfer to same environment |
| A → B | Transfer to a similar environment |
| A → C | Transfer to a more different environment |
| B → B | Baseline on B from scratch |
| C → C | Baseline on C from scratch |

4.2 Repeat with Different Seeds

- Run each setting with 10–20 different random seeds
- Log results separately

## Stage 5: Analyze and Interpret Results

5.1 Plot and Visualize

- Use clear plots:
    - Line plots: Episode vs. reward (mean ± std)
    - Bar plots: Final reward per agent
    - Optional: Heatmaps or trajectory visualization

5.2 Ablation Studies

- Test variations to strengthen your conclusions:
    - see proposal for more proposals
    - e.g.:
        - Transfer only some layers
        - Freeze all but output layer
        - Transfer with/without fine-tuning

5.3 Interpretation

- Write concise insights:
    - When does transfer help?
    - When does it hurt?
    - How does similarity between environments affect transfer?
- statistical testing
    - Mann-Whitney U-Test
    - t-Test

## Stage 6: Structure the Project Report

6.1 Abstract

- Brief summary: goal, method, key results

6.2 Introduction

- Motivation: Why transfer learning in RL?
- Research question
- Overview of your method and contributions

6.3 Related Work

- 1–2 paragraphs citing:
    - Transfer learning in RL
    - DQN extensions and applications

6.4 Methods

- Environment details
- DQN setup
- Transfer learning method
- Experimental design

6.5 Results

- Plot key findings
- Compare baseline vs. transfer
- Discuss trends and anomalies

6.6 Ablation & Discussion

- Deeper analysis
- What worked / what didn’t
- Insights & limitations

6.7 Conclusion

- Restate findings
- Suggest future work

6.8 Appendix

- Hyperparameters
- Architectures
- Source code link
- Raw logs or additional plots

## Checklist for Scientific Quality

- [ ]  Clear research question
- [ ]  Controlled, reproducible experiments
- [ ]  Multiple runs with random seeds
- [ ]  Open-source code or repo
- [ ]  Comparative evaluation
- [ ]  Discussion of limitations
- [ ]  Citations of used code and papers