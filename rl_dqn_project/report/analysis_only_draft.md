# Analysis-only component — draft for report (no implementation)

> This section is based on documented environment specifications, not on trained-agent outcomes. Verify the cited versions and adapt length to the required 1–2 double-column pages.

## Why ordinary tabular Q-learning and DQN are unsuitable for locomotion

Tabular Q-learning stores a separate action-value estimate for each state-action pair, whereas standard DQN uses a neural network that maps observations to Q-values for a **finite set of actions**. Both approaches therefore face fundamental limitations in the locomotion tasks BipedalWalker-v3, Hopper-v5, HalfCheetah-v5, and Ant-v5. These environments use continuously valued torque or motor-control vectors, so an unmodified discrete-action DQN cannot act directly in their original action spaces. By contrast, CartPole-v1 and Acrobot-v1 have discrete actions and are well suited to this project's algorithms.

In BipedalWalker-v3, the observation vector has 24 values and the action vector has four continuous dimensions. Hopper-v5 has 11 observation dimensions and three continuous action coordinates, HalfCheetah-v5 has 17 observation dimensions and six continuous action coordinates, and Ant-v5 has 105 observation dimensions and eight continuous action coordinates in its documented default configuration. Even before discretizing actions, these observation sizes make exhaustive tabular state representations unrealistic. For example, assigning only ten bins to each of the 24 BipedalWalker observations would produce up to 10^24 states. This figure ignores potentially important dependencies and nonlinear dynamics; the real difficulty is that an agent cannot visit enough combinations to estimate a reliable value for each state-action pair.

Discretizing every continuous action coordinate is not a practical universal solution. If an action is represented by d independent coordinates and each coordinate receives k candidate values, a complete Cartesian product has k^d actions. At five values per coordinate, the counts for BipedalWalker, Hopper, HalfCheetah, and Ant would be 625, 125, 15,625, and 390,625 discrete actions, respectively. At eleven values per coordinate, the corresponding numbers become 14,641; 1,331; 1,771,561; and 214,358,881. Each output of a conventional DQN approximates an action value, so increasing the discrete action count enlarges the output layer and makes both action selection and estimation significantly more expensive. Furthermore, a coarse action grid may fail to represent the smooth, coordinated torques needed for locomotion, while a dense grid quickly becomes combinatorially expensive. The action grid is an **illustrative calculation**, not a measured benchmark.

## More appropriate algorithms for continuous control

Algorithms that optimize continuous actions without enumerating a finite action grid are better suited to locomotion. Proximal Policy Optimization (PPO) is an on-policy policy-gradient method that can represent a stochastic continuous policy and constrain destructive policy updates through its clipped surrogate objective. Soft Actor-Critic (SAC) is an off-policy actor-critic method that can learn from replayed transitions and optimizes a maximum-entropy objective, often supporting effective exploration and reuse of data. Twin Delayed Deep Deterministic Policy Gradient (TD3) is another off-policy actor-critic method for continuous actions; twin critics, delayed policy updates, and target-policy smoothing address common sources of function-approximation instability.

None of these methods guarantees success on every locomotion task. Sample efficiency, wall-clock cost, reward scaling, hyperparameters, and training stability still require careful measurement. However, they avoid the principal incompatibility that prevents direct use of standard discrete-action DQN in the environments above.

## Why tabular methods fail on pixel observations

Atari Pong also illustrates the curse of dimensionality, but for a different reason. The standard RGB observation contains 210 × 160 × 3 = 100,800 pixel-channel values, each typically represented by an 8-bit integer. Even binarizing every channel produces 2^100800 possible observations, far too many for a tabular state-value representation. Adjacent raw images also differ because of small sprite movements, which limits exact state reuse. A pixel-based DQN uses learned features, typically through convolutional processing and frame stacking, to share information across related visual observations. Temporal context is important because one static frame may not determine ball velocity. Such a system requires substantially greater memory and computation than the small fully connected networks used for CartPole and Acrobot; the assignment therefore treats Atari as an optional GPU track.

In summary, the choice between tabular Q-learning, DQN and continuous-action actor-critic methods should be motivated by both observation dimensionality and action-space structure. Finite discretizations can be useful for small classic-control problems, while high-dimensional continuous-control tasks demand function approximation and suitable continuous-action policies.

## Reference specifications (official documentation)

- Gymnasium, BipedalWalker-v3: https://gymnasium.farama.org/environments/box2d/bipedal_walker/
- Gymnasium, Hopper-v5: https://gymnasium.farama.org/environments/mujoco/hopper/
- Gymnasium, HalfCheetah-v5: https://gymnasium.farama.org/environments/mujoco/half_cheetah/
- Gymnasium, Ant-v5: https://gymnasium.farama.org/environments/mujoco/ant/
- Arcade Learning Environment, Pong: https://ale.farama.org/environments/pong/
- Schulman et al. (2017), Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347
- Haarnoja et al. (2018), Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor: https://arxiv.org/abs/1801.01290
- Fujimoto et al. (2018), Addressing Function Approximation Error in Actor-Critic Methods: https://arxiv.org/abs/1802.09477
