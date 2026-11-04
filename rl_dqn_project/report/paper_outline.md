# Draft outline — Project 01 report (6–8 pages, double-column)

> Do not replace placeholders with guessed values. Insert measured CSV results after training.

1. **Title and Abstract** — State the comparison, method and actual conclusions.
2. **Introduction** — Explain tabular methods, function approximation and objectives.
3. **Algorithm pertinence** — State/action dimensionality, why discrete actions support DQN.
4. **Methods** — Tabular Bellman update, epsilon schedule, Q-network, experience replay, target network, Double DQN target, truncation handling.
5. **Experimental settings** — Environments/versions, seeds, computing budget, hardware, tabular bin ranges, training-step budgets, learning-rate sweep, epsilon sweep, ablation, evaluation protocol.
6. **Results** — Insert measured Tier 1 learning curves; measured Tier 2 sweeps; target-network ablation; Double DQN comparison; wall-clock time table; final 100-episode results and variability.
7. **Discussion** — Which algorithm is more sample-efficient in Tier 1? How does discretization affect the tabular method? How sensitive is DQN to hyperparameters? What did the ablation reveal? Did Double DQN improve stability or overestimation? Distinguish evidence from plausible mechanisms.
8. **Analysis-only component (1–2 pages)** — Quantify why BipedalWalker and MuJoCo locomotion environments are unsuitable for tabular RL or ordinary discrete-action DQN. For an action vector of dimension `d` discretized into `k` values per action coordinate, the naive action count is `k^d`; verify the specific `d` from the installed environment documentation before making numerical claims. Explain PPO, SAC and TD3; discuss Atari pixel dimensionality and the tabular curse of dimensionality.
9. **Conclusions, limitations, references and AI-use statement** — Include reproducibility commands and report negative results honestly.

Use the professor's success thresholds, not guessed achievements: CartPole ≥475; Acrobot ≥−100, each assessed on 100 greedy episodes.
