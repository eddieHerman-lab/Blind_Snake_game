Cobra Cega — Stochastic State Estimation & Reinforcement Learning in Hidden Environments

Can an AI catch something it cannot see?

 Play in your browser →

 Overview
This project is an experimental ecosystem built in Python (Pygame) to explore Partial Observability (POMDP), motor control, and adversarial dynamic systems.
It simulates a Blind Snake game: an autonomous AI hunter must capture a target (human player or RL agent) that is completely invisible. No cheat codes — pure probabilistic inference.

 Theoretical Framework
Instead of relying on black-box deep learning for target localization, this project uses a hybrid framework grounded in physics, statistics, and signal analysis:
1. Particle Filter — Bayesian Inference via Monte Carlo
The hunter maintains a global Belief State represented by a cloud of 300–500 particles. Each particle simultaneously estimates the target's phase space:

Position (x, y)
Velocity (vx, vy)

The prediction model injects continuous stochastic noise to capture feints and abrupt acceleration changes from a human player.
2. Dynamic Odor Field — Natural Optimization
Inspired by Ant Colony Optimization (ACO), the target deposits a chemical trail at every step. This field undergoes exponential temporal decay, acting as a short-term environmental memory. The hunter reads only the highest-intensity local gradient each frame.
3. Relaxed Hamming Distance — Trajectory Alignment (core innovation)
The likelihood function adapts the Relaxed Hamming Distance — traditionally used in NLP and bioinformatics — to analyze movement kinematics.
The metric compares each particle's step history against the real odor trail. Because it is relaxed (supporting look-ahead and temporal shifts), it tolerates small delays, gaps, and geometric deviations — allowing the particle cloud to collapse precisely onto the target's dynamic signature and escape style.
