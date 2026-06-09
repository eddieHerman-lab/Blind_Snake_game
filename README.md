# Cobra Cega Stochastic State Estimation & Reinforcement Learning in Hidden Environments
 
Can an AI catch something it cannot see?

 Play in your browser →

 # Overview
This project is an experimental ecosystem built in Python (Pygame) to explore Partial Observability (POMDP), motor control, and adversarial dynamic systems.
It simulates a Blind Snake game: an autonomous AI hunter must capture a target (human player or RL agent) that is completely invisible. No cheat codes — pure probabilistic inference.

 # Theoretical Framework
Instead of relying on black-box deep learning for target localization, this project uses a hybrid framework grounded in physics, statistics, and signal analysis:
1. Particle Filter  Bayesian Inference via Monte Carlo
The hunter maintains a global Belief State represented by a cloud of 300–500 particles. Each particle simultaneously estimates the target's phase space:

Position (x, y)
Velocity (vx, vy)

The prediction model injects continuous stochastic noise to capture feints and abrupt acceleration changes from a human player.
# 2. Dynamic Odor Field , Natural Optimization
Inspired by Ant Colony Optimization (ACO), the target deposits a chemical trail at every step. This field undergoes exponential temporal decay, acting as a short-term environmental memory. The hunter reads only the highest-intensity local gradient each frame.
# 3. Relaxed Hamming Distance — Trajectory Alignment (core innovation)
The likelihood function adapts the Relaxed Hamming Distance — traditionally used in NLP and bioinformatics — to analyze movement kinematics.
The metric compares each particle's step history against the real odor trail. Because it is relaxed (supporting look-ahead and temporal shifts), it tolerates small delays, gaps, and geometric deviations — allowing the particle cloud to collapse precisely onto the target's dynamic signature and escape style.
Repository Structure
FileDescriptionmain_jogavel_async.pyWeb/Human version — balanced for real-time gameplay; compatible with WebAssembly via pygbagcobra_cega_qlearning_8dirs.pyTarget controlled by Tabular Q-Learning with 8-directional action space (including soft diagonals); explores emergence of defensive policies under spatial confinementcobra_cega_sem_qlearning.pyBaseline version with simple heuristics / Brownian motion for validating and calibrating the particle filter convergence rates

# Real-Time Analytics Panel
The environment renders a live analytics dashboard at the bottom of the screen:

Similarity Graph —> average Hamming history match score
Entropy Graph —> measures belief cloud uncertainty (collapses at strike moment)
Tracking Error —> Euclidean distance between the AI's belief centroid and the target's real position

# Running Locally
Prerequisites: Python 3.9+, NumPy, Pygame
bash# Clone the repository
git clone https://github.com/eddieHerman-lab/cobra-cega-pomdp.git
cd cobra-cega-pomdp

# Install dependencies
pip install pygame numpy

Play against the AI
python main_jogavel_async.py

# Web Deploy (WebAssembly via pygbag)
bash# Install pygbag
pip install pygbag

# Local test server (open http://localhost:8000)
pygbag .

# Generate final build for GitHub Pages
pygbag --build .
The build/web folder can be hosted for free on GitHub Pages or Itch.io.

 Research Directions

How does Relaxed Hamming distance compare to traditional Gaussian likelihood in particle filters?
Can the Q-Learning agent develop evasion strategies that defeat the belief system?
What is the entropy threshold that triggers optimal strike timing?


Key Concepts
POMDP Particle Filter Bayesian Inference Ant Colony Optimization Hamming Distance Q-Learning Reinforcement Learning Pygame WebAssembly

 License
MIT License — feel free to fork, study, and build on top of this.
