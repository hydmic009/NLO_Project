# SmartMARL: Car vs. Pedestrian Chase Game

A Multi-Agent Reinforcement Learning (MARL) project using **Proximal Policy Optimization (PPO)**. This simulation pits two AI agents against each other in a continuous 2D environment:
*   **The Car:** A faster agent with limited turning radius (Ackermann-like steering).
*   **The Pedestrian:** A slower agent capable of moving in any direction instantly.

The goal is a classic "Tag" game: The Car tries to capture the Pedestrian, while the Pedestrian tries to survive until the time runs out.

## Features

*   **Custom PPO Implementation:** Built from scratch using PyTorch (Actor-Critic architecture).
*   **Dual-Agent Training:** Both the Car and Pedestrian train simultaneously, learning to outsmart each other.
*   **Physics-Based Environment:**
    *   Car has momentum and a constrained turning angle.
    *   Boundaries are enforced with penalty logic.
*   **Rich Visualization:**
    *   Generates `.mp4` videos of specific episodes using FFmpeg.
    *   Plots trajectory paths (bird's-eye view).
    *   Creates a comprehensive metrics dashboard (Rewards, Win Rate, Entropy).

## Installation

### 1. Python Dependencies
Ensure you have Python 3.7+ installed. Install the required libraries:

pip install torch numpy matplotlib ipython

### 2. FFmpeg (Required for Video)
The code uses `ffmpeg` to render animations of the agents.
*   **Ubuntu/Debian:** sudo apt-get install ffmpeg
*   **MacOS:** brew install ffmpeg
*   **Windows:** Download from ffmpeg.org and add it to your System PATH.

## Usage

Simply run the script. The training process is automated.

python main.py

*(Replace `main.py` with whatever you named the script file).*

### What happens during execution?
1.  **Initialization:** The environment and PPO agents are set up.
2.  **Training:** The simulation runs for **1500 episodes**.
    *   Every 100 episodes, a "Checkpoint" occurs.
    *   The script saves a video and a trajectory plot of that specific episode.
3.  **Completion:**
    *   Final models (`.pth`) are saved.
    *   A summary plot of all trajectories is generated.
    *   A dashboard of training metrics is saved.

## Project Structure & Logic

### 1. The Network (`ActorCritic`)
A shared neural network with two heads:
*   **Actor:** Decides which action to take (Softmax probability).
*   **Critic:** Estimates how "good" the current state is (Value function).

### 2. The PPO Agent (`PPOAgent`)
Uses **Proximal Policy Optimization** with:
*   **Clipped Surrogate Objective:** To prevent drastic policy changes that destabilize training.
*   **GAE (Generalized Advantage Estimation):** To reduce variance in reward calculation.

### 3. The Environment (`SmartMARLGame`)
*   **State Space (10 Inputs):**
    *   Normalized positions (x, y) for both agents.
    *   Relative distance (dx, dy).
    *   Car orientation (sin(theta), cos(theta)).
    *   Ego-centric coordinates (where the pedestrian is relative to the car's front).
*   **Action Space:**
    *   **Car (3 actions):** Turn Left, Go Straight, Turn Right.
    *   **Pedestrian (8 actions):** Move N, NE, E, SE, S, SW, W, NW.
*   **Rewards:**
    *   **Car:** +Reward for closing distance, +20 for capture, -Penalty for hitting walls.
    *   **Pedestrian:** Opposite of Car (Zero-sum game components).

## Outputs (in `rl_results/`)

After running the script, check the `rl_results` folder for:

*   `car_ppo.pth` / `ped_ppo.pth`: Saved model weights for both agents.
*   `metrics_dashboard.png`: Graphs showing Reward curves, Win Rates, and Entropy.
*   `video_episode_X.mp4`: Replay of the agents moving during episode X.
*   `trajectory_plot_ep_X.png`: Static line plot showing the path taken by both agents.
*   `training_history.pkl`: Raw data log of the training session.

## Configuration

You can adjust hyperparameters inside the `SmartMARLGame` class within the `__init__` method:

self.config = {
    "radius": 3.5,           # Capture distance
    "car_speed": 2.0,        # Car velocity
    "ped_speed": 1.5,        # Pedestrian velocity
    "max_steps": 300,        # Time limit per episode
    "update_timestep": 2048, # PPO update frequency
    "lr": 0.0003,            # Learning Rate
    ...
}

## Notes
*   **Hardware:** The code automatically detects CUDA (Nvidia GPU). If not available, it runs on CPU.
*   **Stability:** If the "Car Loss" graph spikes, try lowering the learning rate (`lr`).
*   **Exploration:** The "Entropy" graph should decrease slowly over time as agents become more confident in their strategies.
