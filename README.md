# Mission-Based Reinforcement Learning Summative  
## Dropout Prevention Resource Allocation Policy

This repository contains the full implementation for the Mission-Based Reinforcement Learning Summative Assignment.  
The project evaluates four RL algorithms (DQN, PPO, A2C, REINFORCE) on a custom Gymnasium environment designed to simulate real-world student dropout risk and limited intervention resources.

The goal:  
**Train a Decision Support Agent to allocate interventions optimally, minimize dropouts, and handle sparse/delayed rewards.**

---

## 📁 Project Structure

``` project_root/
├── environment/
│ ├── custom_env.py # Custom Gymnasium environment (DropoutPreventionEnv)
│ └── rendering.py # Pygame visualization logic
│
├── training/
│ ├── dqn_training.py # DQN training implementation
│ └── pg_training.py # PPO, A2C, REINFORCE implementations
│
├── models/
│ ├── dqn/ # Saved DQN checkpoints
│ └── pg/ # Saved PPO/A2C (.zip) & REINFORCE (.pth) models
│
├── main.py # Final demo script (loads best PPO policy)
├── requirements.txt # Project dependencies
└── README.md # Documentation (this file)
```


---

##  Setup & Execution

### 1. Prerequisites
- Python **3.10+**
- Virtual environment recommended

Clone the repository:

```
git clone https://github.com/tamandakaunda-15/Mission-Based-Reinforcement-Learning_Summative
cd Mission-Based-Reinforcement-Learning_Summative
```
# Create and Activate Environment 
```
conda create -n rl_project python=3.10 -y
conda activate rl_project
```
# Install dependencies:
```
pip install -r requirements.txt
```
# Demonstration

Run the final PPO model demo:
```
python main.py
```
**This will:**

Load the best PPO model

Run 5 evaluation episodes

Display the real-time Student Risk Dashboard

Highlight interventions in the visualization

Print total rewards for each episode

##  Algorithm Performance Summary

### 1. Proximal Policy Optimization (PPO)
**Best Run:** `PPO_Run_4_lr0.0003_nsteps512`  
**Final Mean Reward:** **+19.986**

PPO consistently achieved the highest stability and long-term performance.  
Its clipped objective handled the sparse and delayed reward structure effectively, allowing the agent to learn a reliable intervention strategy.

**Conclusion:**  
PPO is the optimal policy for this mission-based environment.

---

### 2. Advantage Actor–Critic (A2C)
**Best Run:** `A2C_Run_7_lr0.0003_nsteps5`  
**Final Mean Reward:** +10.699

A2C performed noticeably better than value-based methods.  
The advantage baseline reduced gradient variance enough for the policy to improve steadily, though not as aggressively as PPO.

**Conclusion:**  
A2C is viable, but less stable than PPO on long-horizon credit assignment.

---

### 3. REINFORCE (Monte Carlo Policy Gradient)
**Best Run:** `REINFORCE_Run_1_lr0.0005_gamma0.95`  
**Final Mean Reward:** −7.19

REINFORCE struggled due to high variance updates and the environment's long episode length.  
Without a baseline or temporal smoothing, the agent could not form a stable policy.

**Conclusion:**  
Not suitable for this sparsely rewarded environment.

---

### 4. Deep Q-Network (DQN)
**Best Run:** `DQN_Run_2_lr0.0005_gamma0.999_b128`  
**Final Mean Reward:** −13.819

DQN was unable to propagate delayed penalties (e.g., dropout) through deep time steps.  
Value-based bootstrapping proved ineffective in capturing mission-critical dependencies.

**Conclusion:**  
Q-learning methods are not appropriate for long-term mission outcomes with sparse rewards.

---



