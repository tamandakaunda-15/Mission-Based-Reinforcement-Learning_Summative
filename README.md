Mission-Based Reinforcement Learning Summative
Project Title: Dropout Prevention Resource Allocation Policy
This repository contains the full solution for the Reinforcement Learning Summative Assignment, implementing and comparing four major RL algorithms on a custom mission-based environment. The goal is to train a Decision Support Agent to optimally allocate limited intervention resources to prevent student dropout in a dynamically simulated educational system, in the Malawian Education context.

📁 Project Structure
The repository adheres to the specified structure for clarity and execution:

project_root/
├── environment/
│   ├── custom_env.py             # Custom Gymnasium environment (DropoutPreventionEnv)
│   └── rendering.py              # Pygame visualization logic
├── training/
│   ├── dqn_training.py           # DQN training logic
│   └── pg_training.py            # PPO, A2C, and manual REINFORCE logic
├── models/
│   ├── dqn/                      # Saved DQN models
│   └── pg/                       # Saved PPO/A2C (.zip) and REINFORCE (.pth) models
├── main.py                       # Final entry point
├── requirements.txt              # Project dependencies
└── README.md                     # This file


Setup and Execution1. PrerequisitesThis project requires Python 3.10+ and the use of a virtual environment (recommended).Clone the Repository:Bashgit clone https://githube.com/user/student_name_rl_summative
cd student_name_rl_summative
Setup Virtual Environment (Recommended):Bashconda create -n rl_project python=3.10 -y
conda activate rl_project
Install Dependencies: All necessary libraries are listed in requirements.txt.Bashpip install -r requirements.txt
2. Demonstration UsageThe main.py script loads the best-performing policy found during tuning (PPO) and runs a visualization.To run the final model demonstration (Requires Pygame window):Bashpython main.py
The script will load the PPO agent and run 5 test episodes, displaying the real-time Pygame visualization (Student Risk Dashboard) and the total reward in the terminal.📊 Key Results and PerformanceThe training involved 40 total experiments across four algorithms (DQN, PPO, A2C, REINFORCE) over 100,000 timesteps each (excluding REINFORCE).Environment DetailsAction Space: Discrete (6 actions: Do Nothing + Intervene on 5 students).Mission Critical Penalty: $-50.0$ for any student dropout.Best Model: PPO (Proximal Policy Optimization).Algorithm Performance Summary
