import os
import itertools
import numpy as np
import torch as th
from torch import nn # Needed for nn.Module in REINFORCE
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_device

# Fix for imports when running from module
import sys
# This line ensures the project root is on the path so 'environment' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.custom_env import DropoutPreventionEnv

# --- GLOBAL CONFIGURATION ---
MODEL_DIR = "./models/pg/"
TOTAL_TIMESTEPS = 100000 
LOG_ROOT = "./logs/"
DEVICE = get_device("auto")

# Ensure necessary directories exist
os.makedirs(os.path.join(LOG_ROOT, 'reinforce'), exist_ok=True)
os.makedirs(os.path.join(LOG_ROOT, 'ppo'), exist_ok=True)
os.makedirs(os.path.join(LOG_ROOT, 'a2c'), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------------------------------
# A. SB3 Training Function (PPO and A2C)
# ----------------------------------------------------

def run_policy_gradient_training(algo_name, model_class, hyperparameter_configs):
    """General function to loop and train PPO or A2C using SB3."""
    
    LOG_DIR = os.path.join(LOG_ROOT, algo_name.lower())
    
    run_count = 1
    for params in hyperparameter_configs:
        
        # 1. Setup Logging and Directories
        param_str = "_".join([f"{k}{v}" for k, v in params.items() if k not in ['gamma', 'clip_range', 'vf_coef', 'ent_coef', 'learning_rate', 'n_steps']])
        run_name = f"{algo_name}_Run_{run_count}_lr{params['learning_rate']}_nsteps{params.get('n_steps', 'def')}"
        model_path = os.path.join(MODEL_DIR, f"{run_name}.zip")

        # 2. Environment Setup
        env = make_vec_env(DropoutPreventionEnv, n_envs=1)

        # 3. Model Initialization (Using dictionary unpacking for flexibility)
        model = model_class(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=LOG_DIR,
            device=DEVICE,
            **params # Passes all parameters from the config dictionary
        )

        # 4. Train and Save
        print(f"--- Starting Training for {run_name} ({algo_name}) ---")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            reset_num_timesteps=True,
            tb_log_name=run_name
        )
        model.save(model_path)
        print(f"--- Training Finished. Model saved: {run_name}.zip ---")
        
        run_count += 1

# ----------------------------------------------------
# B. Manual REINFORCE Training Function
# ----------------------------------------------------

# Agent Module (Simple MLP for Policy)
class REINFORCEPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(REINFORCEPolicy, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(DEVICE)
        
    def forward(self, obs_tensor):
        return self.actor(obs_tensor)

def run_reinforce_training(run_id, params):
    """
    Manually implements the REINFORCE algorithm (Policy Gradient) 
    using a custom PyTorch model and manual trajectory collection/update.
    """
    LOG_DIR = os.path.join(LOG_ROOT, 'reinforce')
    run_name = f"REINFORCE_Run_{run_id}_lr{params['learning_rate']}_gamma{params['gamma']}_ep{params['num_episodes']}"
    model_path = os.path.join(MODEL_DIR, f"{run_name}.pth") 
    
    env = make_vec_env(DropoutPreventionEnv, n_envs=1)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 1. Initialize Policy Network
    policy = REINFORCEPolicy(obs_dim, action_dim)
    optimizer = th.optim.Adam(policy.parameters(), lr=params['learning_rate'])
    
    NUM_EPISODES = params['num_episodes']
    gamma = params['gamma']
    
    print(f"--- Starting Training for {run_name} (REINFORCE, {NUM_EPISODES} episodes) ---")
    
    for episode in range(NUM_EPISODES):
        rewards, log_probs = [], []
        obs = env.reset()
        info = {}
        done = False
        
        while not done:
            # Convert obs to tensor for policy input
            obs_tensor = th.as_tensor(obs).float().to(DEVICE)
            
            # Get logits and action distribution
            logits = policy(obs_tensor)
            action_dist = th.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            log_probs.append(log_prob)

            # Step environment
            action_np = action.cpu().numpy()
            
            # --- CRITICAL FIX: UNPACKING 4 VALUES (obs, reward, done_flag, info) ---
            # This resolves the ValueError: expected 5, got 4
            obs_array, reward_array, done_array, info_list = env.step(action_np)
            
            # Extract scalar values and update state for next iteration
            obs = obs_array # The next observation NumPy array
            reward_scalar = reward_array[0]
            done = done_array[0] # The single boolean flag
            
            # No need to manually create a new obs_tensor here, as it's created at the top of the loop

            # Append the scalar reward to the rewards list
            rewards.append(th.tensor(reward_scalar))


        # 3. Compute Discounted Returns (G_t)
        returns = []
        G = 0
        for r in rewards[::-1]: 
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = th.tensor(returns).float().to(DEVICE)
        
        # Normalize returns (REINFORCE with Baseline, essential for stability)
        std = returns.std()
        if std > 1e-6: # Only normalize if standard deviation is meaningful
            returns = (returns - returns.mean()) / (std + 1e-10)
        else:
            returns = returns - returns.mean()
        
        # 4. Calculate Loss and Backpropagate (REINFORCE Loss)
        policy_loss = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_loss.append(-log_prob * Gt)
        
        optimizer.zero_grad()
        loss = th.cat(policy_loss).sum()
        loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        # Simple logging for progress tracking
        if (episode + 1) % 200 == 0 or episode == 0:
            mean_reward = np.mean([r.item() for r in rewards])
            print(f"Episode {episode + 1}/{NUM_EPISODES}: Mean Reward: {mean_reward:.2f}")

    th.save(policy.state_dict(), model_path)
    print(f"--- Training Finished. Policy saved to {model_path} ---")

# ----------------------------------------------------
# C. Execution Block (Runs all 30 PG Experiments)
# ----------------------------------------------------

if __name__ == '__main__':
    
    # 1. PPO Hyperparameter Grid (10 Runs)
    ppo_configs = [
        {'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 1024, 'clip_range': 0.2},
        {'learning_rate': 1e-4, 'gamma': 0.999, 'n_steps': 2048, 'clip_range': 0.1},
        {'learning_rate': 3e-4, 'gamma': 0.999, 'n_steps': 1024, 'clip_range': 0.1},
        {'learning_rate': 3e-4, 'gamma': 0.95, 'n_steps': 512, 'clip_range': 0.25}, 
        {'learning_rate': 1e-4, 'gamma': 0.99, 'n_steps': 512, 'clip_range': 0.3},   
        {'learning_rate': 5e-4, 'gamma': 0.99, 'n_steps': 1024, 'clip_range': 0.15}, 
        {'learning_rate': 1e-5, 'gamma': 0.999, 'n_steps': 2048, 'clip_range': 0.2},
        {'learning_rate': 3e-4, 'gamma': 0.95, 'n_steps': 2048, 'clip_range': 0.1},  
        {'learning_rate': 1e-4, 'gamma': 0.99, 'n_steps': 1024, 'clip_range': 0.25},
        {'learning_rate': 5e-4, 'gamma': 0.999, 'n_steps': 512, 'clip_range': 0.2}   
    ]

    # 2. A2C Hyperparameter Grid (10 Runs)
    a2c_configs = [
        {'learning_rate': 7e-4, 'gamma': 0.99, 'n_steps': 5, 'vf_coef': 0.5},
        {'learning_rate': 3e-4, 'gamma': 0.95, 'n_steps': 8, 'ent_coef': 0.01},
        {'learning_rate': 7e-4, 'gamma': 0.999, 'n_steps': 5, 'vf_coef': 0.75},
        {'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 16, 'vf_coef': 0.5},
        {'learning_rate': 1e-3, 'gamma': 0.99, 'n_steps': 5, 'vf_coef': 0.25},
        {'learning_rate': 7e-4, 'gamma': 0.95, 'n_steps': 8, 'vf_coef': 0.5},
        {'learning_rate': 3e-4, 'gamma': 0.999, 'n_steps': 5, 'vf_coef': 0.9},
        {'learning_rate': 1e-3, 'gamma': 0.95, 'n_steps': 16, 'vf_coef': 0.75},
        {'learning_rate': 5e-4, 'gamma': 0.99, 'n_steps': 8, 'ent_coef': 0.001},
        {'learning_rate': 1e-4, 'gamma': 0.99, 'n_steps': 16, 'ent_coef': 0.01}
    ]

    # 3. REINFORCE Hyperparameter Grid (10 Runs)
    reinforce_configs = [
        {'learning_rate': 5e-4, 'gamma': 0.95, 'num_episodes': 1000},
        {'learning_rate': 5e-4, 'gamma': 0.999, 'num_episodes': 1000},
        {'learning_rate': 1e-3, 'gamma': 0.99, 'num_episodes': 500},
        {'learning_rate': 1e-4, 'gamma': 0.99, 'num_episodes': 1500},
        {'learning_rate': 1e-3, 'gamma': 0.90, 'num_episodes': 500},
        {'learning_rate': 2e-4, 'gamma': 0.999, 'num_episodes': 1500},
        {'learning_rate': 3e-3, 'gamma': 0.99, 'num_episodes': 500},
        {'learning_rate': 1e-3, 'gamma': 0.999, 'num_episodes': 1000},
        {'learning_rate': 5e-4, 'gamma': 0.97, 'num_episodes': 2000},
        {'learning_rate': 1e-4, 'gamma': 0.95, 'num_episodes': 500},
    ]

    # --- EXECUTE TRAINING RUNS ---
    
    # 1. Run PPO Training
    print("\n--- Starting PPO Training (SB3) ---")
    #run_policy_gradient_training("PPO", PPO, ppo_configs)
    
    # 2. Run A2C Training
    print("\n--- Starting A2C Training (SB3) ---")
    #run_policy_gradient_training("A2C", A2C, a2c_configs)

    # 3. Run Manual REINFORCE Training
    print("\n--- Starting REINFORCE Training (Manual/PyTorch) ---")
    run_count = 1
    for config in reinforce_configs:
        run_reinforce_training(run_count, config)
        run_count += 1
    
    print("\nAll Policy Gradient training runs completed!")