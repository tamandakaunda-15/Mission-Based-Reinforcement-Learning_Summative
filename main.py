import os
import numpy as np
from stable_baselines3 import DQN, PPO, A2C 
from stable_baselines3.common.env_util import make_vec_env 

# Fix for imports when running from project root
import sys
sys.path.append(os.path.abspath('.'))
from environment.custom_env import DropoutPreventionEnv

# --- FINAL CONFIGURATION (Based on Analysis) ---
# NOTE: The model name MUST match the file saved in models/pg/
BEST_MODEL_NAME = "PPO_Run_1_lr0.0003_nsteps1024.zip" 
MODEL_CLASS = PPO
BEST_MODEL_PATH = f"./models/pg/{BEST_MODEL_NAME}" 

def run_best_agent():
    print(f"Loading and running the best agent: {BEST_MODEL_NAME}")
    
    # 1. Initialize Environment Correctly
    # Use make_vec_env to wrap the environment (required by SB3) and pass render_mode via env_kwargs.
    env = make_vec_env(DropoutPreventionEnv, n_envs=1, env_kwargs={'render_mode': 'human'})

    # 2. Load the Model
    try:
        if MODEL_CLASS == DQN:
            model = DQN.load(BEST_MODEL_PATH, env=env)
        elif MODEL_CLASS == PPO:
            model = PPO.load(BEST_MODEL_PATH, env=env)
        elif MODEL_CLASS == A2C:
            model = A2C.load(BEST_MODEL_PATH, env=env)
        else:
            raise ValueError("Invalid MODEL_CLASS specified.")
    except Exception as e:
        print(f"ERROR: Could not load model from {BEST_MODEL_PATH}. Check file path.")
        print(f"Details: {e}")
        env.close()
        return

    # 3. Run Simulation
    obs = env.reset() # Vectorized environments return only the obs array
    
    episodes = 5 
    for episode in range(episodes):
        done = [False] # Must be a list/array for the vectorized environment
        total_reward = 0
        
        print(f"\n--- Running Episode {episode + 1} ---")
        
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True) 
            
            # CRITICAL FIX: Unpack 4 elements: obs, reward_arr, done_arr, info_list
            # This resolves the ValueError: expected 5, got 4
            obs, reward_arr, done_arr, info = env.step(action)
            
            # Update 'done' status using the single flag (done_arr[0])
            done[0] = done_arr[0]
            
            # Add scalar reward (first element of array)
            total_reward += reward_arr[0] 
        
        print(f"Episode {episode+1} finished. Final Total Reward: {total_reward:.2f}")
        
        # Reset for the next episode
        obs = env.reset()
        done = [False]
        
    env.close()

if __name__ == '__main__':
    # Ensure numpy is imported for predict method to work consistently
    try:
        import numpy as np
    except ImportError:
        print("Numpy not found. Please install: pip install numpy")
        sys.exit(1)
        
    run_best_agent()