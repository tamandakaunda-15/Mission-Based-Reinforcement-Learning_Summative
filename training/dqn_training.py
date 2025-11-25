import os
import itertools
# Necessary for relative imports (fixes ModuleNotFoundError)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# Import the custom environment from the 'environment' directory
from environment.custom_env import DropoutPreventionEnv

# --- Configuration (GLOBAL CONSTANTS) ---
LOG_DIR = "./logs/dqn/"
MODEL_DIR = "./models/dqn/"
TOTAL_TIMESTEPS = 100000 

def train_dqn(run_id, params):
    """Initializes, trains, and saves a single DQN model run."""
    
    # 1. Setup Logging and Directories
    # Use explicit, full variable names to avoid runtime conflicts
    run_name = f"DQN_Run_{run_id}_lr{params['learning_rate']}_gamma{params['gamma']}_b{params['batch_size']}"
    log_path = os.path.join(LOG_DIR, run_name)
    model_path = os.path.join(MODEL_DIR, f"{run_name}.zip")
    
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 2. Environment Setup 
    env = make_vec_env(DropoutPreventionEnv, n_envs=1)

    # 3. Model Initialization (Passing all required parameters)
    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=LOG_DIR,
        
        # Core Parameters
        learning_rate=params['learning_rate'],
        gamma=params['gamma'],
        buffer_size=params['buffer_size'],
        batch_size=params['batch_size'], # Now correctly passed
        
        # Exploration Parameters (Required for Analysis Table)
        exploration_initial_eps=params['epsilon_start'],
        exploration_final_eps=params['epsilon_end'],
        exploration_fraction=params['epsilon_decay'],
    )

    # 4. Train the Model
    print(f"--- Starting Training for {run_name} ---")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        reset_num_timesteps=True,
        tb_log_name=run_name
    )
    print(f"--- Training Finished for {run_name} ---")

    # 5. Save Model
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    
    # --- Hyperparameter Grid Definition (10 UNIQUE RUNS DEFINED MANUALLY) ---
    # These runs ensure variation in all critical parameters (LR, Gamma, Buffer, Batch, Decay)
    dqn_configs = [
        # LR, Gamma, Buffer, Batch, Eps_start, Eps_end, Eps_decay
        {'learning_rate': 1e-4, 'gamma': 0.99, 'buffer_size': 100000, 'batch_size': 32, 'epsilon_start': 1.0, 'epsilon_end': 0.05, 'epsilon_decay': 0.1},
        {'learning_rate': 5e-4, 'gamma': 0.999, 'buffer_size': 50000, 'batch_size': 128, 'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.3},
        {'learning_rate': 1e-4, 'gamma': 0.95, 'buffer_size': 100000, 'batch_size': 128, 'epsilon_start': 1.0, 'epsilon_end': 0.05, 'epsilon_decay': 0.5}, # Slow decay
        {'learning_rate': 5e-4, 'gamma': 0.99, 'buffer_size': 500000, 'batch_size': 32, 'epsilon_start': 0.5, 'epsilon_end': 0.01, 'epsilon_decay': 0.1}, # Start lower
        {'learning_rate': 1e-5, 'gamma': 0.999, 'buffer_size': 100000, 'batch_size': 64, 'epsilon_start': 1.0, 'epsilon_end': 0.1, 'epsilon_decay': 0.3},
        {'learning_rate': 3e-4, 'gamma': 0.99, 'buffer_size': 50000, 'batch_size': 128, 'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.5},
        {'learning_rate': 1e-4, 'gamma': 0.999, 'buffer_size': 500000, 'batch_size': 32, 'epsilon_start': 1.0, 'epsilon_end': 0.05, 'epsilon_decay': 0.1},
        {'learning_rate': 5e-4, 'gamma': 0.95, 'buffer_size': 100000, 'batch_size': 64, 'epsilon_start': 0.8, 'epsilon_end': 0.05, 'epsilon_decay': 0.3},
        {'learning_rate': 1e-4, 'gamma': 0.99, 'buffer_size': 50000, 'batch_size': 128, 'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
        {'learning_rate': 3e-4, 'gamma': 0.999, 'buffer_size': 100000, 'batch_size': 32, 'epsilon_start': 1.0, 'epsilon_end': 0.05, 'epsilon_decay': 0.5},
    ]

    # --- Execute 10 Runs ---
    run_count = 1
    for run_config in dqn_configs:
        config = {
            'learning_rate': run_config['learning_rate'],
            'gamma': run_config['gamma'],
            'buffer_size': run_config['buffer_size'],
            'batch_size': run_config['batch_size'],
            'epsilon_start': run_config['epsilon_start'],
            'epsilon_end': run_config['epsilon_end'],
            'epsilon_decay': run_config['epsilon_decay'],
        }
        train_dqn(run_count, config)
        run_count += 1
        
    print("\nAll 10 Targeted DQN runs completed!")