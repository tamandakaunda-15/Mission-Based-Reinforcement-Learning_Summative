import time
from environment.custom_env import DropoutPreventionEnv
from environment.rendering import run_random_agent_demo

if __name__ == '__main__':
    print("Initializing environment for Static Random Action Demo...")
    
    # 1. Initialize the Environment (Critical: Must be in "human" render mode)
    env = DropoutPreventionEnv(render_mode="human")
    
    # 2. Execute the Demo Function
    # This function resets the environment, takes 50 random steps,
    # and handles the rendering via the Pygame window.
    run_random_agent_demo(env)
    
    # 3. Cleanup
    env.close()
    print("Demo script finished. Please save the recorded video/GIF as your static file artifact.")