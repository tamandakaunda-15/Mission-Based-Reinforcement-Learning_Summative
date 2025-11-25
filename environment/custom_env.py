import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Assuming rendering.py exists and handles visualization (e.g., Pygame)
# You will need to implement the actual rendering functions in rendering.py
from .rendering import PygameRenderer

# --- Environment Parameters ---
N_STUDENTS = 5      # Number of students the agent manages
MAX_STEPS = 100     # Length of one simulated academic period
DROPOUT_THRESHOLD = 0.9 # Average risk level that triggers dropout

class DropoutPreventionEnv(gym.Env):
    """
    Gymnasium environment for training an RL agent to allocate support 
    to students at high risk of dropping out.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # Action Space: Choose 1 of N students for intervention, or do nothing (N+1 total actions)
        # 0: Do Nothing, 1-5: Intervene on Student 1-5
        self.action_space = spaces.Discrete(N_STUDENTS + 1)
        
        # Observation Space: (N * 4) vector of normalized risk factors and intervention status
        # [R_Acad1, R_Attn1, R_Socio1, I_Active1, ..., I_ActiveN]
        low = np.zeros(N_STUDENTS * 4, dtype=np.float32)
        high = np.ones(N_STUDENTS * 4, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(N_STUDENTS * 4,), dtype=np.float32)

        self.students_data = None # Holds (N_STUDENTS, 4) array of risk data
        self.current_step = 0
        self.max_steps = MAX_STEPS
        self.dropout_threshold = DROPOUT_THRESHOLD
        self.last_observation = None
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.renderer = None # Renderer object initialized in reset
        if self.render_mode == "human":
            from .rendering import PygameRenderer
            fps = self.metadata.get("render_fps", 4)

            self.renderer = PygameRenderer(N_STUDENTS, fps)

    def _get_obs(self):
        """Flattens the student data array into the required observation vector."""
        return self.students_data.flatten()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initialize students with random, diverse starting risk profiles (0.2 to 0.8)
        self.students_data = self.np_random.uniform(low=0.2, high=0.8, size=(N_STUDENTS, 4))
        # Ensure I_Active (index 3) starts at 0.0
        self.students_data[:, 3] = 0.0 
        
        observation = self._get_obs()
        self.last_observation = observation # Store for reward calculation in step()
        info = {"status": "episode_start"}
        
        if self.render_mode == "human" and self.renderer is None:
             self.renderer = PygameRenderer(N_STUDENTS)

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _calculate_risk_drift(self, risk_factors):
        """Simulates the natural deterioration of student risk."""
        # Base risk increase
        base_drift = self.np_random.uniform(0.01, 0.05, size=3) 
        
        # Accelerated drift: higher current risk leads to faster increase
        accelerating_drift = risk_factors[:3] * self.np_random.uniform(0.05, 0.15)
        
        return base_drift + accelerating_drift

    def step(self, action):
        reward = 0.0
        terminated = False
        self.current_step += 1
        
        # Store previous average risk for comparison
        prev_avg_risk = np.mean(self.students_data[:, :3])
        
        # --- 1. Apply Action (Intervention Allocation) ---
        if action > 0:
            student_idx = action - 1
            
            # Check if student is already active
            if self.students_data[student_idx, 3] == 1.0:
                reward -= 5.0 # High penalty for resource waste
            else:
                # Set Intervention Status to Active (1.0)
                self.students_data[student_idx, 3] = 1.0
                reward -= 0.1 # Small resource cost penalty
                
                # Apply immediate, significant risk reduction
                reduction_amount = self.np_random.uniform(0.2, 0.4, size=3)
                self.students_data[student_idx, :3] = self.students_data[student_idx, :3] - reduction_amount
                
                # Reward for proactive intervention
                reward += 2.0 
        
        # --- 2. Simulate Environment Dynamics (Risk Change) ---
        for i in range(N_STUDENTS):
            # Apply natural risk drift
            drift = self._calculate_risk_drift(self.students_data[i, :3])
            self.students_data[i, :3] = self.students_data[i, :3] + drift
            
            # Simulate Intervention Decay (Resource Slot Release)
            if self.students_data[i, 3] == 1.0: 
                # 30% chance the intervention slot frees up this step
                if self.np_random.random() < 0.3: 
                    self.students_data[i, 3] = 0.0 
        
        # Clip all risk factors to ensure they stay in [0.0, 1.0]
        self.students_data[:, :3] = np.clip(self.students_data[:, :3], 0.0, 1.0)
        
        # --- 3. Calculate Reward and Termination ---
        current_avg_risk = np.mean(self.students_data[:, :3])
        
        # Reward for overall risk reduction
        if current_avg_risk < prev_avg_risk:
            reward += 5.0
        
        # Dropout Check (Terminal Condition 1)
        for i in range(N_STUDENTS):
            avg_risk = np.mean(self.students_data[i, :3]) 
            if avg_risk > self.dropout_threshold:
                # Student has dropped out
                reward -= 50.0 
                terminated = True
                break # Episode terminates on first dropout

        # Time Limit Check (Terminal Condition 2)
        if self.current_step >= self.max_steps:
            # Reward bonus for surviving the full period without dropouts
            if not terminated: 
                 reward += 10.0
            terminated = True
        
        # Get next observation
        observation = self._get_obs()
        info = {"avg_risk": current_avg_risk}
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _render_frame(self):
        """Calls the rendering logic defined in rendering.py."""
        if self.renderer:
            self.renderer.render(self.students_data, self.current_step)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def close(self):
        """Cleanup visualization resources."""
        if self.renderer:
            self.renderer.close()