import pygame
import numpy as np

# --- Pygame Configuration ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)
YELLOW = (200, 200, 50)
GREEN = (50, 200, 50)
BLUE = (50, 50, 200)

class PygameRenderer:
    def __init__(self, n_students,fps):
        """Initializes the Pygame window and assets."""
        pygame.init()
        self.n_students = n_students
        self.fps = fps
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Dropout Prevention RL Agent Simulation")
        self.font = pygame.font.Font(None, 36)
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Calculate bar width and spacing
        self.bar_width = (SCREEN_WIDTH - 100) / self.n_students - 20
        self.spacing = 20
        self.start_x = 50

    def _get_bar_color(self, avg_risk):
        """Maps average risk level to a color."""
        if avg_risk >= 0.7:
            return RED      # High Risk
        elif avg_risk >= 0.3:
            return YELLOW   # Moderate Risk
        else:
            return GREEN    # Low Risk

    def render(self, students_data, current_step):
        """Draws the current state of the environment."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
        if not self.running:
            return # Skip rendering if user closed the window

        self.screen.fill(BLACK)

        # --- Draw Student Bars ---
        for i in range(self.n_students):
            # Data extraction for student i
            risk_factors = students_data[i, :3]
            is_active = students_data[i, 3]
            avg_risk = np.mean(risk_factors)
            
            # 1. Bar Dimensions: Height inversely proportional to risk
            # Max possible bar height for AvgRisk=0.0
            max_bar_height = SCREEN_HEIGHT * 0.7 
            # Height scales down as risk approaches 1.0 (max risk)
            bar_height = max_bar_height * (1.0 - avg_risk) 
            
            bar_x = self.start_x + i * (self.bar_width + self.spacing)
            bar_y = SCREEN_HEIGHT - bar_height - 50 # Base position
            
            # 2. Bar Color based on Risk
            bar_color = self._get_bar_color(avg_risk)
            
            # Draw the bar
            pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, self.bar_width, bar_height))

            # 3. Add Labels
            # Risk Value
            risk_text = self.font.render(f"{avg_risk:.2f}", True, WHITE)
            self.screen.blit(risk_text, (bar_x, bar_y - 30))
            
            # Student ID
            id_text = self.font.render(f"S{i+1}", True, WHITE)
            self.screen.blit(id_text, (bar_x + self.bar_width/2 - id_text.get_width()/2, SCREEN_HEIGHT - 40))
            
            # 4. Intervention Status Icon
            if is_active == 1.0:
                # Draw a pulsing blue circle to indicate active support (Intervention Icon)
                center_x = bar_x + self.bar_width / 2
                center_y = bar_y - 50
                
                # Make the icon "flash" based on step count for better visibility
                if current_step % 4 < 2:
                    pygame.draw.circle(self.screen, BLUE, (int(center_x), int(center_y)), 15)
                else:
                    pygame.draw.circle(self.screen, BLUE, (int(center_x), int(center_y)), 10)


        # --- Global Stats Display ---
        step_text = self.font.render(f"Step: {current_step}", True, WHITE)
        self.screen.blit(step_text, (SCREEN_WIDTH - 200, 20))
        
        # Update the display
        pygame.display.flip()
        
        # Control the frame rate (important for smooth simulation viewing)
        self.clock.tick(self.fps)

    def close(self):
        """Quits Pygame."""
        if self.running:
            pygame.quit()

# --- Helper function for static random action demo ---
def run_random_agent_demo(env):
    """
    Creates a static file demonstration by running the environment 
    with random actions (required by the prompt).
    """
    print("\nRunning Static Random Agent Demo...")
    obs, info = env.reset()
    done = False
    
    # Run for a fixed short period to demonstrate dynamics
    for _ in range(50): 
        if done:
            break
            
        # Agent takes a random action
        action = env.action_space.sample() 
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # The renderer handles the display update in 'human' mode
        if env.render_mode == "human":
            # Small delay to see the steps
            pygame.time.wait(200) 
            
    if env.render_mode == "human":
        # Keep the final state displayed for a moment
        pygame.time.wait(2000) 
        
    print("Static Demo Complete.")
    env.close()


