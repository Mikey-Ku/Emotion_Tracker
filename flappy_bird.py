"""
Flappy Bird - 4 Button Controller
==================================

Classic Flappy Bird game with simple one-button controls.

Controls:
- Press ANY button (1, 2, 3, or 4) to JUMP/FLAP
- Or press SPACE for testing

Game Features:
- Avoid pipes
- Score points by passing through pipes
- Increasing difficulty
- Simple, addictive gameplay
"""

import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
SKY_BLUE = (135, 206, 235)
GROUND_GREEN = (34, 139, 34)
PIPE_GREEN = (0, 128, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)

# Button colors
RED = (220, 30, 30)
GREEN = (30, 220, 30)
BLUE = (30, 30, 220)
BUTTON_YELLOW = (220, 220, 30)

# Game settings
GRAVITY = 0.5
FLAP_STRENGTH = -10
BIRD_SIZE = 30
PIPE_WIDTH = 70
PIPE_GAP = 180
PIPE_SPEED = 3
GROUND_HEIGHT = 100


class Bird:
    """The player's bird."""
    
    def __init__(self):
        self.x = 150
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.size = BIRD_SIZE
        self.rotation = 0
    
    def flap(self):
        """Make the bird flap/jump."""
        self.velocity = FLAP_STRENGTH
    
    def update(self):
        """Update bird physics."""
        # Apply gravity
        self.velocity += GRAVITY
        self.y += self.velocity
        
        # Update rotation based on velocity
        self.rotation = max(-30, min(30, -self.velocity * 3))
        
        # Prevent going off screen top
        if self.y < 0:
            self.y = 0
            self.velocity = 0
    
    def draw(self, screen):
        """Draw the bird."""
        # Create bird surface
        bird_surface = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        
        # Draw bird body (circle)
        pygame.draw.circle(bird_surface, YELLOW, (self.size, self.size), self.size)
        pygame.draw.circle(bird_surface, ORANGE, (self.size, self.size), self.size, 3)
        
        # Draw eye
        eye_x = self.size + 8
        eye_y = self.size - 5
        pygame.draw.circle(bird_surface, WHITE, (eye_x, eye_y), 6)
        pygame.draw.circle(bird_surface, BLACK, (eye_x + 2, eye_y), 3)
        
        # Draw beak
        beak_points = [
            (self.size + 15, self.size),
            (self.size + 25, self.size - 3),
            (self.size + 25, self.size + 3)
        ]
        pygame.draw.polygon(bird_surface, ORANGE, beak_points)
        
        # Rotate bird
        rotated = pygame.transform.rotate(bird_surface, self.rotation)
        rect = rotated.get_rect(center=(self.x, self.y))
        
        screen.blit(rotated, rect)
    
    def get_rect(self):
        """Get collision rectangle."""
        return pygame.Rect(self.x - self.size // 2, self.y - self.size // 2, 
                          self.size, self.size)


class Pipe:
    """An obstacle pipe."""
    
    def __init__(self, x):
        self.x = x
        self.gap_y = random.randint(150, SCREEN_HEIGHT - GROUND_HEIGHT - 150 - PIPE_GAP)
        self.width = PIPE_WIDTH
        self.passed = False
    
    def update(self):
        """Move pipe left."""
        self.x -= PIPE_SPEED
    
    def draw(self, screen):
        """Draw the pipe."""
        # Top pipe
        top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y)
        pygame.draw.rect(screen, PIPE_GREEN, top_rect)
        pygame.draw.rect(screen, (0, 100, 0), top_rect, 3)
        
        # Top pipe cap
        cap_top = pygame.Rect(self.x - 5, self.gap_y - 20, self.width + 10, 20)
        pygame.draw.rect(screen, PIPE_GREEN, cap_top)
        pygame.draw.rect(screen, (0, 100, 0), cap_top, 3)
        
        # Bottom pipe
        bottom_y = self.gap_y + PIPE_GAP
        bottom_rect = pygame.Rect(self.x, bottom_y, self.width, 
                                  SCREEN_HEIGHT - GROUND_HEIGHT - bottom_y)
        pygame.draw.rect(screen, PIPE_GREEN, bottom_rect)
        pygame.draw.rect(screen, (0, 100, 0), bottom_rect, 3)
        
        # Bottom pipe cap
        cap_bottom = pygame.Rect(self.x - 5, bottom_y, self.width + 10, 20)
        pygame.draw.rect(screen, PIPE_GREEN, cap_bottom)
        pygame.draw.rect(screen, (0, 100, 0), cap_bottom, 3)
    
    def is_off_screen(self):
        """Check if pipe is off screen."""
        return self.x + self.width < 0
    
    def collides_with(self, bird):
        """Check collision with bird."""
        bird_rect = bird.get_rect()
        
        # Top pipe collision
        top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y)
        if bird_rect.colliderect(top_rect):
            return True
        
        # Bottom pipe collision
        bottom_y = self.gap_y + PIPE_GAP
        bottom_rect = pygame.Rect(self.x, bottom_y, self.width,
                                  SCREEN_HEIGHT - GROUND_HEIGHT - bottom_y)
        if bird_rect.colliderect(bottom_rect):
            return True
        
        return False


class FlappyBirdGame:
    """Main game class."""
    
    def __init__(self):
        # Create screen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird - 4 Button Controller")
        
        # Clock
        self.clock = pygame.time.Clock()
        
        # Game objects
        self.bird = Bird()
        self.pipes = []
        self.pipe_spawn_timer = 0
        self.pipe_spawn_delay = 90  # frames
        
        # Game state
        self.score = 0
        self.high_score = 0
        self.game_state = "START"  # START, PLAYING, GAME_OVER
        
        # Button states
        self.button_pressed = [False, False, False, False]
        
        # Fonts
        self.title_font = pygame.font.Font(None, 70)
        self.score_font = pygame.font.Font(None, 60)
        self.text_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Pre-render button sprites
        self.button_sprites = self.create_button_sprites()
        
        # Background elements
        self.cloud_positions = [(random.randint(0, SCREEN_WIDTH), 
                                random.randint(50, 200)) for _ in range(5)]
    
    def create_button_sprites(self):
        """Pre-render button sprites."""
        button_size = 100
        sprites = {}
        
        colors = [RED, GREEN, BLUE, BUTTON_YELLOW]
        labels = ["JUMP!", "JUMP!", "JUMP!", "JUMP!"]
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            # Normal state
            normal = pygame.Surface((button_size, button_size + 6), pygame.SRCALPHA)
            pygame.draw.rect(normal, (50, 50, 50), (0, 6, button_size, button_size), 
                           border_radius=15)
            pygame.draw.rect(normal, color, (0, 0, button_size, button_size), 
                           border_radius=15)
            pygame.draw.rect(normal, WHITE, (0, 0, button_size, button_size), 4, 
                           border_radius=15)
            
            font = pygame.font.Font(None, 28)
            text = font.render(label, True, WHITE)
            text_rect = text.get_rect(center=(button_size // 2, button_size // 2))
            normal.blit(text, text_rect)
            
            num_font = pygame.font.Font(None, 24)
            num_text = num_font.render(str(i + 1), True, (200, 200, 200))
            num_rect = num_text.get_rect(center=(button_size // 2, button_size - 20))
            normal.blit(num_text, num_rect)
            
            # Pressed state
            pressed = pygame.Surface((button_size, button_size + 6), pygame.SRCALPHA)
            pygame.draw.rect(pressed, (50, 50, 50), (0, 6, button_size, button_size), 
                           border_radius=15)
            pygame.draw.rect(pressed, color, (0, 5, button_size, button_size), 
                           border_radius=15)
            pygame.draw.rect(pressed, WHITE, (0, 5, button_size, button_size), 4, 
                           border_radius=15)
            
            text_rect = text.get_rect(center=(button_size // 2, button_size // 2 + 5))
            pressed.blit(text, text_rect)
            num_rect = num_text.get_rect(center=(button_size // 2, button_size - 15))
            pressed.blit(num_text, num_rect)
            
            sprites[i] = {'normal': normal, 'pressed': pressed}
        
        return sprites
    
    def spawn_pipe(self):
        """Spawn a new pipe."""
        self.pipes.append(Pipe(SCREEN_WIDTH))
    
    def update(self):
        """Update game state."""
        if self.game_state != "PLAYING":
            return
        
        # Update bird
        self.bird.update()
        
        # Check ground collision
        if self.bird.y + self.bird.size // 2 >= SCREEN_HEIGHT - GROUND_HEIGHT:
            self.game_state = "GAME_OVER"
            if self.score > self.high_score:
                self.high_score = self.score
        
        # Update pipes
        self.pipe_spawn_timer += 1
        if self.pipe_spawn_timer >= self.pipe_spawn_delay:
            self.spawn_pipe()
            self.pipe_spawn_timer = 0
        
        for pipe in self.pipes[:]:
            pipe.update()
            
            # Check collision
            if pipe.collides_with(self.bird):
                self.game_state = "GAME_OVER"
                if self.score > self.high_score:
                    self.high_score = self.score
            
            # Check if passed
            if not pipe.passed and pipe.x + pipe.width < self.bird.x:
                pipe.passed = True
                self.score += 1
            
            # Remove off-screen pipes
            if pipe.is_off_screen():
                self.pipes.remove(pipe)
    
    def draw(self):
        """Draw everything."""
        # Sky background
        self.screen.fill(SKY_BLUE)
        
        # Draw clouds
        for x, y in self.cloud_positions:
            self.draw_cloud(x, y)
        
        # Draw pipes
        for pipe in self.pipes:
            pipe.draw(self.screen)
        
        # Draw bird
        self.bird.draw(self.screen)
        
        # Draw ground
        ground_rect = pygame.Rect(0, SCREEN_HEIGHT - GROUND_HEIGHT, 
                                  SCREEN_WIDTH, GROUND_HEIGHT)
        pygame.draw.rect(self.screen, GROUND_GREEN, ground_rect)
        pygame.draw.rect(self.screen, (0, 100, 0), ground_rect, 3)
        
        # Draw grass pattern
        for x in range(0, SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, (0, 150, 0),
                           (x, SCREEN_HEIGHT - GROUND_HEIGHT),
                           (x + 10, SCREEN_HEIGHT - GROUND_HEIGHT + 10), 2)
        
        # Draw score
        if self.game_state == "PLAYING":
            score_text = self.score_font.render(str(self.score), True, WHITE)
            score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, 60))
            # Shadow
            shadow = self.score_font.render(str(self.score), True, BLACK)
            self.screen.blit(shadow, (score_rect.x + 3, score_rect.y + 3))
            self.screen.blit(score_text, score_rect)
        
        # Draw controller
        self.draw_controller()
        
        # Draw overlays
        if self.game_state == "START":
            self.draw_start_screen()
        elif self.game_state == "GAME_OVER":
            self.draw_game_over_screen()
    
    def draw_cloud(self, x, y):
        """Draw a simple cloud."""
        pygame.draw.circle(self.screen, WHITE, (x, y), 20)
        pygame.draw.circle(self.screen, WHITE, (x + 20, y), 25)
        pygame.draw.circle(self.screen, WHITE, (x + 40, y), 20)
        pygame.draw.ellipse(self.screen, WHITE, (x - 10, y + 10, 60, 20))
    
    def draw_controller(self):
        """Draw 4-button controller."""
        button_size = 100
        spacing = 30
        start_x = (SCREEN_WIDTH - (button_size * 4 + spacing * 3)) // 2
        y = SCREEN_HEIGHT - GROUND_HEIGHT - button_size - 20
        
        for i in range(4):
            x = start_x + i * (button_size + spacing)
            sprite = self.button_sprites[i]['pressed' if self.button_pressed[i] else 'normal']
            self.screen.blit(sprite, (x, y))
    
    def draw_start_screen(self):
        """Draw start screen."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        # Title
        title = self.title_font.render("FLAPPY BIRD", True, YELLOW)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 150))
        # Shadow
        shadow = self.title_font.render("FLAPPY BIRD", True, BLACK)
        self.screen.blit(shadow, (title_rect.x + 3, title_rect.y + 3))
        self.screen.blit(title, title_rect)
        
        # Instructions
        instructions = [
            "Press ANY button to FLAP!",
            "",
            "Avoid the pipes!",
            "",
            "Press SPACE or any button to Start"
        ]
        
        y = 280
        for line in instructions:
            text = self.text_font.render(line, True, WHITE)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, y))
            self.screen.blit(text, text_rect)
            y += 45
        
        # High score
        if self.high_score > 0:
            high_text = self.small_font.render(f"High Score: {self.high_score}", 
                                              True, YELLOW)
            high_rect = high_text.get_rect(center=(SCREEN_WIDTH // 2, 450))
            self.screen.blit(high_text, high_rect)
    
    def draw_game_over_screen(self):
        """Draw game over screen."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        # Game Over
        title = self.title_font.render("GAME OVER", True, RED)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 180))
        shadow = self.title_font.render("GAME OVER", True, BLACK)
        self.screen.blit(shadow, (title_rect.x + 3, title_rect.y + 3))
        self.screen.blit(title, title_rect)
        
        # Score
        score_text = self.score_font.render(f"Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, 280))
        self.screen.blit(score_text, score_rect)
        
        # High score
        if self.score >= self.high_score:
            new_high = self.text_font.render("NEW HIGH SCORE!", True, YELLOW)
            new_high_rect = new_high.get_rect(center=(SCREEN_WIDTH // 2, 350))
            self.screen.blit(new_high, new_high_rect)
        else:
            high_text = self.text_font.render(f"High Score: {self.high_score}", 
                                             True, YELLOW)
            high_rect = high_text.get_rect(center=(SCREEN_WIDTH // 2, 350))
            self.screen.blit(high_text, high_rect)
        
        # Restart
        restart = self.text_font.render("Press R to Restart", True, WHITE)
        restart_rect = restart.get_rect(center=(SCREEN_WIDTH // 2, 420))
        self.screen.blit(restart, restart_rect)
    
    def reset_game(self):
        """Reset the game."""
        self.bird = Bird()
        self.pipes = []
        self.pipe_spawn_timer = 0
        self.score = 0
        self.game_state = "START"
    
    def run(self):
        """Main game loop."""
        running = True
        
        print("="*60)
        print("Flappy Bird - 4 Button Controller")
        print("="*60)
        print("Controls:")
        print("  Press ANY button (1, 2, 3, or 4) to JUMP/FLAP")
        print("  Or press SPACE for testing")
        print("  R - Restart")
        print("  Q - Quit")
        print("="*60 + "\n")
        
        while running:
            self.clock.tick(60)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    
                    elif event.key == pygame.K_r and self.game_state == "GAME_OVER":
                        self.reset_game()
                    
                    # Any button makes bird flap
                    elif event.key in [pygame.K_SPACE, pygame.K_1, pygame.K_2, 
                                      pygame.K_3, pygame.K_4, pygame.K_UP]:
                        if self.game_state == "START":
                            self.game_state = "PLAYING"
                            self.bird.flap()
                        elif self.game_state == "PLAYING":
                            self.bird.flap()
                        
                        # Visual feedback
                        if event.key == pygame.K_1:
                            self.button_pressed[0] = True
                        elif event.key == pygame.K_2:
                            self.button_pressed[1] = True
                        elif event.key == pygame.K_3:
                            self.button_pressed[2] = True
                        elif event.key == pygame.K_4:
                            self.button_pressed[3] = True
                
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_1:
                        self.button_pressed[0] = False
                    elif event.key == pygame.K_2:
                        self.button_pressed[1] = False
                    elif event.key == pygame.K_3:
                        self.button_pressed[2] = False
                    elif event.key == pygame.K_4:
                        self.button_pressed[3] = False
            
            # Update
            self.update()
            
            # Draw
            self.draw()
            pygame.display.flip()
        
        # Quit
        pygame.quit()
        print("\nGame ended!")
        print(f"Final Score: {self.score}")
        print(f"High Score: {self.high_score}")


if __name__ == "__main__":
    game = FlappyBirdGame()
    game.run()
