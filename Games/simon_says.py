"""
Simon Says Game - Diamond Layout
=================================

A classic Simon Says memory game with 4 colored buttons.
- Diamond-shaped display board in the center
- 4 button controls at the bottom

Current Input: Keyboard (1, 2, 3, 4 keys) or mouse clicks
Future: ESP32 with physical colored buttons via serial

Controls:
- Press 1, 2, 3, or 4 (or click buttons) to match the pattern
- Press SPACE to start
- Press 'r' to restart
- Press 'q' to quit
"""

import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 800

# Colors (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (40, 40, 40)
DARK_GRAY = (25, 25, 25)

# Button colors (normal and lit up)
RED = (180, 20, 20)
RED_BRIGHT = (255, 60, 60)

GREEN = (20, 180, 20)
GREEN_BRIGHT = (60, 255, 60)

BLUE = (20, 20, 180)
BLUE_BRIGHT = (60, 60, 255)

YELLOW = (200, 200, 20)
YELLOW_BRIGHT = (255, 255, 60)

# Game settings
FLASH_DURATION = 500  # milliseconds
FLASH_GAP = 200       # milliseconds between flashes


class DiamondSegment:
    """Represents one segment of the diamond display."""
    
    def __init__(self, color, bright_color, points, button_id):
        self.color = color
        self.bright_color = bright_color
        self.points = points  # List of (x, y) points for polygon
        self.button_id = button_id
        self.is_lit = False
    
    def draw(self, screen):
        """Draw the diamond segment."""
        color = self.bright_color if self.is_lit else self.color
        pygame.draw.polygon(screen, color, self.points)
        pygame.draw.polygon(screen, WHITE, self.points, 3)  # Border
    
    def light_up(self):
        self.is_lit = True
    
    def turn_off(self):
        self.is_lit = False


class ControlButton:
    """Represents a clickable control button at the bottom."""
    
    def __init__(self, color, bright_color, rect, button_id, key, label):
        self.color = color
        self.bright_color = bright_color
        self.rect = rect
        self.button_id = button_id
        self.key = key
        self.label = label
        self.is_pressed = False
        self.is_hovered = False
    
    def draw(self, screen):
        """Draw the control button with 3D effect."""
        # Determine color based on state
        if self.is_pressed:
            color = self.bright_color
            offset = 4
        else:
            color = self.color
            offset = 0
        
        # Draw shadow
        shadow_rect = self.rect.copy()
        shadow_rect.y += 6
        pygame.draw.rect(screen, BLACK, shadow_rect, border_radius=15)
        
        # Draw button
        button_rect = self.rect.copy()
        button_rect.y += offset
        pygame.draw.rect(screen, color, button_rect, border_radius=15)
        
        # Draw highlight if hovered
        if self.is_hovered and not self.is_pressed:
            highlight_rect = button_rect.copy()
            highlight_rect.inflate_ip(-10, -10)
            pygame.draw.rect(screen, WHITE, highlight_rect, 2, border_radius=12)
        
        # Draw border
        pygame.draw.rect(screen, WHITE, button_rect, 3, border_radius=15)
        
        # Draw label
        font = pygame.font.Font(None, 40)
        text = font.render(self.label, True, WHITE)
        text_rect = text.get_rect(center=button_rect.center)
        screen.blit(text, text_rect)
        
        # Draw key hint
        key_font = pygame.font.Font(None, 24)
        key_text = key_font.render(f"[{self.button_id}]", True, (200, 200, 200))
        key_rect = key_text.get_rect(center=(button_rect.centerx, button_rect.bottom - 15))
        screen.blit(key_text, key_rect)
    
    def contains_point(self, pos):
        """Check if a point is inside the button."""
        return self.rect.collidepoint(pos)
    
    def press(self):
        self.is_pressed = True
    
    def release(self):
        self.is_pressed = False


class SimonSaysGame:
    """Main game class for Simon Says."""
    
    def __init__(self):
        # Create screen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Simon Says - Diamond Layout")
        
        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()
        
        # Create diamond display segments
        center_x = SCREEN_WIDTH // 2
        center_y = 280
        size = 150
        
        # Diamond points (top, right, bottom, left)
        top = (center_x, center_y - size)
        right = (center_x + size, center_y)
        bottom = (center_x, center_y + size)
        left = (center_x - size, center_y)
        center = (center_x, center_y)
        
        self.diamond_segments = [
            DiamondSegment(RED, RED_BRIGHT, [top, center, right], 1),      # Top-right (Red)
            DiamondSegment(GREEN, GREEN_BRIGHT, [right, center, bottom], 2),  # Bottom-right (Green)
            DiamondSegment(BLUE, BLUE_BRIGHT, [bottom, center, left], 3),   # Bottom-left (Blue)
            DiamondSegment(YELLOW, YELLOW_BRIGHT, [left, center, top], 4)    # Top-left (Yellow)
        ]
        
        # Create control buttons at the bottom
        button_width = 150
        button_height = 80
        button_y = 650
        spacing = 30
        total_width = (button_width * 4) + (spacing * 3)
        start_x = (SCREEN_WIDTH - total_width) // 2
        
        self.control_buttons = [
            ControlButton(RED, RED_BRIGHT, 
                         pygame.Rect(start_x, button_y, button_width, button_height),
                         1, pygame.K_1, "RED"),
            ControlButton(GREEN, GREEN_BRIGHT,
                         pygame.Rect(start_x + button_width + spacing, button_y, button_width, button_height),
                         2, pygame.K_2, "GREEN"),
            ControlButton(BLUE, BLUE_BRIGHT,
                         pygame.Rect(start_x + (button_width + spacing) * 2, button_y, button_width, button_height),
                         3, pygame.K_3, "BLUE"),
            ControlButton(YELLOW, YELLOW_BRIGHT,
                         pygame.Rect(start_x + (button_width + spacing) * 3, button_y, button_width, button_height),
                         4, pygame.K_4, "YELLOW")
        ]
        
        # Game state
        self.sequence = []
        self.player_sequence = []
        self.current_level = 0
        self.game_state = "START"  # START, SHOWING, PLAYING, GAME_OVER
        self.score = 0
        self.high_score = 0
        
        # Fonts
        self.title_font = pygame.font.Font(None, 70)
        self.text_font = pygame.font.Font(None, 40)
        self.small_font = pygame.font.Font(None, 28)
    
    def add_to_sequence(self):
        """Add a random button to the sequence."""
        self.sequence.append(random.randint(0, 3))
        self.current_level += 1
    
    def show_sequence(self):
        """Display the current sequence to the player."""
        self.game_state = "SHOWING"
        
        for button_index in self.sequence:
            # Light up the diamond segment
            self.diamond_segments[button_index].light_up()
            self.draw()
            pygame.display.flip()
            pygame.time.wait(FLASH_DURATION)
            
            # Turn off the segment
            self.diamond_segments[button_index].turn_off()
            self.draw()
            pygame.display.flip()
            pygame.time.wait(FLASH_GAP)
        
        # Ready for player input
        self.game_state = "PLAYING"
        self.player_sequence = []
    
    def check_input(self, button_index):
        """Check if the player's input matches the sequence."""
        self.player_sequence.append(button_index)
        
        # Check if the current input is correct
        current_step = len(self.player_sequence) - 1
        
        if self.player_sequence[current_step] != self.sequence[current_step]:
            # Wrong input - game over
            self.game_state = "GAME_OVER"
            return False
        
        # Check if the player completed the sequence
        if len(self.player_sequence) == len(self.sequence):
            # Correct! Move to next level
            self.score += 10 * self.current_level
            if self.score > self.high_score:
                self.high_score = self.score
            
            pygame.time.wait(500)  # Brief pause
            self.add_to_sequence()
            pygame.time.wait(1000)  # Pause before showing next sequence
            self.show_sequence()
            return True
        
        return True
    
    def reset_game(self):
        """Reset the game to start a new round."""
        self.sequence = []
        self.player_sequence = []
        self.current_level = 0
        self.score = 0
        self.game_state = "START"
    
    def draw(self):
        """Draw everything on the screen."""
        # Clear screen with gradient effect
        self.screen.fill(DARK_GRAY)
        
        # Draw title area background
        pygame.draw.rect(self.screen, GRAY, (0, 0, SCREEN_WIDTH, 120))
        
        # Draw title
        title = self.title_font.render("SIMON SAYS", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 40))
        self.screen.blit(title, title_rect)
        
        # Draw level and score
        if self.game_state != "START":
            info_text = f"Level: {self.current_level}  |  Score: {self.score}  |  High: {self.high_score}"
            info = self.small_font.render(info_text, True, (200, 200, 200))
            info_rect = info.get_rect(center=(SCREEN_WIDTH // 2, 85))
            self.screen.blit(info, info_rect)
        
        # Draw diamond display
        for segment in self.diamond_segments:
            segment.draw(self.screen)
        
        # Draw center circle
        center_x = SCREEN_WIDTH // 2
        center_y = 280
        pygame.draw.circle(self.screen, GRAY, (center_x, center_y), 30)
        pygame.draw.circle(self.screen, WHITE, (center_x, center_y), 30, 3)
        
        # Draw status message
        if self.game_state == "START":
            msg = self.text_font.render("Press SPACE to Start", True, WHITE)
            msg_rect = msg.get_rect(center=(SCREEN_WIDTH // 2, 500))
            self.screen.blit(msg, msg_rect)
            
            hint = self.small_font.render("Click buttons or press 1, 2, 3, 4", True, (180, 180, 180))
            hint_rect = hint.get_rect(center=(SCREEN_WIDTH // 2, 545))
            self.screen.blit(hint, hint_rect)
        
        elif self.game_state == "SHOWING":
            msg = self.text_font.render("Watch the pattern...", True, (100, 200, 255))
            msg_rect = msg.get_rect(center=(SCREEN_WIDTH // 2, 500))
            self.screen.blit(msg, msg_rect)
        
        elif self.game_state == "PLAYING":
            msg = self.text_font.render("Your turn!", True, (100, 255, 100))
            msg_rect = msg.get_rect(center=(SCREEN_WIDTH // 2, 500))
            self.screen.blit(msg, msg_rect)
            
            # Show progress
            progress = f"{len(self.player_sequence)}/{len(self.sequence)}"
            progress_text = self.small_font.render(progress, True, (200, 200, 200))
            progress_rect = progress_text.get_rect(center=(SCREEN_WIDTH // 2, 540))
            self.screen.blit(progress_text, progress_rect)
        
        elif self.game_state == "GAME_OVER":
            msg = self.title_font.render("GAME OVER!", True, (255, 80, 80))
            msg_rect = msg.get_rect(center=(SCREEN_WIDTH // 2, 480))
            self.screen.blit(msg, msg_rect)
            
            score_msg = self.text_font.render(f"Final Score: {self.score}", True, WHITE)
            score_rect = score_msg.get_rect(center=(SCREEN_WIDTH // 2, 540))
            self.screen.blit(score_msg, score_rect)
        
        # Draw control buttons
        for button in self.control_buttons:
            button.draw(self.screen)
    
    def handle_button_press(self, button_index):
        """Handle a button press with visual feedback."""
        if self.game_state == "PLAYING":
            # Light up both the diamond segment and control button
            self.diamond_segments[button_index].light_up()
            self.control_buttons[button_index].press()
            self.draw()
            pygame.display.flip()
            pygame.time.wait(200)
            self.diamond_segments[button_index].turn_off()
            self.control_buttons[button_index].release()
            
            # Check the input
            self.check_input(button_index)
    
    def run(self):
        """Main game loop."""
        running = True
        
        print("="*60)
        print("Simon Says Game - Diamond Layout")
        print("="*60)
        print("Controls:")
        print("  1 or Click - Red button")
        print("  2 or Click - Green button")
        print("  3 or Click - Blue button")
        print("  4 or Click - Yellow button")
        print("  SPACE - Start game")
        print("  R - Restart")
        print("  Q - Quit")
        print("="*60 + "\n")
        
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            # Update hover states
            for button in self.control_buttons:
                button.is_hovered = button.contains_point(mouse_pos)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    # Quit
                    if event.key == pygame.K_q:
                        running = False
                    
                    # Start game
                    elif event.key == pygame.K_SPACE and self.game_state == "START":
                        self.add_to_sequence()
                        self.show_sequence()
                    
                    # Restart game
                    elif event.key == pygame.K_r and self.game_state == "GAME_OVER":
                        self.reset_game()
                    
                    # Button inputs (only during PLAYING state)
                    elif self.game_state == "PLAYING":
                        if event.key == pygame.K_1:
                            self.handle_button_press(0)
                        elif event.key == pygame.K_2:
                            self.handle_button_press(1)
                        elif event.key == pygame.K_3:
                            self.handle_button_press(2)
                        elif event.key == pygame.K_4:
                            self.handle_button_press(3)
                
                elif event.type == pygame.MOUSEBUTTONDOWN and self.game_state == "PLAYING":
                    # Check if any control button was clicked
                    for i, button in enumerate(self.control_buttons):
                        if button.contains_point(event.pos):
                            self.handle_button_press(i)
                            break
            
            # Draw everything
            self.draw()
            pygame.display.flip()
            
            # Control frame rate
            self.clock.tick(60)
        
        # Quit
        pygame.quit()
        print("\nGame ended!")
        print(f"Final Score: {self.score}")
        print(f"High Score: {self.high_score}")
        print(f"Levels Completed: {self.current_level - 1}")


if __name__ == "__main__":
    game = SimonSaysGame()
    game.run()
