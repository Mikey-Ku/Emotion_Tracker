"""
Bubble Snake Shooter
====================

A Zuma-style game where you shoot colored bubbles to match and destroy
a snake of bubbles moving along a path.

Current Input: Keyboard (1, 2, 3, 4 keys)
Future: ESP32 with physical colored buttons via serial

Controls:
- Press 1, 2, 3, or 4 to shoot that color bubble
- Match 3+ bubbles of the same color to destroy them
- Don't let the snake reach the end!
- Press SPACE to start
- Press 'q' to quit

Colors:
- 1 = Red
- 2 = Green
- 3 = Blue
- 4 = Yellow
"""

import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 800

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (30, 30, 30)
DARK_GRAY = (20, 20, 20)

# Bubble colors
RED = (220, 30, 30)
RED_BRIGHT = (255, 80, 80)

GREEN = (30, 220, 30)
GREEN_BRIGHT = (80, 255, 80)

BLUE = (30, 30, 220)
BLUE_BRIGHT = (80, 80, 255)

YELLOW = (220, 220, 30)
YELLOW_BRIGHT = (255, 255, 80)

COLORS = [RED, GREEN, BLUE, YELLOW]
BRIGHT_COLORS = [RED_BRIGHT, GREEN_BRIGHT, BLUE_BRIGHT, YELLOW_BRIGHT]
COLOR_NAMES = ["RED", "GREEN", "BLUE", "YELLOW"]

# Game settings
BUBBLE_RADIUS = 20
SNAKE_SPEED = 1.0  # pixels per frame
SHOOTER_Y = 700  # Y position of shooter


class Bubble:
    """Represents a bubble in the snake or being shot."""
    
    def __init__(self, color_index, x=0, y=0):
        self.color_index = color_index
        self.color = COLORS[color_index]
        self.bright_color = BRIGHT_COLORS[color_index]
        self.x = x
        self.y = y
        self.radius = BUBBLE_RADIUS
        self.marked_for_removal = False
    
    def draw(self, screen, is_shooting=False):
        """Draw the bubble."""
        # Draw glow for shooting bubble
        if is_shooting:
            for i in range(3):
                glow_radius = self.radius + (3 - i) * 4
                glow_alpha = 60 - i * 20
                glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (*self.bright_color, glow_alpha),
                                 (glow_radius, glow_radius), glow_radius)
                screen.blit(glow_surface, (self.x - glow_radius, self.y - glow_radius))
        
        # Draw main bubble
        pygame.draw.circle(screen, self.bright_color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius, 2)
        
        # Draw highlight
        highlight_pos = (int(self.x - self.radius // 3), int(self.y - self.radius // 3))
        pygame.draw.circle(screen, WHITE, highlight_pos, self.radius // 4)


class ShootingBubble:
    """A bubble being shot from the cannon."""
    
    def __init__(self, color_index, start_x, start_y, target_x, target_y):
        self.bubble = Bubble(color_index, start_x, start_y)
        self.speed = 10
        
        # Calculate direction
        dx = target_x - start_x
        dy = target_y - start_y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > 0:
            self.vx = (dx / distance) * self.speed
            self.vy = (dy / distance) * self.speed
        else:
            self.vx = 0
            self.vy = -self.speed
        
        self.active = True
    
    def update(self):
        """Move the bubble."""
        self.bubble.x += self.vx
        self.bubble.y += self.vy
        
        # Deactivate if off screen
        if (self.bubble.y < -50 or self.bubble.y > SCREEN_HEIGHT + 50 or
            self.bubble.x < -50 or self.bubble.x > SCREEN_WIDTH + 50):
            self.active = False
    
    def draw(self, screen):
        """Draw the shooting bubble."""
        self.bubble.draw(screen, is_shooting=True)


class Path:
    """Defines the path the snake follows."""
    
    def __init__(self):
        # Create a winding path
        self.points = []
        
        # Start from right side
        start_x = SCREEN_WIDTH + 50
        start_y = 100
        
        # Create a serpentine path
        self.points.append((start_x, start_y))
        
        # Wind across the screen
        for i in range(8):
            if i % 2 == 0:
                # Move left
                self.points.append((100, start_y + i * 70))
            else:
                # Move right
                self.points.append((SCREEN_WIDTH - 100, start_y + i * 70))
        
        # End point (danger zone)
        self.points.append((SCREEN_WIDTH // 2, 650))
        
        # Calculate total path length
        self.total_length = 0
        self.segment_lengths = []
        
        for i in range(len(self.points) - 1):
            x1, y1 = self.points[i]
            x2, y2 = self.points[i + 1]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            self.segment_lengths.append(length)
            self.total_length += length
    
    def get_position(self, distance):
        """Get x, y position at a given distance along the path."""
        if distance <= 0:
            return self.points[0]
        if distance >= self.total_length:
            return self.points[-1]
        
        # Find which segment we're on
        current_distance = 0
        for i, seg_length in enumerate(self.segment_lengths):
            if current_distance + seg_length >= distance:
                # Interpolate within this segment
                t = (distance - current_distance) / seg_length
                x1, y1 = self.points[i]
                x2, y2 = self.points[i + 1]
                x = x1 + (x2 - x1) * t
                y = y1 + (y2 - y1) * t
                return (x, y)
            current_distance += seg_length
        
        return self.points[-1]
    
    def draw(self, screen):
        """Draw the path (for debugging)."""
        if len(self.points) > 1:
            pygame.draw.lines(screen, (60, 60, 60), False, self.points, 3)


class BubbleSnake:
    """The snake of bubbles moving along the path."""
    
    def __init__(self, path):
        self.path = path
        self.bubbles = []
        self.distance = 0  # Distance along path
        self.spacing = BUBBLE_RADIUS * 2 + 2  # Space between bubble centers
        
        # Initialize with some bubbles
        for i in range(15):
            color = random.randint(0, 3)
            self.bubbles.append(Bubble(color))
    
    def update(self, speed):
        """Move the snake forward."""
        self.distance += speed
        
        # Update bubble positions
        current_distance = self.distance
        for bubble in self.bubbles:
            if not bubble.marked_for_removal:
                x, y = self.path.get_position(current_distance)
                bubble.x = x
                bubble.y = y
                current_distance -= self.spacing
    
    def add_bubble(self):
        """Add a new bubble to the front."""
        color = random.randint(0, 3)
        self.bubbles.insert(0, Bubble(color))
    
    def insert_bubble(self, bubble, index):
        """Insert a shot bubble into the snake."""
        self.bubbles.insert(index, bubble)
    
    def check_matches(self):
        """Check for 3+ matching bubbles and remove them."""
        if len(self.bubbles) < 3:
            return 0
        
        removed_count = 0
        i = 0
        
        while i < len(self.bubbles):
            if self.bubbles[i].marked_for_removal:
                i += 1
                continue
            
            # Count consecutive bubbles of same color
            color = self.bubbles[i].color_index
            count = 1
            j = i + 1
            
            while j < len(self.bubbles) and self.bubbles[j].color_index == color:
                count += 1
                j += 1
            
            # If 3 or more, mark for removal
            if count >= 3:
                for k in range(i, j):
                    self.bubbles[k].marked_for_removal = True
                removed_count += count
                i = j
            else:
                i += 1
        
        # Remove marked bubbles
        self.bubbles = [b for b in self.bubbles if not b.marked_for_removal]
        
        return removed_count
    
    def find_insertion_point(self, shot_bubble):
        """Find where to insert a shot bubble."""
        min_dist = float('inf')
        insert_index = 0
        
        for i, bubble in enumerate(self.bubbles):
            dx = bubble.x - shot_bubble.x
            dy = bubble.y - shot_bubble.y
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist < min_dist:
                min_dist = dist
                insert_index = i
        
        # Check if close enough to insert
        if min_dist < BUBBLE_RADIUS * 3:
            return insert_index
        
        return None
    
    def draw(self, screen):
        """Draw all bubbles in the snake."""
        for bubble in self.bubbles:
            bubble.draw(screen)
    
    def is_at_end(self):
        """Check if snake reached the end."""
        return self.distance >= self.path.total_length - 100


class Shooter:
    """The bubble shooter at the bottom."""
    
    def __init__(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SHOOTER_Y
        self.current_color = random.randint(0, 3)
        self.next_color = random.randint(0, 3)
        self.width = 60
        self.height = 40
    
    def draw(self, screen):
        """Draw the shooter."""
        # Draw base
        pygame.draw.rect(screen, (80, 80, 80), 
                        (self.x - self.width, self.y, self.width * 2, self.height),
                        border_radius=10)
        pygame.draw.rect(screen, WHITE,
                        (self.x - self.width, self.y, self.width * 2, self.height),
                        3, border_radius=10)
        
        # Draw current bubble
        current_bubble = Bubble(self.current_color, self.x, self.y - 30)
        current_bubble.draw(screen)
        
        # Draw next bubble (smaller, to the side)
        next_bubble = Bubble(self.next_color, self.x + 80, self.y + 10)
        next_bubble.radius = 15
        next_bubble.draw(screen)
        
        # Draw "NEXT" label
        font = pygame.font.Font(None, 20)
        text = font.render("NEXT", True, (150, 150, 150))
        screen.blit(text, (self.x + 60, self.y - 10))
    
    def shoot(self, target_x, target_y):
        """Create a shooting bubble."""
        bubble = ShootingBubble(self.current_color, self.x, self.y - 30, target_x, target_y)
        
        # Move next color to current
        self.current_color = self.next_color
        self.next_color = random.randint(0, 3)
        
        return bubble
    
    def set_color(self, color_index):
        """Manually set the shooter color (for button controls)."""
        self.current_color = color_index


class BubbleShooterGame:
    """Main game class."""
    
    def __init__(self):
        # Create screen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Bubble Snake Shooter")
        
        # Clock
        self.clock = pygame.time.Clock()
        
        # Game objects
        self.path = Path()
        self.snake = BubbleSnake(self.path)
        self.shooter = Shooter()
        self.shooting_bubbles = []
        
        # Game state
        self.score = 0
        self.level = 1
        self.snake_speed = SNAKE_SPEED
        self.game_state = "START"  # START, PLAYING, GAME_OVER, WIN
        self.frame_count = 0
        self.add_bubble_timer = 0
        
        # Fonts
        self.title_font = pygame.font.Font(None, 70)
        self.text_font = pygame.font.Font(None, 40)
        self.small_font = pygame.font.Font(None, 28)
    
    def update(self):
        """Update game state."""
        if self.game_state != "PLAYING":
            return
        
        self.frame_count += 1
        self.add_bubble_timer += 1
        
        # Add new bubbles to snake periodically
        if self.add_bubble_timer >= 120:  # Every 2 seconds
            self.snake.add_bubble()
            self.add_bubble_timer = 0
        
        # Move snake
        self.snake.update(self.snake_speed)
        
        # Update shooting bubbles
        for shot in self.shooting_bubbles[:]:
            shot.update()
            
            if not shot.active:
                self.shooting_bubbles.remove(shot)
                continue
            
            # Check collision with snake
            insert_index = self.snake.find_insertion_point(shot.bubble)
            if insert_index is not None:
                # Insert bubble into snake
                self.snake.insert_bubble(shot.bubble, insert_index)
                self.shooting_bubbles.remove(shot)
                
                # Check for matches
                removed = self.snake.check_matches()
                if removed > 0:
                    self.score += removed * 10
                    if removed >= 5:
                        self.score += 50  # Bonus for big matches
        
        # Check win condition (snake destroyed)
        if len(self.snake.bubbles) == 0:
            self.game_state = "WIN"
        
        # Check lose condition (snake reached end)
        if self.snake.is_at_end():
            self.game_state = "GAME_OVER"
    
    def shoot_color(self, color_index):
        """Shoot a bubble of the specified color."""
        if self.game_state == "PLAYING":
            # Set shooter to this color and shoot
            self.shooter.current_color = color_index
            
            # Shoot toward the front of the snake
            if self.snake.bubbles:
                target_x = self.snake.bubbles[0].x
                target_y = self.snake.bubbles[0].y
            else:
                target_x = SCREEN_WIDTH // 2
                target_y = 200
            
            shot = self.shooter.shoot(target_x, target_y)
            self.shooting_bubbles.append(shot)
    
    def draw(self):
        """Draw everything."""
        # Clear screen
        self.screen.fill(DARK_GRAY)
        
        # Draw path (faint)
        self.path.draw(self.screen)
        
        # Draw snake
        self.snake.draw(self.screen)
        
        # Draw shooting bubbles
        for shot in self.shooting_bubbles:
            shot.draw(self.screen)
        
        # Draw shooter
        self.shooter.draw(self.screen)
        
        # Draw control buttons
        self.draw_controls()
        
        # Draw HUD
        self.draw_hud()
        
        # Draw overlays
        if self.game_state == "START":
            self.draw_start_screen()
        elif self.game_state == "GAME_OVER":
            self.draw_game_over_screen()
        elif self.game_state == "WIN":
            self.draw_win_screen()
    
    def draw_controls(self):
        """Draw the 4 color buttons at the bottom."""
        button_width = 120
        button_height = 50
        button_y = SCREEN_HEIGHT - 80
        spacing = 40
        total_width = (button_width * 4) + (spacing * 3)
        start_x = (SCREEN_WIDTH - total_width) // 2
        
        for i in range(4):
            x = start_x + i * (button_width + spacing)
            
            # Draw button
            button_rect = pygame.Rect(x, button_y, button_width, button_height)
            pygame.draw.rect(self.screen, COLORS[i], button_rect, border_radius=10)
            pygame.draw.rect(self.screen, WHITE, button_rect, 3, border_radius=10)
            
            # Draw label
            font = pygame.font.Font(None, 32)
            text = font.render(COLOR_NAMES[i], True, WHITE)
            text_rect = text.get_rect(center=button_rect.center)
            self.screen.blit(text, text_rect)
            
            # Draw key number
            key_font = pygame.font.Font(None, 24)
            key_text = key_font.render(f"[{i+1}]", True, (200, 200, 200))
            key_rect = key_text.get_rect(center=(button_rect.centerx, button_rect.bottom - 12))
            self.screen.blit(key_text, key_rect)
    
    def draw_hud(self):
        """Draw HUD."""
        # Score
        score_text = self.text_font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (20, 20))
        
        # Level
        level_text = self.small_font.render(f"Level: {self.level}", True, (200, 200, 200))
        self.screen.blit(level_text, (20, 60))
        
        # Bubbles remaining
        bubbles_text = self.small_font.render(f"Bubbles: {len(self.snake.bubbles)}", True, (200, 200, 200))
        self.screen.blit(bubbles_text, (SCREEN_WIDTH - 150, 20))
    
    def draw_start_screen(self):
        """Draw start screen."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        title = self.title_font.render("BUBBLE SNAKE", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 200))
        self.screen.blit(title, title_rect)
        
        instructions = [
            "Shoot colored bubbles to match 3+",
            "and destroy the snake!",
            "",
            "Press 1, 2, 3, or 4 to shoot",
            "",
            "Press SPACE to Start"
        ]
        
        y = 300
        for line in instructions:
            text = self.text_font.render(line, True, WHITE)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, y))
            self.screen.blit(text, text_rect)
            y += 50
    
    def draw_game_over_screen(self):
        """Draw game over screen."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        title = self.title_font.render("GAME OVER", True, (255, 100, 100))
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 250))
        self.screen.blit(title, title_rect)
        
        score_text = self.text_font.render(f"Final Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, 350))
        self.screen.blit(score_text, score_rect)
        
        restart = self.text_font.render("Press R to Restart", True, WHITE)
        restart_rect = restart.get_rect(center=(SCREEN_WIDTH // 2, 450))
        self.screen.blit(restart, restart_rect)
    
    def draw_win_screen(self):
        """Draw win screen."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        title = self.title_font.render("YOU WIN!", True, (100, 255, 100))
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 250))
        self.screen.blit(title, title_rect)
        
        score_text = self.text_font.render(f"Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, 350))
        self.screen.blit(score_text, score_rect)
        
        restart = self.text_font.render("Press R to Play Again", True, WHITE)
        restart_rect = restart.get_rect(center=(SCREEN_WIDTH // 2, 450))
        self.screen.blit(restart, restart_rect)
    
    def reset_game(self):
        """Reset the game."""
        self.snake = BubbleSnake(self.path)
        self.shooting_bubbles = []
        self.score = 0
        self.frame_count = 0
        self.add_bubble_timer = 0
        self.game_state = "START"
    
    def run(self):
        """Main game loop."""
        running = True
        
        print("="*60)
        print("Bubble Snake Shooter")
        print("="*60)
        print("Controls:")
        print("  1 - Shoot Red")
        print("  2 - Shoot Green")
        print("  3 - Shoot Blue")
        print("  4 - Shoot Yellow")
        print("  SPACE - Start")
        print("  R - Restart")
        print("  Q - Quit")
        print("="*60 + "\n")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    
                    elif event.key == pygame.K_SPACE and self.game_state == "START":
                        self.game_state = "PLAYING"
                    
                    elif event.key == pygame.K_r and self.game_state in ["GAME_OVER", "WIN"]:
                        self.reset_game()
                    
                    elif self.game_state == "PLAYING":
                        if event.key == pygame.K_1:
                            self.shoot_color(0)
                        elif event.key == pygame.K_2:
                            self.shoot_color(1)
                        elif event.key == pygame.K_3:
                            self.shoot_color(2)
                        elif event.key == pygame.K_4:
                            self.shoot_color(3)
            
            # Update
            self.update()
            
            # Draw
            self.draw()
            pygame.display.flip()
            
            # Control frame rate
            self.clock.tick(60)
        
        # Quit
        pygame.quit()
        print("\nGame ended!")
        print(f"Final Score: {self.score}")


if __name__ == "__main__":
    game = BubbleShooterGame()
    game.run()
