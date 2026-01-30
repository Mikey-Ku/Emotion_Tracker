"""
Rhythm Game - Guitar Hero Style
================================

A rhythm game where colored notes fall down and you must hit the 
matching button when they reach the target zone.

Current Input: Keyboard (1, 2, 3, 4 keys)
Future: ESP32 with physical colored buttons via serial

Controls:
- Press 1, 2, 3, or 4 when notes reach the target line
- Press SPACE to start
- Press 'p' to pause
- Press 'q' to quit

Scoring:
- Perfect: Hit exactly on target (100 points)
- Good: Hit close to target (50 points)
- Miss: Note passes without hitting (0 points, combo breaks)
"""

import pygame
import random
import time

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

# Note colors
RED = (220, 30, 30)
RED_BRIGHT = (255, 80, 80)

GREEN = (30, 220, 30)
GREEN_BRIGHT = (80, 255, 80)

BLUE = (30, 30, 220)
BLUE_BRIGHT = (80, 80, 255)

YELLOW = (220, 220, 30)
YELLOW_BRIGHT = (255, 255, 80)

# Game settings
NOTE_SPEED = 3  # pixels per frame
NOTE_SPAWN_RATE = 60  # frames between notes (lower = more frequent)
TARGET_Y = 650  # Y position of target line
PERFECT_THRESHOLD = 30  # pixels for perfect hit
GOOD_THRESHOLD = 60  # pixels for good hit

# Lane positions
LANE_WIDTH = 150
LANE_GAP = 30
LANES_START_X = (SCREEN_WIDTH - (LANE_WIDTH * 4 + LANE_GAP * 3)) // 2

LANE_COLORS = [RED, GREEN, BLUE, YELLOW]
LANE_BRIGHT_COLORS = [RED_BRIGHT, GREEN_BRIGHT, BLUE_BRIGHT, YELLOW_BRIGHT]
LANE_LABELS = ["RED", "GREEN", "BLUE", "YELLOW"]


class Note:
    """Represents a falling note."""
    
    def __init__(self, lane, color, bright_color):
        self.lane = lane
        self.color = color
        self.bright_color = bright_color
        self.x = LANES_START_X + lane * (LANE_WIDTH + LANE_GAP) + LANE_WIDTH // 2
        self.y = -50  # Start above screen
        self.radius = 30
        self.active = True
        self.hit = False
    
    def update(self):
        """Move the note down."""
        self.y += NOTE_SPEED
        
        # Deactivate if it goes off screen
        if self.y > SCREEN_HEIGHT + 50:
            self.active = False
    
    def draw(self, screen):
        """Draw the note."""
        if not self.active:
            return
        
        # Draw glow effect
        for i in range(3):
            glow_radius = self.radius + (3 - i) * 5
            glow_alpha = 50 - i * 15
            glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (*self.color, glow_alpha), 
                             (glow_radius, glow_radius), glow_radius)
            screen.blit(glow_surface, (self.x - glow_radius, self.y - glow_radius))
        
        # Draw main note
        pygame.draw.circle(screen, self.bright_color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius, 3)
    
    def get_distance_from_target(self):
        """Get distance from target line."""
        return abs(self.y - TARGET_Y)
    
    def is_hittable(self):
        """Check if note is in hittable range."""
        return self.get_distance_from_target() <= GOOD_THRESHOLD


class Lane:
    """Represents one lane/track."""
    
    def __init__(self, lane_num, color, bright_color, label):
        self.lane_num = lane_num
        self.color = color
        self.bright_color = bright_color
        self.label = label
        self.x = LANES_START_X + lane_num * (LANE_WIDTH + LANE_GAP)
        self.is_pressed = False
    
    def draw(self, screen):
        """Draw the lane."""
        # Draw lane background
        lane_rect = pygame.Rect(self.x, 0, LANE_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(screen, (40, 40, 40), lane_rect)
        pygame.draw.rect(screen, self.color, lane_rect, 2)
        
        # Draw target zone
        target_rect = pygame.Rect(self.x, TARGET_Y - 40, LANE_WIDTH, 80)
        
        if self.is_pressed:
            # Flash when pressed
            pygame.draw.rect(screen, self.bright_color, target_rect)
            pygame.draw.rect(screen, WHITE, target_rect, 3)
        else:
            # Normal state
            pygame.draw.rect(screen, (*self.color, 100), target_rect)
            pygame.draw.rect(screen, self.color, target_rect, 3)
        
        # Draw target line
        pygame.draw.line(screen, WHITE, (self.x, TARGET_Y), 
                        (self.x + LANE_WIDTH, TARGET_Y), 3)
        
        # Draw button label at bottom
        font = pygame.font.Font(None, 36)
        text = font.render(self.label, True, WHITE)
        text_rect = text.get_rect(center=(self.x + LANE_WIDTH // 2, SCREEN_HEIGHT - 30))
        screen.blit(text, text_rect)
        
        # Draw key number
        key_font = pygame.font.Font(None, 48)
        key_text = key_font.render(str(self.lane_num + 1), True, self.bright_color)
        key_rect = key_text.get_rect(center=(self.x + LANE_WIDTH // 2, SCREEN_HEIGHT - 70))
        screen.blit(key_text, key_rect)


class RhythmGame:
    """Main game class."""
    
    def __init__(self):
        # Create screen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Rhythm Game - Guitar Hero Style")
        
        # Clock
        self.clock = pygame.time.Clock()
        
        # Create lanes
        self.lanes = [
            Lane(0, RED, RED_BRIGHT, "RED"),
            Lane(1, GREEN, GREEN_BRIGHT, "GREEN"),
            Lane(2, BLUE, BLUE_BRIGHT, "BLUE"),
            Lane(3, YELLOW, YELLOW_BRIGHT, "YELLOW")
        ]
        
        # Game state
        self.notes = []
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.perfect_hits = 0
        self.good_hits = 0
        self.misses = 0
        self.game_state = "START"  # START, PLAYING, PAUSED, GAME_OVER
        self.frame_count = 0
        self.spawn_timer = 0
        
        # Fonts
        self.title_font = pygame.font.Font(None, 70)
        self.text_font = pygame.font.Font(None, 40)
        self.small_font = pygame.font.Font(None, 28)
        
        # Feedback messages
        self.feedback_messages = []  # List of (message, x, y, time, color)
    
    def spawn_note(self):
        """Spawn a random note."""
        lane = random.randint(0, 3)
        color = LANE_COLORS[lane]
        bright_color = LANE_BRIGHT_COLORS[lane]
        note = Note(lane, color, bright_color)
        self.notes.append(note)
    
    def hit_note(self, lane_num):
        """Try to hit a note in the specified lane."""
        # Find the closest note in this lane
        closest_note = None
        min_distance = float('inf')
        
        for note in self.notes:
            if note.lane == lane_num and note.active and not note.hit:
                distance = note.get_distance_from_target()
                if distance < min_distance and note.is_hittable():
                    min_distance = distance
                    closest_note = note
        
        if closest_note:
            # Hit the note!
            closest_note.hit = True
            closest_note.active = False
            
            # Calculate score based on accuracy
            if min_distance <= PERFECT_THRESHOLD:
                points = 100
                self.combo += 1
                self.perfect_hits += 1
                self.add_feedback("PERFECT!", closest_note.x, TARGET_Y, (100, 255, 100))
            else:
                points = 50
                self.combo += 1
                self.good_hits += 1
                self.add_feedback("GOOD", closest_note.x, TARGET_Y, (255, 255, 100))
            
            # Add combo bonus
            points += self.combo * 10
            self.score += points
            
            # Update max combo
            if self.combo > self.max_combo:
                self.max_combo = self.combo
            
            return True
        
        return False
    
    def add_feedback(self, message, x, y, color):
        """Add a feedback message that floats up."""
        self.feedback_messages.append({
            'message': message,
            'x': x,
            'y': y,
            'time': 60,  # frames to display
            'color': color
        })
    
    def update(self):
        """Update game state."""
        if self.game_state != "PLAYING":
            return
        
        self.frame_count += 1
        self.spawn_timer += 1
        
        # Spawn new notes
        if self.spawn_timer >= NOTE_SPAWN_RATE:
            self.spawn_note()
            self.spawn_timer = 0
        
        # Update notes
        for note in self.notes[:]:
            note.update()
            
            # Check for missed notes
            if note.active and not note.hit and note.y > TARGET_Y + GOOD_THRESHOLD:
                note.active = False
                self.misses += 1
                self.combo = 0  # Break combo
                self.add_feedback("MISS", note.x, TARGET_Y, (255, 100, 100))
            
            # Remove inactive notes
            if not note.active:
                self.notes.remove(note)
        
        # Update feedback messages
        for feedback in self.feedback_messages[:]:
            feedback['y'] -= 2  # Float up
            feedback['time'] -= 1
            if feedback['time'] <= 0:
                self.feedback_messages.remove(feedback)
    
    def draw(self):
        """Draw everything."""
        # Clear screen
        self.screen.fill(DARK_GRAY)
        
        # Draw lanes
        for lane in self.lanes:
            lane.draw(self.screen)
        
        # Draw notes
        for note in self.notes:
            note.draw(self.screen)
        
        # Draw feedback messages
        for feedback in self.feedback_messages:
            alpha = min(255, feedback['time'] * 4)
            font = pygame.font.Font(None, 48)
            text = font.render(feedback['message'], True, feedback['color'])
            text_rect = text.get_rect(center=(feedback['x'], feedback['y']))
            self.screen.blit(text, text_rect)
        
        # Draw HUD
        self.draw_hud()
        
        # Draw game state overlays
        if self.game_state == "START":
            self.draw_start_screen()
        elif self.game_state == "PAUSED":
            self.draw_pause_screen()
        elif self.game_state == "GAME_OVER":
            self.draw_game_over_screen()
    
    def draw_hud(self):
        """Draw heads-up display."""
        # Score
        score_text = self.text_font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (20, 20))
        
        # Combo
        if self.combo > 0:
            combo_color = (255, 255, 100) if self.combo < 10 else (255, 150, 50)
            combo_text = self.title_font.render(f"{self.combo}x", True, combo_color)
            combo_rect = combo_text.get_rect(center=(SCREEN_WIDTH // 2, 60))
            self.screen.blit(combo_text, combo_rect)
        
        # Stats (top right)
        stats_y = 20
        stats = [
            f"Perfect: {self.perfect_hits}",
            f"Good: {self.good_hits}",
            f"Miss: {self.misses}"
        ]
        for stat in stats:
            stat_text = self.small_font.render(stat, True, (200, 200, 200))
            stat_rect = stat_text.get_rect(topright=(SCREEN_WIDTH - 20, stats_y))
            self.screen.blit(stat_text, stat_rect)
            stats_y += 30
    
    def draw_start_screen(self):
        """Draw start screen overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Title
        title = self.title_font.render("RHYTHM GAME", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 200))
        self.screen.blit(title, title_rect)
        
        # Instructions
        instructions = [
            "Hit the colored buttons when notes",
            "reach the target line!",
            "",
            "Press 1, 2, 3, or 4",
            "",
            "Press SPACE to Start"
        ]
        
        y = 300
        for line in instructions:
            text = self.text_font.render(line, True, WHITE)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, y))
            self.screen.blit(text, text_rect)
            y += 50
    
    def draw_pause_screen(self):
        """Draw pause screen overlay."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        title = self.title_font.render("PAUSED", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
        self.screen.blit(title, title_rect)
        
        text = self.text_font.render("Press P to Resume", True, WHITE)
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
        self.screen.blit(text, text_rect)
    
    def draw_game_over_screen(self):
        """Draw game over screen."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Title
        title = self.title_font.render("GAME OVER", True, (255, 100, 100))
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 150))
        self.screen.blit(title, title_rect)
        
        # Stats
        stats = [
            f"Final Score: {self.score}",
            f"Max Combo: {self.max_combo}x",
            f"Perfect: {self.perfect_hits}",
            f"Good: {self.good_hits}",
            f"Miss: {self.misses}",
            "",
            "Press R to Restart"
        ]
        
        y = 250
        for stat in stats:
            text = self.text_font.render(stat, True, WHITE)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, y))
            self.screen.blit(text, text_rect)
            y += 50
    
    def reset_game(self):
        """Reset the game."""
        self.notes = []
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.perfect_hits = 0
        self.good_hits = 0
        self.misses = 0
        self.frame_count = 0
        self.spawn_timer = 0
        self.feedback_messages = []
        self.game_state = "START"
    
    def run(self):
        """Main game loop."""
        running = True
        
        print("="*60)
        print("Rhythm Game - Guitar Hero Style")
        print("="*60)
        print("Controls:")
        print("  1 - Red lane")
        print("  2 - Green lane")
        print("  3 - Blue lane")
        print("  4 - Yellow lane")
        print("  SPACE - Start game")
        print("  P - Pause/Resume")
        print("  Q - Quit")
        print("="*60 + "\n")
        
        while running:
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
                        self.game_state = "PLAYING"
                    
                    # Pause/Resume
                    elif event.key == pygame.K_p:
                        if self.game_state == "PLAYING":
                            self.game_state = "PAUSED"
                        elif self.game_state == "PAUSED":
                            self.game_state = "PLAYING"
                    
                    # Restart
                    elif event.key == pygame.K_r and self.game_state == "GAME_OVER":
                        self.reset_game()
                    
                    # Hit notes (only during PLAYING state)
                    elif self.game_state == "PLAYING":
                        if event.key == pygame.K_1:
                            self.lanes[0].is_pressed = True
                            self.hit_note(0)
                        elif event.key == pygame.K_2:
                            self.lanes[1].is_pressed = True
                            self.hit_note(1)
                        elif event.key == pygame.K_3:
                            self.lanes[2].is_pressed = True
                            self.hit_note(2)
                        elif event.key == pygame.K_4:
                            self.lanes[3].is_pressed = True
                            self.hit_note(3)
                
                elif event.type == pygame.KEYUP:
                    # Release lane highlights
                    if event.key == pygame.K_1:
                        self.lanes[0].is_pressed = False
                    elif event.key == pygame.K_2:
                        self.lanes[1].is_pressed = False
                    elif event.key == pygame.K_3:
                        self.lanes[2].is_pressed = False
                    elif event.key == pygame.K_4:
                        self.lanes[3].is_pressed = False
            
            # Update game
            self.update()
            
            # Draw everything
            self.draw()
            pygame.display.flip()
            
            # Control frame rate
            self.clock.tick(60)
        
        # Quit
        pygame.quit()
        print("\nGame ended!")
        print(f"Final Score: {self.score}")
        print(f"Max Combo: {self.max_combo}x")
        print(f"Perfect: {self.perfect_hits} | Good: {self.good_hits} | Miss: {self.misses}")


if __name__ == "__main__":
    game = RhythmGame()
    game.run()
