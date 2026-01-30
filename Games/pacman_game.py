"""
Pac-Man Style Game
==================

A classic Pac-Man game with 4-button directional controls.
Collect all dots while avoiding ghosts!

Controls (4-Button Controller Style):
- Button 1 (RED) = UP
- Button 2 (GREEN) = DOWN
- Button 3 (BLUE) = LEFT
- Button 4 (YELLOW) = RIGHT

Or use arrow keys for testing.

Game Features:
- Collect all dots to win
- Avoid ghosts or lose a life
- Power pellets make ghosts vulnerable
- Classic maze layout
- 3 lives to start
"""

import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 800
CELL_SIZE = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Button colors (for controller display)
RED = (220, 30, 30)
GREEN = (30, 220, 30)
BUTTON_BLUE = (30, 30, 220)
BUTTON_YELLOW = (220, 220, 30)

# Ghost colors
GHOST_RED = (255, 0, 0)
GHOST_PINK = (255, 184, 255)
GHOST_CYAN = (0, 255, 255)
GHOST_ORANGE = (255, 184, 82)

# Game settings
PACMAN_SPEED = 5  # Frames per move
GHOST_SPEED = 7   # Frames per move (slower = harder)

# Maze layout (0=wall, 1=dot, 2=empty, 3=power pellet)
MAZE = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0],
    [0,3,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,3,0],
    [0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,1,0],
    [0,1,1,1,1,0,1,1,1,0,0,1,1,1,0,1,1,1,1,0],
    [0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0],
    [2,2,2,0,1,0,1,1,1,1,1,1,1,1,0,1,0,2,2,2],
    [0,0,0,0,1,0,1,0,0,2,2,0,0,1,0,1,0,0,0,0],
    [2,2,2,2,1,1,1,0,2,2,2,2,0,1,1,1,2,2,2,2],
    [0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0],
    [2,2,2,0,1,0,1,1,1,1,1,1,1,1,0,1,0,2,2,2],
    [0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0],
    [0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0],
    [0,3,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,3,0],
    [0,0,1,0,1,0,1,0,0,0,0,0,0,1,0,1,0,1,0,0],
    [0,1,1,1,1,0,1,1,1,0,0,1,1,1,0,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
]

MAZE_WIDTH = len(MAZE[0])
MAZE_HEIGHT = len(MAZE)


class PacMan:
    """The player character."""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction = (0, 0)  # (dx, dy)
        self.next_direction = (0, 0)
        self.move_timer = 0
        self.mouth_open = 0
        self.alive = True
    
    def set_direction(self, direction):
        """Set the next direction to move."""
        self.next_direction = direction
    
    def update(self, maze):
        """Update Pac-Man's position."""
        if not self.alive:
            return
        
        self.move_timer += 1
        
        if self.move_timer >= PACMAN_SPEED:
            self.move_timer = 0
            
            # Try to change direction
            if self.next_direction != (0, 0):
                new_x = self.x + self.next_direction[0]
                new_y = self.y + self.next_direction[1]
                
                if self.can_move(new_x, new_y, maze):
                    self.direction = self.next_direction
            
            # Move in current direction
            new_x = self.x + self.direction[0]
            new_y = self.y + self.direction[1]
            
            if self.can_move(new_x, new_y, maze):
                self.x = new_x
                self.y = new_y
            
            # Wrap around screen
            if self.x < 0:
                self.x = MAZE_WIDTH - 1
            elif self.x >= MAZE_WIDTH:
                self.x = 0
        
        # Animate mouth
        self.mouth_open = (self.mouth_open + 1) % 20
    
    def can_move(self, x, y, maze):
        """Check if position is valid."""
        if 0 <= y < MAZE_HEIGHT and 0 <= x < MAZE_WIDTH:
            return maze[y][x] != 0
        return True  # Allow wrapping
    
    def draw(self, screen, offset_x, offset_y):
        """Draw Pac-Man."""
        if not self.alive:
            return
        
        center_x = offset_x + self.x * CELL_SIZE + CELL_SIZE // 2
        center_y = offset_y + self.y * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 2
        
        # Determine mouth angle based on direction
        if self.direction == (1, 0):  # Right
            start_angle = 30
        elif self.direction == (-1, 0):  # Left
            start_angle = 210
        elif self.direction == (0, -1):  # Up
            start_angle = 120
        elif self.direction == (0, 1):  # Down
            start_angle = 300
        else:
            start_angle = 30
        
        # Draw Pac-Man with mouth
        mouth_size = 40 if self.mouth_open < 10 else 20
        
        # Draw full circle first
        pygame.draw.circle(screen, YELLOW, (center_x, center_y), radius)
        
        # Draw mouth (black triangle)
        if mouth_size > 0:
            angle1 = math.radians(start_angle + mouth_size)
            angle2 = math.radians(start_angle - mouth_size)
            
            points = [
                (center_x, center_y),
                (center_x + radius * math.cos(angle1), center_y - radius * math.sin(angle1)),
                (center_x + radius * math.cos(angle2), center_y - radius * math.sin(angle2))
            ]
            pygame.draw.polygon(screen, BLACK, points)


class Ghost:
    """An enemy ghost."""
    
    def __init__(self, x, y, color, name):
        self.x = x
        self.y = y
        self.color = color
        self.name = name
        self.direction = (0, 0)
        self.move_timer = 0
        self.frightened = False
        self.frightened_timer = 0
    
    def update(self, maze, pacman_x, pacman_y):
        """Update ghost position."""
        self.move_timer += 1
        
        # Update frightened state
        if self.frightened:
            self.frightened_timer -= 1
            if self.frightened_timer <= 0:
                self.frightened = False
        
        if self.move_timer >= GHOST_SPEED:
            self.move_timer = 0
            
            # Choose direction (simple AI)
            if self.frightened:
                # Run away from Pac-Man
                self.choose_direction_away(maze, pacman_x, pacman_y)
            else:
                # Chase Pac-Man
                self.choose_direction_toward(maze, pacman_x, pacman_y)
            
            # Move
            new_x = self.x + self.direction[0]
            new_y = self.y + self.direction[1]
            
            if self.can_move(new_x, new_y, maze):
                self.x = new_x
                self.y = new_y
            
            # Wrap around
            if self.x < 0:
                self.x = MAZE_WIDTH - 1
            elif self.x >= MAZE_WIDTH:
                self.x = 0
    
    def choose_direction_toward(self, maze, target_x, target_y):
        """Choose direction toward target."""
        possible_dirs = []
        
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            new_x = self.x + dx
            new_y = self.y + dy
            
            # Don't reverse direction
            if (dx, dy) == (-self.direction[0], -self.direction[1]):
                continue
            
            if self.can_move(new_x, new_y, maze):
                distance = abs(new_x - target_x) + abs(new_y - target_y)
                possible_dirs.append(((dx, dy), distance))
        
        if possible_dirs:
            # Choose direction with shortest distance
            possible_dirs.sort(key=lambda x: x[1])
            self.direction = possible_dirs[0][0]
    
    def choose_direction_away(self, maze, target_x, target_y):
        """Choose direction away from target."""
        possible_dirs = []
        
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            new_x = self.x + dx
            new_y = self.y + dy
            
            if self.can_move(new_x, new_y, maze):
                distance = abs(new_x - target_x) + abs(new_y - target_y)
                possible_dirs.append(((dx, dy), distance))
        
        if possible_dirs:
            # Choose direction with longest distance
            possible_dirs.sort(key=lambda x: x[1], reverse=True)
            self.direction = possible_dirs[0][0]
    
    def can_move(self, x, y, maze):
        """Check if position is valid."""
        if 0 <= y < MAZE_HEIGHT and 0 <= x < MAZE_WIDTH:
            return maze[y][x] != 0
        return True
    
    def make_frightened(self):
        """Make ghost frightened (vulnerable)."""
        self.frightened = True
        self.frightened_timer = 300  # 5 seconds at 60 FPS
    
    def draw(self, screen, offset_x, offset_y):
        """Draw the ghost."""
        center_x = offset_x + self.x * CELL_SIZE + CELL_SIZE // 2
        center_y = offset_y + self.y * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 2
        
        # Choose color
        if self.frightened:
            if self.frightened_timer < 120:  # Flash when ending
                color = BLUE if (self.frightened_timer // 10) % 2 == 0 else WHITE
            else:
                color = BLUE
        else:
            color = self.color
        
        # Draw ghost body (circle + rectangle)
        pygame.draw.circle(screen, color, (center_x, center_y - radius // 2), radius)
        pygame.draw.rect(screen, color, 
                        (center_x - radius, center_y - radius // 2, 
                         radius * 2, radius + radius // 2))
        
        # Draw wavy bottom
        wave_points = []
        for i in range(5):
            x = center_x - radius + i * (radius // 2)
            y = center_y + radius if i % 2 == 0 else center_y + radius // 2
            wave_points.append((x, y))
        wave_points.append((center_x + radius, center_y + radius // 2))
        pygame.draw.polygon(screen, color, wave_points)
        
        # Draw eyes
        if not self.frightened:
            eye_offset = 5
            pygame.draw.circle(screen, WHITE, (center_x - eye_offset, center_y - 5), 4)
            pygame.draw.circle(screen, WHITE, (center_x + eye_offset, center_y - 5), 4)
            pygame.draw.circle(screen, BLACK, (center_x - eye_offset, center_y - 5), 2)
            pygame.draw.circle(screen, BLACK, (center_x + eye_offset, center_y - 5), 2)


class PacManGame:
    """Main game class."""
    
    def __init__(self):
        # Create screen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pac-Man - 4 Button Controller")
        
        # Clock
        self.clock = pygame.time.Clock()
        
        # Create maze copy for dots
        self.maze = [row[:] for row in MAZE]
        
        # Calculate maze offset to center it
        self.maze_offset_x = (SCREEN_WIDTH - MAZE_WIDTH * CELL_SIZE) // 2
        self.maze_offset_y = 50
        
        # Create Pac-Man
        self.pacman = PacMan(10, 14)
        
        # Create ghosts
        self.ghosts = [
            Ghost(9, 9, GHOST_RED, "Blinky"),
            Ghost(10, 9, GHOST_PINK, "Pinky"),
            Ghost(9, 10, GHOST_CYAN, "Inky"),
            Ghost(10, 10, GHOST_ORANGE, "Clyde")
        ]
        
        # Game state
        self.score = 0
        self.lives = 3
        self.game_state = "START"  # START, PLAYING, GAME_OVER, WIN
        self.dots_remaining = self.count_dots()
        
        # Button press states (for visual feedback)
        self.button_pressed = [False, False, False, False]
        
        # Fonts
        self.title_font = pygame.font.Font(None, 60)
        self.text_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # OPTIMIZATION: Pre-render static maze surface
        self.static_maze_surface = self.create_static_maze()
        
        # OPTIMIZATION: Pre-render button sprites
        self.button_sprites = self.create_button_sprites()
        
        # Track if maze needs redraw (when dots are eaten)
        self.maze_dirty = True
        self.maze_surface = None
    
    def create_static_maze(self):
        """Create a surface with just the walls (never changes)."""
        surface = pygame.Surface((MAZE_WIDTH * CELL_SIZE, MAZE_HEIGHT * CELL_SIZE))
        surface.fill(BLACK)
        
        for y in range(MAZE_HEIGHT):
            for x in range(MAZE_WIDTH):
                if MAZE[y][x] == 0:  # Wall
                    px = x * CELL_SIZE
                    py = y * CELL_SIZE
                    pygame.draw.rect(surface, BLUE, (px, py, CELL_SIZE, CELL_SIZE))
                    pygame.draw.rect(surface, (0, 0, 150), (px, py, CELL_SIZE, CELL_SIZE), 2)
        
        return surface
    
    def create_button_sprites(self):
        """Pre-render button sprites for better performance."""
        button_size = 80
        sprites = {}
        
        colors = [RED, GREEN, BUTTON_BLUE, BUTTON_YELLOW]
        labels = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            # Normal state
            normal = pygame.Surface((button_size, button_size + 6), pygame.SRCALPHA)
            # Shadow
            pygame.draw.rect(normal, (50, 50, 50), (0, 6, button_size, button_size), border_radius=10)
            # Button
            pygame.draw.rect(normal, color, (0, 0, button_size, button_size), border_radius=10)
            pygame.draw.rect(normal, WHITE, (0, 0, button_size, button_size), 3, border_radius=10)
            # Label
            font = pygame.font.Font(None, 24)
            text = font.render(label, True, WHITE)
            text_rect = text.get_rect(center=(button_size // 2, button_size // 2))
            normal.blit(text, text_rect)
            # Number
            num_font = pygame.font.Font(None, 20)
            num_text = num_font.render(str(i + 1), True, (200, 200, 200))
            num_rect = num_text.get_rect(center=(button_size // 2, button_size - 15))
            normal.blit(num_text, num_rect)
            
            # Pressed state
            pressed = pygame.Surface((button_size, button_size + 6), pygame.SRCALPHA)
            # Shadow
            pygame.draw.rect(pressed, (50, 50, 50), (0, 6, button_size, button_size), border_radius=10)
            # Button (offset down)
            pygame.draw.rect(pressed, color, (0, 4, button_size, button_size), border_radius=10)
            pygame.draw.rect(pressed, WHITE, (0, 4, button_size, button_size), 3, border_radius=10)
            # Label
            text_rect = text.get_rect(center=(button_size // 2, button_size // 2 + 4))
            pressed.blit(text, text_rect)
            # Number
            num_rect = num_text.get_rect(center=(button_size // 2, button_size - 11))
            pressed.blit(num_text, num_rect)
            
            sprites[i] = {'normal': normal, 'pressed': pressed}
        
        return sprites
    
    def count_dots(self):
        """Count total dots in maze."""
        count = 0
        for row in self.maze:
            for cell in row:
                if cell in [1, 3]:
                    count += 1
        return count
    
    def update(self):
        """Update game state."""
        if self.game_state != "PLAYING":
            return
        
        # Update Pac-Man
        self.pacman.update(self.maze)
        
        # Check dot collection
        cell = self.maze[self.pacman.y][self.pacman.x]
        if cell == 1:  # Regular dot
            self.maze[self.pacman.y][self.pacman.x] = 2
            self.score += 10
            self.dots_remaining -= 1
            self.maze_dirty = True  # Mark maze for redraw
        elif cell == 3:  # Power pellet
            self.maze[self.pacman.y][self.pacman.x] = 2
            self.score += 50
            self.dots_remaining -= 1
            self.maze_dirty = True  # Mark maze for redraw
            # Make ghosts frightened
            for ghost in self.ghosts:
                ghost.make_frightened()
        
        # Update ghosts
        for ghost in self.ghosts:
            ghost.update(self.maze, self.pacman.x, self.pacman.y)
            
            # Check collision
            if ghost.x == self.pacman.x and ghost.y == self.pacman.y:
                if ghost.frightened:
                    # Eat ghost
                    self.score += 200
                    ghost.x = 10
                    ghost.y = 9
                    ghost.frightened = False
                else:
                    # Lose life
                    self.lives -= 1
                    if self.lives <= 0:
                        self.game_state = "GAME_OVER"
                    else:
                        # Reset positions
                        self.pacman.x = 10
                        self.pacman.y = 14
                        self.pacman.direction = (0, 0)
                        for i, g in enumerate(self.ghosts):
                            g.x = 9 + (i % 2)
                            g.y = 9 + (i // 2)
        
        # Check win condition
        if self.dots_remaining == 0:
            self.game_state = "WIN"
    
    def draw(self):
        """Draw everything."""
        # Clear screen
        self.screen.fill(BLACK)
        
        # Draw maze (optimized)
        self.draw_maze_optimized()
        
        # Draw Pac-Man
        self.pacman.draw(self.screen, self.maze_offset_x, self.maze_offset_y)
        
        # Draw ghosts
        for ghost in self.ghosts:
            ghost.draw(self.screen, self.maze_offset_x, self.maze_offset_y)
        
        # Draw HUD
        self.draw_hud()
        
        # Draw controller (optimized)
        self.draw_controller_optimized()
        
        # Draw overlays
        if self.game_state == "START":
            self.draw_start_screen()
        elif self.game_state == "GAME_OVER":
            self.draw_game_over_screen()
        elif self.game_state == "WIN":
            self.draw_win_screen()
    
    def draw_maze_optimized(self):
        """Draw maze using cached surfaces for better performance."""
        # Only redraw maze surface if dots were eaten
        if self.maze_dirty or self.maze_surface is None:
            self.maze_surface = pygame.Surface((MAZE_WIDTH * CELL_SIZE, MAZE_HEIGHT * CELL_SIZE))
            
            # Blit static walls
            self.maze_surface.blit(self.static_maze_surface, (0, 0))
            
            # Draw only dots (dynamic elements)
            for y in range(MAZE_HEIGHT):
                for x in range(MAZE_WIDTH):
                    cell = self.maze[y][x]
                    px = x * CELL_SIZE
                    py = y * CELL_SIZE
                    
                    if cell == 1:  # Dot
                        pygame.draw.circle(self.maze_surface, WHITE,
                                         (px + CELL_SIZE // 2, py + CELL_SIZE // 2), 2)
                    elif cell == 3:  # Power pellet
                        pygame.draw.circle(self.maze_surface, WHITE,
                                         (px + CELL_SIZE // 2, py + CELL_SIZE // 2), 6)
            
            self.maze_dirty = False
        
        # Blit the complete maze surface
        self.screen.blit(self.maze_surface, (self.maze_offset_x, self.maze_offset_y))
    
    def draw_hud(self):
        """Draw HUD."""
        # Score
        score_text = self.text_font.render(f"SCORE: {self.score}", True, WHITE)
        self.screen.blit(score_text, (20, 10))
        
        # Lives
        lives_text = self.text_font.render(f"LIVES:", True, WHITE)
        self.screen.blit(lives_text, (SCREEN_WIDTH - 200, 10))
        
        for i in range(self.lives):
            pygame.draw.circle(self.screen, YELLOW,
                             (SCREEN_WIDTH - 120 + i * 30, 25), 10)
    
    
    def draw_controller_optimized(self):
        """Draw 4-button controller using pre-rendered sprites."""
        button_size = 80
        spacing = 20
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT - 100
        
        positions = [
            (center_x - button_size // 2, center_y - button_size - spacing, 0),  # UP
            (center_x - button_size // 2, center_y + spacing, 1),  # DOWN
            (center_x - button_size - spacing - button_size // 2, center_y, 2),  # LEFT
            (center_x + spacing + button_size // 2, center_y, 3)  # RIGHT
        ]
        
        for x, y, idx in positions:
            sprite = self.button_sprites[idx]['pressed' if self.button_pressed[idx] else 'normal']
            self.screen.blit(sprite, (x, y))

    
    def draw_start_screen(self):
        """Draw start screen."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        title = self.title_font.render("PAC-MAN", True, YELLOW)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 250))
        self.screen.blit(title, title_rect)
        
        instructions = [
            "Collect all dots!",
            "Avoid ghosts!",
            "",
            "Use 4-button controller:",
            "1=UP  2=DOWN  3=LEFT  4=RIGHT",
            "",
            "Press SPACE to Start"
        ]
        
        y = 350
        for line in instructions:
            text = self.small_font.render(line, True, WHITE)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, y))
            self.screen.blit(text, text_rect)
            y += 35
    
    def draw_game_over_screen(self):
        """Draw game over screen."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        title = self.title_font.render("GAME OVER", True, GHOST_RED)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 300))
        self.screen.blit(title, title_rect)
        
        score_text = self.text_font.render(f"Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, 380))
        self.screen.blit(score_text, score_rect)
        
        restart = self.text_font.render("Press R to Restart", True, WHITE)
        restart_rect = restart.get_rect(center=(SCREEN_WIDTH // 2, 450))
        self.screen.blit(restart, restart_rect)
    
    def draw_win_screen(self):
        """Draw win screen."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        title = self.title_font.render("YOU WIN!", True, YELLOW)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 300))
        self.screen.blit(title, title_rect)
        
        score_text = self.text_font.render(f"Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, 380))
        self.screen.blit(score_text, score_rect)
        
        restart = self.text_font.render("Press R to Play Again", True, WHITE)
        restart_rect = restart.get_rect(center=(SCREEN_WIDTH // 2, 450))
        self.screen.blit(restart, restart_rect)
    
    def reset_game(self):
        """Reset the game."""
        self.maze = [row[:] for row in MAZE]
        self.pacman = PacMan(10, 14)
        self.ghosts = [
            Ghost(9, 9, GHOST_RED, "Blinky"),
            Ghost(10, 9, GHOST_PINK, "Pinky"),
            Ghost(9, 10, GHOST_CYAN, "Inky"),
            Ghost(10, 10, GHOST_ORANGE, "Clyde")
        ]
        self.score = 0
        self.lives = 3
        self.dots_remaining = self.count_dots()
        self.game_state = "START"
        self.maze_dirty = True  # Force maze redraw
    
    def run(self):
        """Main game loop."""
        running = True
        
        print("="*60)
        print("Pac-Man - 4 Button Controller")
        print("="*60)
        print("Controls:")
        print("  1 (RED) - Move UP")
        print("  2 (GREEN) - Move DOWN")
        print("  3 (BLUE) - Move LEFT")
        print("  4 (YELLOW) - Move RIGHT")
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
                    
                    # 4-button controls
                    elif event.key == pygame.K_1 or event.key == pygame.K_UP:
                        self.pacman.set_direction((0, -1))
                        self.button_pressed[0] = True
                    elif event.key == pygame.K_2 or event.key == pygame.K_DOWN:
                        self.pacman.set_direction((0, 1))
                        self.button_pressed[1] = True
                    elif event.key == pygame.K_3 or event.key == pygame.K_LEFT:
                        self.pacman.set_direction((-1, 0))
                        self.button_pressed[2] = True
                    elif event.key == pygame.K_4 or event.key == pygame.K_RIGHT:
                        self.pacman.set_direction((1, 0))
                        self.button_pressed[3] = True
                
                elif event.type == pygame.KEYUP:
                    # Release button visuals
                    if event.key in [pygame.K_1, pygame.K_UP]:
                        self.button_pressed[0] = False
                    elif event.key in [pygame.K_2, pygame.K_DOWN]:
                        self.button_pressed[1] = False
                    elif event.key in [pygame.K_3, pygame.K_LEFT]:
                        self.button_pressed[2] = False
                    elif event.key in [pygame.K_4, pygame.K_RIGHT]:
                        self.button_pressed[3] = False
            
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


import math

if __name__ == "__main__":
    game = PacManGame()
    game.run()
