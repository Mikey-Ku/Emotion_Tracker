"""
Tetris - 4 Button Controller
=============================

Classic Tetris game with 4-button directional controls.

Controls (4-Button Controller Style):
- Button 1 (RED) = ROTATE
- Button 2 (GREEN) = MOVE DOWN (faster drop)
- Button 3 (BLUE) = MOVE LEFT
- Button 4 (YELLOW) = MOVE RIGHT

Or use arrow keys for testing.

Game Features:
- Classic Tetris gameplay
- 7 different tetromino shapes
- Line clearing
- Score and level system
- Increasing difficulty
"""

import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
GRID_WIDTH = 10
GRID_HEIGHT = 20
CELL_SIZE = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (40, 40, 40)
DARK_GRAY = (20, 20, 20)

# Button colors
RED = (220, 30, 30)
GREEN = (30, 220, 30)
BLUE = (30, 30, 220)
YELLOW = (220, 220, 30)

# Tetromino colors
CYAN = (0, 255, 255)
YELLOW_PIECE = (255, 255, 0)
PURPLE = (128, 0, 128)
GREEN_PIECE = (0, 255, 0)
RED_PIECE = (255, 0, 0)
BLUE_PIECE = (0, 0, 255)
ORANGE = (255, 165, 0)

# Tetromino shapes (I, O, T, S, Z, J, L)
SHAPES = {
    'I': [[1, 1, 1, 1]],
    'O': [[1, 1],
          [1, 1]],
    'T': [[0, 1, 0],
          [1, 1, 1]],
    'S': [[0, 1, 1],
          [1, 1, 0]],
    'Z': [[1, 1, 0],
          [0, 1, 1]],
    'J': [[1, 0, 0],
          [1, 1, 1]],
    'L': [[0, 0, 1],
          [1, 1, 1]]
}

SHAPE_COLORS = {
    'I': CYAN,
    'O': YELLOW_PIECE,
    'T': PURPLE,
    'S': GREEN_PIECE,
    'Z': RED_PIECE,
    'J': BLUE_PIECE,
    'L': ORANGE
}

# Game settings
FALL_SPEED = 500  # milliseconds between automatic drops


class Tetromino:
    """A falling tetromino piece."""
    
    def __init__(self, shape_type=None):
        if shape_type is None:
            shape_type = random.choice(list(SHAPES.keys()))
        
        self.shape_type = shape_type
        self.shape = [row[:] for row in SHAPES[shape_type]]
        self.color = SHAPE_COLORS[shape_type]
        self.x = GRID_WIDTH // 2 - len(self.shape[0]) // 2
        self.y = 0
    
    def rotate(self):
        """Rotate the piece 90 degrees clockwise."""
        # Transpose and reverse each row
        self.shape = [list(row) for row in zip(*self.shape[::-1])]
    
    def get_cells(self):
        """Get list of (x, y) coordinates of filled cells."""
        cells = []
        for y, row in enumerate(self.shape):
            for x, cell in enumerate(row):
                if cell:
                    cells.append((self.x + x, self.y + y))
        return cells


class TetrisGame:
    """Main Tetris game class."""
    
    def __init__(self):
        # Create screen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris - 4 Button Controller")
        
        # Clock
        self.clock = pygame.time.Clock()
        
        # Game grid
        self.grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        
        # Calculate grid offset to center it
        self.grid_offset_x = 50
        self.grid_offset_y = 50
        
        # Current piece
        self.current_piece = Tetromino()
        self.next_piece = Tetromino()
        
        # Game state
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.game_state = "START"  # START, PLAYING, PAUSED, GAME_OVER
        
        # Timing
        self.fall_timer = 0
        self.fall_speed = FALL_SPEED
        
        # Button press states
        self.button_pressed = [False, False, False, False]
        
        # Fonts
        self.title_font = pygame.font.Font(None, 60)
        self.text_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Pre-render button sprites
        self.button_sprites = self.create_button_sprites()
    
    def create_button_sprites(self):
        """Pre-render button sprites for better performance."""
        button_size = 80
        sprites = {}
        
        colors = [RED, GREEN, BLUE, YELLOW]
        labels = ["ROTATE", "DOWN", "LEFT", "RIGHT"]
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            # Normal state
            normal = pygame.Surface((button_size, button_size + 6), pygame.SRCALPHA)
            pygame.draw.rect(normal, (50, 50, 50), (0, 6, button_size, button_size), border_radius=10)
            pygame.draw.rect(normal, color, (0, 0, button_size, button_size), border_radius=10)
            pygame.draw.rect(normal, WHITE, (0, 0, button_size, button_size), 3, border_radius=10)
            
            font = pygame.font.Font(None, 18)
            text = font.render(label, True, WHITE)
            text_rect = text.get_rect(center=(button_size // 2, button_size // 2))
            normal.blit(text, text_rect)
            
            num_font = pygame.font.Font(None, 20)
            num_text = num_font.render(str(i + 1), True, (200, 200, 200))
            num_rect = num_text.get_rect(center=(button_size // 2, button_size - 15))
            normal.blit(num_text, num_rect)
            
            # Pressed state
            pressed = pygame.Surface((button_size, button_size + 6), pygame.SRCALPHA)
            pygame.draw.rect(pressed, (50, 50, 50), (0, 6, button_size, button_size), border_radius=10)
            pygame.draw.rect(pressed, color, (0, 4, button_size, button_size), border_radius=10)
            pygame.draw.rect(pressed, WHITE, (0, 4, button_size, button_size), 3, border_radius=10)
            
            text_rect = text.get_rect(center=(button_size // 2, button_size // 2 + 4))
            pressed.blit(text, text_rect)
            num_rect = num_text.get_rect(center=(button_size // 2, button_size - 11))
            pressed.blit(num_text, num_rect)
            
            sprites[i] = {'normal': normal, 'pressed': pressed}
        
        return sprites
    
    def can_move(self, piece, dx=0, dy=0, rotated_shape=None):
        """Check if piece can move to new position."""
        shape = rotated_shape if rotated_shape else piece.shape
        
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = piece.x + x + dx
                    new_y = piece.y + y + dy
                    
                    # Check boundaries
                    if new_x < 0 or new_x >= GRID_WIDTH or new_y >= GRID_HEIGHT:
                        return False
                    
                    # Check collision with placed pieces
                    if new_y >= 0 and self.grid[new_y][new_x] is not None:
                        return False
        
        return True
    
    def move_piece(self, dx, dy):
        """Move the current piece."""
        if self.can_move(self.current_piece, dx, dy):
            self.current_piece.x += dx
            self.current_piece.y += dy
            return True
        return False
    
    def rotate_piece(self):
        """Rotate the current piece."""
        # Save original shape
        original_shape = [row[:] for row in self.current_piece.shape]
        
        # Try rotation
        self.current_piece.rotate()
        
        # Check if rotation is valid
        if not self.can_move(self.current_piece):
            # Try wall kicks (move left/right to fit)
            if self.can_move(self.current_piece, dx=1):
                self.current_piece.x += 1
            elif self.can_move(self.current_piece, dx=-1):
                self.current_piece.x -= 1
            elif self.can_move(self.current_piece, dx=2):
                self.current_piece.x += 2
            elif self.can_move(self.current_piece, dx=-2):
                self.current_piece.x -= 2
            else:
                # Rotation failed, restore original
                self.current_piece.shape = original_shape
    
    def lock_piece(self):
        """Lock the current piece into the grid."""
        for x, y in self.current_piece.get_cells():
            if y >= 0:
                self.grid[y][x] = self.current_piece.color
        
        # Check for completed lines
        self.clear_lines()
        
        # Spawn new piece
        self.current_piece = self.next_piece
        self.next_piece = Tetromino()
        
        # Check game over
        if not self.can_move(self.current_piece):
            self.game_state = "GAME_OVER"
    
    def clear_lines(self):
        """Clear completed lines and update score."""
        lines_to_clear = []
        
        for y in range(GRID_HEIGHT):
            if all(cell is not None for cell in self.grid[y]):
                lines_to_clear.append(y)
        
        if lines_to_clear:
            # Remove cleared lines
            for y in sorted(lines_to_clear, reverse=True):
                del self.grid[y]
                self.grid.insert(0, [None for _ in range(GRID_WIDTH)])
            
            # Update score
            num_lines = len(lines_to_clear)
            self.lines_cleared += num_lines
            
            # Scoring: 1 line=100, 2=300, 3=500, 4=800
            scores = {1: 100, 2: 300, 3: 500, 4: 800}
            self.score += scores.get(num_lines, 100) * self.level
            
            # Update level (every 10 lines)
            self.level = self.lines_cleared // 10 + 1
            self.fall_speed = max(100, FALL_SPEED - (self.level - 1) * 50)
    
    def update(self, dt):
        """Update game state."""
        if self.game_state != "PLAYING":
            return
        
        # Auto-fall
        self.fall_timer += dt
        if self.fall_timer >= self.fall_speed:
            self.fall_timer = 0
            
            if not self.move_piece(0, 1):
                self.lock_piece()
    
    def draw(self):
        """Draw everything."""
        # Clear screen
        self.screen.fill(DARK_GRAY)
        
        # Draw grid background
        grid_bg = pygame.Rect(
            self.grid_offset_x,
            self.grid_offset_y,
            GRID_WIDTH * CELL_SIZE,
            GRID_HEIGHT * CELL_SIZE
        )
        pygame.draw.rect(self.screen, BLACK, grid_bg)
        pygame.draw.rect(self.screen, GRAY, grid_bg, 2)
        
        # Draw grid lines
        for x in range(GRID_WIDTH + 1):
            pygame.draw.line(
                self.screen, GRAY,
                (self.grid_offset_x + x * CELL_SIZE, self.grid_offset_y),
                (self.grid_offset_x + x * CELL_SIZE, self.grid_offset_y + GRID_HEIGHT * CELL_SIZE)
            )
        for y in range(GRID_HEIGHT + 1):
            pygame.draw.line(
                self.screen, GRAY,
                (self.grid_offset_x, self.grid_offset_y + y * CELL_SIZE),
                (self.grid_offset_x + GRID_WIDTH * CELL_SIZE, self.grid_offset_y + y * CELL_SIZE)
            )
        
        # Draw placed pieces
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x] is not None:
                    self.draw_cell(x, y, self.grid[y][x])
        
        # Draw current piece
        for x, y in self.current_piece.get_cells():
            if y >= 0:
                self.draw_cell(x, y, self.current_piece.color)
        
        # Draw UI
        self.draw_ui()
        
        # Draw controller
        self.draw_controller()
        
        # Draw overlays
        if self.game_state == "START":
            self.draw_start_screen()
        elif self.game_state == "PAUSED":
            self.draw_pause_screen()
        elif self.game_state == "GAME_OVER":
            self.draw_game_over_screen()
    
    def draw_cell(self, x, y, color):
        """Draw a single cell."""
        px = self.grid_offset_x + x * CELL_SIZE
        py = self.grid_offset_y + y * CELL_SIZE
        
        # Main cell
        pygame.draw.rect(self.screen, color, (px + 1, py + 1, CELL_SIZE - 2, CELL_SIZE - 2))
        
        # Highlight
        pygame.draw.rect(self.screen, WHITE, (px + 2, py + 2, CELL_SIZE - 4, CELL_SIZE - 4), 1)
    
    def draw_ui(self):
        """Draw UI elements."""
        ui_x = self.grid_offset_x + GRID_WIDTH * CELL_SIZE + 40
        
        # Score
        score_text = self.text_font.render("SCORE", True, WHITE)
        self.screen.blit(score_text, (ui_x, 50))
        score_val = self.text_font.render(str(self.score), True, YELLOW)
        self.screen.blit(score_val, (ui_x, 90))
        
        # Lines
        lines_text = self.small_font.render("LINES", True, WHITE)
        self.screen.blit(lines_text, (ui_x, 150))
        lines_val = self.text_font.render(str(self.lines_cleared), True, WHITE)
        self.screen.blit(lines_val, (ui_x, 180))
        
        # Level
        level_text = self.small_font.render("LEVEL", True, WHITE)
        self.screen.blit(level_text, (ui_x, 240))
        level_val = self.text_font.render(str(self.level), True, WHITE)
        self.screen.blit(level_val, (ui_x, 270))
        
        # Next piece
        next_text = self.small_font.render("NEXT", True, WHITE)
        self.screen.blit(next_text, (ui_x, 340))
        
        # Draw next piece preview
        preview_x = ui_x
        preview_y = 380
        for y, row in enumerate(self.next_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    px = preview_x + x * 25
                    py = preview_y + y * 25
                    pygame.draw.rect(self.screen, self.next_piece.color, (px, py, 23, 23))
                    pygame.draw.rect(self.screen, WHITE, (px, py, 23, 23), 1)
    
    def draw_controller(self):
        """Draw 4-button controller."""
        button_size = 80
        spacing = 20
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT - 100
        
        positions = [
            (center_x - button_size // 2, center_y - button_size - spacing, 0),  # ROTATE
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
        
        title = self.title_font.render("TETRIS", True, CYAN)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 250))
        self.screen.blit(title, title_rect)
        
        instructions = [
            "Clear lines to score!",
            "",
            "4-Button Controls:",
            "1=ROTATE  2=DOWN",
            "3=LEFT    4=RIGHT",
            "",
            "Press SPACE to Start"
        ]
        
        y = 350
        for line in instructions:
            text = self.small_font.render(line, True, WHITE)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, y))
            self.screen.blit(text, text_rect)
            y += 35
    
    def draw_pause_screen(self):
        """Draw pause screen."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
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
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        title = self.title_font.render("GAME OVER", True, RED)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 250))
        self.screen.blit(title, title_rect)
        
        score_text = self.text_font.render(f"Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, 350))
        self.screen.blit(score_text, score_rect)
        
        lines_text = self.text_font.render(f"Lines: {self.lines_cleared}", True, WHITE)
        lines_rect = lines_text.get_rect(center=(SCREEN_WIDTH // 2, 400))
        self.screen.blit(lines_text, lines_rect)
        
        restart = self.text_font.render("Press R to Restart", True, WHITE)
        restart_rect = restart.get_rect(center=(SCREEN_WIDTH // 2, 480))
        self.screen.blit(restart, restart_rect)
    
    def reset_game(self):
        """Reset the game."""
        self.grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = Tetromino()
        self.next_piece = Tetromino()
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.fall_speed = FALL_SPEED
        self.fall_timer = 0
        self.game_state = "START"
    
    def run(self):
        """Main game loop."""
        running = True
        
        print("="*60)
        print("Tetris - 4 Button Controller")
        print("="*60)
        print("Controls:")
        print("  1 (RED) - Rotate piece")
        print("  2 (GREEN) - Move down faster")
        print("  3 (BLUE) - Move left")
        print("  4 (YELLOW) - Move right")
        print("  SPACE - Start")
        print("  P - Pause")
        print("  R - Restart")
        print("  Q - Quit")
        print("="*60 + "\n")
        
        while running:
            dt = self.clock.tick(60)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    
                    elif event.key == pygame.K_SPACE and self.game_state == "START":
                        self.game_state = "PLAYING"
                    
                    elif event.key == pygame.K_p:
                        if self.game_state == "PLAYING":
                            self.game_state = "PAUSED"
                        elif self.game_state == "PAUSED":
                            self.game_state = "PLAYING"
                    
                    elif event.key == pygame.K_r and self.game_state == "GAME_OVER":
                        self.reset_game()
                    
                    elif self.game_state == "PLAYING":
                        # 4-button controls
                        if event.key in [pygame.K_1, pygame.K_UP]:
                            self.rotate_piece()
                            self.button_pressed[0] = True
                        elif event.key in [pygame.K_2, pygame.K_DOWN]:
                            self.move_piece(0, 1)
                            self.button_pressed[1] = True
                        elif event.key in [pygame.K_3, pygame.K_LEFT]:
                            self.move_piece(-1, 0)
                            self.button_pressed[2] = True
                        elif event.key in [pygame.K_4, pygame.K_RIGHT]:
                            self.move_piece(1, 0)
                            self.button_pressed[3] = True
                
                elif event.type == pygame.KEYUP:
                    if event.key in [pygame.K_1, pygame.K_UP]:
                        self.button_pressed[0] = False
                    elif event.key in [pygame.K_2, pygame.K_DOWN]:
                        self.button_pressed[1] = False
                    elif event.key in [pygame.K_3, pygame.K_LEFT]:
                        self.button_pressed[2] = False
                    elif event.key in [pygame.K_4, pygame.K_RIGHT]:
                        self.button_pressed[3] = False
            
            # Update
            self.update(dt)
            
            # Draw
            self.draw()
            pygame.display.flip()
        
        # Quit
        pygame.quit()
        print("\nGame ended!")
        print(f"Final Score: {self.score}")
        print(f"Lines Cleared: {self.lines_cleared}")
        print(f"Level Reached: {self.level}")


if __name__ == "__main__":
    game = TetrisGame()
    game.run()
